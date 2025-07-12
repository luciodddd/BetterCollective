import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

print("""As a baseline, I'm going to use a Kaplan-Meier survival estimator. 
    \n This model can predict churn for 0-60 months, not just a single probability: S(t)=P(still active at t).
    \n This model is good when there is right-centered data, as I checked for deposit and handles, so their tru churn time is probably higher than the observed
    \n It's fast and scalable and it's easy to interpret.
    \n I also considered Parametric survival models but I had to assume a distribution and Monthly logistic regression (“churn yes/no each month”) but it requires long-term data
    \n It sets a solid baseline: any feature-rich model built next must demonstrate that it predicts individual churn better than simply following the empirical survival curve.
""")

# -----------------------------------------------------------------
# 1. Load cleaned data ------------------------------------------------
#Path actual:
_current_dir = Path(__file__).parent.parent
_clean_data_path = _current_dir.parent / "data/cleaned_data_for_fcst.parquet"

_df = pd.read_parquet(_clean_data_path, engine="pyarrow")
print("Data loaded")

# ---------------------------------------------------------------
# 2. Build duration + event. 
# duration  = max months_active seen for each player
# event     = 1 if player churned (≥3-month gap to snapshot), 0 otherwise
_snapshot_period = _df["activity_month"].max().to_period("M")

_agg = (_df.groupby("account_id")
        .agg(duration=("months_active", "max"),
            last_month=("activity_month", "max")) # The most recent activity
        .reset_index())

# Months gap, how long ago was the last activity of the player, considering we are on the snapshot
_months_gap = (_snapshot_period.ordinal - _agg["last_month"].dt.to_period("M").astype(int))

_agg["event"] = (_months_gap >= 3).astype(int)
_durations = _agg["duration"].clip(lower=0)
_events    = _agg["event"]

# ---------------------------------------------------------------
# 3. Train/Test split.
print("I used 20 percent of the data to test, got randomly.")
_train_idx, _test_idx = train_test_split(
    _agg.index, test_size=0.2, stratify=_events, random_state=42
)

kmf = KaplanMeierFitter()
kmf.fit(_durations.loc[_train_idx], _events.loc[_train_idx], label="KM-baseline")

# ---------------------------------------------------------------
# 4. Evaluation
print("To evaluate I used the Concordance index. This is a rank-based metric that tells you how well a model's" \
" predicted scores order pairs of observations in the same way their true _outcomes are ordered." \
"\n It would be something like 'Given two players, how often does the one my model says is riskier actually experience the chrun sooner?'")
_c_index = concordance_index(
    _durations.loc[_test_idx],
    -kmf.predict(_durations.loc[_test_idx]),
    _events.loc[_test_idx]
)
# With this function 
def brier_at(_month):
    # scalar survival prob from KM model
    surv_prob = kmf.predict(_month)
    
    # 1 if the event (churn) actually happened by this horizon
    y_true = ((_durations.loc[_test_idx] <= _month) & (_events.loc[_test_idx] == 1)).astype(int)
    
    # same predicted event-probability for every player in the test fold
    y_pred = np.full_like(y_true, 1 - surv_prob, dtype=float)
    
    return brier_score_loss(y_true, y_pred)

scores = {m: brier_at(m) for m in (1, 3, 6, 12)}
## ---------- Brier Graph ----------
horizons = [1, 3, 6, 12]
briers   = [brier_at(h) for h in horizons]

plt.figure(figsize=(6, 4))
plt.bar([str(h) for h in horizons], briers)
plt.title("Brier Score at Selected Horizons")
plt.xlabel("Months ahead")
plt.ylabel("Brier score (lower = better)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.show()
# --------- EVALUATION CONCLUTIONS ----
print(f"Concordance index : {_c_index:.3f}", "\n This means the model orders players by churn-risk better than random (0.50) but far from perfect – leaves head-room for feature-rich models.")
print("Brier scores       :", scores,"\n The brier results show that long-term retention is more reliable to predict with this model than short-term")
print("""Early drop-off is severe - 40 % churn after month 0. interventions must target weeks 0-4.
\n Long-tail whales - a 2 % slice stays 30 + months.""")

# ---------------------------------------------------------------
# 5. Export survival curve (0–60 m)
print("I'll export the survival curve in .csv but I could export it any other way, upload it to a sql server, s3, etc.")
_timeline = np.arange(0, 61)
_out = (kmf.survival_function_at_times(_timeline)
        .reset_index()
        .rename(columns={"index": "month", "KM-baseline": "survival_prob"}))

_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts"
_ARTIFACT.mkdir(exist_ok=True)
_out.to_csv(_ARTIFACT / "km_baseline_survival.csv", index=False)
print("Saved as km_baseline_survival.csv")