# ---------------------------------------------------------------
# Imports & config
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,confusion_matrix, classification_report)
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

print(""" NOTE: I'll be using an Early-churn classifier, XGBoost model. Since it:
      * Handles a mix of categorical and int values
      * Captures non-linear interactions. For example a $0 deposit vs. $10 has a different meaning than $500 vs. $510.
      * It's faster than Random Forest to handle class imbalances, and the data has a mild imbalances, as I defined quick churn as snapshot - last activity  <= 3 months.
      * For small datasets XGBoost has a good performance even with cross-validation.
      * I also considered Random Forest and Logistic Regression
      * As the goal is to predict early churn, in the first months of activity, I create a 'early-churn model'.
    """)

SEED = 42 #to split the same data. I would normally create files for parameters common in multiple .py files, so there is always a tracking and never a hardcoding
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------
# 1  Load cleaned monthly data
_current_dir = Path(__file__).parent.parent
_clean_data_path = _current_dir.parent / "data/cleaned_data_for_fcst.parquet"
_df = pd.read_parquet(_clean_data_path, engine="pyarrow")
print("Data loaded")

# ---------------------------------------------------------------
# 2  Label definition  ➟  “quick churn = 1 if player is gone by month-3”. I will keep the churn definition I used at the beginning
# ---------------------------------------------------------------
_snap = _df["activity_month"].max().to_period("M")

# compute lifetime & churn flag again (≥3-month _gap ⇒ churn)
_agg = (_df.groupby("account_id")
        .agg(lifetime=("months_active", "max"), #In case the player has many activity months (entries)
            last_month=("activity_month", "max"),
            first_month=("activity_month", "min"))
        .reset_index())

#Inactivity period:
_gap = _snap.ordinal - _agg["last_month"].dt.to_period("M").astype(int) 
_agg["churned"]      = (_gap >= 3).astype(int)         # 1 = event occurred
_agg["quick_churn"]  = ((_agg["churned"] == 1) & (_agg["lifetime"] <= 3)).astype(int) #Quick churn means tje player didn't use the sportsbook for many periods

print(_agg[["lifetime", "churned", "quick_churn"]].head())

# ---------------------------------------------------------------
# 3  Feature table 
# ---------------------------------------------------------------
print("""I want to predict churn as soon as a player's first month closes ('early-warning'). All later months would leak future information, so I discard them.""")
_first_rows = (_df.sort_values(["account_id", "activity_month"])
                .groupby("account_id").head(1)             # earliest row
                .merge(_agg[["account_id", "quick_churn"]], on="account_id")) # Add "quick_churn", the target variable...

# ---- Basic engineered features
_first_rows["has_deposit"]      = (_first_rows["total_deposit"]  > 0).astype(int)
_first_rows["has_handle"]       = (_first_rows["total_handle"]   > 0).astype(int)
print("""Deposits and handle are extremely right-skewed. Log-scale compresses the tail so large outliers don't dominate gradient boosting.""")
_first_rows["total_deposit_log"] = np.log1p(_first_rows["total_deposit"])
_first_rows["total_handle_log"]  = np.log1p(_first_rows["total_handle"])
_first_rows["abs_ngr_log"]       = np.sign(_first_rows["total_ngr"]) * np.log1p(np.abs(_first_rows["total_ngr"]))

_feat_cols_num  = ["total_deposit_log", "total_handle_log", "abs_ngr_log", "has_deposit", "has_handle"]
feat_cols_cat  = ["brand_id", "ben_login_id", "player_reg_product"]

X = _first_rows[_feat_cols_num + feat_cols_cat]
y = _first_rows["quick_churn"]

# ---------------------------------------------------------------
# 4  Train / test split (at player-level). I use the same seed than on the baseline training.
# ---------------------------------------------------------------
_X_train, _X_test, _y_train, _y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# ---------------------------------------------------------------
# 5  Pre-processing + model pipeline
# ---------------------------------------------------------------
_pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), feat_cols_cat)], # I added handle_unknown="ignore" so unseen categories don't crash the model
    remainder="passthrough"
)

_clf = XGBClassifier(
    n_estimators=400, #boosting rounds
    max_depth=4, #shallow trees to avoid over-fitting
    learning_rate=0.05, #small step size, more trees but smoother fit
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic", #0/1
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1
)

_pipe = Pipeline(
    steps=[
        ("prep", _pre),
        ("clf",  _clf)
    ]
)
_pipe.fit(_X_train, _y_train)

# ---------------------------------------------------------------
# 6  Evaluation
# ---------------------------------------------------------------
_proba_test = _pipe.predict_proba(_X_test)[:, 1] # column 1 = P(quick-churn)
_pred_test  = (_proba_test >= 0.5).astype(int) # The threshold 0.50 is a common starting point; it could be moved later on to favour recall or precision later.
print(""" To evaluate it I analyzed the Receiver-Operating-Characteristic, which features True-Positive Rate (TPR) in the X-axis and False-Positive Rate (FPR) in the Y-axis and its trade-off.
    \n Bad models tend to hug the diagonal line while good models tend to  climb quickly toward the top-left corner (high TPR, low FPR). I used AUC-ROC and AUC-PR to check this.
    * AUC-ROC: This is equivalent to the probability the model ranks a random positive higher than a random negative. This is treshold-independent as we don't use it in the formula.
    * AUC-PR: Is the average precision over all recall levels, tends to be more informative than ROC when classes are imbalanced. Shows how well the model captures true churners without flooding with false positives.
""")
_auc_roc = roc_auc_score(_y_test, _proba_test) #Random guessing is close to 0.5, perfect model is close to 1.0
_auc_pr  = average_precision_score(_y_test, _proba_test)

print(f"AUC-ROC : {_auc_roc:.3f}")
print(f"AUC-PR  : {_auc_pr :.3f}\n")
print("Confusion matrix (0.5 threshold):")
print(confusion_matrix(_y_test, _pred_test))
print("\nDetailed report:")
print(classification_report(_y_test, _pred_test, digits=3))

# Graph ROC Curve (saves as "ROC_Curve.png"):
_fpr, _tpr, thresh = roc_curve(_y_test, _proba_test)
roc_auc = auc(_fpr, _tpr)
plt.figure(figsize=(6, 4))
plt.plot(_fpr, _tpr, label=f"XGBoost (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess (AUC = 0.5)")
plt.xlabel("False-positive rate")
plt.ylabel("True-positive rate")
plt.title("ROC curve - early-churn classifier")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(""" Some conclutions of the evaluation:
    * ROC-curve bows clearly above the grey diagonal, which means the model ranks churners far better than random across every threshold.
    * AUC-ROC = 0.770. Means 77% of the time the model will rank a churner ahead of a non-churner. A solid discrimination for a first-month-only model, but could be improved.
    * On the confusion matrix. I got:
        - True 0 (stayers): 2461 correctly kept, 758 false alarms.
        - True 1 (quick-churn): 1927 caught, 1193 missed.
    * Precision = 71.8%. When the model identifies a churn, it's right almost 72% of the time.
    * Compared to baseline KM, that can't rank individuals (implicit AUC = 0.50). This model lifts AUC-ROC to 0.77 and AUC-PR to 0.78, demonstrating clear value from using first-month behaviour.
    * At the default 0.5 threshold the model identifies ~62% of churners while keeping precision at ~ 72%. Marketing could intervene early with targeted promos, accepting ~ 24% false alarms among stayers.
    * If the cost of missing a churner if high (because it's really hard to get new players for example) I could lower the threshold, accepting more false-positives.
    """)

# ---------------------------------------------------------------
# 7  Save artefacts
# ---------------------------------------------------------------
_art = Path(__file__).resolve().parents[1] / "artifacts"
_art.mkdir(exist_ok=True)

# Store the Pipepline and results. I coudl also store it in a repository, sql server, s3, etc. I'm doing it locally for the excersice.
_pipe.named_steps["_clf"].save_model(_art / "xgb_quick_churn.model")
pd.Series(_feat_cols_num + list(_pipe.named_steps["prep"]
                            .get_feature_names_out())).to_csv(_art / "xgb_feature_list.txt", index=False, header=False)
print("\n Model + feature list saved in artifacts")