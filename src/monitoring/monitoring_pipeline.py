import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

# ----------------------------------------------------------------------------
# Configuration (this are sample parameters, I would fit it to match the firm necesities, and maybe I would leave parameters in a separate "parameter-table" on a sql or sth like that, where I can see all global parameter related to predictions and controls)
# ----------------------------------------------------------------------------
_METRICS_WINDOW_DAYS = 60  # size of rolling window for evaluation
_AUC_DROP_THRESHOLD = 0.05  # trigger retrain if AUC-ROC drops > 5 pp
_ECE_THRESHOLD = 0.05       # trigger recalibration if ECE > 0.05
_PSI_THRESHOLD = 0.2        # trigger segment retrain if PSI > 0.2

# Where batch scores + (eventual) labels are stored. The nightly batch-scoring job woul' dwrite today's scores to this file/table and back-fill the "label_quick_churn" column when the 3-month silence period has passed.
_PREDICTIONS_PATH = Path("artifacts/churn_predictions.parquet")
# Where _reference feature distributions from training are stored. This was not created in this exercise but would contain the _reference distribution for any feature I'd like to monitor for drift, like "total_deposit_log" for example.
__REF_DIST_PATH = Path("artifacts/ref_feature_dist.json")
# Where to save computed metrics. The script writes to this file, it appends one row per run with the fresh AUC, Brier, ECE, PSI, record-count, and date.
_METRICS_OUT = Path("artifacts/churn_model_metrics.parquet")

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def expected_calibration_error(_df: pd.DataFrame, n_bins: int = 10) -> float:
    """Compute ECE on probability column 'score' vs. true label 'label'.
        The goal of this function is answer: "When the model says 70 % churn-risk, are about 70% of those players actually churning?"
        * _p_pred: average predicted probability in the bin.
        * _p_true = fraction of real churners in the bin.
    """
    _df = _df.copy()
    _df["bucket"], bins = pd.qcut(_df["score"], q=n_bins, retbins=True, duplicates="drop")
    _grouped = _df.groupby("bucket", observed=False)
    _p_true = _grouped["label"].mean()
    _p_pred = _grouped["score"].mean()
    _n_per_bin = _grouped.size()
    _ece = (_n_per_bin * (_p_true - _p_pred).abs()).sum() / _n_per_bin.sum()
    return _ece


def psi_feature(_current: np.ndarray, _reference: np.ndarray, bins: int = 10) -> float:
    """Compute a simple Population Stability Index (approx. KL divergence).
    Purpose: Detect data drift. PSI (Population-Stability Index) tells whether today's feature distribution has shifted away from the training set.
    * _current: live data.
    * _reference: training data.
    """
    _edges = np.percentile(np.concatenate([_current, _reference]), q=np.linspace(0, 100, bins + 1)) #Build identical bin edges from the combined old + new samples
    _cur_counts, _ = np.histogram(_current, bins=_edges) #Count how many observations fall in each bin for reference and current data. Adds -1 smoothing avoids zero counts.
    _ref_counts, _ = np.histogram(_reference, bins=_edges)
    _cur_dist = (_cur_counts + 1) / (_cur_counts.sum() + bins) # Turns the counts into probabilities
    _ref_dist = (_ref_counts + 1) / (_ref_counts.sum() + bins)
    return float(entropy(_ref_dist, _cur_dist))  # KL divergence

# ----------------------------------------------------------------------------
# Main evaluation routine
# ----------------------------------------------------------------------------

def evaluate_once(_snapshot_date: Optional[datetime] = None) -> Tuple[pd.Series, pd.DataFrame]:
    """Evaluate model performance on the last '_METRICS_WINDOW_DAYS' days.
    Purpose: The core nightly/weekly job. Pulls the last N days of predictions that now have a label, computes all health metrics, appends them to a log, and returns a _headline summary plus per-segment PSI table.
    Returns
    -------
    * _headline : pd.Series: One-row Series with _headline metrics.
    * detail   : pd.DataFrame: Per-segment (partner/state) PSI table.
    """
    if _snapshot_date is None:
        _snapshot_date = datetime.now(timezone.utc).date()

    _start_date = _snapshot_date - timedelta(days=_METRICS_WINDOW_DAYS)

    # Load predictions table (must contain: account_id, partner_id, state, score_ts, score, label)
    _df = pd.read_parquet(_PREDICTIONS_PATH)

    # Filter fully-labelled rows within window
    _mask = (_df["score_ts"] >= pd.Timestamp(_start_date)) & (_df["score_ts"] <= pd.Timestamp(_snapshot_date)) #Keeps rows whose score_ts falls inside the 60-day window and already have a non-null label (the 3-month silence period has finished)
    _df_window = _df.loc[_mask & _df["label"].notna()].copy()

    if _df_window.empty:
        raise ValueError("No labelled rows available for evaluation window.")

    # -------------------------------------------------------------
    # Core global metrics (same metrics we used to evaluate the model trainig)
    # -------------------------------------------------------------
    _auc_roc = roc_auc_score(_df_window["label"], _df_window["score"])
    _auc_pr = average_precision_score(_df_window["label"], _df_window["score"])
    brier = brier_score_loss(_df_window["label"], _df_window["score"])
    ece = expected_calibration_error(_df_window)

    # -------------------------------------------------------------
    # Data-drift check (example on total_deposit_log)
    # -------------------------------------------------------------
    _ref_dist = json.loads(Path(__REF_DIST_PATH).read_text())
    _cur_feature = _df_window["feature_total_deposit_log"].to_numpy()
    psi = psi_feature(_cur_feature, np.array(_ref_dist["total_deposit_log"]))

    _headline = pd.Series({
        "snapshot": _snapshot_date,
        "window_days": _METRICS_WINDOW_DAYS,
        "_auc_roc": _auc_roc,
        "_auc_pr": _auc_pr,
        "brier": brier,
        "ece": ece,
        "psi_total_deposit_log": psi,
        "n_records": len(_df_window)
    })

    # -------------------------------------------------------------
    # Per-segment PSI
    # -------------------------------------------------------------
    _seg_psi = []
    for (partner, state), grp in _df_window.groupby(["partner_id", "state"]):
        cur = grp["feature_total_deposit_log"].to_numpy()
        _seg_psi.append({
            "snapshot": _snapshot_date,
            "partner_id": partner,
            "state": state,
            "psi_total_deposit_log": psi_feature(cur, np.array(_ref_dist["total_deposit_log"]))
        })
    seg__df = pd.DataFrame(_seg_psi)

    # Persist _headline metrics, it could be loaded in any format/connection:
    if _METRICS_OUT.exists():
        _all_metrics = pd.read_parquet(_METRICS_OUT)
        _all_metrics = pd.concat([_all_metrics, _headline.to_frame().T], ignore_index=True)
    else:
        _all_metrics = _headline.to_frame().T
    _all_metrics.to_parquet(_METRICS_OUT, index=False)

    return _headline, seg__df


if __name__ == "__main__": #Allows me to run the code later on Dagster
    _headline_metrics, _segment_detail = evaluate_once()
    print("\n_Headline metrics:\n", _headline_metrics)
    print("\nTop segments by PSI:\n", _segment_detail.sort_values("psi_total_deposit_log", ascending=False).head())
