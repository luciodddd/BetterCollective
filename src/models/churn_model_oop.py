"""churn_model_oop.py . Object-oriented training & inference wrapper for partner-specific early-churn models.
Usage (train):
    python churn_model_oop.py
        --input  data/cleaned_data_for_fcst.parquet
        --target quick_churn
        --output artifacts/AZ01

Usage (score):
    from churn_model_oop import EarlyChurnModel
    model = EarlyChurnModel.load("artifacts/AZ01")
    probs = model.predict_proba(_new_df)

    Internal functions, simbolized with a initial underscore, are at the bottom of the class.
"""

from __future__ import annotations
import argparse
import json
import joblib
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import  average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# Core OOP wrapper
# -----------------------------------------------------------------------------
class EarlyChurnModel:
    """Encapsulates preprocessing + XGBoost model in a single object."""

    def __init__( #Constructor to check variables. I also added an alternative constructor which is simpler to use as you only need to pass a dataframe with data to it.
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
        xgb_params: Optional[dict] = None,
    ) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.xgb_params = xgb_params or { #These are the parameters I used before and could be set as default if no new parameters are passed to the object. It gives flexibility and I could store params in a common file and read them as default instead or hardcoding them in the class definition
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": 42,
        }
        self.pipeline: Optional[Pipeline] = None   #Will hold the scikit-learn Pipeline (preprocessing + XGBoost) once we fit. Starts as None of course.

    # ----------------------- public API -------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EarlyChurnModel":
        """Train the pipeline on provided data."""
        self.pipeline = self._build_pipeline() # Builds the XGBClassifier pipeline
        self.pipeline.fit(X[self.all_cols], y) # Trains it on the provided data
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._assert_fitted()
        return self.pipeline.predict_proba(X[self.all_cols])[:, 1] #Predicts, returns churn probabilities.

    def save(self, output_dir: Union[str, Path]) -> None:
        """Persist model + metadata to 'output_dir'."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, output_dir / "pipeline.joblib")
        meta = {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "xgb_params": self.xgb_params,
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "EarlyChurnModel":   # -- ALTERNATIVE CONSTRUCTOR
        """Load a saved model from 'model_dir'."""
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        obj = cls(meta["numeric_cols"], meta["categorical_cols"], meta["xgb_params"])
        obj.pipeline = joblib.load(model_dir / "pipeline.joblib")
        return obj

    # ----------------------- internal helpers ------------------------------
    @property
    def all_cols(self) -> List[str]:
        return self.numeric_cols + self.categorical_cols

    def _build_pipeline(self) -> Pipeline:
        pre = ColumnTransformer(
            [("cat",OneHotEncoder(handle_unknown="ignore"),self.categorical_cols,)],
            remainder="passthrough",
        )
        clf = XGBClassifier(**self.xgb_params)
        return Pipeline([("prep", pre), ("clf", clf)])

    def _assert_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call .fit() first or load a saved model.")


# -----------------------------------------------------------------------------
# Command-line utility to train & save per-partner model
# -----------------------------------------------------------------------------
_DEF_NUMERIC = [
    "total_deposit_log",
    "total_handle_log",
    "abs_ngr_log",
    "has_deposit",
    "has_handle",
]
_DEF_CATEGORICAL = ["brand_id", "ben_login_id", "player_reg_product"]


# This allows to use the class and train-save a partner-specific model just by running a shell command.
def main() -> None: #It return none, as it simply stores metrics, pipeline and predictions.
    parser = argparse.ArgumentParser(description="Train partner-specific early churn model")
    parser.add_argument("--input", required=True, help="Path to cleaned parquet with features & target") #Required, path to new data's df
    parser.add_argument("--target", default="quick_churn", help="Name of target column")
    parser.add_argument("--output", required=True, help="Directory to save model artefacts") #Required, path to save model artifacts
    parser.add_argument("--numeric", nargs="*", default=_DEF_NUMERIC, help="Numeric feature columns")
    parser.add_argument("--categorical", nargs="*", default=_DEF_CATEGORICAL, help="Categorical feature columns")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    y = df[args.target]
    X = df[args.numeric + args.categorical]

    model = EarlyChurnModel(args.numeric, args.categorical) 
    model.fit(X, y)

    # Basic training metrics on full data (for quick sanity-check)
    prob = model.predict_proba(X)
    metrics = {
        "auc_roc": roc_auc_score(y, prob),
        "auc_pr": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob),
        "n_train": len(y),
    }
    print("Training metrics:", metrics)

    output_dir = Path(args.output)
    model.save(output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Model artefacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
