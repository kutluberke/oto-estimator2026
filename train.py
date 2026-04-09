# train.py — RF + GradientBoosting + XGBoost ensemble trainer
#
# Usage:
#   from train import EnsembleTrainer, EnsemblePredictor
#   trainer = EnsembleTrainer()
#   trainer.train(X_train, y_train, X_test, y_test, feature_columns)
#   trainer.save()
#
#   predictor = EnsemblePredictor.load()
#   prices = predictor.predict(X_df)

import logging
import warnings
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from config import (
    RF_MODEL_PATH, GB_MODEL_PATH, XGB_MODEL_PATH,
    ENSEMBLE_WEIGHTS_PATH, FEATURE_IMP_PATH, FEATURE_COLUMNS_PATH,
    RF_PARAMS_NEW, GB_PARAMS, XGB_PARAMS,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# EnsembleTrainer
# ─────────────────────────────────────────────

class EnsembleTrainer:
    """Train RF + GradientBoosting + XGBoost on log1p(price) target.

    Weight optimisation uses an internal validation slice (last 15% of
    X_train / y_train) via scipy.optimize.minimize (SLSQP).
    """

    def __init__(self):
        self.rf  = RandomForestRegressor(**RF_PARAMS_NEW)
        self.gb  = GradientBoostingRegressor(**GB_PARAMS)
        self.xgb = XGBRegressor(**XGB_PARAMS)
        self.weights: np.ndarray = np.array([1/3, 1/3, 1/3])
        self.feature_columns: list = []
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_columns: list,
    ) -> dict:
        """Fit ensemble and optimise weights.  Returns metrics dict."""
        self.feature_columns = feature_columns

        # --- internal val split for weight optimisation ----------------
        val_size = max(1, int(len(X_train) * 0.15))
        X_tr  = X_train.iloc[:-val_size]
        y_tr  = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]

        logger.info("Fitting RF  (%d samples) ...", len(X_tr))
        self.rf.fit(X_tr, y_tr)

        logger.info("Fitting GB  (%d samples) ...", len(X_tr))
        self.gb.fit(X_tr, y_tr)

        logger.info("Fitting XGB (%d samples) ...", len(X_tr))
        self.xgb.fit(X_tr, y_tr)

        # --- optimise weights on val set --------------------------------
        val_preds = np.stack([
            self.rf.predict(X_val),
            self.gb.predict(X_val),
            self.xgb.predict(X_val),
        ], axis=1)  # (n_val, 3)

        def neg_r2(w):
            blended = val_preds @ w
            return -r2_score(y_val, blended)

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        bounds = [(0.0, 1.0)] * 3
        x0 = np.array([1/3, 1/3, 1/3])

        result = minimize(neg_r2, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        if result.success:
            self.weights = result.x
            logger.info(
                "Ensemble weights — RF: %.3f  GB: %.3f  XGB: %.3f",
                *self.weights,
            )
        else:
            logger.warning("Weight optimisation failed; using equal weights.")

        # --- refit on full train set (after weight optimisation) --------
        logger.info("Refitting on full training set (%d samples) ...", len(X_train))
        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)

        # --- evaluate ---------------------------------------------------
        train_pred_log = self._predict_log(X_train)
        test_pred_log  = self._predict_log(X_test)

        train_r2 = r2_score(y_train, train_pred_log)
        test_r2  = r2_score(y_test,  test_pred_log)

        train_prices = np.expm1(y_train)
        test_prices  = np.expm1(y_test)
        train_pred_prices = np.expm1(train_pred_log)
        test_pred_prices  = np.expm1(test_pred_log)

        train_mae  = mean_absolute_error(train_prices, train_pred_prices)
        test_mae   = mean_absolute_error(test_prices,  test_pred_prices)
        test_mape  = float(np.mean(np.abs(
            (test_prices - test_pred_prices) / (test_prices + 1e-9)
        ))) * 100

        if train_r2 - test_r2 > 0.15:
            logger.warning(
                "OVERFITTING DETECTED: train_R²=%.3f  test_R²=%.3f  gap=%.3f",
                train_r2, test_r2, train_r2 - test_r2,
            )

        self.metrics = {
            "train_r2":   round(train_r2,  4),
            "test_r2":    round(test_r2,   4),
            "train_mae":  round(train_mae,  0),
            "test_mae":   round(test_mae,   0),
            "test_mape":  round(test_mape,  2),
            "n_train":    len(X_train),
            "n_test":     len(X_test),
            "weights":    self.weights.tolist(),
        }

        logger.info(
            "Metrics — train R²: %.3f  test R²: %.3f  "
            "test MAE: ₺{:,.0f}  test MAPE: %.1f%%".format(test_mae),
            train_r2, test_r2, test_mape,
        )
        return self.metrics

    # ------------------------------------------------------------------
    def _predict_log(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack([
            self.rf.predict(X),
            self.gb.predict(X),
            self.xgb.predict(X),
        ], axis=1)
        return preds @ self.weights

    # ------------------------------------------------------------------
    def save(self):
        for path in [RF_MODEL_PATH, GB_MODEL_PATH, XGB_MODEL_PATH,
                     ENSEMBLE_WEIGHTS_PATH, FEATURE_IMP_PATH,
                     FEATURE_COLUMNS_PATH]:
            path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.rf,  RF_MODEL_PATH)
        joblib.dump(self.gb,  GB_MODEL_PATH)
        joblib.dump(self.xgb, XGB_MODEL_PATH)
        joblib.dump(self.weights, ENSEMBLE_WEIGHTS_PATH)
        joblib.dump(self.feature_columns, FEATURE_COLUMNS_PATH)

        self._save_feature_importances()
        logger.info("Models saved.")

    def _save_feature_importances(self):
        cols = self.feature_columns
        imp_rf  = self.rf.feature_importances_
        imp_gb  = self.gb.feature_importances_
        imp_xgb = self.xgb.feature_importances_

        # Average across models (equal weight for interpretability)
        avg_imp = (imp_rf + imp_gb + imp_xgb) / 3.0

        df_imp = pd.DataFrame({
            "feature":    cols,
            "importance": avg_imp,
            "rf_imp":     imp_rf,
            "gb_imp":     imp_gb,
            "xgb_imp":    imp_xgb,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        df_imp.to_csv(FEATURE_IMP_PATH, index=False)
        logger.info("Feature importances saved to %s", FEATURE_IMP_PATH)

    # ------------------------------------------------------------------
    def get_feature_importances(self) -> pd.DataFrame:
        cols = self.feature_columns
        avg = (
            self.rf.feature_importances_ +
            self.gb.feature_importances_ +
            self.xgb.feature_importances_
        ) / 3.0
        return (
            pd.DataFrame({"feature": cols, "importance": avg})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────
# EnsemblePredictor  (inference only)
# ─────────────────────────────────────────────

class EnsemblePredictor:
    """Load saved models and predict car prices."""

    def __init__(self, rf, gb, xgb, weights, feature_columns):
        self.rf  = rf
        self.gb  = gb
        self.xgb = xgb
        self.weights = weights
        self.feature_columns = feature_columns

    # ------------------------------------------------------------------
    @classmethod
    def load(cls) -> "EnsemblePredictor":
        for path, name in [
            (RF_MODEL_PATH,          "rf_model"),
            (GB_MODEL_PATH,          "gb_model"),
            (XGB_MODEL_PATH,         "xgb_model"),
            (ENSEMBLE_WEIGHTS_PATH,  "ensemble_weights"),
            (FEATURE_COLUMNS_PATH,   "feature_columns"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}. Run training first."
                )

        rf      = joblib.load(RF_MODEL_PATH)
        gb      = joblib.load(GB_MODEL_PATH)
        xgb     = joblib.load(XGB_MODEL_PATH)
        weights = joblib.load(ENSEMBLE_WEIGHTS_PATH)
        cols    = joblib.load(FEATURE_COLUMNS_PATH)
        return cls(rf, gb, xgb, weights, cols)

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted prices in TL (expm1 applied)."""
        X_aligned = X.reindex(columns=self.feature_columns, fill_value=0)
        log_preds = np.stack([
            self.rf.predict(X_aligned),
            self.gb.predict(X_aligned),
            self.xgb.predict(X_aligned),
        ], axis=1) @ self.weights
        return np.expm1(log_preds)

    # ------------------------------------------------------------------
    def get_feature_importances(self) -> pd.DataFrame:
        cols = self.feature_columns
        avg = (
            self.rf.feature_importances_ +
            self.gb.feature_importances_ +
            self.xgb.feature_importances_
        ) / 3.0
        return (
            pd.DataFrame({"feature": cols, "importance": avg})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return (
            self.rf is not None and
            self.gb is not None and
            self.xgb is not None and
            len(self.feature_columns) > 0
        )
