# model_trainer.py — Model eğitimi, cross-validation ve değerlendirme

import json
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logging.warning("LightGBM yüklü değil. `pip install lightgbm` ile yükleyin.")

from config import (
    MODEL_PATH, FEATURE_INFO_PATH,
    LGBM_PARAMS_SMALL, LGBM_PARAMS_MEDIUM, LGBM_PARAMS_LARGE,
    RF_PARAMS,
)

logger = logging.getLogger(__name__)


def _select_lgbm_params(n_samples: int) -> dict:
    """Dataset büyüklüğüne göre LightGBM hiperparametrelerini seç."""
    if n_samples < 200:
        return LGBM_PARAMS_SMALL
    elif n_samples < 1000:
        return LGBM_PARAMS_MEDIUM
    else:
        return LGBM_PARAMS_LARGE


class ModelTrainer:
    """
    Birden fazla modeli eğitir, cross-validation ile karşılaştırır,
    en iyisini seçer ve diske kaydeder.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

        self.models: dict = {}
        self.cv_scores: dict = {}
        self.best_model_name: str = ""
        self.best_model = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names: list = []

        self.train_rmse: float = 0.0   # visualizer için
        self.test_metrics: dict = {}

    # ===== EĞİTİM =====

    @staticmethod
    def _adaptive_test_size(n: int) -> float:
        """
        Dataset büyüklüğüne göre test oranı:
          < 40  → 0.30  (küçük dataset'te daha az test örneği bırak)
          40–99 → 0.25
          ≥ 100 → 0.20
        Minimum 5, maksimum test sample garantisi de kontrol edilir.
        """
        if n < 40:
            return 0.30
        elif n < 100:
            return 0.25
        else:
            return 0.20

    def train(self, X: pd.DataFrame, y: pd.Series, feature_names: list) -> None:
        """
        Tüm pipeline:
        1. Train/test split (dataset büyüklüğüne adaptif)
        2. Birden fazla model CV
        3. En iyi modeli seç
        4. Test seti değerlendirmesi
        """
        self.feature_names = feature_names
        n = len(X)
        logger.info(f"Eğitim başlıyor: {n} örnek, {len(feature_names)} feature")

        # --- Train / Test split (adaptif) ---
        adaptive_test = self._adaptive_test_size(n)
        logger.info(f"Test size: {adaptive_test:.0%} ({int(n * adaptive_test)} örnek test, "
                    f"{int(n * (1 - adaptive_test))} örnek train)")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=adaptive_test,
            random_state=self.random_state,
        )

        # --- CV folds (küçük dataset'te fold sayısını düşür) ---
        cv_folds = 5 if len(self.X_train) >= 50 else (3 if len(self.X_train) >= 20 else 2)

        # =====================================================
        # MODEL 1: Ridge (Baseline — hızlı, interpretable)
        # =====================================================
        ridge = Ridge(alpha=1.0)
        ridge_cv = cross_val_score(
            ridge, self.X_train, self.y_train,
            cv=cv_folds, scoring="r2", n_jobs=-1,
        )
        self.cv_scores["Ridge"] = ridge_cv.mean()
        ridge.fit(self.X_train, self.y_train)
        self.models["Ridge"] = ridge
        logger.info(f"Ridge CV R² = {ridge_cv.mean():.4f} ± {ridge_cv.std():.4f}")

        # =====================================================
        # MODEL 2: Random Forest
        # =====================================================
        rf = RandomForestRegressor(**RF_PARAMS)
        rf_cv = cross_val_score(
            rf, self.X_train, self.y_train,
            cv=cv_folds, scoring="r2", n_jobs=-1,
        )
        self.cv_scores["RandomForest"] = rf_cv.mean()
        rf.fit(self.X_train, self.y_train)
        self.models["RandomForest"] = rf
        logger.info(f"RandomForest CV R² = {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

        # =====================================================
        # MODEL 3: LightGBM (Primary)
        # =====================================================
        if LGBM_AVAILABLE:
            lgbm_params = _select_lgbm_params(n)
            lgbm = LGBMRegressor(**lgbm_params)
            lgbm_cv = cross_val_score(
                lgbm, self.X_train, self.y_train,
                cv=cv_folds, scoring="r2", n_jobs=-1,
            )
            self.cv_scores["LightGBM"] = lgbm_cv.mean()
            lgbm.fit(self.X_train, self.y_train)
            self.models["LightGBM"] = lgbm
            logger.info(f"LightGBM CV R² = {lgbm_cv.mean():.4f} ± {lgbm_cv.std():.4f}")

        # --- En iyi modeli seç ---
        self.best_model_name = max(self.cv_scores, key=self.cv_scores.get)
        self.best_model = self.models[self.best_model_name]

        # --- Test seti değerlendirmesi ---
        self._evaluate_on_test()

        # --- Overfitting kontrolü ---
        self._check_overfitting()

        # --- Sonuçları yazdır ---
        self._print_results()

    # ===== DEĞERLENDİRME =====

    def _evaluate_on_test(self) -> None:
        y_pred = self.best_model.predict(self.X_test)
        r2   = r2_score(self.y_test, y_pred)
        mae  = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test.clip(lower=1))) * 100

        self.test_metrics = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }
        self.train_rmse = rmse  # visualizer için referans

    def _check_overfitting(self) -> None:
        cv_r2   = self.cv_scores[self.best_model_name]
        test_r2 = self.test_metrics["r2"]
        gap = cv_r2 - test_r2

        if gap > 0.25:
            logger.warning(
                f"⚠️  OVERFITTING: CV R²={cv_r2:.3f} → Test R²={test_r2:.3f} "
                f"(fark={gap:.3f}). "
                f"Daha fazla veri çek veya max_depth/min_data_in_leaf ayarla."
            )
        elif test_r2 < 0.50:
            logger.warning(
                f"⚠️  DÜŞÜK PERFORMANS: Test R²={test_r2:.3f}. "
                f"Veri kalitesini veya feature'ları kontrol et."
            )
        else:
            logger.info(f"✅ Model sağlıklı görünüyor (CV→Test gap={gap:.3f})")

    def _print_results(self) -> None:
        print(f"\n{'='*58}")
        print(f"  MODEL KARŞILAŞTIRMA (Cross-Validation R², {5}-Fold)")
        print(f"{'='*58}")
        sorted_scores = sorted(self.cv_scores.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_scores:
            marker = " ◀ SEÇILDI" if name == self.best_model_name else ""
            print(f"  {name:<18} R² = {score:.4f}{marker}")

        print(f"\n{'='*58}")
        print(f"  TEST SETİ SONUÇLARI  ({self.best_model_name})")
        print(f"{'='*58}")
        m = self.test_metrics
        print(f"  R² Skoru      : {m['r2']:.4f}  (1.0 = mükemmel)")
        print(f"  MAE           : {m['mae']:>12,.0f} TL  (ortalama mutlak hata)")
        print(f"  RMSE          : {m['rmse']:>12,.0f} TL  (kök ortalama kare hata)")
        print(f"  MAPE          : {m['mape']:.1f}%  (ortalama % hata)")
        print(f"{'='*58}\n")

    # ===== FEATURE IMPORTANCE =====

    def get_feature_importances(self) -> pd.DataFrame:
        """
        En iyi modelin feature importance değerlerini döner.
        Sütunlar: feature, importance
        """
        model = self.best_model
        if not hasattr(model, "feature_importances_"):
            logger.warning(f"{self.best_model_name} feature_importances_ desteklemiyor.")
            return pd.DataFrame()

        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return fi_df

    # ===== KAYDET / YÜKLE =====

    def save(self, path: str = MODEL_PATH) -> None:
        """En iyi modeli diske kaydet."""
        joblib.dump(self.best_model, path)
        logger.info(f"Model kaydedildi: {path}  ({self.best_model_name})")

        # Metadata JSON'a ekle
        try:
            with open(FEATURE_INFO_PATH, "r", encoding="utf-8") as f:
                info = json.load(f)
        except FileNotFoundError:
            info = {}

        info["best_model"]    = self.best_model_name
        info["test_metrics"]  = self.test_metrics
        info["cv_scores"]     = self.cv_scores
        info["feature_names"] = self.feature_names
        info["train_rmse"]    = self.train_rmse

        with open(FEATURE_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logger.info(f"Model metadata kaydedildi: {FEATURE_INFO_PATH}")

    @staticmethod
    def load_model(path: str = MODEL_PATH):
        """Kaydedilmiş modeli yükle."""
        model = joblib.load(path)
        logger.info(f"Model yüklendi: {path}")
        return model


# ===== TEST =====
if __name__ == "__main__":
    from feature_engineer import FeatureEngineer

    df = pd.read_csv("data/cleaned_data.csv")
    fe = FeatureEngineer()
    X, y, features = fe.fit_transform(df)

    trainer = ModelTrainer()
    trainer.train(X, y, features)
    trainer.save()

    fi = trainer.get_feature_importances()
    print("\nFeature Importance:")
    print(fi.to_string(index=False))
