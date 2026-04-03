# feature_engineer.py — Feature engineering, encoding ve scaling
#
# Arabam.com listing kartlarında mevcut olan feature'lar:
#   km, year, model, paket
# Detay sayfasından eklenen feature'lar (opsiyonel - NaN olabilir):
#   errors (hata), repaints (boya), changed_parts (değişen), heavy_damage (ağır hasar)

import json
import joblib
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import TARGET_COLUMN, DATA_BOUNDS, ENGINEER_PATH, FEATURE_INFO_PATH

logger = logging.getLogger(__name__)

# Modele verilecek feature listesi (sabit sıra önemli)
FINAL_FEATURES = [
    "km",
    "year",
    "model_enc",
    "paket_enc",
    "errors",
    "repaints",
    "changed_parts",
    "heavy_damage",
]

# Opsiyonel kondisyon feature'ları — NaN → medyan ile doldurulur
CONDITION_FEATURES = ["errors", "repaints", "changed_parts", "heavy_damage"]


class FeatureEngineer:
    """
    Temiz DataFrame'i ML-ready X, y tensörlerine dönüştürür.
    fit_transform() → eğitim verisi üzerinde öğren + dönüştür
    transform()     → tahmin zamanı aynı dönüşümü uygula
    """

    def __init__(self):
        self.label_encoders: dict = {}
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self._is_fitted: bool = False
        self._rare_threshold = DATA_BOUNDS["rare_location_threshold"]
        self._rare_categories: dict = {}
        self._numeric_medians: dict = {}

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _fill_numeric_nulls(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        fill_cols = ["km", "year"] + CONDITION_FEATURES
        for col in fill_cols:
            if col not in df.columns:
                continue
            if fit:
                median = df[col].median()
                # Eğer sütunun tamamı NaN ise 0 kullan
                self._numeric_medians[col] = median if not pd.isna(median) else 0
            df[col] = df[col].fillna(self._numeric_medians.get(col, 0))
        return df

    def _group_rare_categories(self, df: pd.DataFrame, col: str, fit: bool) -> pd.DataFrame:
        if col not in df.columns:
            return df
        df = df.copy()
        df[col] = df[col].fillna("Bilinmiyor").astype(str)
        if fit:
            counts = df[col].value_counts()
            self._rare_categories[col] = counts[counts < self._rare_threshold].index.tolist()
        rare = self._rare_categories.get(col, [])
        df[col] = df[col].apply(lambda x: "Diğer" if x in rare else x)
        return df

    def _encode(self, df: pd.DataFrame, col: str, out_col: str, fit: bool) -> pd.DataFrame:
        if col not in df.columns:
            df[out_col] = 0
            return df
        df = df.copy()
        df[col] = df[col].fillna("Bilinmiyor").astype(str)
        if fit:
            le = LabelEncoder()
            le.fit(df[col])
            self.label_encoders[col] = le
            df[out_col] = le.transform(df[col])
        else:
            le = self.label_encoders.get(col)
            if le is None:
                df[out_col] = 0
            else:
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                df[out_col] = le.transform(df[col])
        return df

    # ── Ana fonksiyonlar ─────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame):
        """
        Eğitim verisi üzerinde öğren + dönüştür.
        Döner: X (DataFrame), y (Series), feature_names (list)
        """
        logger.info("FeatureEngineer: fit_transform başlıyor")
        df = df.copy()

        # Kondisyon sütunları yoksa sıfırla oluştur
        for col in CONDITION_FEATURES:
            if col not in df.columns:
                df[col] = float("nan")

        df = self._fill_numeric_nulls(df, fit=True)

        for col in ["model", "paket"]:
            df = self._group_rare_categories(df, col, fit=True)

        df = self._encode(df, "model", "model_enc", fit=True)
        df = self._encode(df, "paket", "paket_enc", fit=True)

        # Sadece mevcut sütunlarla feature listesi oluştur
        features = [f for f in FINAL_FEATURES if f in df.columns]

        self.feature_names = features
        self._is_fitted = True

        X = df[features].copy()
        y = df[TARGET_COLUMN].copy()

        condition_found = df[CONDITION_FEATURES].notna().any(axis=1).sum()
        logger.info(
            f"Feature engineering tamamlandı. Features: {features} | "
            f"Kondisyon verisi olan ilan: {condition_found}/{len(df)}"
        )
        return X, y, features

    def transform(self, input_dict: dict) -> pd.DataFrame:
        """
        Tahmin zamanı: kullanıcı input dict'ini feature DataFrame'e çevir.
        input_dict → {
            'year': 2021, 'km': 50000,
            'model': 'Tonale', 'paket': '1.5 Hybrid Veloce',
            'errors': 0, 'repaints': 1, 'changed_parts': 2, 'heavy_damage': 0  # opsiyonel
        }
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer fit edilmedi. Önce fit_transform çalıştır.")

        df = pd.DataFrame([input_dict])

        # Kondisyon sütunları yoksa NaN bırak — medyan ile doldurulacak
        for col in CONDITION_FEATURES:
            if col not in df.columns:
                df[col] = float("nan")

        df = self._fill_numeric_nulls(df, fit=False)

        for col in ["model", "paket"]:
            df = self._group_rare_categories(df, col, fit=False)

        df = self._encode(df, "model", "model_enc", fit=False)
        df = self._encode(df, "paket", "paket_enc", fit=False)

        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0

        return df[self.feature_names]

    # ── Kaydet / Yükle ───────────────────────────────────────────────────────

    def save(self, path: str = ENGINEER_PATH) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"FeatureEngineer kaydedildi: {path}")

        info = {
            "feature_names": self.feature_names,
            "label_encoders": {
                col: list(le.classes_)
                for col, le in self.label_encoders.items()
            },
        }
        with open(FEATURE_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = ENGINEER_PATH) -> "FeatureEngineer":
        return joblib.load(path)


# ── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    fe = FeatureEngineer()
    X, y, features = fe.fit_transform(df)
    print(f"Features: {features}")
    print(f"X shape : {X.shape}")
    print(X.head(3))
