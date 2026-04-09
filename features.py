# features.py — Feature engineering: OHE, log1p target, time-sorted split
#
# Pipeline:
#   fit_transform(df) → X_train, X_test, y_train, y_test, feature_columns
#   transform(input_dict) → X (single-row DataFrame for prediction)
#
# Categorical encoding: one-hot (drop_first=True) — prevents dummy trap
# Target: log1p(price) — inverse: expm1(prediction)
# Split: sort by scrape_timestamp, last 15% = test (no random shuffle)

import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

from config import (
    TARGET_COLUMN, DATA_BOUNDS,
    ENGINEER_PATH, FEATURE_COLUMNS_PATH, MODEL_DIR,
)

logger = logging.getLogger(__name__)

# ── Renk paleti: ilk 15, geri kalanı "Diğer" ──────────────────────────────
TOP_N_COLORS = 15

# Damage feature isimleri (feature importance'da kırmızı renk için)
DAMAGE_FEATURES = [
    "total_damage_score",
    "painted_panel_count",
    "changed_panel_count",
    "has_original_paint",
    "has_local_paint",
    "damage_x_age",
    "damage_x_km",
    "paint_x_brand_median",
]


class FeatureEngineer:
    """
    Temiz DataFrame'i ML-ready X, y tensörlerine dönüştürür.

    fit_transform(df) → (X_train, X_test, y_train, y_test, feature_cols)
    transform(input_dict) → single-row DataFrame, production-ready
    """

    def __init__(self):
        self._is_fitted: bool = False

        # Fit sırasında öğrenilen şeyler
        self._top_colors: List[str] = []
        self._ohe_columns: List[str] = []    # pd.get_dummies sonrası sütunlar
        self._feature_columns: List[str] = []  # nihai model feature listesi
        self._brand_median_damage: dict = {}   # brand → median total_damage_score
        self._hp_medians: dict = {}            # (brand, model) → median hp
        self._global_hp_median: float = 100.0
        self._numeric_medians: dict = {}       # genel numerik medyanlar

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _build_numeric(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Numerik feature'ları hesapla ve eksikleri doldur."""
        df = df.copy()

        # km_log
        df["km"] = pd.to_numeric(df["km"], errors="coerce")
        df["km_log"] = np.log1p(df["km"].clip(lower=0))

        # year → car_age, year_sq
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["car_age"] = 2025 - df["year"]
        df["year_sq"] = df["year"] ** 2

        # Hasar alanları: eksik → 0
        damage_raw = [
            "total_damage_score", "painted_panel_count", "changed_panel_count",
            "has_original_paint", "has_local_paint",
        ]
        for col in damage_raw:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # total_damage_score: yeniden hesapla (tutarlılık için)
        if "painted_panel_count" in df.columns and "changed_panel_count" in df.columns:
            computed = df["painted_panel_count"] + df["changed_panel_count"] * 2
            # NaN kalırsa 0 yaz
            df["total_damage_score"] = computed.fillna(0)

        # Boolean: warranty, from_dealer
        for col in ["warranty", "from_dealer"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # hp: brand-model medyan ile doldur
        df["hp"] = pd.to_numeric(df.get("hp", pd.Series(dtype=float)), errors="coerce")
        if fit:
            if "brand" in df.columns and "model" in df.columns:
                bm_median = (
                    df.groupby(["brand", "model"])["hp"]
                    .median()
                    .to_dict()
                )
                self._hp_medians = bm_median
            self._global_hp_median = df["hp"].median()
            if pd.isna(self._global_hp_median):
                self._global_hp_median = 100.0

        if "brand" in df.columns and "model" in df.columns:
            for idx, row in df[df["hp"].isna()].iterrows():
                key = (row.get("brand"), row.get("model"))
                med = self._hp_medians.get(key)
                df.at[idx, "hp"] = med if pd.notna(med) else self._global_hp_median
        df["hp"] = df["hp"].fillna(self._global_hp_median)

        # Interaction features
        df["damage_x_age"] = df["total_damage_score"] * df["car_age"]
        df["damage_x_km"] = df["total_damage_score"] * df["km_log"]

        # paint_x_brand_median: damage_score / brand_median_damage
        if fit:
            if "brand" in df.columns:
                self._brand_median_damage = (
                    df.groupby("brand")["total_damage_score"].median().to_dict()
                )
        if "brand" in df.columns:
            brand_med = df["brand"].map(self._brand_median_damage).fillna(1.0)
            brand_med = brand_med.replace(0, 1.0)
            df["paint_x_brand_median"] = df["total_damage_score"] / brand_med
        else:
            df["paint_x_brand_median"] = df["total_damage_score"]

        # Genel numerik medyanlar (km, car_age vb.)
        for col in ["km_log", "car_age", "year_sq", "hp",
                    "total_damage_score", "painted_panel_count", "changed_panel_count",
                    "damage_x_age", "damage_x_km", "paint_x_brand_median"]:
            if fit:
                med = df[col].median()
                self._numeric_medians[col] = med if pd.notna(med) else 0.0
            df[col] = df[col].fillna(self._numeric_medians.get(col, 0.0))

        return df

    def _build_categorical(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Kategorik sütunları OHE ile encode et."""
        df = df.copy()

        # Renk: top 15, geri kalanı "Diğer"
        if "color" not in df.columns:
            df["color"] = "Bilinmiyor"
        df["color"] = df["color"].fillna("Bilinmiyor").astype(str)
        if fit:
            top_colors = df["color"].value_counts().head(TOP_N_COLORS).index.tolist()
            self._top_colors = top_colors
        df["color"] = df["color"].apply(
            lambda x: x if x in self._top_colors else "Diğer"
        )

        # Kategorik sütunlar
        cat_cols = ["brand", "fuel_type", "transmission", "body_type", "location", "color"]
        for col in cat_cols:
            if col not in df.columns:
                df[col] = "Bilinmiyor"
            df[col] = df[col].fillna("Bilinmiyor").astype(str)

        if fit:
            df_ohe = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
            self._ohe_columns = list(df_ohe.columns)
            df = pd.concat([df.drop(columns=cat_cols), df_ohe], axis=1)
        else:
            df_ohe = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
            # Eğitimde gördüğü sütunları garantile, fazlaları at
            for col in self._ohe_columns:
                if col not in df_ohe.columns:
                    df_ohe[col] = 0.0
            df_ohe = df_ohe[self._ohe_columns]
            df = pd.concat([df.drop(columns=cat_cols, errors="ignore"), df_ohe], axis=1)

        return df

    # ── Ana fonksiyonlar ─────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Eğitim verisi üzerinde öğren, train/test olarak split et ve döndür.

        Dönüş: X_train, X_test, y_train, y_test, feature_columns
        """
        logger.info(f"FeatureEngineer fit_transform: {len(df)} satır")
        df = df.copy()

        # ── Zaman-sıralı train/test split (leak-free) ──────────────────────
        if "scrape_timestamp" in df.columns:
            df = df.sort_values("scrape_timestamp").reset_index(drop=True)
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Split: {len(train_df)} train, {len(test_df)} test")

        # ── Target: log1p(price) ───────────────────────────────────────────
        y_train = np.log1p(train_df[TARGET_COLUMN].astype(float))
        y_test = np.log1p(test_df[TARGET_COLUMN].astype(float))

        # ── Numeric + interaction features (fit on train only) ─────────────
        train_df = self._build_numeric(train_df, fit=True)
        test_df = self._build_numeric(test_df, fit=False)

        # ── Categorical OHE (fit on train only) ────────────────────────────
        train_df = self._build_categorical(train_df, fit=True)
        test_df = self._build_categorical(test_df, fit=False)

        # ── Feature sütunlarını seç ────────────────────────────────────────
        drop_cols = {
            TARGET_COLUMN, "price", "title", "paket", "detail_url",
            "km", "year", "scrape_timestamp",
            "errors", "repaints", "changed_parts", "heavy_damage",
            "engine_cc",  # string, model için uygun değil
        }
        feature_cols = [
            c for c in train_df.columns
            if c not in drop_cols and train_df[c].dtype != object
        ]
        self._feature_columns = feature_cols
        self._is_fitted = True

        X_train = train_df[feature_cols].astype(float)
        X_test = test_df[feature_cols].astype(float)

        logger.info(
            f"Feature engineering tamamlandı. "
            f"Features: {len(feature_cols)} | "
            f"Train: {X_train.shape} | Test: {X_test.shape}"
        )

        return X_train, X_test, y_train, y_test, feature_cols

    def transform(self, input_dict: dict) -> pd.DataFrame:
        """
        Tahmin zamanı: tek araç input dict'ini feature DataFrame'e çevir.

        input_dict anahtarları (zorunlu):
            brand, year, km, fuel_type, transmission, body_type, location
        Opsiyonel:
            hp, color, warranty, from_dealer,
            has_original_paint, painted_panel_count, changed_panel_count,
            has_local_paint, total_damage_score
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer henüz fit edilmedi.")

        df = pd.DataFrame([input_dict])

        df = self._build_numeric(df, fit=False)
        df = self._build_categorical(df, fit=False)

        # Eksik feature'ları 0 ile doldur
        for col in self._feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        return df[self._feature_columns].astype(float)

    # ── Kaydet / Yükle ───────────────────────────────────────────────────────

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ENGINEER_PATH)
        joblib.dump(self._feature_columns, FEATURE_COLUMNS_PATH)

        # JSON metadata
        meta = {
            "feature_columns": self._feature_columns,
            "ohe_columns": self._ohe_columns,
            "top_colors": self._top_colors,
            "damage_features": DAMAGE_FEATURES,
            "brand_median_damage": {
                str(k): float(v) for k, v in self._brand_median_damage.items()
                if pd.notna(v)
            },
        }
        meta_path = MODEL_DIR / "feature_info.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"FeatureEngineer kaydedildi: {ENGINEER_PATH}")

    @classmethod
    def load(cls) -> "FeatureEngineer":
        return joblib.load(ENGINEER_PATH)


# ── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    csv_path = Path("data/raw_listings.csv")
    if not csv_path.exists():
        print("data/raw_listings.csv bulunamadı. Önce scraper çalıştır.")
    else:
        df = pd.read_csv(csv_path)
        fe = FeatureEngineer()
        X_tr, X_te, y_tr, y_te, cols = fe.fit_transform(df)
        print(f"Train: {X_tr.shape}  Test: {X_te.shape}")
        print(f"Feature sayısı: {len(cols)}")
        print("İlk 5 feature:", cols[:5])
