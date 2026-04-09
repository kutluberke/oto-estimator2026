# data_cleaner.py — Ham veri temizleme ve validasyon

import pandas as pd
import numpy as np
import logging

from config import DATA_BOUNDS, CLEANED_DATA_PATH

logger = logging.getLogger(__name__)


# ===== TİP DÖNÜŞÜMLERİ =====

def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham string sütunlarını doğru tiplere dönüştür.
    Hatalı parse'lar NaN olur (exception fırlatmaz).
    """
    df = df.copy()

    numeric_cols = {
        "price":               "float",
        "km":                  "float",
        "year":                "int",
        # Legacy damage
        "errors":              "float",
        "repaints":            "float",
        "changed_parts":       "float",
        "heavy_damage":        "float",
        # New damage
        "has_original_paint":  "float",
        "painted_panel_count": "float",
        "changed_panel_count": "float",
        "has_local_paint":     "float",
        "total_damage_score":  "float",
        # Spec fields
        "hp":                  "float",
        "warranty":            "float",
        "from_dealer":         "float",
        "num_owners":          "float",
    }

    for col, dtype in numeric_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Yıl: float → nullable int
    if "year" in df.columns:
        df["year"] = df["year"].astype("Int64")

    # String sütunlar: boşluk temizle
    str_cols = ["brand", "model", "paket", "location", "title",
                "fuel_type", "transmission", "body_type", "color", "engine_cc"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": None, "None": None, "": None})

    return df


# ===== OUTLIER DETECTION =====

def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantıksal sınır + IQR yöntemiyle aykırı değerleri kaldır.
    """
    df = df.copy()
    initial = len(df)

    # 1. Mantıksal sınırlar
    df = df[df["price"].between(DATA_BOUNDS["price_min"], DATA_BOUNDS["price_max"])]
    df = df[df["km"].between(DATA_BOUNDS["km_min"], DATA_BOUNDS["km_max"])]
    df = df[df["year"].between(DATA_BOUNDS["year_min"], DATA_BOUNDS["year_max"])]

    # 2. IQR bazlı fiyat outlier temizliği (brand bazında)
    if len(df) > 20 and "brand" in df.columns:
        brand_groups = []
        for brand, grp in df.groupby("brand"):
            if len(grp) < 10:
                brand_groups.append(grp)
                continue
            q_lo = grp["price"].quantile(0.025)
            q_hi = grp["price"].quantile(0.975)
            brand_groups.append(grp[grp["price"].between(q_lo, q_hi)])
        df = pd.concat(brand_groups, ignore_index=True)
    elif len(df) > 20:
        Q1 = df["price"].quantile(0.10)
        Q3 = df["price"].quantile(0.90)
        IQR = Q3 - Q1
        lower = Q1 - 2.5 * IQR
        upper = Q3 + 2.5 * IQR
        df = df[df["price"].between(lower, upper)]

    removed = initial - len(df)
    if removed > 0:
        logger.info(f"Outlier temizleme: {removed} satır kaldırıldı")

    return df


# ===== MISSING VALUE YÖNETİMİ =====

def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kritik alanlar eksikse satırı sil.
    Opsiyonel alanlar eksikse NaN bırak (feature engineer halleder).
    """
    df = df.copy()
    initial = len(df)

    df = df.dropna(subset=["price", "km"])
    df = df.dropna(subset=["year"])

    if "location" in df.columns:
        df["location"] = df["location"].fillna("Bilinmiyor")

    removed = initial - len(df)
    if removed > 0:
        logger.info(f"Missing value temizleme: {removed} satır kaldırıldı")

    return df


# ===== DUPLIKASYON =====

def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial = len(df)

    subset = [c for c in ["title", "price", "km"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")

    removed = initial - len(df)
    if removed > 0:
        logger.info(f"Duplikasyon temizleme: {removed} satır kaldırıldı")

    return df


# ===== KONUM NORMALİZASYONU =====

def _normalize_location(df: pd.DataFrame) -> pd.DataFrame:
    if "location" not in df.columns:
        return df

    df = df.copy()
    df["location"] = df["location"].replace({
        "istanbul": "İstanbul",
        "ISTANBUL": "İstanbul",
        "İstanbull": "İstanbul",
        "Ankara ": "Ankara",
        "izmir": "İzmir",
        "IZMIR": "İzmir",
    })
    df["location"] = df["location"].apply(
        lambda x: x.strip().title() if isinstance(x, str) else x
    )
    return df


# ===== VERİ KALİTE KONTROLÜ =====

def validate_dataset(df: pd.DataFrame, min_listings: int = 10) -> None:
    issues = []

    if len(df) < min_listings:
        issues.append(
            f"Yetersiz veri: {len(df)} ilan var, minimum {min_listings} gerekli."
        )

    if df["price"].std() == 0:
        issues.append("Fiyat varyasyonu sıfır — tüm ilanlar aynı fiyatta.")

    if df["price"].isna().any():
        issues.append("Temizleme sonrası hâlâ fiyat eksik olan satırlar var.")

    if issues:
        raise ValueError("\n".join(issues))

    logger.info(f"Dataset validasyonu geçti: {len(df)} ilan")


# ===== ÖZET RAPOR =====

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print(f"  VERİ ÖZETİ")
    print(f"{'='*55}")
    print(f"  Toplam ilan        : {len(df)}")
    print(f"  Fiyat (min)        : {df['price'].min():>12,.0f} TL")
    print(f"  Fiyat (max)        : {df['price'].max():>12,.0f} TL")
    print(f"  Fiyat (ortalama)   : {df['price'].mean():>12,.0f} TL")
    print(f"  Fiyat (medyan)     : {df['price'].median():>12,.0f} TL")
    print(f"  KM (min)           : {df['km'].min():>12,.0f}")
    print(f"  KM (max)           : {df['km'].max():>12,.0f}")
    print(f"  Yıl aralığı        : {int(df['year'].min())} – {int(df['year'].max())}")

    if "brand" in df.columns:
        brands = df["brand"].value_counts().head(5)
        print(f"\n  En sık marka:")
        for brand, cnt in brands.items():
            print(f"    {brand:<22} {cnt} ilan")

    if "location" in df.columns:
        top_locations = df["location"].value_counts().head(5)
        print(f"\n  En sık konum:")
        for loc, cnt in top_locations.items():
            print(f"    {loc:<22} {cnt} ilan")

    print(f"\n  Eksik değerler:")
    missing = df.isnull().sum()
    for col, cnt in missing[missing > 0].items():
        pct = cnt / len(df) * 100
        print(f"    {col:<25} {cnt} ({pct:.1f}%)")

    print(f"{'='*55}\n")


# ===== ANA FONKSİYON =====

def clean_data(raw_df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Ham DataFrame'i alır, temizler, validate eder ve döner.
    """
    logger.info(f"Temizleme başlıyor: {len(raw_df)} ham ilan")

    df = raw_df.copy()

    df = _convert_types(df)
    df = _remove_duplicates(df)
    df = _handle_missing(df)
    df = _remove_outliers(df)
    df = _normalize_location(df)

    df = df.reset_index(drop=True)

    validate_dataset(df)
    print_summary(df)

    if save:
        try:
            CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CLEANED_DATA_PATH, index=False, encoding="utf-8-sig")
            logger.info(f"Temiz veri kaydedildi: {CLEANED_DATA_PATH}")
        except OSError as e:
            logger.warning(f"CSV kaydetme hatası (devam ediliyor): {e}")

    return df


# ===== TEST =====
if __name__ == "__main__":
    from pathlib import Path
    csv_path = Path("data/raw_listings.csv")
    df_raw = pd.read_csv(csv_path)
    df_clean = clean_data(df_raw, save=True)
    print(df_clean.head())
