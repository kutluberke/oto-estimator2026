# config.py — Sabitler, URL şablonları, scraping ayarları

import os

# ===== URL YAPILANDIRMASI =====
BASE_URL = "https://www.arabam.com"

def build_search_url(marka: str, model: str, page: int = 1) -> str:
    """
    arabam.com arama URL'si oluştur.
    Örnek: alfa-romeo / tonale → https://www.arabam.com/ikinci-el/otomobil/alfa-romeo-tonale?page=1
    """
    return f"{BASE_URL}/ikinci-el/otomobil/{marka}-{model}?page={page}"

# ===== SCRAPING AYARLARI =====
SCRAPE_CONFIG = {
    "delay_min": 0.6,          # İstekler arası min bekleme (saniye)
    "delay_max": 1.2,          # İstekler arası max bekleme (saniye)
    "timeout": 12,             # HTTP timeout (saniye)
    "max_retries": 3,          # Başarısız istek için max retry sayısı
    "max_pages": 20,           # Scrape edilecek max sayfa (güvenlik sınırı)
    "min_listings": 10,        # Model eğitimi için minimum ilan sayısı
}

# ===== HTTP HEADERS =====
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ===== FEATURE TANIMLARI =====
NUMERIC_FEATURES = ["km", "year", "errors", "repaints", "changed_parts"]
CATEGORICAL_FEATURES = ["model", "paket", "location"]
TARGET_COLUMN = "price"

# Feature importance için görüntülenecek feature isimleri (Türkçe)
FEATURE_DISPLAY_NAMES = {
    "km":            "Kilometre",
    "year":          "Yıl",
    "age":           "Araç Yaşı",
    "model":         "Model",
    "paket":         "Paket / Donanım",
    "errors":        "Hata",
    "repaints":      "Boya",
    "changed_parts": "Değişen",
    "heavy_damage":  "Ağır Hasar Kaydı",
    "condition_score": "Kondisyon Skoru",
    "tramer":        "Tramer",
}

# ===== VERİ TEMİZLEME SINIRLAR =====
DATA_BOUNDS = {
    "km_min": 0,
    "km_max": 600_000,
    "year_min": 1990,
    "year_max": 2026,
    "price_min": 500_000,          # 500k TL altı saçma (2025-2026 Türkiye piyasası)
    "price_max": 15_000_000,       # 15M TL üstü exotik (outlier)
    "rare_location_threshold": 3,  # 3'ten az ilanı olan il → 'Diğer'
}

# ===== TRAMER ANAHTAR KELİMELER =====
# İlan açıklamasında bu ifadeler aranır (case-insensitive)
TRAMER_NEGATIVE_KEYWORDS = [
    "tramer yok", "trameri yok", "0 tramer", "tramer kaydı yok",
    "hasar kaydı yok", "hasarsız", "hasar yok",
]
TRAMER_POSITIVE_KEYWORDS = [
    "tramer var", "trameri var", "tramer kaydı var",
    "hasar kaydı var", "hasarlı",
]

# ===== DOSYA YOLLARI =====
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

RAW_DATA_PATH       = os.path.join(DATA_DIR, "raw_listings.csv")
CLEANED_DATA_PATH   = os.path.join(DATA_DIR, "cleaned_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

MODEL_PATH       = os.path.join(MODEL_DIR, "model.pkl")
ENGINEER_PATH    = os.path.join(MODEL_DIR, "feature_engineer.pkl")
FEATURE_INFO_PATH = os.path.join(MODEL_DIR, "feature_info.json")

LOG_PATH = os.path.join(LOGS_DIR, "scrape_log.txt")

# ===== MODEL HİPERPARAMETRELERİ =====
# Dataset büyüklüğüne göre otomatik seçilir (model_trainer.py kullanır)
LGBM_PARAMS_SMALL = {   # < 200 sample
    "n_estimators": 80,
    "learning_rate": 0.1,
    "max_depth": 4,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "random_state": 42,
}
LGBM_PARAMS_MEDIUM = {  # 200–1000 sample
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_data_in_leaf": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "random_state": 42,
}
LGBM_PARAMS_LARGE = {   # > 1000 sample
    "n_estimators": 250,
    "learning_rate": 0.03,
    "max_depth": 8,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "random_state": 42,
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}
