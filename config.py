# config.py — Sabitler, URL şablonları, scraping ayarları

import random as _random
from pathlib import Path

# ===== PROJE KÖKÜ =====
BASE_DIR = Path(__file__).parent

# ===== URL YAPILANDIRMASI =====
BASE_URL = "https://www.arabam.com"

# Genel kategori sayfaları (scrape_category() için)
CATEGORY_URLS = [
    "/ikinci-el/otomobil",
    "/ikinci-el/suv-arazi-arac",
]


def build_search_url(marka: str, model: str, page: int = 1) -> str:
    return f"{BASE_URL}/ikinci-el/otomobil/{marka}-{model}?page={page}"


def build_category_url(category_path: str, page: int = 1) -> str:
    return f"{BASE_URL}{category_path}?page={page}"


# ===== VERİ SINIRLAR =====
DATA_BOUNDS = {
    "price_min": 50_000,
    "price_max": 15_000_000,
    "km_min": 0,
    "km_max": 500_000,
    "year_min": 1990,
    "year_max": 2026,
    "rare_location_threshold": 5,
}

TARGET_COLUMN = "price"

# ===== SCRAPING AYARLARI =====
SCRAPE_CONFIG = {
    "delay_min": 1.5,
    "delay_max": 3.0,
    "delay_sigma": 0.4,
    "timeout": 15,
    "max_retries": 4,
    "max_pages": 20,
    "min_listings": 10,
    "detail_workers": 2,
    "backoff_base": 8,
}

# ===== USER-AGENT HAVUZU =====
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 OPR/110.0.0.0",
]


def get_random_headers(referer: str = None) -> dict:
    ua = _random.choice(UA_POOL)
    is_firefox = "Firefox" in ua
    is_edge = "Edg/" in ua

    headers = {
        "User-Agent": ua,
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    if referer:
        headers["Referer"] = referer

    fetch_site = "same-origin" if referer and "arabam.com" in referer else "none"

    if is_firefox:
        headers.update({
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": fetch_site,
            "sec-fetch-user": "?1",
        })
    else:
        version = "124"
        if is_edge:
            headers["sec-ch-ua"] = (
                f'"Microsoft Edge";v="{version}", "Chromium";v="{version}", "Not-A.Brand";v="99"'
            )
        else:
            headers["sec-ch-ua"] = (
                f'"Chromium";v="{version}", "Google Chrome";v="{version}", "Not-A.Brand";v="99"'
            )
        headers.update({
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": fetch_site,
            "sec-fetch-user": "?1",
        })

    return headers


# ===== HTTP HEADERS (geriye dönük uyumluluk) =====
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.arabam.com/",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
}

# ===== TRAMER ANAHTAR KELİMELER =====
TRAMER_NEGATIVE_KEYWORDS = [
    "tramer yok", "trameri yok", "0 tramer", "tramer kaydı yok",
    "hasar kaydı yok", "hasarsız", "hasar yok",
]
TRAMER_POSITIVE_KEYWORDS = [
    "tramer var", "trameri var", "tramer kaydı var",
    "hasar kaydı var", "hasarlı",
]

# ===== DOSYA YOLLARI =====
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
PLOTS_DIR  = BASE_DIR / "plots"
LOGS_DIR   = BASE_DIR / "logs"

RAW_DATA_PATH        = DATA_DIR / "raw_listings.csv"
CLEANED_DATA_PATH    = DATA_DIR / "cleaned_data.csv"
PROCESSED_DATA_PATH  = DATA_DIR / "processed_data.csv"

MODEL_PATH            = MODEL_DIR / "model.pkl"
ENGINEER_PATH         = MODEL_DIR / "feature_engineer.pkl"
FEATURE_INFO_PATH     = MODEL_DIR / "feature_info.json"
FEATURE_COLUMNS_PATH  = MODEL_DIR / "feature_columns.pkl"

RF_MODEL_PATH         = MODEL_DIR / "rf_model.pkl"
GB_MODEL_PATH         = MODEL_DIR / "gb_model.pkl"
XGB_MODEL_PATH        = MODEL_DIR / "xgb_model.pkl"
ENSEMBLE_WEIGHTS_PATH = MODEL_DIR / "ensemble_weights.pkl"
FEATURE_IMP_PATH      = MODEL_DIR / "feature_importances.csv"

LOG_PATH = LOGS_DIR / "scrape_log.txt"

# ===== LEGACY MODEL PARAMETRELERİ =====
LGBM_PARAMS_SMALL = {
    "n_estimators": 80, "learning_rate": 0.1, "max_depth": 4,
    "num_leaves": 15, "min_data_in_leaf": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "verbose": -1, "random_state": 42,
}
LGBM_PARAMS_MEDIUM = {
    "n_estimators": 150, "learning_rate": 0.05, "max_depth": 6,
    "num_leaves": 31, "min_data_in_leaf": 15, "subsample": 0.8,
    "colsample_bytree": 0.8, "verbose": -1, "random_state": 42,
}
LGBM_PARAMS_LARGE = {
    "n_estimators": 250, "learning_rate": 0.03, "max_depth": 8,
    "num_leaves": 63, "min_data_in_leaf": 20, "subsample": 0.8,
    "colsample_bytree": 0.8, "verbose": -1, "random_state": 42,
}
RF_PARAMS = {
    "n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
    "min_samples_leaf": 3, "random_state": 42, "n_jobs": -1,
}

# ===== YENİ ENSEMBLE PARAMETRELERİ =====
RF_PARAMS_NEW = {
    "n_estimators": 300, "max_depth": 6, "min_samples_leaf": 8,
    "random_state": 42, "n_jobs": -1,
}
GB_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "min_samples_leaf": 8,
    "random_state": 42,
}
XGB_PARAMS = {
    "n_estimators": 300, "max_depth": 5, "min_child_weight": 8,
    "learning_rate": 0.05, "random_state": 42,
    "eval_metric": "rmse", "verbosity": 0,
}
