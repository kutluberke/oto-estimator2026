#!/usr/bin/env python3
"""
arabam_pipeline.py — arabam.com ikinci el araç fiyat tahmin pipeline
Tek dosya: scraping → temizlik → feature engineering → model → tahmin CLI
Veri bellekte tutulur; CSV veya veritabanı kullanılmaz.
"""

import re
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import time
import random
import warnings
import joblib
import numpy as np
import pandas as pd
from curl_cffi import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — SCRAPING
# ─────────────────────────────────────────────────────────────────────────────

_BASE_URL = "https://www.arabam.com/ikinci-el/otomobil"
_TARGET   = 700  # toplam ilan hedefi

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _make_headers(referer: str = None) -> dict:
    ua = random.choice(_USER_AGENTS)
    headers = {
        "User-Agent":              ua,
        "Accept":                  "text/html,application/xhtml+xml,application/xml;"
                                   "q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language":         "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding":         "gzip, deflate, br",
        "Connection":              "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT":                     "1",
    }
    if referer:
        headers["Referer"] = referer
    if "Firefox" not in ua:
        headers.update({
            "sec-ch-ua":          '"Chromium";v="124","Google Chrome";v="124","Not-A.Brand";v="99"',
            "sec-ch-ua-mobile":   "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest":     "document",
            "sec-fetch-mode":     "navigate",
            "sec-fetch-site":     "same-origin" if referer else "none",
            "sec-fetch-user":     "?1",
        })
    return headers


def _fetch_html(session: requests.Session, url: str, max_retries: int = 3) -> str | None:
    """URL'yi çek; 403/429/503'te üstel geri çekilme ile yeniden dene."""
    for attempt in range(max_retries):
        try:
            resp = session.get(
                url,
                headers=_make_headers(referer="https://www.arabam.com/"),
                impersonate="chrome",
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (403, 429, 503):
                wait = (2 ** attempt) * 3 + random.uniform(0, 2)
                print(f"  HTTP {resp.status_code} — {wait:.1f}s bekleniyor (deneme {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"  Beklenmeyen durum kodu: {resp.status_code} — {url}")
            return None
        except requests.RequestException as e:
            wait = (2 ** attempt) * 2
            print(f"  İstek hatası: {e} — {wait:.1f}s bekleniyor")
            time.sleep(wait)
    return None


def _parse_price(text: str) -> float | None:
    cleaned = re.sub(r"[^\d]", "", text or "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_km(text: str) -> float | None:
    cleaned = re.sub(r"[^\d]", "", text or "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_year(text: str) -> int | None:
    text = (text or "").strip()
    try:
        val = int(text)
        if 1960 <= val <= 2026:
            return val
    except ValueError:
        pass
    m = re.search(r"\b(20[0-2]\d|19[6-9]\d)\b", text)
    return int(m.group(1)) if m else None


def _extract_brand_model(row, model_full: str) -> tuple[str, str]:
    """Marka ve modeli ilan href'inden çıkar; başarısız olursa td metnini kullan."""
    detail_a = row.find("a", href=re.compile(r"/ilan/"))
    if detail_a:
        href = detail_a.get("href", "")
        # Örnek: /ikinci-el/otomobil/volkswagen-golf/...
        m = re.search(
            r"/ikinci-el/[^/]+/([a-z0-9]+(?:-[a-z0-9]+)*?)"
            r"-([a-z0-9]+(?:-[a-z0-9]+)*?)(?:/|\?|$)",
            href,
        )
        if m:
            brand = m.group(1).replace("-", " ").title()
            model = m.group(2).replace("-", " ").title()
            return brand, model

    # Yedek: td metnini böl
    parts = model_full.strip().split()
    brand = parts[0].title() if parts else "Bilinmiyor"
    model = " ".join(parts[1:2]).title() if len(parts) > 1 else "Bilinmiyor"
    return brand, model


def _parse_listing_row(row) -> dict | None:
    """Tek <tr class='listing-list-item'> satırını parse et."""
    try:
        tds = row.find_all("td")
        if len(tds) < 7:
            return None

        model_full   = tds[1].get_text(strip=True)
        title_div    = tds[2].find("div", class_="listing-title-lines")
        title        = title_div.get_text(strip=True) if title_div else tds[2].get_text(strip=True)
        year         = _parse_year(tds[3].get_text(strip=True))
        km           = _parse_km(tds[4].get_text(strip=True))

        price_span   = tds[6].find("span", class_="listing-price")
        price_text   = price_span.get_text(strip=True) if price_span else tds[6].get_text(strip=True)
        price        = _parse_price(price_text)

        location     = tds[8].get_text(separator=" ", strip=True).split()[0] if len(tds) > 8 else None
        listing_date = tds[7].get_text(strip=True) if len(tds) > 7 else None

        brand, model = _extract_brand_model(row, model_full)

        return {
            "title":        title,
            "brand":        brand,
            "model":        model,
            "year":         year,
            "km":           km,
            "price":        price,
            "location":     location,
            "listing_date": listing_date,
        }
    except Exception:
        return None


def scrape() -> pd.DataFrame:
    print("\n[SCRAPING] arabam.com/ikinci-el/otomobil taranıyor…")
    session  = requests.Session()
    all_rows = []
    page     = 1

    # Warmup: ana sayfa
    session.get("https://www.arabam.com/", headers=_make_headers(), impersonate="chrome", timeout=10)
    time.sleep(random.uniform(1.0, 2.0))

    while len(all_rows) < _TARGET:
        url  = f"{_BASE_URL}?take=50&page={page}"
        html = _fetch_html(session, url)

        if html is None:
            print(f"  Sayfa {page}: çekilemedi, duruyorum.")
            break

        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr", class_="listing-list-item")

        if not rows:
            print(f"  Sayfa {page}: ilan satırı bulunamadı — site engelliyor olabilir.")
            break

        parsed = 0
        for row in rows:
            rec = _parse_listing_row(row)
            if rec:
                all_rows.append(rec)
                parsed += 1
            if len(all_rows) >= _TARGET:
                break

        print(f"  Sayfa {page:>3}: {parsed:>2} ilan parse edildi -> toplam {len(all_rows)}")
        page += 1
        time.sleep(random.uniform(1.5, 3.5))

    df = pd.DataFrame(all_rows)
    print(f"[SCRAPING] Tamamlandı: {len(df)} ham ilan")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[CLEANING] Veri temizleniyor…")
    n0 = len(df)

    # price ve km null satırları at
    df = df.dropna(subset=["price", "km"])

    # Fiyat aykırı değerleri
    df = df[(df["price"] >= 100_000) & (df["price"] <= 10_000_000)]

    # KM aykırı değerleri
    df = df[df["km"] <= 500_000]

    # Yıl parse + filtrele
    df["year"] = df["year"].apply(
        lambda v: _parse_year(str(v)) if not isinstance(v, int) else v
    )
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]

    # Sayısal tipleri garantile
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["km"]    = pd.to_numeric(df["km"],    errors="coerce")
    df = df.dropna(subset=["price", "km"])

    # Türetilmiş özellik
    df["vehicle_age"] = 2025 - df["year"]

    # location / brand / model boşlarını doldur
    df["location"] = df["location"].fillna("Bilinmiyor")
    df["brand"]    = df["brand"].fillna("Bilinmiyor")
    df["model"]    = df["model"].fillna("Bilinmiyor")

    print(f"[CLEANING] {n0} -> {len(df)} ilan ({n0 - len(df)} satır silindi)")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

_CAT_COLS = ["brand", "model", "location"]

def engineer_features(df: pd.DataFrame):
    """
    Döner: (X_train, X_test, y_train, y_test, encoders, most_frequent)
    encoders       = {col: LabelEncoder}
    most_frequent  = {col: most_frequent raw value}
    """
    print("\n[TRAINING] Feature engineering…")
    encoders     = {}
    most_frequent = {}

    for col in _CAT_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col]    = le
        most_frequent[col] = df[col].mode()[0]

    feature_cols = ["brand_enc", "model_enc", "vehicle_age", "km", "location_enc"]
    X = df[feature_cols].values
    y = np.log1p(df["price"].values)  # log-transform hedef

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Eğitim: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, encoders, most_frequent


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4 — MODELING
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(model, X_test, y_test) -> dict:
    y_pred_log = model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_test)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_models(X_train, X_test, y_train, y_test):
    """Üç model eğit, karşılaştır, en iyisini döndür."""
    print("\n[TRAINING] Modeller eğitiliyor…")

    candidates = {
        "Random Forest":      RandomForestRegressor(
                                  n_estimators=200, max_depth=10,
                                  n_jobs=-1, random_state=42),
        "XGBoost":            XGBRegressor(
                                  n_estimators=200, max_depth=6,
                                  learning_rate=0.05, random_state=42,
                                  verbosity=0, eval_metric="rmse"),
        "Gradient Boosting":  GradientBoostingRegressor(
                                  n_estimators=200, max_depth=5,
                                  random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        print(f"  {name} eğitiliyor…", end=" ", flush=True)
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        results[name] = {"model": model, **metrics}
        print(f"R²={metrics['R2']:.4f}")

    # Karşılaştırma tablosu
    print("\n  ── Model Karşılaştırması ───────────────────────────────")
    print(f"  {'Model':<22}  {'MAE':>12}  {'RMSE':>14}  {'R²':>8}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*14}  {'─'*8}")
    for name, res in results.items():
        print(
            f"  {name:<22}  {res['MAE']:>12,.0f}  {res['RMSE']:>14,.0f}  {res['R2']:>8.4f}"
        )
    print()

    best_name  = max(results, key=lambda n: results[n]["R2"])
    best_model = results[best_name]["model"]
    print(f"  En iyi model: {best_name}  (R²={results[best_name]['R2']:.4f})")

    joblib.dump(best_model, "best_model.pkl")
    print("  Kaydedildi: best_model.pkl")

    return best_model, best_name


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 5 — PREDICTION CLI
# ─────────────────────────────────────────────────────────────────────────────

def _encode_input(value: str, col: str, encoders: dict, most_frequent: dict) -> int:
    le = encoders[col]
    try:
        return int(le.transform([value])[0])
    except ValueError:
        fallback = most_frequent[col]
        print(f"  Uyarı: '{value}' eğitim verisinde görülmemiş ({col}). "
              f"En sık değer kullanılıyor: '{fallback}'")
        return int(le.transform([fallback])[0])


def predict_loop(best_model, encoders: dict, most_frequent: dict) -> None:
    print("\n[PREDICTING] Tahmin modu — çıkmak için 'exit' yazın\n")

    while True:
        try:
            brand    = input("Marka       : ").strip()
            if brand.lower() == "exit":
                break
            model_in = input("Model       : ").strip()
            if model_in.lower() == "exit":
                break
            year_in  = input("Yıl         : ").strip()
            if year_in.lower() == "exit":
                break
            km_in    = input("KM          : ").strip()
            if km_in.lower() == "exit":
                break
            loc_in   = input("Konum       : ").strip()
            if loc_in.lower() == "exit":
                break

            year_val = _parse_year(year_in)
            if year_val is None:
                print("  Geçersiz yıl, tekrar deneyin.\n")
                continue

            km_val = _parse_km(km_in)
            if km_val is None:
                print("  Geçersiz KM değeri, tekrar deneyin.\n")
                continue

            brand_enc = _encode_input(brand.title(),    "brand",    encoders, most_frequent)
            model_enc = _encode_input(model_in.title(), "model",    encoders, most_frequent)
            loc_enc   = _encode_input(loc_in.title(),   "location", encoders, most_frequent)
            age_val   = 2025 - year_val

            X = np.array([[brand_enc, model_enc, age_val, km_val, loc_enc]])
            price = np.expm1(best_model.predict(X)[0])
            print(f"\n  Tahmini fiyat: {price:>15,.0f} TL\n")

        except (KeyboardInterrupt, EOFError):
            break

    print("\nÇıkılıyor.")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Phase 1
    df_raw = scrape()

    # Phase 2
    df = clean(df_raw)
    if len(df) < 200:
        raise RuntimeError(
            f"Temizleme sonrası yalnızca {len(df)} satır kaldı "
            f"(minimum 200 gerekli). Scraping başarısız olmuş olabilir."
        )

    # Phase 3 + 4
    X_train, X_test, y_train, y_test, encoders, most_frequent = engineer_features(df)
    best_model, best_name = train_models(X_train, X_test, y_train, y_test)

    # Phase 5
    predict_loop(best_model, encoders, most_frequent)


if __name__ == "__main__":
    main()
