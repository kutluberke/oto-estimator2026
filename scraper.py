# scraper.py — arabam.com ikinci el ilan scraper
#
# Gerçek HTML yapısı (debug ile doğrulandı):
#   <tr class='listing-list-item should-hover bg-white'>
#     td[0] → boş (resim)
#     td[1] class='listing-modelname pr'          → Model adı + detay linki
#     td[2] class='horizontal-half-padder-minus'  → İlan başlığı
#     td[3] class='listing-text'                  → Yıl
#     td[4] class='listing-text'                  → KM
#     td[5] class='listing-text'                  → Renk
#     td[6] class=''                               → Fiyat
#     td[7] class='listing-text tac'              → Tarih
#     td[8] class='listing-text'                  → Konum

import re
import time
import random
import logging
import threading
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable

try:
    import cloudscraper as _cloudscraper
    _CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    _CLOUDSCRAPER_AVAILABLE = False

from config import (
    HEADERS, SCRAPE_CONFIG, DATA_BOUNDS,
    BASE_URL, RAW_DATA_PATH, build_search_url,
    get_random_headers,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(get_random_headers())
    return s


_session = _make_session()


def _warmup_session() -> None:
    """İki adımlı warm-up: ana sayfa → kategori sayfası. Çerez oluşturur, insan gibi görünür."""
    steps = [
        ("https://www.arabam.com/", None),
        ("https://www.arabam.com/ikinci-el/otomobil", "https://www.arabam.com/"),
    ]
    for url, referer in steps:
        try:
            _session.headers.update(get_random_headers(referer=referer))
            _session.get(url, timeout=10)
            delay = max(2.0, random.gauss(3.0, 0.6))
            time.sleep(delay)
        except Exception:
            pass


try:
    _warmup_session()
except Exception:
    pass

_rate_lock = threading.Lock()


def _dump_debug_html(url: str, content: bytes, path: str = "debug_page.html") -> None:
    """Ham HTML yanıtını dosyaya yaz. --debug bayrağı aktifken tetiklenir."""
    try:
        with open(path, "wb") as f:
            f.write(content)
        logger.info(f"Debug HTML kaydedildi: {path}  (URL: {url})")
    except Exception as e:
        logger.warning(f"Debug HTML kaydedilemedi: {e}")


# ═══════════════════════════════════════════════════════
#  YARDIMCI PARSE FONKSİYONLARI
# ═══════════════════════════════════════════════════════

def _parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = re.sub(r"[^\d]", "", text.strip())
    if not cleaned:
        return None
    try:
        val = float(cleaned)
        if DATA_BOUNDS["price_min"] <= val <= DATA_BOUNDS["price_max"]:
            return val
    except ValueError:
        pass
    return None


def _parse_km(text: str) -> Optional[float]:
    """
    "115.000" (nokta = binlik ayraç, km suffix YOK) → 115000.0
    """
    if not text:
        return None
    cleaned = re.sub(r"[^\d]", "", text.strip())
    if not cleaned:
        return None
    try:
        val = float(cleaned)
        if DATA_BOUNDS["km_min"] <= val <= DATA_BOUNDS["km_max"]:
            return val
    except ValueError:
        pass
    return None


def _parse_year(text: str) -> Optional[int]:
    if not text:
        return None
    try:
        val = int(text.strip())
        if DATA_BOUNDS["year_min"] <= val <= DATA_BOUNDS["year_max"]:
            return val
    except ValueError:
        pass
    m = re.search(r"\b(20[0-2]\d|199\d)\b", text)
    return int(m.group(1)) if m else None


def _parse_location(text: str) -> Optional[str]:
    """
    "İstanbul Bağcılar Karşılaştır..." → "İstanbul"
    """
    if not text:
        return None
    noise = {"karşılaştır", "favorilerimde", "favoriye", "ekle", "çıkar", "gi̇", "gir"}
    words = text.strip().split()
    for word in words:
        if word.lower() in noise:
            break
        # İl = ilk kelime
        return word
    return None


def _extract_paket(model_full: str, marka: str, model_query: str) -> str:
    """
    "Alfa Romeo Tonale 1.5 Hybrid Veloce" → "1.5 Hybrid Veloce"
    """
    text = model_full.strip()
    for word in marka.replace("-", " ").split():
        text = re.sub(re.escape(word), "", text, flags=re.IGNORECASE)
    for word in model_query.replace("-", " ").split():
        text = re.sub(re.escape(word), "", text, flags=re.IGNORECASE)
    paket = re.sub(r"\s+", " ", text).strip(" -–|")
    return paket if paket else "Standart"


def _parse_condition_text(text: str) -> dict:
    """
    "0 boya 2 değişen 0 hata" → {'repaints': 0, 'changed_parts': 2, 'errors': 0}
    "Ağır Hasar Kaydı: Var"   → {'heavy_damage': 1}

    Repaints/errors/changed_parts için maksimum 2 basamak sınırı var
    (detay sayfasındaki fiyat rakamlarının yanlışlıkla eşleşmesini önler).
    """
    result = {
        "errors": None, "repaints": None,
        "changed_parts": None, "heavy_damage": None,
    }

    # 1-2 basamaklı sayı + "hata" (max 30 parça mantıklı sınır)
    m = re.search(r"\b(\d{1,2})\s*hata\b", text, re.IGNORECASE)
    if m and int(m.group(1)) <= 30:
        result["errors"] = int(m.group(1))

    # 1-2 basamaklı sayı + "boya"
    m = re.search(r"\b(\d{1,2})\s*boya\b", text, re.IGNORECASE)
    if m and int(m.group(1)) <= 30:
        result["repaints"] = int(m.group(1))

    # 1-2 basamaklı sayı + "değişen/degisen"
    m = re.search(r"\b(\d{1,2})\s*de[gğ]i[şs]en\b", text, re.IGNORECASE)
    if m and int(m.group(1)) <= 30:
        result["changed_parts"] = int(m.group(1))

    # Ağır hasar kaydı — "Var" → 1, "Yok" → 0
    m = re.search(
        r"a[gğ][iı]r\s+hasar\s+kayd[iı]\s*[:\-]?\s*(var|yok)",
        text, re.IGNORECASE
    )
    if m:
        result["heavy_damage"] = 1 if m.group(1).lower() == "var" else 0
    else:
        # Alternatif: sadece "ağır hasar var" ifadesi
        if re.search(r"a[gğ][iı]r\s+hasar\s+var", text, re.IGNORECASE):
            result["heavy_damage"] = 1
        elif re.search(r"a[gğ][iı]r\s+hasar\s+yok|a[gğ][iı]r\s+hasar\s+kayd[iı]\s+bulunmamak", text, re.IGNORECASE):
            result["heavy_damage"] = 0

    return result


# ═══════════════════════════════════════════════════════
#  HTTP
# ═══════════════════════════════════════════════════════

def _get(
    url: str,
    delay_after: bool = True,
    use_lock: bool = False,
    referer: str = None,
    debug: bool = False,
) -> Optional[requests.Response]:
    """
    Session tabanlı HTTP GET — UA rotasyonu, üstel geri çekilme, cloudscraper fallback.
    use_lock=True → paralel thread'lerde lock ile sıralı istek; sleep lock dışında.
    """
    cfg = SCRAPE_CONFIG
    backoff_base = cfg.get("backoff_base", 8)

    for attempt in range(cfg["max_retries"]):
        try:
            _session.headers.update(get_random_headers(referer=referer))

            if use_lock:
                with _rate_lock:
                    resp = _session.get(url, timeout=cfg["timeout"])
                time.sleep(max(0.3, random.gauss(0.5, 0.15)))
            else:
                resp = _session.get(url, timeout=cfg["timeout"])

            if resp.status_code == 404:
                return None

            if resp.status_code == 403:
                logger.warning(f"403 Forbidden (deneme {attempt + 1}): {url}")
                # Son denemede cloudscraper fallback dene
                if _CLOUDSCRAPER_AVAILABLE and attempt == cfg["max_retries"] - 1:
                    logger.info("cloudscraper fallback deneniyor…")
                    try:
                        cs = _cloudscraper.create_scraper()
                        cs.headers.update(get_random_headers(referer=referer))
                        cs_resp = cs.get(url, timeout=cfg["timeout"])
                        if cs_resp.status_code == 200:
                            return cs_resp
                    except Exception as ce:
                        logger.warning(f"cloudscraper hatası: {ce}")
                wait = backoff_base * (2 ** attempt) + random.gauss(0, 1.5)
                time.sleep(max(wait, 1.0))
                continue

            if resp.status_code in (429, 503):
                wait = backoff_base * (2 ** attempt) + random.gauss(0, 2.0)
                logger.warning(f"Rate limit ({resp.status_code}). {wait:.1f}s bekleniyor…")
                time.sleep(max(wait, 5.0))
                continue

            resp.raise_for_status()

            if debug:
                _dump_debug_html(url, resp.content)

            if delay_after and not use_lock:
                sigma = cfg.get("delay_sigma", 0.4)
                mu = (cfg["delay_min"] + cfg["delay_max"]) / 2
                delay = max(cfg["delay_min"], random.gauss(mu, sigma))
                time.sleep(delay)
            return resp

        except requests.Timeout:
            wait = backoff_base * (2 ** attempt)
            time.sleep(wait)
        except requests.RequestException as e:
            logger.warning(f"Request hatası (deneme {attempt + 1}): {e}")
            wait = backoff_base * (2 ** attempt)
            time.sleep(wait)
    return None


# ═══════════════════════════════════════════════════════
#  DETAY SAYFASI — hata / boya / değişen
# ═══════════════════════════════════════════════════════

def _get_detail_url(row) -> Optional[str]:
    """
    Listing satırındaki detay sayfası URL'sini bul.
    arabam.com'da td[1] veya td[2] içindeki <a> ile '/ilan/' path'i.
    """
    for a_tag in row.find_all("a", href=True):
        href = a_tag["href"]
        if "/ilan/" in href:
            return href if href.startswith("http") else BASE_URL + href
    return None


def _scrape_detail(url: str) -> dict:
    """
    Detay sayfasından hata/boya/değişen parse et.
    use_lock=True: paralel thread'lerde rate limiting için lock kullanır.
    Başarısız olursa hepsi None döner.
    """
    empty = {"errors": None, "repaints": None, "changed_parts": None}
    if not url:
        return empty

    resp = _get(url, delay_after=False, use_lock=True)
    if resp is None:
        return empty

    soup = BeautifulSoup(resp.content, "html.parser")
    page_text = soup.get_text(separator=" ", strip=True)
    condition = _parse_condition_text(page_text)

    if any(v is not None for v in condition.values()):
        return condition

    return empty


# ═══════════════════════════════════════════════════════
#  LİSTİNG SATIRI PARSE
# ═══════════════════════════════════════════════════════

def _parse_row(row, marka: str, model_query: str, debug: bool = False) -> Optional[dict]:
    tds = []
    try:
        tds = row.find_all("td")
        if len(tds) < 6:
            if debug:
                logger.debug(
                    f"Yetersiz TD ({len(tds)}): "
                    f"{[td.get_text(strip=True)[:30] for td in tds]}"
                )
            return None

        model_full = tds[1].get_text(strip=True)

        # Başlık artık td[1] içindeki div'de
        title_div = tds[1].find("div", class_="listing-title-lines")
        title = title_div.get_text(strip=True) if title_div else model_full

        year = _parse_year(tds[2].get_text(strip=True))  # eskisi: tds[3]
        km = _parse_km(tds[3].get_text(strip=True))  # eskisi: tds[4]

        if km is None:
            return None

        price_td = tds[5]  # eskisi: tds[6]
        price_span = price_td.find("span", class_="listing-price")
        price_text = price_span.get_text(strip=True) if price_span else price_td.get_text(strip=True)
        price = _parse_price(price_text)

        if price is None:
            return None

        location_text = tds[7].get_text(separator=" ", strip=True) if len(tds) > 7 else ""  # eskisi: tds[8]
        location = _parse_location(location_text)
        paket = _extract_paket(model_full, marka, model_query)
        detail_url = _get_detail_url(row)

        return {
            "title": title,
            "price": price,
            "km": km,
            "year": year,
            "model": model_query.title(),
            "paket": paket,
            "location": location,
            "detail_url": detail_url,
            "errors": None,
            "repaints": None,
            "changed_parts": None,
            "heavy_damage": None,
        }

    except Exception as e:
        if debug:
            logger.debug(
                f"Satır parse hatası: {e} | "
                f"TDs: {[td.get_text(strip=True)[:30] for td in tds]}"
            )
        else:
            logger.debug(f"Satır parse hatası: {e}")
        return None
# ═══════════════════════════════════════════════════════
#  ANA SCRAPE FONKSİYONU
# ═══════════════════════════════════════════════════════

def scrape_listings(
    marka: str,
    model_query: str,
    save: bool = True,
    max_pages: int = None,
    fetch_details: bool = True,
    progress_callback: Callable = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    arabam.com'dan marka+model için ilanları çek.

    Parametreler:
        marka            : URL formatı. Örn: "alfa-romeo"
        model_query      : URL formatı. Örn: "tonale"
        save             : True ise data/raw_listings.csv kaydeder
        max_pages        : Sayfa limiti (None = config default)
        fetch_details    : True ise detay sayfasından hata/boya/değişen çeker
        progress_callback: (page, total, status_msg) callable
        debug            : True ise ilk başarılı sayfanın HTML'ini debug_page.html'e yazar
    """
    logger.info(f"Scraping: {marka} / {model_query} | detay={fetch_details} | debug={debug}")

    cfg   = SCRAPE_CONFIG
    limit = max_pages or cfg["max_pages"]
    all_listings: list = []
    consecutive_empty = 0
    _debug_dumped = False  # HTML'i yalnızca ilk kez yaz

    # ── Listing sayfaları ────────────────────────────────
    for page in range(1, limit + 1):
        url = build_search_url(marka, model_query, page)
        referer = build_search_url(marka, model_query, page - 1) if page > 1 else "https://www.arabam.com/ikinci-el/otomobil"

        if progress_callback:
            progress_callback(page, len(all_listings), f"Sayfa {page}/{limit} taranıyor…")

        resp = _get(url, referer=referer, debug=debug and not _debug_dumped)
        if resp is not None and debug and not _debug_dumped:
            _debug_dumped = True
        if resp is None:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue

        soup = BeautifulSoup(resp.content, "html.parser")
        rows = soup.find_all("tr", class_="listing-list-item")

        if not rows:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue

        consecutive_empty = 0
        for row in rows:
            listing = _parse_row(row, marka, model_query, debug=debug)
            if listing:
                all_listings.append(listing)

        logger.info(f"Sayfa {page}: {len(rows)} satır → toplam {len(all_listings)} ilan")

    if not all_listings:
        logger.error("Hiç ilan çekilemedi!")
        return pd.DataFrame()

    df = pd.DataFrame(all_listings)

    # ── Detay sayfaları — paralel ThreadPoolExecutor ────
    if fetch_details:
        total   = len(df)
        found   = 0
        done    = 0
        MAX_WORKERS = SCRAPE_CONFIG.get("detail_workers", 2)

        logger.info(f"Detay sayfaları paralel çekiliyor ({total} ilan, {MAX_WORKERS} thread)…")

        # (idx, url) çiftlerini hazırla
        tasks = [
            (idx, row["detail_url"])
            for idx, row in df.iterrows()
            if row.get("detail_url")
        ]

        # Thread-safe sayaç
        _done_lock = threading.Lock()

        def _fetch_and_return(idx_url):
            idx, url = idx_url
            return idx, _scrape_detail(url)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_and_return, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    idx, cond = future.result()
                    df.at[idx, "errors"]        = cond["errors"]
                    df.at[idx, "repaints"]      = cond["repaints"]
                    df.at[idx, "changed_parts"] = cond["changed_parts"]
                    df.at[idx, "heavy_damage"]  = cond["heavy_damage"]
                    if any(v is not None for v in cond.values()):
                        found += 1
                except Exception as e:
                    logger.debug(f"Detay future hatası: {e}")

                with _done_lock:
                    done += 1
                    _done_snap = done
                    _found_snap = found

                if progress_callback:
                    progress_callback(
                        None, total,
                        f"Detay verisi: {_done_snap}/{total} ilan "
                        f"({_found_snap} kondisyon bulundu)"
                    )

        logger.info(f"Detay verisi tamamlandı: {found}/{total} ilandan kondisyon bulundu")

    # ── Temizle & kaydet ────────────────────────────────
    df = df.drop("detail_url", axis=1, errors="ignore")
    df = df.drop_duplicates(subset=["title", "price", "km"], keep="first")

    if save:
        import os
        os.makedirs("data", exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"Kaydedildi: {RAW_DATA_PATH} ({len(df)} ilan)")

    return df


# ── Test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    marka = sys.argv[1] if len(sys.argv) > 1 else "alfa-romeo"
    model = sys.argv[2] if len(sys.argv) > 2 else "tonale"
    df = scrape_listings(marka, model, fetch_details=True, max_pages=3)
    print(df[["title", "price", "km", "year", "paket", "errors", "repaints", "changed_parts"]].to_string())
