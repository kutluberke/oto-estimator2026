# scraper.py — arabam.com ikinci el ilan scraper (genişletilmiş)
#
# HTML yapısı (debug ile doğrulandı):
#   <tr class='listing-list-item should-hover bg-white'>
#     td[0] → boş (resim)
#     td[1] → Model adı + detay linki
#     td[2] → Yıl
#     td[3] → KM
#     td[5] → Fiyat
#     td[7] → Konum

import re
import time
import random
import logging
import threading
from datetime import datetime, timezone
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
    BASE_URL, RAW_DATA_PATH, build_search_url, build_category_url,
    get_random_headers, CATEGORY_URLS,
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
    steps = [
        ("https://www.arabam.com/", None),
        ("https://www.arabam.com/ikinci-el/otomobil", "https://www.arabam.com/"),
    ]
    for url, referer in steps:
        try:
            _session.headers.update(get_random_headers(referer=referer))
            _session.get(url, timeout=10)
            time.sleep(max(2.0, random.gauss(3.0, 0.6)))
        except Exception:
            pass


try:
    _warmup_session()
except Exception:
    pass

_rate_lock = threading.Lock()


def _dump_debug_html(url: str, content: bytes, path: str = "debug_page.html") -> None:
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
    if not text:
        return None
    noise = {"karşılaştır", "favorilerimde", "favoriye", "ekle", "çıkar", "gi̇", "gir"}
    words = text.strip().split()
    for word in words:
        if word.lower() in noise:
            break
        return word
    return None


def _extract_paket(model_full: str, marka: str, model_query: str) -> str:
    text = model_full.strip()
    for word in marka.replace("-", " ").split():
        text = re.sub(re.escape(word), "", text, flags=re.IGNORECASE)
    for word in model_query.replace("-", " ").split():
        text = re.sub(re.escape(word), "", text, flags=re.IGNORECASE)
    paket = re.sub(r"\s+", " ", text).strip(" -–|")
    return paket if paket else "Standart"


def _parse_condition_text(text: str) -> dict:
    result = {"errors": None, "repaints": None, "changed_parts": None, "heavy_damage": None}

    m = re.search(r"\b(\d{1,2})\s*hata\b", text, re.IGNORECASE)
    if m and int(m.group(1)) <= 30:
        result["errors"] = int(m.group(1))

    m = re.search(r"\b(\d{1,2})\s*boya\b", text, re.IGNORECASE)
    if m and int(m.group(1)) <= 30:
        result["repaints"] = int(m.group(1))

    m = re.search(r"\b(\d{1,2})\s*de[gğ]i[şs]en\b", text, re.IGNORECASE)
    if m and int(m.group(1)) <= 30:
        result["changed_parts"] = int(m.group(1))

    m = re.search(
        r"a[gğ][iı]r\s+hasar\s+kayd[iı]\s*[:\-]?\s*(var|yok)",
        text, re.IGNORECASE
    )
    if m:
        result["heavy_damage"] = 1 if m.group(1).lower() == "var" else 0
    else:
        if re.search(r"a[gğ][iı]r\s+hasar\s+var", text, re.IGNORECASE):
            result["heavy_damage"] = 1
        elif re.search(
            r"a[gğ][iı]r\s+hasar\s+yok|a[gğ][iı]r\s+hasar\s+kayd[iı]\s+bulunmamak",
            text, re.IGNORECASE
        ):
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
            time.sleep(backoff_base * (2 ** attempt))
        except requests.RequestException as e:
            logger.warning(f"Request hatası (deneme {attempt + 1}): {e}")
            time.sleep(backoff_base * (2 ** attempt))
    return None


# ═══════════════════════════════════════════════════════
#  DETAY SAYFASI
# ═══════════════════════════════════════════════════════

def _get_detail_url(row) -> Optional[str]:
    for a_tag in row.find_all("a", href=True):
        href = a_tag["href"]
        if "/ilan/" in href:
            return href if href.startswith("http") else BASE_URL + href
    return None


def _parse_damage_table(soup: BeautifulSoup) -> dict:
    result = {
        "has_original_paint": None,
        "painted_panel_count": None,
        "changed_panel_count": None,
        "has_local_paint": None,
        "total_damage_score": None,
    }

    painted = 0
    changed = 0
    has_local = False
    found_table = False

    damage_table = soup.find("table", class_=re.compile(r"damage|hasar", re.I))
    if damage_table is None:
        for tag in soup.find_all(["h2", "h3", "div"],
                                  string=re.compile(r"hasar|damage", re.I)):
            parent = tag.find_parent(["section", "div", "table"])
            if parent:
                damage_table = parent
                break

    if damage_table:
        found_table = True
        cells = damage_table.find_all(["td", "li", "span"])
        for cell in cells:
            text = cell.get_text(strip=True).lower()
            if "boyalı" in text or ("boya" in text and "orijinal" not in text
                                    and "lokal" not in text):
                try:
                    num = int(re.search(r"\d+", text).group())
                    painted += num
                except (AttributeError, ValueError):
                    painted += 1
            elif "değişen" in text or "degisen" in text:
                try:
                    num = int(re.search(r"\d+", text).group())
                    changed += num
                except (AttributeError, ValueError):
                    changed += 1
            elif "lokal" in text:
                has_local = True

    page_text = soup.get_text(separator=" ", strip=True)
    cond = _parse_condition_text(page_text)

    if not found_table:
        if cond["repaints"] is not None:
            painted = cond["repaints"]
        if cond["changed_parts"] is not None:
            changed = cond["changed_parts"]

    has_original = (painted == 0 and changed == 0)
    if re.search(r"boyasız|tramer\s+yok|hasar\s+kayd[iı]\s+yok", page_text, re.IGNORECASE):
        has_original = True

    result["has_original_paint"] = int(has_original)
    result["painted_panel_count"] = painted
    result["changed_panel_count"] = changed
    result["has_local_paint"] = int(has_local)
    result["total_damage_score"] = painted + changed * 2

    return result


def _parse_spec_table(soup: BeautifulSoup) -> dict:
    result = {
        "fuel_type": None, "transmission": None, "body_type": None,
        "engine_cc": None, "hp": None, "color": None,
        "warranty": None, "from_dealer": None, "num_owners": None,
    }

    page_text = soup.get_text(separator="\n", strip=True)

    fuel_map = {
        "benzin": "Benzin", "dizel": "Dizel", "lpg": "LPG",
        "hibrit": "Hibrit", "hybrid": "Hibrit",
        "elektrik": "Elektrik", "electric": "Elektrik",
    }
    for kw, val in fuel_map.items():
        if re.search(kw, page_text, re.IGNORECASE):
            result["fuel_type"] = val
            break

    if re.search(r"otomatik|automatic", page_text, re.IGNORECASE):
        result["transmission"] = "Otomatik"
    elif re.search(r"yarı\s*otomatik|semi[\s-]auto", page_text, re.IGNORECASE):
        result["transmission"] = "Yarı Otomatik"
    elif re.search(r"manuel|manual|düz\s*vites", page_text, re.IGNORECASE):
        result["transmission"] = "Manuel"

    body_map = {
        r"sedan": "Sedan",
        r"hatchback|hatch\s*back": "Hatchback",
        r"suv|arazi": "SUV",
        r"station\s*wagon|steyşın|station": "Station Wagon",
        r"coupe|kupe": "Coupe",
        r"cabrio|convertible|cabriolet": "Cabrio",
        r"mpv|minivan|van": "MPV",
        r"pickup|kamyonet": "Pickup",
    }
    for pattern, val in body_map.items():
        if re.search(pattern, page_text, re.IGNORECASE):
            result["body_type"] = val
            break

    m = re.search(r"\b(\d[.,]\d)\s*(?:lt?|litre?|cc|cm3)?\b", page_text, re.IGNORECASE)
    if m:
        result["engine_cc"] = m.group(1).replace(",", ".")

    m = re.search(r"\b(\d{2,4})\s*(?:hp|beygir|bg|ps|kw)\b", page_text, re.IGNORECASE)
    if m:
        try:
            hp_val = int(m.group(1))
            if 30 <= hp_val <= 1500:
                result["hp"] = hp_val
        except ValueError:
            pass

    color_keywords = [
        "Beyaz", "Siyah", "Gri", "Gümüş", "Kırmızı", "Mavi", "Yeşil",
        "Sarı", "Turuncu", "Kahverengi", "Bej", "Bordo", "Lacivert",
        "Mor", "Pembe", "Altın", "Bronz",
    ]
    for color in color_keywords:
        if re.search(color, page_text, re.IGNORECASE):
            result["color"] = color
            break

    result["warranty"] = 1 if re.search(
        r"garantili|garanti\s+var|fabrika\s+garanti", page_text, re.IGNORECASE
    ) else 0

    if re.search(r"galeriden|galeri\s+ilan|yetkili\s+sat", page_text, re.IGNORECASE):
        result["from_dealer"] = 1
    elif re.search(r"sahibinden|bireysel\s+ilan", page_text, re.IGNORECASE):
        result["from_dealer"] = 0

    m = re.search(r"(\d+)\.\s*el\b|(\d+)\s*el\b", page_text, re.IGNORECASE)
    if m:
        try:
            owners = int(m.group(1) or m.group(2))
            if 1 <= owners <= 10:
                result["num_owners"] = owners
        except (ValueError, TypeError):
            pass

    return result


def _scrape_detail(url: str) -> dict:
    empty = {
        "errors": None, "repaints": None, "changed_parts": None, "heavy_damage": None,
        "has_original_paint": None, "painted_panel_count": None,
        "changed_panel_count": None, "has_local_paint": None, "total_damage_score": None,
        "fuel_type": None, "transmission": None, "body_type": None,
        "engine_cc": None, "hp": None, "color": None,
        "warranty": None, "from_dealer": None, "num_owners": None,
    }
    if not url:
        return empty

    resp = _get(url, delay_after=False, use_lock=True)
    if resp is None:
        return empty

    try:
        soup = BeautifulSoup(resp.content, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)

        cond = _parse_condition_text(page_text)
        result = dict(empty)
        result.update(cond)
        result.update(_parse_damage_table(soup))
        result.update(_parse_spec_table(soup))

        return result

    except Exception as e:
        logger.debug(f"Detay parse hatası ({url}): {e}")
        return empty


# ═══════════════════════════════════════════════════════
#  LİSTİNG SATIRI PARSE
# ═══════════════════════════════════════════════════════

def _empty_listing_fields() -> dict:
    return {
        "fuel_type": None, "transmission": None, "body_type": None,
        "engine_cc": None, "hp": None, "color": None,
        "warranty": None, "from_dealer": None, "num_owners": None,
        "has_original_paint": None, "painted_panel_count": None,
        "changed_panel_count": None, "has_local_paint": None, "total_damage_score": None,
        "errors": None, "repaints": None, "changed_parts": None, "heavy_damage": None,
    }


def _parse_row(row, marka: str, model_query: str, debug: bool = False) -> Optional[dict]:
    tds = []
    try:
        tds = row.find_all("td")
        if len(tds) < 6:
            return None

        model_full = tds[1].get_text(strip=True)
        title_div  = tds[1].find("div", class_="listing-title-lines")
        title      = title_div.get_text(strip=True) if title_div else model_full

        year  = _parse_year(tds[2].get_text(strip=True))
        km    = _parse_km(tds[3].get_text(strip=True))
        if km is None:
            return None

        price_td   = tds[5]
        price_span = price_td.find("span", class_="listing-price")
        price_text = price_span.get_text(strip=True) if price_span else price_td.get_text(strip=True)
        price      = _parse_price(price_text)
        if price is None:
            return None

        location_text = tds[7].get_text(separator=" ", strip=True) if len(tds) > 7 else ""
        location      = _parse_location(location_text)
        paket         = _extract_paket(model_full, marka, model_query)
        detail_url    = _get_detail_url(row)

        listing = {
            "brand": marka.replace("-", " ").title(),
            "title": title,
            "price": price,
            "km": km,
            "year": year,
            "model": model_query.title(),
            "paket": paket,
            "location": location,
            "detail_url": detail_url,
        }
        listing.update(_empty_listing_fields())
        return listing

    except Exception as e:
        logger.debug(f"Satır parse hatası: {e}")
        return None


def _parse_category_row(row, category_slug: str, debug: bool = False) -> Optional[dict]:
    tds = []
    try:
        tds = row.find_all("td")
        if len(tds) < 6:
            return None

        model_full = tds[1].get_text(strip=True)
        title_div  = tds[1].find("div", class_="listing-title-lines")
        title      = title_div.get_text(strip=True) if title_div else model_full

        # brand/model'i URL'den çıkar
        detail_a = tds[1].find("a", href=True)
        brand = ""
        model = ""
        if detail_a:
            href = detail_a.get("href", "")
            m = re.search(r"/ikinci-el/[^/]+/([^/]+)-([^/?]+)", href)
            if m:
                brand = m.group(1).replace("-", " ").title()
                model = m.group(2).replace("-", " ").title()
        if not brand:
            parts = model_full.split()
            brand = parts[0] if parts else ""
            model = " ".join(parts[1:2]) if len(parts) > 1 else ""

        year = _parse_year(tds[2].get_text(strip=True))
        km   = _parse_km(tds[3].get_text(strip=True))
        if km is None:
            return None

        price_td   = tds[5]
        price_span = price_td.find("span", class_="listing-price")
        price_text = price_span.get_text(strip=True) if price_span else price_td.get_text(strip=True)
        price      = _parse_price(price_text)
        if price is None:
            return None

        location_text = tds[7].get_text(separator=" ", strip=True) if len(tds) > 7 else ""
        location      = _parse_location(location_text)
        detail_url    = _get_detail_url(row)

        listing = {
            "brand": brand,
            "title": title,
            "price": price,
            "km": km,
            "year": year,
            "model": model,
            "paket": model_full,
            "location": location,
            "detail_url": detail_url,
        }
        listing.update(_empty_listing_fields())
        return listing

    except Exception as e:
        logger.debug(f"Kategori satır parse hatası: {e}")
        return None


# ═══════════════════════════════════════════════════════
#  DETAY SAYFALARINI PARALEL DOLDUR
# ═══════════════════════════════════════════════════════

DETAIL_FIELDS = [
    "errors", "repaints", "changed_parts", "heavy_damage",
    "has_original_paint", "painted_panel_count", "changed_panel_count",
    "has_local_paint", "total_damage_score",
    "fuel_type", "transmission", "body_type",
    "engine_cc", "hp", "color",
    "warranty", "from_dealer", "num_owners",
]


def _fill_detail_pages(df: pd.DataFrame, progress_callback: Callable = None) -> pd.DataFrame:
    total      = len(df)
    found      = 0
    done       = 0
    MAX_WORKERS = SCRAPE_CONFIG.get("detail_workers", 2)

    logger.info(f"Detay sayfaları çekiliyor ({total} ilan, {MAX_WORKERS} thread)…")

    tasks = [
        (idx, row["detail_url"])
        for idx, row in df.iterrows()
        if row.get("detail_url")
    ]

    _done_lock = threading.Lock()

    def _fetch(idx_url):
        return idx_url[0], _scrape_detail(idx_url[1])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                idx, detail = future.result()
                for field in DETAIL_FIELDS:
                    if field in detail:
                        df.at[idx, field] = detail[field]
                if any(v is not None for v in detail.values()):
                    found += 1
            except Exception as e:
                logger.debug(f"Detay future hatası: {e}")

            with _done_lock:
                done += 1
                snap_done  = done
                snap_found = found

            if progress_callback:
                progress_callback(
                    None, total,
                    f"Detay verisi: {snap_done}/{total} ilan ({snap_found} bulundu)"
                )

    logger.info(f"Detay tamamlandı: {found}/{total} ilandan veri bulundu")
    return df


# ═══════════════════════════════════════════════════════
#  ANA SCRAPE FONKSİYONLARI
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
    """Tek marka+model için ilanları çek (geriye dönük uyumlu)."""
    logger.info(f"Scraping: {marka}/{model_query} | detay={fetch_details}")

    cfg       = SCRAPE_CONFIG
    limit     = max_pages or cfg["max_pages"]
    all_rows  = []
    consec    = 0
    dumped    = False
    ts        = datetime.now(timezone.utc).isoformat()

    for page in range(1, limit + 1):
        url     = build_search_url(marka, model_query, page)
        referer = (build_search_url(marka, model_query, page - 1)
                   if page > 1 else "https://www.arabam.com/ikinci-el/otomobil")

        if progress_callback:
            progress_callback(page, len(all_rows), f"Sayfa {page}/{limit} taranıyor…")

        resp = _get(url, referer=referer, debug=debug and not dumped)
        if resp is not None and debug and not dumped:
            dumped = True
        if resp is None:
            consec += 1
            if consec >= 2:
                break
            continue

        soup = BeautifulSoup(resp.content, "html.parser")
        rows = soup.find_all("tr", class_="listing-list-item")

        if not rows:
            consec += 1
            if consec >= 2:
                break
            continue

        consec = 0
        for row in rows:
            listing = _parse_row(row, marka, model_query, debug=debug)
            if listing:
                listing["scrape_timestamp"] = ts
                all_rows.append(listing)

        logger.info(f"Sayfa {page}: {len(rows)} satır → toplam {len(all_rows)} ilan")

    if not all_rows:
        logger.error("Hiç ilan çekilemedi!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    if fetch_details:
        df = _fill_detail_pages(df, progress_callback)

    df = df.drop("detail_url", axis=1, errors="ignore")
    df = df.drop_duplicates(subset=["title", "price", "km"], keep="first")

    if save:
        try:
            RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
            logger.info(f"Kaydedildi: {RAW_DATA_PATH} ({len(df)} ilan)")
        except OSError as e:
            logger.warning(f"CSV kaydetme hatası: {e}")

    return df


def scrape_category(
    category_url: str,
    max_pages: int = 30,
    fetch_details: bool = True,
    progress_callback: Callable = None,
    save: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Genel kategori sayfasını scrape et (/ikinci-el/otomobil vb.)."""
    logger.info(f"Kategori scraping: {category_url} | sayfa_limit={max_pages}")

    cfg          = SCRAPE_CONFIG
    all_rows     = []
    consec       = 0
    dumped       = False
    ts           = datetime.now(timezone.utc).isoformat()
    cat_slug     = category_url.strip("/").split("/")[-1]

    for page in range(1, max_pages + 1):
        url     = build_category_url(category_url, page)
        referer = (build_category_url(category_url, page - 1)
                   if page > 1 else "https://www.arabam.com/")

        if progress_callback:
            progress_callback(page, len(all_rows), f"[{cat_slug}] Sayfa {page}/{max_pages}…")

        resp = _get(url, referer=referer, debug=debug and not dumped)
        if resp is not None and debug and not dumped:
            dumped = True
        if resp is None:
            consec += 1
            if consec >= 2:
                break
            continue

        soup = BeautifulSoup(resp.content, "html.parser")
        rows = soup.find_all("tr", class_="listing-list-item")

        if not rows:
            consec += 1
            if consec >= 2:
                break
            continue

        consec = 0
        for row in rows:
            listing = _parse_category_row(row, cat_slug, debug=debug)
            if listing:
                listing["scrape_timestamp"] = ts
                all_rows.append(listing)

        logger.info(f"[{cat_slug}] Sayfa {page}: {len(rows)} satır → toplam {len(all_rows)}")

    if not all_rows:
        logger.error(f"[{cat_slug}] Hiç ilan çekilemedi!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    if fetch_details:
        df = _fill_detail_pages(df, progress_callback)

    df = df.drop("detail_url", axis=1, errors="ignore")
    df = df.drop_duplicates(subset=["title", "price", "km"], keep="first")

    if save:
        try:
            RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            if RAW_DATA_PATH.exists():
                existing = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["title", "price", "km"], keep="first")
            df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
            logger.info(f"Kaydedildi: {RAW_DATA_PATH} ({len(df)} toplam ilan)")
        except OSError as e:
            logger.warning(f"CSV kaydetme hatası: {e}")

    return df


def scrape_all_categories(
    max_pages_per_category: int = 30,
    fetch_details: bool = True,
    progress_callback: Callable = None,
) -> pd.DataFrame:
    """Tüm kategori URL'lerini (otomobil + suv) scrape et ve birleştir."""
    all_dfs = []
    for cat_url in CATEGORY_URLS:
        df = scrape_category(
            category_url=cat_url,
            max_pages=max_pages_per_category,
            fetch_details=fetch_details,
            progress_callback=progress_callback,
            save=False,
        )
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["title", "price", "km"], keep="first")

    try:
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"Tüm kategoriler kaydedildi: {RAW_DATA_PATH} ({len(combined)} ilan)")
    except OSError as e:
        logger.warning(f"CSV kaydetme hatası: {e}")

    return combined


# ── Test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    marka = sys.argv[1] if len(sys.argv) > 1 else "alfa-romeo"
    model = sys.argv[2] if len(sys.argv) > 2 else "tonale"
    df = scrape_listings(marka, model, fetch_details=True, max_pages=3)
    cols = [c for c in ["title", "price", "km", "year", "fuel_type",
                         "transmission", "total_damage_score"] if c in df.columns]
    print(df[cols].to_string())
