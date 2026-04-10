# scraper.py — arabam.com ikinci el ilan scraper (requests + BeautifulSoup)
#
# HTML yapısı (2026-04 itibarıyla doğrulandı):
#   <tr class='listing-list-item'>
#     td[0] → resim / link
#     td[1] → Model adı (kısa)
#     td[2] → Tam başlık / açıklama
#     td[3] → Yıl
#     td[4] → KM  ("125.000")
#     td[5] → Renk
#     td[6] → Fiyat  (listing-price span, "1.299.000 TL")
#     td[7] → Tarih
#     td[8] → Konum + butonlar

import re
import time
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional, Callable

import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import (
    SCRAPE_CONFIG, DATA_BOUNDS,
    BASE_URL, RAW_DATA_PATH, build_search_url, build_category_url,
    CATEGORY_URLS,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP katmanı
# ─────────────────────────────────────────────────────────────────────────────

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

# Detay istekeleri için thread-local session (her thread kendi session'ını kullanır)
_thread_local = threading.local()

# Rate-limiting için global kilit (detay sayfaları paralel ama kibar olsun)
_rate_lock = threading.Lock()


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
        "Referer":                 referer or "https://www.arabam.com/",
    }
    if "Firefox" not in ua:
        headers.update({
            "sec-ch-ua":          '"Chromium";v="124","Google Chrome";v="124","Not-A.Brand";v="99"',
            "sec-ch-ua-mobile":   "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest":     "document",
            "sec-fetch-mode":     "navigate",
            "sec-fetch-site":     "same-origin",
            "sec-fetch-user":     "?1",
        })
    return headers


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_make_headers())
    return s


def _fetch_html(
    session: requests.Session,
    url: str,
    max_retries: int = None,
) -> Optional[str]:
    """GET isteği at; 403/429/503'te üstel geri çekilme ile yeniden dene."""
    cfg     = SCRAPE_CONFIG
    retries = max_retries if max_retries is not None else cfg.get("max_retries", 3)

    for attempt in range(retries):
        try:
            resp = session.get(
                url,
                headers=_make_headers(referer="https://www.arabam.com/"),
                timeout=cfg.get("timeout", 15),
            )
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (403, 429, 503):
                wait = cfg.get("backoff_base", 8) * (2 ** attempt) + random.uniform(0, 2)
                logger.warning("HTTP %d — %.1fs bekleniyor (deneme %d/%d): %s",
                               resp.status_code, wait, attempt + 1, retries, url)
                time.sleep(wait)
                continue
            logger.warning("HTTP %d: %s", resp.status_code, url)
            return None
        except requests.RequestException as exc:
            wait = 4 * (2 ** attempt)
            logger.warning("İstek hatası (deneme %d/%d): %s — %.1fs bekleniyor",
                           attempt + 1, retries, exc, wait)
            time.sleep(wait)

    logger.error("Tüm denemeler başarısız: %s", url)
    return None


def _fetch_detail_html(url: str) -> str:
    """Thread-local session ile detay sayfasını çek."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _make_session()

    with _rate_lock:
        time.sleep(random.uniform(
            SCRAPE_CONFIG.get("delay_min", 1.5),
            SCRAPE_CONFIG.get("delay_max", 3.0),
        ))

    html = _fetch_html(_thread_local.session, url, max_retries=2)
    return html or ""


def _dump_debug_html(url: str, html: str, path: str = "debug_page.html") -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Debug HTML kaydedildi: %s  (URL: %s)", path, url)
    except Exception as e:
        logger.warning("Debug HTML kaydedilemedi: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
#  Parse yardımcı fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────

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
    for word in text.strip().split():
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
        text, re.IGNORECASE,
    )
    if m:
        result["heavy_damage"] = 1 if m.group(1).lower() == "var" else 0
    else:
        if re.search(r"a[gğ][iı]r\s+hasar\s+var", text, re.IGNORECASE):
            result["heavy_damage"] = 1
        elif re.search(
            r"a[gğ][iı]r\s+hasar\s+yok|a[gğ][iı]r\s+hasar\s+kayd[iı]\s+bulunmamak",
            text, re.IGNORECASE,
        ):
            result["heavy_damage"] = 0

    return result


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
        for cell in damage_table.find_all(["td", "li", "span"]):
            text = cell.get_text(strip=True).lower()
            if "boyalı" in text or (
                "boya" in text and "orijinal" not in text and "lokal" not in text
            ):
                try:
                    painted += int(re.search(r"\d+", text).group())
                except (AttributeError, ValueError):
                    painted += 1
            elif "değişen" in text or "degisen" in text:
                try:
                    changed += int(re.search(r"\d+", text).group())
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

    has_original = painted == 0 and changed == 0
    if re.search(r"boyasız|tramer\s+yok|hasar\s+kayd[iı]\s+yok", page_text, re.IGNORECASE):
        has_original = True

    result["has_original_paint"]  = int(has_original)
    result["painted_panel_count"] = painted
    result["changed_panel_count"] = changed
    result["has_local_paint"]     = int(has_local)
    result["total_damage_score"]  = painted + changed * 2
    return result


def _parse_spec_table(soup: BeautifulSoup) -> dict:
    result = {
        "fuel_type": None, "transmission": None, "body_type": None,
        "engine_cc": None, "hp": None, "color": None,
        "warranty": None, "from_dealer": None, "num_owners": None,
    }

    page_text = soup.get_text(separator="\n", strip=True)

    for kw, val in {
        "benzin": "Benzin", "dizel": "Dizel", "lpg": "LPG",
        "hibrit": "Hibrit", "hybrid": "Hibrit",
        "elektrik": "Elektrik", "electric": "Elektrik",
    }.items():
        if re.search(kw, page_text, re.IGNORECASE):
            result["fuel_type"] = val
            break

    if re.search(r"otomatik|automatic", page_text, re.IGNORECASE):
        result["transmission"] = "Otomatik"
    elif re.search(r"yarı\s*otomatik|semi[\s-]auto", page_text, re.IGNORECASE):
        result["transmission"] = "Yarı Otomatik"
    elif re.search(r"manuel|manual|düz\s*vites", page_text, re.IGNORECASE):
        result["transmission"] = "Manuel"

    for pattern, val in {
        r"sedan":                      "Sedan",
        r"hatchback|hatch\s*back":     "Hatchback",
        r"suv|arazi":                  "SUV",
        r"station\s*wagon|steyşın":    "Station Wagon",
        r"coupe|kupe":                 "Coupe",
        r"cabrio|convertible":         "Cabrio",
        r"mpv|minivan|van":            "MPV",
        r"pickup|kamyonet":            "Pickup",
    }.items():
        if re.search(pattern, page_text, re.IGNORECASE):
            result["body_type"] = val
            break

    # Motor hacmi: 1.0, 1.4, 1.6, 2.0 veya 1598 cc gibi
    m = re.search(
        r"\b(?:Motor|Silindir)\s+Hacmi[^\d]*(\d{3,4}|\d\.\d{1,2})\b"
        r"|\b(\d{3,4}|\d\.\d{1,2})\s*(?:cc|cm[3³])\b"
        r"|\b(\d\.\d{1,2})\s*(?:lt?|litre)\b",
        page_text, re.IGNORECASE
    )
    if m:
        val = next(v for v in m.groups() if v)
        result["engine_cc"] = val.replace(",", ".")

    m = re.search(r"\b(\d{2,4})\s*(?:hp|beygir|bg|ps|kw)\b", page_text, re.IGNORECASE)
    if m:
        try:
            hp_val = int(m.group(1))
            if 30 <= hp_val <= 1500:
                result["hp"] = hp_val
        except ValueError:
            pass

    for color in ["Beyaz", "Siyah", "Gri", "Gümüş", "Kırmızı", "Mavi", "Yeşil",
                  "Sarı", "Turuncu", "Kahverengi", "Bej", "Bordo", "Lacivert",
                  "Mor", "Pembe", "Altın", "Bronz"]:
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


# ─────────────────────────────────────────────────────────────────────────────
#  Detay HTML parse (URL almaz, HTML string alır)
# ─────────────────────────────────────────────────────────────────────────────

def _empty_detail() -> dict:
    return {
        "errors": None, "repaints": None, "changed_parts": None, "heavy_damage": None,
        "has_original_paint": None, "painted_panel_count": None,
        "changed_panel_count": None, "has_local_paint": None, "total_damage_score": None,
        "fuel_type": None, "transmission": None, "body_type": None,
        "engine_cc": None, "hp": None, "color": None,
        "warranty": None, "from_dealer": None, "num_owners": None,
    }


def _parse_detail_html(html: str) -> dict:
    """Detay sayfası HTML'inden hasar + spec verisini çıkar."""
    if not html:
        return _empty_detail()
    try:
        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)

        result = _empty_detail()
        result.update(_parse_condition_text(page_text))
        result.update(_parse_damage_table(soup))
        result.update(_parse_spec_table(soup))
        return result
    except Exception as e:
        logger.debug("Detay parse hatası: %s", e)
        return _empty_detail()


# ─────────────────────────────────────────────────────────────────────────────
#  Listing satırı parse
# ─────────────────────────────────────────────────────────────────────────────

def _empty_listing_fields() -> dict:
    return {
        "fuel_type": None, "transmission": None, "body_type": None,
        "engine_cc": None, "hp": None, "color": None,
        "warranty": None, "from_dealer": None, "num_owners": None,
        "has_original_paint": None, "painted_panel_count": None,
        "changed_panel_count": None, "has_local_paint": None, "total_damage_score": None,
        "errors": None, "repaints": None, "changed_parts": None, "heavy_damage": None,
    }


def _get_detail_url(row) -> Optional[str]:
    for a_tag in row.find_all("a", href=True):
        href = a_tag["href"]
        if "/ilan/" in href:
            return href if href.startswith("http") else BASE_URL + href
    return None


def _parse_row(row, marka: str, model_query: str, debug: bool = False) -> Optional[dict]:
    try:
        tds = row.find_all("td")
        if len(tds) < 7:
            return None

        model_full = tds[1].get_text(strip=True)
        title_div  = tds[2].find("div", class_="listing-title-lines")
        title      = (title_div.get_text(strip=True) if title_div
                      else tds[2].get_text(strip=True) or model_full)

        year  = _parse_year(tds[3].get_text(strip=True))
        km    = _parse_km(tds[4].get_text(strip=True))
        if km is None:
            return None

        price_td   = tds[6]
        price_span = price_td.find("span", class_="listing-price")
        price_text = price_span.get_text(strip=True) if price_span else price_td.get_text(strip=True)
        price      = _parse_price(price_text)
        if price is None:
            return None

        location_text = tds[8].get_text(separator=" ", strip=True) if len(tds) > 8 else ""
        listing = {
            "brand":      marka.replace("-", " ").title(),
            "title":      title,
            "price":      price,
            "km":         km,
            "year":       year,
            "model":      model_query.title(),
            "paket":      _extract_paket(model_full, marka, model_query),
            "location":   _parse_location(location_text),
            "detail_url": _get_detail_url(row),
        }
        listing.update(_empty_listing_fields())
        return listing
    except Exception as e:
        logger.debug("Satır parse hatası: %s", e)
        return None


def _parse_category_row(row, category_slug: str, debug: bool = False) -> Optional[dict]:
    try:
        tds = row.find_all("td")
        if len(tds) < 7:
            return None

        model_full = tds[1].get_text(strip=True)
        title_div  = tds[2].find("div", class_="listing-title-lines")
        title      = (title_div.get_text(strip=True) if title_div
                      else tds[2].get_text(strip=True) or model_full)

        detail_a = row.find("a", href=re.compile(r"/ilan/"))
        brand = model = ""
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

        year = _parse_year(tds[3].get_text(strip=True))
        km   = _parse_km(tds[4].get_text(strip=True))
        if km is None:
            return None

        price_td   = tds[6]
        price_span = price_td.find("span", class_="listing-price")
        price_text = price_span.get_text(strip=True) if price_span else price_td.get_text(strip=True)
        price      = _parse_price(price_text)
        if price is None:
            return None

        location_text = tds[8].get_text(separator=" ", strip=True) if len(tds) > 8 else ""
        listing = {
            "brand":      brand,
            "title":      title,
            "price":      price,
            "km":         km,
            "year":       year,
            "model":      model,
            "paket":      model_full,
            "location":   _parse_location(location_text),
            "detail_url": _get_detail_url(row),
        }
        listing.update(_empty_listing_fields())
        return listing
    except Exception as e:
        logger.debug("Kategori satır parse hatası: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Detay sayfalarını paralel çek (ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

DETAIL_FIELDS = [
    "errors", "repaints", "changed_parts", "heavy_damage",
    "has_original_paint", "painted_panel_count", "changed_panel_count",
    "has_local_paint", "total_damage_score",
    "fuel_type", "transmission", "body_type",
    "engine_cc", "hp", "color",
    "warranty", "from_dealer", "num_owners",
]


def _fill_detail_pages(
    df: pd.DataFrame,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    workers  = SCRAPE_CONFIG.get("detail_workers", 3)
    tasks    = [(idx, row["detail_url"])
                for idx, row in df.iterrows() if row.get("detail_url")]
    total    = len(df)
    done     = 0
    found    = 0
    lock     = threading.Lock()

    logger.info("Detay sayfaları çekiliyor (%d ilan, %d worker)…", len(tasks), workers)

    def fetch_one(idx_url):
        idx, url = idx_url
        html   = _fetch_detail_html(url)
        detail = _parse_detail_html(html)
        return idx, detail

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                idx, detail = future.result()
                with lock:
                    done += 1
                    if any(v is not None for v in detail.values()):
                        found += 1
                    snap_done, snap_found = done, found

                for field in DETAIL_FIELDS:
                    if field in detail:
                        df.at[idx, field] = detail[field]

                if progress_callback:
                    progress_callback(
                        snap_done, total,
                        f"Detay: {snap_done}/{total} ilan ({snap_found} bulundu)",
                    )
            except Exception as exc:
                logger.debug("Detay future hatası: %s", exc)

    logger.info("Detay tamamlandı: %d/%d ilandan veri bulundu", found, total)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Çekirdek scraping fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_listings_core(
    marka: str,
    model_query: str,
    max_pages: int,
    fetch_details: bool,
    progress_callback: Optional[Callable],
    debug: bool,
) -> pd.DataFrame:
    cfg      = SCRAPE_CONFIG
    all_rows = []
    consec   = 0
    dumped   = False
    ts       = datetime.now(timezone.utc).isoformat()
    session  = _make_session()

    # Warmup: ana sayfayı ziyaret et
    try:
        session.get("https://www.arabam.com/",
                    headers=_make_headers(), timeout=10)
        time.sleep(random.uniform(1.0, 2.0))
    except Exception:
        pass

    for page_num in range(1, max_pages + 1):
        url = build_search_url(marka, model_query, page_num)

        if progress_callback:
            progress_callback(page_num, max_pages,
                              f"Sayfa {page_num}/{max_pages} taranıyor...")

        logger.info("Sayfa %d/%d: %s", page_num, max_pages, url)
        html = _fetch_html(session, url)

        if html is None:
            consec += 1
            if consec >= 2:
                break
            continue

        if debug and not dumped:
            _dump_debug_html(url, html)
            dumped = True

        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr", class_="listing-list-item")

        if not rows:
            consec += 1
            logger.warning("Sayfa %d: ilan satırı bulunamadı (consec=%d)", page_num, consec)
            if consec >= 2:
                break
            continue

        consec = 0
        for row in rows:
            listing = _parse_row(row, marka, model_query, debug=debug)
            if listing:
                listing["scrape_timestamp"] = ts
                all_rows.append(listing)

        logger.info("Sayfa %d: %d satır -> toplam %d ilan",
                    page_num, len(rows), len(all_rows))

        time.sleep(random.uniform(
            cfg.get("delay_min", 1.5),
            cfg.get("delay_max", 3.0),
        ))

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    if fetch_details:
        df = _fill_detail_pages(df, progress_callback)

    return df


def _scrape_category_core(
    category_url: str,
    max_pages: int,
    fetch_details: bool,
    progress_callback: Optional[Callable],
    debug: bool,
) -> pd.DataFrame:
    cfg      = SCRAPE_CONFIG
    all_rows = []
    consec   = 0
    dumped   = False
    ts       = datetime.now(timezone.utc).isoformat()
    cat_slug = category_url.strip("/").split("/")[-1]
    session  = _make_session()

    try:
        session.get("https://www.arabam.com/",
                    headers=_make_headers(), timeout=10)
        time.sleep(random.uniform(1.0, 2.0))
    except Exception:
        pass

    for page_num in range(1, max_pages + 1):
        url = build_category_url(category_url, page_num)

        if progress_callback:
            progress_callback(page_num, max_pages,
                              f"[{cat_slug}] Sayfa {page_num}/{max_pages}...")

        logger.info("[%s] Sayfa %d/%d", cat_slug, page_num, max_pages)
        html = _fetch_html(session, url)

        if html is None:
            consec += 1
            if consec >= 2:
                break
            continue

        if debug and not dumped:
            _dump_debug_html(url, html)
            dumped = True

        soup = BeautifulSoup(html, "html.parser")
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

        logger.info("[%s] Sayfa %d: %d satir -> toplam %d ilan",
                    cat_slug, page_num, len(rows), len(all_rows))

        time.sleep(random.uniform(
            cfg.get("delay_min", 1.5),
            cfg.get("delay_max", 3.0),
        ))

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    if fetch_details:
        df = _fill_detail_pages(df, progress_callback)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Herkese açık API
# ─────────────────────────────────────────────────────────────────────────────

def scrape_listings(
    marka: str,
    model_query: str,
    save: bool = True,
    max_pages: int = None,
    fetch_details: bool = True,
    progress_callback: Optional[Callable] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Tek marka+model için ilanları çek."""
    logger.info("Scraping: %s/%s | detay=%s", marka, model_query, fetch_details)

    df = _scrape_listings_core(
        marka, model_query,
        max_pages or SCRAPE_CONFIG["max_pages"],
        fetch_details, progress_callback, debug,
    )

    if df is None or df.empty:
        logger.error("Hic ilan cekilemedi!")
        return pd.DataFrame()

    df = df.drop("detail_url", axis=1, errors="ignore")
    df = df.drop_duplicates(subset=["title", "price", "km"], keep="first")

    if save:
        try:
            RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
            logger.info("Kaydedildi: %s (%d ilan)", RAW_DATA_PATH, len(df))
        except OSError as e:
            logger.warning("CSV kaydetme hatasi: %s", e)

    return df


def scrape_category(
    category_url: str,
    max_pages: int = 30,
    fetch_details: bool = True,
    progress_callback: Optional[Callable] = None,
    save: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Genel kategori sayfasini scrape et (/ikinci-el/otomobil vb.)."""
    logger.info("Kategori scraping: %s | sayfa_limit=%d", category_url, max_pages)

    df = _scrape_category_core(
        category_url, max_pages, fetch_details, progress_callback, debug,
    )

    if df is None or df.empty:
        logger.error("[%s] Hic ilan cekilemedi!", category_url)
        return pd.DataFrame()

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
            logger.info("Kaydedildi: %s (%d toplam ilan)", RAW_DATA_PATH, len(df))
        except OSError as e:
            logger.warning("CSV kaydetme hatasi: %s", e)

    return df


def scrape_all_categories(
    max_pages_per_category: int = 30,
    fetch_details: bool = True,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """Tum CATEGORY_URLS'leri (otomobil + suv) scrape et ve birlestir."""
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
        logger.info("Tum kategoriler kaydedildi: %s (%d ilan)",
                    RAW_DATA_PATH, len(combined))
    except OSError as e:
        logger.warning("CSV kaydetme hatasi: %s", e)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="arabam.com scraper (requests+BS4)")
    parser.add_argument("--mode", choices=["single", "category", "all"], default="single")
    parser.add_argument("--marka", default="volkswagen")
    parser.add_argument("--model", default="golf")
    parser.add_argument("--pages", type=int, default=30)
    parser.add_argument("--no-details", action="store_true",
                        help="Detay sayfalarini cekme")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    fetch_details = not args.no_details

    if args.mode == "single":
        df = scrape_listings(args.marka, args.model,
                             max_pages=args.pages,
                             fetch_details=fetch_details,
                             debug=args.debug)
    elif args.mode == "category":
        df = scrape_category("/ikinci-el/otomobil",
                             max_pages=args.pages,
                             fetch_details=fetch_details,
                             debug=args.debug)
    else:
        df = scrape_all_categories(max_pages_per_category=args.pages,
                                   fetch_details=fetch_details)

    print(f"Toplam {len(df)} ilan cekildi -> data/raw_listings.csv")
    if not df.empty:
        cols = [c for c in ["title", "price", "km", "year", "fuel_type",
                             "transmission", "total_damage_score"] if c in df.columns]
        print(df[cols].head(10).to_string())
