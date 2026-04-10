# scraper.py — arabam.com ikinci el ilan scraper (nodriver tabanlı)
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
#
# Kurulum:
#   pip install nodriver

import re
import random
import asyncio
import logging
from datetime import datetime, timezone
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional, Callable

import nodriver as uc

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
#  Browser yardımcıları (nodriver)
# ─────────────────────────────────────────────────────────────────────────────

async def _get_browser() -> uc.Browser:
    browser = await uc.start(
        headless=False,
        browser_args=[
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
        ],
    )
    return browser


async def _fetch_page(browser: uc.Browser, url: str, wait_selector: str = None) -> str:
    """URL'yi mevcut sekmede aç, isteğe bağlı selector bekle, HTML döndür."""
    tab = await browser.get(url)
    await asyncio.sleep(random.uniform(2.0, 4.0))
    if wait_selector:
        try:
            await tab.find(wait_selector, timeout=10)
        except Exception:
            pass
    try:
        content = await tab.get_content()
    except Exception as e:
        logger.warning("get_content hatası (%s): %s", url, e)
        content = ""
    return content


async def _fetch_detail_tab(browser: uc.Browser, url: str) -> str:
    """Detay sayfasını YENİ sekmede aç, HTML döndür, sekmeyi kapat."""
    tab = None
    try:
        tab = await browser.get(url, new_tab=True)
        await asyncio.sleep(random.uniform(2.0, 3.5))
        try:
            await tab.find(".technical-properties", timeout=8)
        except Exception:
            try:
                await tab.find(".classified-detail", timeout=5)
            except Exception:
                pass
        try:
            content = await tab.get_content()
        except Exception as e:
            logger.debug("Detay get_content hatası (%s): %s", url, e)
            content = ""
        return content
    except Exception as e:
        logger.debug("Detay sekme hatası (%s): %s", url, e)
        return ""
    finally:
        if tab is not None:
            try:
                await tab.close()
            except Exception:
                pass


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

        # brand/model'i ilan URL'sinden çıkar
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
#  Detay sayfalarını paralel çek (asyncio.gather + Semaphore, max 3 tab)
# ─────────────────────────────────────────────────────────────────────────────

DETAIL_FIELDS = [
    "errors", "repaints", "changed_parts", "heavy_damage",
    "has_original_paint", "painted_panel_count", "changed_panel_count",
    "has_local_paint", "total_damage_score",
    "fuel_type", "transmission", "body_type",
    "engine_cc", "hp", "color",
    "warranty", "from_dealer", "num_owners",
]


async def _fill_detail_pages_async(
    browser: uc.Browser,
    df: pd.DataFrame,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    total      = len(df)
    semaphore  = asyncio.Semaphore(3)  # max 3 eşzamanlı sekme
    done_count = 0
    found_count = 0
    count_lock = asyncio.Lock()

    tasks = [
        (idx, row["detail_url"])
        for idx, row in df.iterrows()
        if row.get("detail_url")
    ]
    logger.info(
        "Detay sayfaları çekiliyor (%d ilan, max 3 eşzamanlı)…", len(tasks)
    )

    async def fetch_one(idx: int, url: str):
        nonlocal done_count, found_count
        async with semaphore:
            html   = await _fetch_detail_tab(browser, url)
            detail = _parse_detail_html(html)

        async with count_lock:
            done_count += 1
            if any(v is not None for v in detail.values()):
                found_count += 1
            snap_done  = done_count
            snap_found = found_count

        logger.debug("Detay: %d/%d (%d bulundu)", snap_done, total, snap_found)
        return idx, detail

    results = await asyncio.gather(
        *[fetch_one(idx, url) for idx, url in tasks],
        return_exceptions=True,
    )

    for result in results:
        if isinstance(result, Exception):
            logger.debug("Detay future hatası: %s", result)
            continue
        idx, detail = result
        for field in DETAIL_FIELDS:
            if field in detail:
                df.at[idx, field] = detail[field]

    logger.info("Detay tamamlandı: %d/%d ilandan veri bulundu", found_count, total)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Async çekirdek fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

async def _async_scrape_listings(
    marka: str,
    model_query: str,
    max_pages: int,
    fetch_details: bool,
    progress_callback: Optional[Callable],
    debug: bool,
) -> pd.DataFrame:
    cfg      = SCRAPE_CONFIG
    limit    = max_pages or cfg["max_pages"]
    all_rows = []
    consec   = 0
    dumped   = False
    ts       = datetime.now(timezone.utc).isoformat()

    browser = await _get_browser()

    try:
        # Warmup: ana sayfayı ziyaret et
        logger.info("Warmup: arabam.com ana sayfası açılıyor…")
        try:
            await browser.get("https://www.arabam.com/")
            await asyncio.sleep(random.uniform(3.0, 5.0))
        except Exception as e:
            logger.warning("Warmup hatası: %s", e)

        for page_num in range(1, limit + 1):
            url = build_search_url(marka, model_query, page_num)
            logger.info("Sayfa %d/%d: %s", page_num, limit, url)

            html = await _fetch_page(browser, url, wait_selector="tr.listing-list-item")

            if not html:
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

            logger.info("Sayfa %d: %d satır → toplam %d ilan",
                        page_num, len(rows), len(all_rows))

            await asyncio.sleep(random.uniform(1.5, 3.0))

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        if fetch_details:
            df = await _fill_detail_pages_async(browser, df, progress_callback)

    finally:
        try:
            browser.stop()
        except Exception:
            pass

    return df


async def _async_scrape_category(
    category_url: str,
    max_pages: int,
    fetch_details: bool,
    progress_callback: Optional[Callable],
    debug: bool,
) -> pd.DataFrame:
    all_rows = []
    consec   = 0
    dumped   = False
    ts       = datetime.now(timezone.utc).isoformat()
    cat_slug = category_url.strip("/").split("/")[-1]

    browser = await _get_browser()

    try:
        logger.info("Warmup: arabam.com ana sayfası açılıyor…")
        try:
            await browser.get("https://www.arabam.com/")
            await asyncio.sleep(random.uniform(3.0, 5.0))
        except Exception as e:
            logger.warning("Warmup hatası: %s", e)

        for page_num in range(1, max_pages + 1):
            url = build_category_url(category_url, page_num)
            logger.info("[%s] Sayfa %d/%d", cat_slug, page_num, max_pages)

            html = await _fetch_page(browser, url, wait_selector="tr.listing-list-item")

            if not html:
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

            logger.info("[%s] Sayfa %d: %d satır → toplam %d ilan",
                        cat_slug, page_num, len(rows), len(all_rows))

            await asyncio.sleep(random.uniform(1.5, 3.0))

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        if fetch_details:
            df = await _fill_detail_pages_async(browser, df, progress_callback)

    finally:
        try:
            browser.stop()
        except Exception:
            pass

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Herkese açık API (senkron sarmalayıcılar)
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

    df = asyncio.run(_async_scrape_listings(
        marka, model_query,
        max_pages or SCRAPE_CONFIG["max_pages"],
        fetch_details, progress_callback, debug,
    ))

    if df is None or df.empty:
        logger.error("Hiç ilan çekilemedi!")
        return pd.DataFrame()

    df = df.drop("detail_url", axis=1, errors="ignore")
    df = df.drop_duplicates(subset=["title", "price", "km"], keep="first")

    if save:
        try:
            RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
            logger.info("Kaydedildi: %s (%d ilan)", RAW_DATA_PATH, len(df))
        except OSError as e:
            logger.warning("CSV kaydetme hatası: %s", e)

    return df


def scrape_category(
    category_url: str,
    max_pages: int = 30,
    fetch_details: bool = True,
    progress_callback: Optional[Callable] = None,
    save: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Genel kategori sayfasını scrape et (/ikinci-el/otomobil vb.)."""
    logger.info("Kategori scraping: %s | sayfa_limit=%d", category_url, max_pages)

    df = asyncio.run(_async_scrape_category(
        category_url, max_pages, fetch_details, progress_callback, debug,
    ))

    if df is None or df.empty:
        logger.error("[%s] Hiç ilan çekilemedi!", category_url)
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
            logger.warning("CSV kaydetme hatası: %s", e)

    return df


def scrape_all_categories(
    max_pages_per_category: int = 30,
    fetch_details: bool = True,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """Tüm CATEGORY_URLS'leri (otomobil + suv) scrape et ve birleştir."""
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
        logger.info("Tüm kategoriler kaydedildi: %s (%d ilan)",
                    RAW_DATA_PATH, len(combined))
    except OSError as e:
        logger.warning("CSV kaydetme hatası: %s", e)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="arabam.com scraper (nodriver)")
    parser.add_argument("--mode", choices=["single", "category", "all"], default="single")
    parser.add_argument("--marka", default="volkswagen")
    parser.add_argument("--model", default="golf")
    parser.add_argument("--pages", type=int, default=30)
    parser.add_argument("--no-details", action="store_true",
                        help="Detay sayfalarını çekme")
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
