# app.py — arabam.com Fiyat Tahmin Uygulaması (Streamlit)
#
# Çalıştırma: streamlit run app.py
# Ortam değişkeni: GROQ_API_KEY (.env veya st.secrets["GROQ_API_KEY"])

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
import traceback
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── proje kökü sys.path'e eklenir ─────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from config import (
    RAW_DATA_PATH, CLEANED_DATA_PATH, CATEGORY_URLS,
    RF_MODEL_PATH, ENGINEER_PATH, FEATURE_COLUMNS_PATH,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sayfa yapılandırması
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Arabam Fiyat Tahmini",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Koyu tema, premium görünüm
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Temel renkler ── */
:root {
  --bg:        #070B12;
  --card:      #0C111B;
  --border:    #1E2738;
  --accent:    #F59E0B;
  --accent2:   #3B82F6;
  --text:      #C1CEDE;
  --textdim:   #6B7C93;
  --success:   #10B981;
  --warning:   #F59E0B;
  --danger:    #EF4444;
  --radius:    12px;
}

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: var(--bg); }
section[data-testid="stSidebar"] { background: var(--card) !important; border-right: 1px solid var(--border); }
.block-container { padding: 1.5rem 2rem !important; max-width: 1200px; }

/* ── Kartlar ── */
.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.5rem;
  text-align: center;
  transition: border-color .2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card .label { font-size: .75rem; color: var(--textdim); text-transform: uppercase; letter-spacing:.06em; margin-bottom:.3rem; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
.metric-card .sub   { font-size: .8rem; color: var(--textdim); margin-top:.2rem; }

/* ── Fiyat kutusu ── */
.price-box {
  background: linear-gradient(135deg, #0f1929 0%, #0c111b 100%);
  border: 2px solid var(--accent);
  border-radius: 16px;
  padding: 2rem;
  text-align: center;
  box-shadow: 0 0 40px rgba(245,158,11,.08);
}
.price-box .price-label { font-size:.85rem; color:var(--textdim); text-transform:uppercase; letter-spacing:.08em; margin-bottom:.5rem; }
.price-box .price-main  { font-size: 2.8rem; font-weight: 800; color: var(--accent); line-height:1.1; }
.price-box .price-range { font-size:.9rem; color:var(--textdim); margin-top:.6rem; }

/* ── Badge ── */
.badge {
  display:inline-block; padding:.3rem .8rem;
  border-radius:20px; font-size:.8rem; font-weight:600; letter-spacing:.04em;
}
.badge-green  { background:#0d2e22; color:#10B981; border:1px solid #10B981; }
.badge-yellow { background:#2b2106; color:#F59E0B; border:1px solid #F59E0B; }
.badge-orange { background:#2b1a06; color:#F97316; border:1px solid #F97316; }
.badge-red    { background:#2b0a0a; color:#EF4444; border:1px solid #EF4444; }

/* ── Hero bar ── */
.hero {
  background: linear-gradient(135deg, #0c111b 0%, #111827 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin-bottom: 1.5rem;
  display: flex; align-items: center; gap: 1rem;
}
.hero-icon { font-size:2.4rem; }
.hero-title { font-size:1.6rem; font-weight:800; color:#E5E7EB; margin:0; }
.hero-sub   { font-size:.9rem; color:var(--textdim); margin:0; }

/* ── Adım göstergesi ── */
.steps { display:flex; gap:.5rem; margin-bottom:1.5rem; }
.step-item {
  flex:1; padding:.6rem 1rem; border-radius:8px; text-align:center;
  font-size:.8rem; font-weight:600; border:1px solid var(--border);
  color:var(--textdim); background:var(--card);
}
.step-item.active { border-color:var(--accent); color:var(--accent); background:#1a1400; }
.step-item.done   { border-color:var(--success); color:var(--success); background:#091a13; }

/* ── Tablo ── */
.sim-table { width:100%; border-collapse:collapse; font-size:.85rem; }
.sim-table th { color:var(--textdim); border-bottom:1px solid var(--border); padding:.5rem .8rem; text-align:left; }
.sim-table td { padding:.5rem .8rem; border-bottom:1px solid var(--border); color:var(--text); }
.sim-table tr:hover td { background:rgba(245,158,11,.04); }

/* ── Hasar rozeti ── */
.damage-section {
  background: var(--card); border: 1px solid var(--border);
  border-radius:var(--radius); padding:1rem 1.2rem; margin-top:.5rem;
}
.hasar-diff {
  background:#0d1a2e; border:1px solid var(--accent2);
  border-radius:var(--radius); padding:1rem 1.5rem; text-align:center;
}
.hasar-diff .label { font-size:.8rem; color:var(--textdim); }
.hasar-diff .amount { font-size:1.4rem; font-weight:700; color:var(--accent2); }

/* ── Genel buton ── */
.stButton > button {
  border-radius: 8px !important; font-weight: 600 !important;
  transition: all .2s !important;
}

/* ── Expander ── */
.streamlit-expanderHeader { color: var(--text) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Marka / Model listesi
# ─────────────────────────────────────────────────────────────────────────────

MARKA_MODELLER: dict[str, list[str]] = {
    "alfa-romeo":  ["giulia", "stelvio", "tonale", "giulietta", "156"],
    "audi":        ["a1", "a3", "a4", "a5", "a6", "a7", "a8", "q2", "q3", "q5", "q7", "q8", "e-tron"],
    "bmw":         ["1-serisi", "2-serisi", "3-serisi", "4-serisi", "5-serisi", "7-serisi", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
    "chery":       ["tiggo-4-pro", "tiggo-7-pro", "tiggo-8-pro"],
    "citroen":     ["c3", "c3-aircross", "c4", "c5-aircross", "berlingo"],
    "dacia":       ["duster", "sandero", "logan", "jogger", "spring"],
    "fiat":        ["egea", "doblo", "panda", "500", "tipo"],
    "ford":        ["fiesta", "focus", "mondeo", "kuga", "puma", "mustang-mach-e", "transit-connect"],
    "honda":       ["civic", "jazz", "hr-v", "cr-v", "accord"],
    "hyundai":     ["i10", "i20", "i30", "tucson", "kona", "santa-fe", "ioniq5", "ioniq6", "bayon"],
    "jeep":        ["renegade", "compass", "wrangler", "gladiator", "grand-cherokee"],
    "kia":         ["picanto", "rio", "ceed", "xceed", "stonic", "sportage", "sorento", "ev6"],
    "land-rover":  ["defender", "discovery", "discovery-sport", "range-rover", "range-rover-sport", "range-rover-evoque"],
    "mazda":       ["mazda2", "mazda3", "mazda6", "cx-3", "cx-30", "cx-5", "cx-60"],
    "mercedes":    ["a-serisi", "b-serisi", "c-serisi", "e-serisi", "s-serisi", "cla", "gla", "glb", "glc", "gle", "gls"],
    "mg":          ["zs", "hs", "5", "4"],
    "mitsubishi":  ["eclipse-cross", "outlander", "asx", "l200"],
    "nissan":      ["micra", "juke", "qashqai", "x-trail", "ariya", "leaf"],
    "opel":        ["corsa", "astra", "mokka", "crossland", "grandland", "zafira"],
    "peugeot":     ["208", "308", "408", "508", "2008", "3008", "5008"],
    "porsche":     ["cayenne", "macan", "taycan", "panamera", "911"],
    "renault":     ["clio", "megane", "taliant", "zoe", "captur", "kadjar", "koleos", "scenic"],
    "seat":        ["ibiza", "leon", "arona", "ateca", "tarraco"],
    "skoda":       ["fabia", "octavia", "superb", "kamiq", "karoq", "kodiaq", "enyaq"],
    "subaru":      ["forester", "outback", "xv", "impreza"],
    "suzuki":      ["swift", "vitara", "s-cross", "jimny", "baleno"],
    "tesla":       ["model-3", "model-s", "model-x", "model-y"],
    "toyota":      ["yaris", "corolla", "camry", "c-hr", "rav4", "highlander", "bz4x", "land-cruiser"],
    "volkswagen":  ["polo", "golf", "passat", "jetta", "arteon", "t-cross", "t-roc", "tiguan", "tiguan-allspace", "touareg", "id4"],
    "volvo":       ["s60", "s90", "v40", "v60", "v90", "xc40", "xc60", "xc90"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Session state başlangıcı
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "step":          1,
        "raw_df":        None,
        "clean_df":      None,
        "feature_eng":   None,
        "predictor":     None,
        "train_metrics": None,
        "feature_cols":  None,
        "training_brands": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı bileşenler
# ─────────────────────────────────────────────────────────────────────────────

def _hero():
    st.markdown("""
    <div class="hero">
      <span class="hero-icon">🚗</span>
      <div>
        <p class="hero-title">arabam.com Fiyat Tahmini</p>
        <p class="hero-sub">Gerçek ilan verisiyle eğitilmiş yapay zeka fiyat motoru</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _step_indicator(current: int):
    steps = ["1 · Veri Toplama", "2 · Model Eğitimi", "3 · Fiyat Tahmini"]
    parts = []
    for i, label in enumerate(steps, 1):
        cls = "active" if i == current else ("done" if i < current else "")
        parts.append(f'<div class="step-item {cls}">{label}</div>')
    st.markdown(f'<div class="steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def _metric_card(label: str, value: str, sub: str = ""):
    return f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="sub">{sub}</div>
    </div>"""


def _damage_badge(score: int) -> str:
    if score == 0:
        return '<span class="badge badge-green">🟢 Hasarsız</span>'
    elif score <= 3:
        return '<span class="badge badge-yellow">🟡 Hafif Hasar</span>'
    elif score <= 7:
        return '<span class="badge badge-orange">🟠 Orta Hasar</span>'
    else:
        return '<span class="badge badge-red">🔴 Ağır Hasar</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Önbellek yükleyiciler
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_raw_data(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return None


@st.cache_resource(show_spinner=False)
def _load_predictor():
    try:
        from train import EnsemblePredictor
        return EnsemblePredictor.load()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Scraping
# ─────────────────────────────────────────────────────────────────────────────

def _run_scrape(mode: str, marka: str, model: str, max_pages: int):
    """Scraping çalıştırır ve DataFrame döndürür."""
    from scraper import scrape_listings, scrape_all_categories

    status = st.empty()
    progress = st.progress(0)
    log_area = st.empty()
    log_lines: list[str] = []

    def cb(msg: str, pct: float = 0.0):
        log_lines.append(msg)
        if len(log_lines) > 6:
            log_lines.pop(0)
        log_area.code("\n".join(log_lines))
        progress.progress(min(pct, 1.0))

    status.info("Scraping başlıyor...")
    try:
        if mode == "category":
            df = scrape_all_categories(
                max_pages=max_pages,
                fetch_details=True,
                progress_callback=cb,
            )
        else:
            df = scrape_listings(
                marka=marka,
                model_query=model,
                max_pages=max_pages,
                fetch_details=True,
                progress_callback=cb,
            )
    except Exception as e:
        status.error(f"Scraping hatası: {e}")
        st.text(traceback.format_exc())
        return None

    progress.progress(1.0)
    if df is None or df.empty:
        status.warning("Hiç ilan bulunamadı.")
        return None

    status.success(f"{len(df):,} ilan indirildi.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Temizleme
# ─────────────────────────────────────────────────────────────────────────────

def _run_clean(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        from data_cleaner import clean_data
        clean = clean_data(df)
        return clean
    except Exception as e:
        st.error(f"Veri temizleme hatası: {e}")
        st.text(traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Eğitim
# ─────────────────────────────────────────────────────────────────────────────

def _run_train(clean_df: pd.DataFrame):
    """FeatureEngineer + EnsembleTrainer çalıştırır."""
    from features import FeatureEngineer
    from train import EnsembleTrainer

    status = st.empty()

    status.info("Özellikler çıkarılıyor...")
    fe = FeatureEngineer()
    try:
        X_train, X_test, y_train, y_test, feature_cols = fe.fit_transform(clean_df)
    except Exception as e:
        status.error(f"Feature engineering hatası: {e}")
        st.text(traceback.format_exc())
        return None, None, None

    status.info(
        f"Eğitim seti: {len(X_train):,} satır  |  Test seti: {len(X_test):,} satır  |  {len(feature_cols)} özellik"
    )

    status.info("Ensemble modeli eğitiliyor (RF + GB + XGB)...")
    trainer = EnsembleTrainer()
    try:
        metrics = trainer.train(X_train, y_train, X_test, y_test, feature_cols)
    except Exception as e:
        status.error(f"Model eğitim hatası: {e}")
        st.text(traceback.format_exc())
        return None, None, None

    trainer.save()
    fe.save()

    status.success("Eğitim tamamlandı ve modeller kaydedildi.")

    from train import EnsemblePredictor
    predictor = EnsemblePredictor.load()

    return fe, predictor, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Tahmin yardımcıları
# ─────────────────────────────────────────────────────────────────────────────

def _build_input(
    brand: str, year: int, km: int, fuel_type: str, transmission: str,
    body_type: str, city: str, hp: int, color: str, engine_cc: str,
    warranty: bool, from_dealer: bool,
    has_original_paint: bool, painted_panel_count: int,
    changed_panel_count: int, has_local_paint: bool,
    model_str: str = "",
) -> dict:
    total_damage = painted_panel_count + changed_panel_count * 2
    return {
        "brand":               brand,
        "model":               model_str,
        "year":                year,
        "km":                  km,
        "fuel_type":           fuel_type,
        "transmission":        transmission,
        "body_type":           body_type,
        "location":            city,
        "hp":                  hp,
        "color":               color,
        "engine_cc":           engine_cc,
        "warranty":            int(warranty),
        "from_dealer":         int(from_dealer),
        "has_original_paint":  int(has_original_paint),
        "painted_panel_count": painted_panel_count,
        "changed_panel_count": changed_panel_count,
        "has_local_paint":     int(has_local_paint),
        "total_damage_score":  total_damage,
        "num_owners":          1,
        # legacy fields
        "errors":              0,
        "repaints":            painted_panel_count,
        "changed_parts":       changed_panel_count,
        "heavy_damage":        0,
    }


def _find_similar(
    df: pd.DataFrame, brand: str, year: int, km: int, n: int = 5
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    mask = (
        (df["brand"].str.lower() == brand.lower()) &
        (df["year"].between(year - 2, year + 2)) &
        (df["km"].between(max(0, km - 30_000), km + 30_000))
    )
    sub = df[mask].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("price").head(n)
    cols = [c for c in ["brand", "model", "year", "km", "price", "location", "fuel_type"] if c in sub.columns]
    return sub[cols].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Groq AI yorumu
# ─────────────────────────────────────────────────────────────────────────────

def _get_groq_commentary(
    brand: str, year: int, km: int, predicted_price: float,
    low: float, high: float, damage_score: int, test_r2: float | None,
) -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    if not api_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        hasar_text = {0: "hiç hasarsız", 1: "hafif hasarlı (skor 1-3)"}.get(
            damage_score,
            "orta hasarlı (skor 4-7)" if damage_score <= 7 else "ağır hasarlı (skor 8+)"
        )
        r2_text = f"%.1f%%" % (test_r2 * 100) if test_r2 else "bilinmiyor"

        prompt = (
            f"Sen deneyimli bir Türk araç değerleme uzmanısın. "
            f"Bir {year} model {brand} aracının fiyatı tahmin edildi. "
            f"Araç {km:,} km'de, {hasar_text}. "
            f"Tahmin edilen fiyat: {predicted_price:,.0f} TL "
            f"(güven aralığı: {low:,.0f} – {high:,.0f} TL). "
            f"Model doğruluğu (R²): {r2_text}. "
            f"Bu tahmini değerlendir: fiyat piyasaya göre uygun mu, "
            f"hasar durumu fiyatı nasıl etkiler, alıcıya ne tavsiye edersin? "
            f"3-4 cümle, sade Türkçe, madde işareti kullanma."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.getLogger(__name__).warning("Groq hatası: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance grafiği
# ─────────────────────────────────────────────────────────────────────────────

def _feature_importance_chart(predictor, top_n: int = 15):
    from features import DAMAGE_FEATURES
    try:
        df_imp = predictor.get_feature_importances().head(top_n)
    except Exception:
        return

    colors = [
        "#F87171" if any(d in row["feature"] for d in DAMAGE_FEATURES)
        else "#F59E0B"
        for _, row in df_imp.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0C111B")
    ax.set_facecolor("#0C111B")

    bars = ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1],
                   color=colors[::-1], edgecolor="none", height=0.6)
    ax.set_xlabel("Önem", color="#6B7C93", fontsize=9)
    ax.tick_params(colors="#C1CEDE", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color("#6B7C93")
    ax.grid(axis="x", color="#1E2738", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        '<span style="color:#F87171">■</span> Hasar özellikleri  '
        '<span style="color:#F59E0B">■</span> Diğer özellikler',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Veri özeti kartları
# ─────────────────────────────────────────────────────────────────────────────

def _show_data_summary(df: pd.DataFrame):
    cols = st.columns(4)
    n = len(df)
    avg_price = df["price"].mean() if "price" in df.columns else 0
    avg_km    = df["km"].mean()    if "km"    in df.columns else 0
    n_brands  = df["brand"].nunique() if "brand" in df.columns else 0

    cards = [
        ("İlan Sayısı",    f"{n:,}",              "toplam kayıt"),
        ("Ort. Fiyat",     f"₺{avg_price:,.0f}",  ""),
        ("Ort. KM",        f"{avg_km:,.0f}",       "km"),
        ("Marka Sayısı",   f"{n_brands}",          ""),
    ]
    for col, (label, val, sub) in zip(cols, cards):
        col.markdown(_metric_card(label, val, sub), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ADIM 1 — Veri Toplama
# ─────────────────────────────────────────────────────────────────────────────

def _render_step1():
    _step_indicator(1)
    st.subheader("Veri Toplama")

    # Mod seçimi
    mode = st.radio(
        "Scraping modu",
        ["Tek Marka / Model", "Genel Kategori (Otomobil + SUV)"],
        horizontal=True,
    )

    col1, col2 = st.columns(2)
    marka = model_q = ""

    if mode == "Tek Marka / Model":
        with col1:
            marka = st.selectbox("Marka", sorted(MARKA_MODELLER.keys()), index=list(sorted(MARKA_MODELLER.keys())).index("volkswagen"))
        with col2:
            modeller = MARKA_MODELLER.get(marka, [""])
            model_q = st.selectbox("Model", modeller)
        scrape_mode = "single"
    else:
        st.info("Her iki kategori taranacak: `/ikinci-el/otomobil` + `/ikinci-el/suv-arazi-arac`")
        scrape_mode = "category"

    max_pages = st.slider("Maksimum sayfa sayısı", 1, 50, 20)

    # Önceki veri varsa göster
    existing = _load_raw_data(str(RAW_DATA_PATH))
    if existing is not None:
        with st.expander(f"Mevcut veri: {len(existing):,} ilan — yeniden scrape etmek için aşağıdaki butonu kullan"):
            _show_data_summary(existing)
            if st.button("Bu veriyi kullan (scrape atla)"):
                st.session_state.raw_df = existing
                clean = _run_clean(existing)
                if clean is not None:
                    st.session_state.clean_df = clean
                    st.session_state.step = 2
                    st.rerun()

    if st.button("🔍 Veri Topla", type="primary"):
        df = _run_scrape(scrape_mode, marka, model_q, max_pages)
        if df is not None:
            st.session_state.raw_df = df
            clean = _run_clean(df)
            if clean is not None:
                st.session_state.clean_df = clean
                _load_raw_data.clear()
                st.session_state.step = 2
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ADIM 2 — Model Eğitimi
# ─────────────────────────────────────────────────────────────────────────────

def _render_step2():
    _step_indicator(2)
    st.subheader("Model Eğitimi")

    clean_df = st.session_state.get("clean_df")
    if clean_df is None:
        st.warning("Önce veri toplayın.")
        if st.button("← Geri"):
            st.session_state.step = 1
            st.rerun()
        return

    _show_data_summary(clean_df)
    st.caption(f"Eğitim verisi: {len(clean_df):,} temizlenmiş ilan")

    if st.button("🤖 Modeli Eğit", type="primary"):
        fe, predictor, metrics = _run_train(clean_df)
        if fe and predictor and metrics:
            st.session_state.feature_eng = fe
            st.session_state.predictor   = predictor
            st.session_state.train_metrics = metrics

            # Markalar
            if "brand" in clean_df.columns:
                st.session_state.training_brands = sorted(clean_df["brand"].dropna().unique().tolist())

            # Metrik kartları
            st.markdown("### Model Performansı")
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(_metric_card("Test R²",   f"{metrics['test_r2']:.3f}",  ""), unsafe_allow_html=True)
            c2.markdown(_metric_card("Test MAE",  f"₺{metrics['test_mae']:,.0f}", ""), unsafe_allow_html=True)
            c3.markdown(_metric_card("Test MAPE", f"{metrics['test_mape']:.1f}%", ""), unsafe_allow_html=True)
            c4.markdown(_metric_card("Eğitim",    f"{metrics['n_train']:,}", "kayıt"), unsafe_allow_html=True)

            # Ağırlıklar
            w = metrics.get("weights", [1/3, 1/3, 1/3])
            st.caption(
                f"Ensemble ağırlıkları → RF: {w[0]:.2f}  GB: {w[1]:.2f}  XGB: {w[2]:.2f}"
            )

            st.session_state.step = 3
            st.rerun()

    if st.button("← Geri"):
        st.session_state.step = 1
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ADIM 3 — Fiyat Tahmini
# ─────────────────────────────────────────────────────────────────────────────

def _render_step3():
    _step_indicator(3)

    predictor = st.session_state.get("predictor")
    fe        = st.session_state.get("feature_eng")
    metrics   = st.session_state.get("train_metrics", {})

    # Yüklü model yoksa diskten dene
    if predictor is None:
        predictor = _load_predictor()
        if predictor:
            st.session_state.predictor = predictor

    if predictor is None or fe is None:
        st.warning("Model yüklü değil. Lütfen önce eğitim adımını tamamlayın.")
        col1, col2 = st.columns(2)
        if col1.button("← Eğitime Dön"):
            st.session_state.step = 2
            st.rerun()
        if col2.button("← Veriye Dön"):
            st.session_state.step = 1
            st.rerun()
        return

    # ── Sidebar giriş formu ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔧 Araç Özellikleri")

        brands = st.session_state.get("training_brands") or sorted(MARKA_MODELLER.keys())
        brand = st.selectbox("Marka", brands)

        col_a, col_b = st.columns(2)
        year = col_a.slider("Yıl", 2005, 2025, 2020)
        km   = col_b.number_input("Kilometre", min_value=0, max_value=500_000,
                                   value=80_000, step=5_000)

        fuel_type    = st.selectbox("Yakıt Tipi",  ["Dizel", "Benzin", "LPG", "Elektrik", "Hibrit"])
        transmission = st.selectbox("Vites",        ["Otomatik", "Manuel", "Yarı Otomatik"])
        body_type    = st.selectbox("Kasa Tipi",    ["Sedan", "Hatchback", "SUV", "Crossover", "Station Wagon", "Coupe", "Cabrio", "Pickup", "Van"])
        city         = st.selectbox("Şehir",        [
            "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya",
            "Adana", "Konya", "Mersin", "Kocaeli", "Gaziantep",
            "Eskişehir", "Samsun", "Kayseri", "Denizli", "Diğer",
        ])

        col_c, col_d = st.columns(2)
        hp       = col_c.number_input("Beygir Gücü", min_value=50, max_value=600, value=120, step=5)
        engine_s = col_d.selectbox("Motor", ["1.0", "1.2", "1.4", "1.5", "1.6", "2.0", "2.5", "3.0", "Diğer"])

        color = st.selectbox("Renk", [
            "Beyaz", "Siyah", "Gri", "Gümüş", "Mavi", "Kırmızı",
            "Kahverengi", "Yeşil", "Sarı", "Turuncu", "Bordo", "Bej",
            "Mor", "Lacivert", "Diğer",
        ])

        col_e, col_f = st.columns(2)
        warranty    = col_e.checkbox("Garantili", value=False)
        from_dealer = col_f.checkbox("Galeriden", value=False)

        st.markdown("---")
        with st.expander("🔧 Hasar Bilgisi"):
            has_original_paint  = st.checkbox("Orijinal boya (tüm paneller)", value=True)
            painted_panel_count = st.slider("Boyalı panel sayısı", 0, 10, 0)
            changed_panel_count = st.slider("Değişen parça sayısı", 0, 6, 0)
            has_local_paint     = st.checkbox("Lokal boya var", value=False)

            total_damage = painted_panel_count + changed_panel_count * 2
            st.markdown(
                f'<div style="margin-top:.5rem">{_damage_badge(total_damage)}'
                f'<span style="color:#6B7C93; font-size:.8rem; margin-left:.6rem">'
                f'Hasar skoru: {total_damage}</span></div>',
                unsafe_allow_html=True,
            )

        predict_btn = st.button("🔍 Tahmin Et", type="primary", use_container_width=True)

    # ── Ana panel ───────────────────────────────────────────────────────────
    st.subheader("Fiyat Tahmini")

    if not predict_btn:
        st.info("Sol panelden araç özelliklerini girin ve **Tahmin Et** butonuna tıklayın.")
        return

    # Build input dict
    inp = _build_input(
        brand=brand, year=year, km=km,
        fuel_type=fuel_type, transmission=transmission,
        body_type=body_type, city=city, hp=hp,
        color=color, engine_cc=engine_s,
        warranty=warranty, from_dealer=from_dealer,
        has_original_paint=has_original_paint,
        painted_panel_count=painted_panel_count,
        changed_panel_count=changed_panel_count,
        has_local_paint=has_local_paint,
    )

    try:
        X = fe.transform(inp)
        price = float(predictor.predict(X)[0])
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
        st.text(traceback.format_exc())
        return

    low  = price * 0.85
    high = price * 1.15

    # ── Fiyat kutusu ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="price-box">
      <div class="price-label">Tahmini Piyasa Değeri</div>
      <div class="price-main">₺{price:,.0f}</div>
      <div class="price-range">Güven aralığı: ₺{low:,.0f} — ₺{high:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # ── Hasar Etkisi ─────────────────────────────────────────────────
        if total_damage > 0:
            inp_nodmg = {**inp,
                         "has_original_paint":  1,
                         "painted_panel_count": 0,
                         "changed_panel_count": 0,
                         "has_local_paint":     0,
                         "total_damage_score":  0}
            try:
                X_nd = fe.transform(inp_nodmg)
                price_nodmg = float(predictor.predict(X_nd)[0])
                diff = price_nodmg - price
                st.markdown(f"""
                <div class="hasar-diff">
                  <div class="label">Hasarsız olsaydı tahmini değer</div>
                  <div class="amount">₺{price_nodmg:,.0f}</div>
                  <div class="label" style="margin-top:.4rem">
                    Hasar nedeniyle kayıp:
                    <strong style="color:#EF4444">-₺{diff:,.0f}</strong>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            except Exception:
                pass

        # ── Feature importance ────────────────────────────────────────────
        st.markdown("#### Özellik Önem Sırası (Top 15)")
        _feature_importance_chart(predictor, top_n=15)

    with col_right:
        # ── Benzer ilanlar ────────────────────────────────────────────────
        clean_df = st.session_state.get("clean_df") or _load_raw_data(str(CLEANED_DATA_PATH))
        sim = _find_similar(clean_df, brand, year, km)
        if sim is not None:
            st.markdown("#### Benzer İlanlar")
            rows = ""
            for _, row in sim.iterrows():
                price_str = f"₺{row['price']:,.0f}" if pd.notna(row.get("price")) else "-"
                km_str    = f"{row['km']:,.0f} km"  if pd.notna(row.get("km"))    else "-"
                rows += (
                    f"<tr><td>{row.get('brand','')}</td>"
                    f"<td>{row.get('year','')}</td>"
                    f"<td>{km_str}</td>"
                    f"<td><strong>{price_str}</strong></td>"
                    f"<td>{row.get('location','')}</td></tr>"
                )
            st.markdown(f"""
            <table class="sim-table">
              <thead><tr>
                <th>Marka</th><th>Yıl</th><th>KM</th><th>Fiyat</th><th>Şehir</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

        # ── Groq yorumu ───────────────────────────────────────────────────
        test_r2 = metrics.get("test_r2") if metrics else None
        commentary = _get_groq_commentary(
            brand=brand, year=year, km=km,
            predicted_price=price, low=low, high=high,
            damage_score=total_damage, test_r2=test_r2,
        )
        if commentary:
            st.markdown("#### Yapay Zeka Yorumu")
            st.info(commentary)
        elif not os.getenv("GROQ_API_KEY"):
            st.caption("💡 AI yorumu için `.env` dosyasına `GROQ_API_KEY` ekleyin.")


# ─────────────────────────────────────────────────────────────────────────────
# Ana akış
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _init_session()
    _hero()

    step = st.session_state.step

    if step == 1:
        _render_step1()
    elif step == 2:
        _render_step2()
    elif step == 3:
        _render_step3()


if __name__ == "__main__":
    main()
