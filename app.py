"""
app.py — Arabam.com Fiyat Tahmin Modeli — Streamlit Arayüzü

Akış:
  Adım 1 → Marka / Model gir, veriyi çek
  Adım 2 → Modeli eğit, sonuçları gör
  Adım 3 → Araç özelliklerini gir, fiyat tahmini al
"""

import io
import logging
import warnings

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OtoEstimate — Araç Fiyat AI",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium Automotive Dark Theme CSS ───────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════════════════════════════
   CSS DEĞİŞKENLERİ — tek yerden yönet
   ═══════════════════════════════════════════════ */
:root {
    --bg:          #070B12;
    --surface:     #0C111B;
    --surface2:    #111826;
    --border:      #1C2535;
    --border2:     #263040;
    --amber:       #F59E0B;
    --amber-dim:   rgba(245,158,11,0.12);
    --amber-glow:  rgba(245,158,11,0.30);
    --teal:        #2DD4BF;
    --green:       #10B981;
    --red:         #F87171;
    --text-hi:     #EEF2FF;
    --text-md:     #C1CEDE;
    --text-lo:     #7A8FA6;
    --text-dim:    #3D5068;
    --font-display: 'Syne', sans-serif;
    --font-body:    'Outfit', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ═══════════════════════════════════════════════
   GLOBAL RESET — icon fontlarına dokunmadan
   ═══════════════════════════════════════════════ */
html, body {
    font-family: var(--font-body) !important;
    box-sizing: border-box;
}
/* Sadece metin taşıyan elementleri hedefle — SVG/icon span'larına dokunma */
p, h1, h2, h3, h4, h5, h6,
input, textarea, select,
button, label, a, li, td, th,
.stMarkdown, .stMarkdown *,
[data-testid="stText"], [data-testid="stText"] *,
[data-testid="stAlert"] p,
[data-testid="stCaptionContainer"] p {
    font-family: var(--font-body) !important;
}
/* Streamlit widget container'ları — ama icon span'larını değil */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    font-family: var(--font-body) !important;
}

.stApp {
    background-color: var(--bg) !important;
    color: var(--text-md) !important;
    /* subtle grid lines — engineering drawing vibe */
    background-image:
        linear-gradient(rgba(28,37,53,0.4) 1px, transparent 1px),
        linear-gradient(90deg, rgba(28,37,53,0.4) 1px, transparent 1px);
    background-size: 40px 40px;
}

#MainMenu, footer, header { visibility: hidden !important; }

.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 1080px !important;
}

/* ═══════════════════════════════════════════════
   LABEL / WIDGET TEXT — kapsamlı override
   Streamlit'in tüm varyantlarını yakala
   ═══════════════════════════════════════════════ */
label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stSlider label,
.stCheckbox label,
div[class*="Widget"] label,
div[class*="widget"] label,
p[class*="label"],
span[class*="label"] {
    color: var(--text-md) !important;
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: none !important;
    opacity: 1 !important;
}

/* ═══════════════════════════════════════════════
   HERO BAR
   ═══════════════════════════════════════════════ */
.hero-wrap {
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
    margin-bottom: 24px;
}
.hero-logo {
    font-family: var(--font-display);
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--amber);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    line-height: 1;
    /* amber text glow */
    text-shadow: 0 0 40px rgba(245,158,11,0.35);
}
.hero-sub {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-lo);
    margin-top: 5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--amber-dim);
    border: 1px solid rgba(245,158,11,0.25);
    color: var(--amber);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 3px;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--amber);
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.7); }
}

/* ═══════════════════════════════════════════════
   STEP INDICATOR
   ═══════════════════════════════════════════════ */
.step-rail {
    display: flex;
    align-items: center;
    margin-bottom: 28px;
}
.step-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    flex: 0 0 auto;
    min-width: 90px;
}
.step-circle {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-display);
    font-size: 0.9rem;
    font-weight: 700;
    transition: all 0.3s ease;
}
.step-circle.done   { background: var(--green);  color: #070B12; box-shadow: 0 0 14px rgba(16,185,129,0.45); }
.step-circle.active { background: var(--amber);  color: #070B12; box-shadow: 0 0 20px var(--amber-glow); }
.step-circle.idle   { background: var(--surface2); color: var(--text-dim); border: 1px solid var(--border); }
.step-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; }
.step-label.done   { color: var(--green) !important; }
.step-label.active { color: var(--amber) !important; }
.step-label.idle   { color: var(--text-dim) !important; }
.step-connector {
    flex: 1;
    height: 1px;
    background: var(--border);
    margin: 0 6px;
    margin-bottom: 24px;
    transition: background 0.4s;
}
.step-connector.done { background: linear-gradient(90deg, var(--green), var(--border)); opacity: 0.6; }

/* ═══════════════════════════════════════════════
   SECTION HEADER
   ═══════════════════════════════════════════════ */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-hi);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.section-accent {
    width: 3px; height: 16px;
    background: var(--amber);
    border-radius: 2px;
    flex-shrink: 0;
    box-shadow: 0 0 8px var(--amber-glow);
}

/* ═══════════════════════════════════════════════
   METRIC TILES
   ═══════════════════════════════════════════════ */
.metric-row {
    display: flex;
    gap: 8px;
    margin: 14px 0;
    flex-wrap: wrap;
}
.metric-tile {
    flex: 1;
    min-width: 100px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    text-align: center;
    position: relative;
    transition: border-color 0.2s;
}
.metric-tile:hover { border-color: var(--border2); }
.metric-tile::after {
    content: '';
    position: absolute;
    bottom: 0; left: 20%; right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--amber-dim), transparent);
}
.metric-val {
    font-family: var(--font-mono);
    font-size: 1.25rem;
    font-weight: 500;
    color: var(--text-hi);
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.metric-lbl {
    font-size: 0.62rem;
    color: var(--text-lo);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ═══════════════════════════════════════════════
   PRICE RESULT CARD
   ═══════════════════════════════════════════════ */
.price-result {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-top: 2px solid var(--amber);
    border-radius: 8px;
    padding: 28px 22px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.price-result::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(245,158,11,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.price-eyebrow {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    color: var(--amber);
    text-transform: uppercase;
    margin-bottom: 10px;
}
.price-main {
    font-family: var(--font-display);
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--text-hi);
    line-height: 1;
    letter-spacing: -0.01em;
    text-shadow: 0 0 30px rgba(238,242,255,0.1);
}
.price-range {
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--text-lo);
    margin-top: 12px;
    line-height: 1.8;
}
.price-range b { color: var(--text-md); font-weight: 500; }

/* ═══════════════════════════════════════════════
   INPUT FIELDS
   ═══════════════════════════════════════════════ */
input, textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background-color: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 5px !important;
    color: var(--text-hi) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
input:focus, textarea:focus,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(245,158,11,0.18) !important;
    outline: none !important;
}
input::placeholder { color: var(--text-dim) !important; }

/* Selectbox */
div[data-testid="stSelectbox"] > div > div,
div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 5px !important;
    color: var(--text-hi) !important;
}
div[data-baseweb="select"] svg { fill: var(--text-lo) !important; }

/* Slider */
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--amber) !important;
    border-color: var(--amber) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] > div:first-child {
    background: var(--amber) !important;
}

/* ═══════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════ */
button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--amber) !important;
    color: #070B12 !important;
    border: none !important;
    font-family: var(--font-display) !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 5px !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #FBB92A !important;
    box-shadow: 0 4px 24px var(--amber-glow) !important;
    transform: translateY(-2px) !important;
}

button:not([kind="primary"]),
div[data-testid="stButton"] > button:not([kind="primary"]) {
    background: transparent !important;
    color: var(--text-lo) !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
}
button:not([kind="primary"]):hover,
div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
}

/* ═══════════════════════════════════════════════
   PROGRESS BAR
   ═══════════════════════════════════════════════ */
[data-testid="stProgress"] > div {
    background: var(--border) !important;
    border-radius: 3px !important;
    height: 3px !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--amber), #FCD34D) !important;
    border-radius: 3px !important;
}

/* ═══════════════════════════════════════════════
   ALERTS
   ═══════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
}
[data-testid="stAlert"][data-type="info"],
[data-testid="stAlert"] {
    background: rgba(245,158,11,0.07) !important;
    border: 1px solid rgba(245,158,11,0.2) !important;
    color: var(--text-md) !important;
}
[data-testid="stAlert"][data-type="success"] {
    background: rgba(16,185,129,0.07) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
}
[data-testid="stAlert"][data-type="error"] {
    background: rgba(248,113,113,0.07) !important;
    border: 1px solid rgba(248,113,113,0.2) !important;
}

/* ═══════════════════════════════════════════════
   EXPANDER
   ═══════════════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--surface) !important;
    overflow: hidden;
}
/* Summary'nin sadece renk/boyut'unu etkile — font-family'ye dokunma */
[data-testid="stExpander"] summary {
    color: var(--text-lo) !important;
    font-size: 0.82rem !important;
    padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--text-md) !important;
}
[data-testid="stExpander"] summary svg {
    fill: var(--text-lo) !important;
}

/* ═══════════════════════════════════════════════
   DATAFRAME
   ═══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden;
}

/* ═══════════════════════════════════════════════
   DIVIDER
   ═══════════════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 24px 0 !important;
}

/* ═══════════════════════════════════════════════
   CAPTION / SMALL TEXT
   ═══════════════════════════════════════════════ */
small,
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] {
    color: var(--text-lo) !important;
    font-size: 0.74rem !important;
    font-family: var(--font-body) !important;
}

/* ═══════════════════════════════════════════════
   MARKDOWN TEXT (genel okunabilirlik)
   ═══════════════════════════════════════════════ */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
    color: var(--text-md) !important;
    font-family: var(--font-body) !important;
}
.stMarkdown code {
    background: var(--surface2) !important;
    color: var(--teal) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 1px 5px !important;
    font-family: var(--font-mono) !important;
}

/* ═══════════════════════════════════════════════
   DOWNLOAD BUTTON
   ═══════════════════════════════════════════════ */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--text-lo) !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    font-size: 0.78rem !important;
    font-family: var(--font-body) !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    color: var(--amber) !important;
    border-color: var(--amber) !important;
}

/* ═══════════════════════════════════════════════
   SPINNER
   ═══════════════════════════════════════════════ */
[data-testid="stSpinner"] > div { border-top-color: var(--amber) !important; }

/* ═══════════════════════════════════════════════
   SCROLLBAR
   ═══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }

/* ═══════════════════════════════════════════════
   CONDITION TAG
   ═══════════════════════════════════════════════ */
.cond-tag {
    display: inline-flex;
    align-items: center;
    background: var(--amber-dim);
    border: 1px solid rgba(245,158,11,0.25);
    color: var(--amber);
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 3px;
    margin-left: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ── Arabam.com marka / model listesi (URL slug formatı) ─────────────────────
MARKA_MODELLER: dict[str, list[str]] = {
    "Alfa Romeo":    ["giulia", "giulietta", "stelvio", "tonale"],
    "Audi":          ["a1", "a3", "a4", "a5", "a6", "a7", "a8", "e-tron",
                      "q2", "q3", "q4-e-tron", "q5", "q7", "q8", "tt"],
    "BMW":           ["1-serisi", "2-serisi", "3-serisi", "4-serisi", "5-serisi",
                      "6-serisi", "7-serisi", "8-serisi",
                      "i3", "i4", "i5", "i7", "ix",
                      "x1", "x2", "x3", "x4", "x5", "x6", "x7", "z4"],
    "Buick":         ["enclave", "encore", "envision"],
    "BYD":           ["atto-3", "dolphin", "han", "seal", "seagull", "song-plus", "tang"],
    "Cadillac":      ["ct4", "ct5", "escalade", "xt4", "xt5", "xt6"],
    "Chery":         ["tiggo-4", "tiggo-7", "tiggo-8"],
    "Chevrolet":     ["aveo", "camaro", "captiva", "corvette", "cruze",
                      "equinox", "malibu", "spark", "trax"],
    "Chrysler":      ["300c", "300s", "grand-voyager", "pacifica", "sebring", "voyager"],
    "Citroen":       ["berlingo", "c1", "c3", "c3-aircross", "c4", "c4-cactus",
                      "c5", "c5-aircross", "c5-x", "ds3", "ds4", "ds5"],
    "Cupra":         ["born", "formentor", "leon", "terramar"],
    "Dacia":         ["duster", "jogger", "logan", "sandero", "spring"],
    "DS":            ["ds3", "ds4", "ds5", "ds7", "ds9"],
    "Fiat":          ["124-spider", "500", "500x", "bravo", "doblo", "egea",
                      "fiorino", "linea", "panda", "punto", "tipo"],
    "Ford":          ["b-max", "bronco", "c-max", "ecosport", "edge", "fiesta",
                      "focus", "galaxy", "kuga", "mondeo", "mustang",
                      "puma", "ranger", "tourneo"],
    "Haval":         ["h6", "jolion"],
    "Honda":         ["accord", "civic", "cr-v", "e", "hr-v",
                      "insight", "jazz", "legend"],
    "Hyundai":       ["accent", "bayon", "elantra", "i10", "i20", "i30",
                      "ioniq", "ioniq-5", "ioniq-6", "kona", "nexo",
                      "santa-fe", "tucson", "veloster"],
    "Infiniti":      ["q30", "q50", "q60", "q70", "qx30",
                      "qx50", "qx55", "qx60", "qx80"],
    "Isuzu":         ["d-max", "mu-x"],
    "Jaguar":        ["e-pace", "e-type", "f-pace", "f-type",
                      "i-pace", "xe", "xf", "xj"],
    "Jeep":          ["avenger", "cherokee", "compass", "gladiator",
                      "grand-cherokee", "renegade", "wrangler"],
    "Kia":           ["ceed", "ev6", "ev9", "niro", "picanto", "pro-ceed",
                      "sorento", "soul", "sportage", "stinger", "stonic", "xceed"],
    "Lamborghini":   ["huracan", "urus"],
    "Land Rover":    ["defender", "discovery", "discovery-sport", "freelander",
                      "range-rover", "range-rover-evoque",
                      "range-rover-sport", "range-rover-velar"],
    "Lancia":        ["delta", "musa", "thema", "ypsilon"],
    "Lexus":         ["ct", "es", "gx", "is", "lc", "ls", "lx",
                      "nx", "rc", "rx", "ux"],
    "Maserati":      ["ghibli", "granturismo", "grecale", "levante", "quattroporte"],
    "Mazda":         ["2", "3", "5", "6", "cx-3", "cx-30",
                      "cx-5", "cx-60", "mx-5"],
    "Mercedes-Benz": ["a-serisi", "b-serisi", "c-serisi", "cla", "clk", "cls",
                      "e-serisi", "eqa", "eqb", "eqc", "eqe", "eqs",
                      "g-serisi", "gl", "gla", "glb", "glc", "gle", "gls",
                      "s-serisi", "sl", "slk", "sprinter", "v-serisi"],
    "MG":            ["4", "5", "hs", "zs"],
    "Mini":          ["clubman", "convertible", "countryman",
                      "coupe", "hatch", "paceman", "roadster"],
    "Mitsubishi":    ["asx", "colt", "eclipse-cross", "galant",
                      "i-miev", "l200", "lancer", "outlander", "pajero"],
    "Nissan":        ["ariya", "e-nv200", "juke", "kicks", "leaf", "micra",
                      "navara", "note", "nv200", "pathfinder",
                      "qashqai", "x-trail"],
    "Opel":          ["adam", "agila", "antara", "astra", "cascada", "corsa",
                      "crossland", "frontera", "grandland", "insignia",
                      "meriva", "mokka", "omega", "signum", "vectra", "zafira"],
    "Peugeot":       ["106", "107", "108", "2008", "207", "208", "3008", "301",
                      "306", "307", "308", "4008", "407", "408", "5008",
                      "508", "e-208", "e-2008", "partner", "rifter"],
    "Polestar":      ["2", "3", "4"],
    "Porsche":       ["718", "911", "cayenne", "cayman",
                      "macan", "panamera", "taycan"],
    "Renault":       ["arkana", "austral", "captur", "clio", "duster",
                      "espace", "fluence", "kadjar", "kangoo", "laguna",
                      "megane", "safrane", "scenic", "symbol",
                      "talisman", "zoe"],
    "Saab":          ["9-3", "9-5"],
    "Seat":          ["arona", "ateca", "cordoba", "ibiza",
                      "leon", "tarraco", "toledo"],
    "Skoda":         ["fabia", "kamiq", "karoq", "kodiaq",
                      "octavia", "rapid", "scala", "superb", "yeti"],
    "Smart":         ["eq-fortwo", "forfour", "fortwo"],
    "Subaru":        ["forester", "impreza", "legacy", "outback", "xv"],
    "Suzuki":        ["baleno", "grand-vitara", "ignis", "jimny",
                      "swift", "sx4", "vitara"],
    "Tesla":         ["model-3", "model-s", "model-x", "model-y"],
    "Togg":          ["t10f", "t10x"],
    "Toyota":        ["auris", "avensis", "aygo", "bz4x", "c-hr", "camry",
                      "corolla", "corolla-cross", "hilux", "land-cruiser",
                      "prius", "proace", "rav4", "urban-cruiser",
                      "yaris", "yaris-cross"],
    "Volkswagen":    ["arteon", "caddy", "california", "golf",
                      "id-3", "id-4", "id-7", "jetta", "multivan",
                      "passat", "polo", "sharan", "t-cross",
                      "t-roc", "tiguan", "touareg", "touran", "up"],
    "Volvo":         ["c30", "c40", "c70", "ex30", "ex90",
                      "s40", "s60", "s80", "s90",
                      "v40", "v60", "v70", "v90",
                      "xc40", "xc60", "xc70", "xc90"],
}

# Görünen ad → URL slug dönüşümü
def _marka_slug(display: str) -> str:
    return display.lower().replace(" ", "-").replace("ş","s").replace("ı","i") \
                          .replace("ğ","g").replace("ü","u").replace("ö","o").replace("ç","c")


# ── Cache'li scraping — aynı marka/model/max_pages için tekrar çekmez ──────
@st.cache_data(show_spinner=False, ttl=3600)   # 1 saat geçerli
def _cached_scrape(marka: str, model_query: str, max_pages: int, _callback):
    """
    Streamlit cache: aynı parametrelerle çağrılırsa arabam.com'a gitmez.
    _callback underscore ile başladığı için cache key'e dahil edilmez.
    ttl=3600 → 1 saat sonra otomatik expire.
    """
    from scraper import scrape_listings
    return scrape_listings(
        marka, model_query,
        save=False,
        max_pages=max_pages,
        fetch_details=True,
        progress_callback=_callback,
    )


# ── Session state başlangıç değerleri ───────────────────────────────────────
def _init_state():
    defaults = {
        "step":        1,
        "clean_df":    None,
        "trainer":     None,
        "fe":          None,
        "y_all":       None,
        "X_all":       None,
        "fi_df":       None,
        "marka":       "",
        "model_query": "",
        "last_pred":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Yardımcılar ─────────────────────────────────────────────────────────────

def _fmt_tl(amount: float) -> str:
    return f"{amount:,.0f} TL".replace(",", ".")

def _reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    _init_state()


# ── Hero bar & step indicator ────────────────────────────────────────────────

def _render_hero():
    marka   = st.session_state.get("marka", "")
    model_q = st.session_state.get("model_query", "")
    badge   = f"{marka.upper()} · {model_q.upper()}" if marka else "ARABAM.COM · CANLI VERİ"

    hero_col, reset_col = st.columns([5, 1])
    with hero_col:
        st.markdown(f"""
        <div class="hero-wrap">
            <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
                <div>
                    <div class="hero-logo">OtoEstimate</div>
                    <div class="hero-sub">// ikinci el araç fiyat analizi &amp; tahmini</div>
                </div>
                <div class="hero-badge">{badge}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with reset_col:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        if st.button("↺ YENİ ARAMA", use_container_width=True, key="hero_reset_btn"):
            _reset()
            st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


def _render_step_indicator():
    step = st.session_state["step"]

    def _cls(n):
        if step > n:  return "done"
        if step == n: return "active"
        return "idle"

    def _icon(n):
        if step > n:  return "✓"
        return str(n)

    steps = [(1, "Veri Çek"), (2, "Eğit"), (3, "Tahmin")]
    parts = []
    for i, (n, label) in enumerate(steps):
        c = _cls(n)
        parts.append(f"""
        <div class="step-node">
            <div class="step-circle {c}">{_icon(n)}</div>
            <div class="step-label {c}">{label}</div>
        </div>""")
        if i < len(steps) - 1:
            conn_cls = "done" if step > n else ""
            parts.append(f'<div class="step-connector {conn_cls}"></div>')

    st.markdown(f'<div class="step-rail">{"".join(parts)}</div>', unsafe_allow_html=True)


def _section_hdr(text: str):
    st.markdown(f"""
    <div class="section-hdr">
        <span class="section-accent"></span>{text}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 1 — Veri Çekme
# ══════════════════════════════════════════════════════════════════════════════

def render_step1():
    _section_hdr("Araç Seç — Veri Çek")

    marka_listesi = list(MARKA_MODELLER.keys())

    # Önceki seçimi hatırla
    onceki_marka_slug = st.session_state.get("marka", "alfa-romeo")
    onceki_model_slug = st.session_state.get("model_query", "tonale")

    # Slug → display name eşleştir
    onceki_marka_display = next(
        (m for m in marka_listesi if _marka_slug(m) == onceki_marka_slug),
        marka_listesi[0]
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        secilen_marka = st.selectbox(
            "Marka",
            options=marka_listesi,
            index=marka_listesi.index(onceki_marka_display),
            help="Listeden marka seç",
        )

    model_listesi = MARKA_MODELLER.get(secilen_marka, [])
    onceki_model_idx = model_listesi.index(onceki_model_slug) \
        if onceki_model_slug in model_listesi else 0

    with col2:
        secilen_model = st.selectbox(
            "Model",
            options=model_listesi,
            index=onceki_model_idx,
            help="Markanın modelleri",
        )

    marka      = _marka_slug(secilen_marka)
    model_query = secilen_model

    col_sl, col_btn = st.columns([3, 1])
    with col_sl:
        max_pages = st.slider(
            "Sayfa limiti",
            min_value=2, max_value=20, value=8, step=1,
            help="Her sayfada ~20 ilan. 8 sayfa ≈ 160 ilan (+ detay sayfaları).",
        )
        st.caption(
            f"~{max_pages * 20} ilan taranır  •  "
            f"tahmini süre {int(max_pages * 1.5)}–{int(max_pages * 2.5)} dk  •  "
            f"hata/boya/değişen detay sayfalarından çekilir"
        )
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        scrape_btn = st.button("VERİ ÇEK", use_container_width=True, type="primary")

    # Seçili araç önizlemesi
    st.markdown(f"""
    <div style="
        display:inline-flex; align-items:center; gap:8px;
        margin: 4px 0 16px 0;
        font-family:'JetBrains Mono',monospace;
        font-size:0.72rem; color:var(--text-lo);
        letter-spacing:0.08em;
    ">
        <span style="color:var(--amber);">▸</span>
        arabam.com/ikinci-el/otomobil/
        <span style="color:var(--text-md);">{marka}-{model_query}</span>
    </div>
    """, unsafe_allow_html=True)

    if scrape_btn:
        st.session_state["marka"]       = marka
        st.session_state["model_query"] = model_query
        # Yeni arama başladığında model state'i temizle
        st.session_state["trainer"]   = None
        st.session_state["fe"]        = None
        st.session_state["last_pred"] = None

        from data_cleaner import clean_data

        prog_bar    = st.progress(0, text="Bağlanıyor...")
        status_slot = st.empty()

        def update_progress(page, total_so_far, status_msg: str = ""):
            if page is not None:
                pct = min(page / max_pages, 1.0) * 0.5
                prog_bar.progress(pct, text=status_msg or f"Sayfa {page}/{max_pages}")
                status_slot.caption(f"✔ {total_so_far} ilan bulundu")
            else:
                try:
                    done_str  = status_msg.split(":")[1].strip().split("/")[0].strip()
                    total_str = status_msg.split("/")[1].split()[0].strip()
                    pct = 0.5 + (int(done_str) / max(int(total_str), 1)) * 0.5
                    prog_bar.progress(min(pct, 0.99), text=status_msg)
                except Exception:
                    prog_bar.progress(0.6, text=status_msg or "Detay sayfaları çekiliyor…")
                status_slot.caption(status_msg)

        try:
            raw_df = _cached_scrape(
                marka, model_query,
                max_pages, update_progress,
            )

            prog_bar.progress(1.0, text="Tamamlandı!")
            status_slot.empty()

            if raw_df.empty:
                prog_bar.empty()
                st.error(
                    "Hiç ilan bulunamadı. Marka/model adlarını kontrol et.\n\n"
                    "Örn: `alfa-romeo` / `tonale` (arabam.com URL formatı)"
                )
                return

            clean_df = clean_data(raw_df, save=False)
            st.session_state["clean_df"] = clean_df
            st.session_state["step"]     = 2

        except Exception as e:
            prog_bar.empty()
            st.error(f"Scraping hatası: {e}")
            return

        st.rerun()

    if st.session_state["clean_df"] is not None:
        _show_data_summary(st.session_state["clean_df"], key_suffix="step1")


def _show_data_summary(df: pd.DataFrame, key_suffix: str = "default"):
    cond_pct = 0
    if "errors" in df.columns:
        cond_pct = int(df["errors"].notna().sum() / len(df) * 100)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="metric-val">{len(df)}</div>
            <div class="metric-lbl">Toplam İlan</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{_fmt_tl(df["price"].mean())}</div>
            <div class="metric-lbl">Ortalama Fiyat</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{_fmt_tl(df["price"].median())}</div>
            <div class="metric-lbl">Medyan Fiyat</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{int(df["year"].min())}–{int(df["year"].max())}</div>
            <div class="metric-lbl">Yıl Aralığı</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{cond_pct}%</div>
            <div class="metric-lbl">Kondisyon Verisi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        with st.expander("Ham veriye bak (ilk 20 satır)"):
            show_cols = [c for c in
                         ["title", "price", "km", "year", "paket",
                          "errors", "repaints", "changed_parts", "heavy_damage"]
                         if c in df.columns]
            st.dataframe(
                df[show_cols].head(20).style.format(
                    {"price": "{:,.0f}", "km": "{:,.0f}"}
                ),
                use_container_width=True,
            )
    with col_b:
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇ CSV İndir",
            data=csv_buf.getvalue(),
            file_name=f"arabam_{st.session_state['marka']}_{st.session_state['model_query']}.csv",
            mime="text/csv",
            key=f"download_csv_btn_{key_suffix}",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 2 — Model Eğitimi
# ══════════════════════════════════════════════════════════════════════════════

def render_step2():
    _section_hdr("Model Eğitimi")

    df = st.session_state["clean_df"]
    if df is None:
        st.warning("Önce veri çek (Adım 1).")
        return

    if st.session_state["trainer"] is None:
        _show_data_summary(df, key_suffix="step2")
        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("MODELİ EĞİT", type="primary", use_container_width=False)
        if not train_btn:
            return

        with st.spinner("Ridge • Random Forest • LightGBM karşılaştırılıyor…"):
            try:
                from feature_engineer import FeatureEngineer
                from model_trainer import ModelTrainer

                fe = FeatureEngineer()
                X, y, features = fe.fit_transform(df)

                trainer = ModelTrainer()
                trainer.train(X, y, features)

                st.session_state["trainer"] = trainer
                st.session_state["fe"]      = fe
                st.session_state["y_all"]   = y
                st.session_state["X_all"]   = X
                st.session_state["fi_df"]   = trainer.get_feature_importances()
                st.session_state["step"]    = 3

            except Exception as e:
                st.error(f"Eğitim hatası: {e}")
                return

        st.rerun()
    else:
        _show_training_results()


def _show_training_results():
    trainer = st.session_state["trainer"]
    if trainer is None:
        return

    m = trainer.test_metrics

    st.markdown(f"""
    <div style="margin-bottom:14px;">
        <span style="
            font-family:'Syne',sans-serif;
            font-size:0.95rem;
            font-weight:700;
            color:#10B981;
            letter-spacing:0.08em;
            text-transform:uppercase;
        ">✓ {trainer.best_model_name} seçildi</span>
    </div>
    <div class="metric-row">
        <div class="metric-tile">
            <div class="metric-val">{m['r2']:.3f}</div>
            <div class="metric-lbl">R² Skoru</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{_fmt_tl(m['mae'])}</div>
            <div class="metric-lbl">MAE</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{_fmt_tl(m['rmse'])}</div>
            <div class="metric-lbl">RMSE</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{m['mape']:.1f}%</div>
            <div class="metric-lbl">MAPE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fi_df = st.session_state.get("fi_df")
    if fi_df is not None and not fi_df.empty:
        _section_hdr("Feature Önem Sıralaması")
        _plot_fi_inline(fi_df)


def _plot_fi_inline(fi_df: pd.DataFrame):
    from config import FEATURE_DISPLAY_NAMES

    df = fi_df.head(10).copy().sort_values("importance", ascending=True)
    df["label"] = df["feature"].apply(lambda f: FEATURE_DISPLAY_NAMES.get(f, f))
    max_val = df["importance"].max()
    df["norm"] = (df["importance"] / max_val * 100) if max_val > 0 else df["importance"]

    fig, ax = plt.subplots(figsize=(8, max(3.2, len(df) * 0.42)))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    colors = ["#F0A500" if i == len(df) - 1 else "#1E3A5A" for i in range(len(df))]
    bars = ax.barh(df["label"], df["norm"], color=colors, edgecolor="none", height=0.55)

    for bar, val in zip(bars, df["norm"]):
        ax.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", ha="left", fontsize=8, color="#8894A4"
        )

    ax.set_xlabel("Normalize Önem Skoru", fontsize=8, color="#5A6880")
    ax.set_xlim(0, 120)
    ax.tick_params(colors="#8894A4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#0D1117")
    ax.xaxis.set_tick_params(color="#1A2030")
    ax.yaxis.set_tick_params(color="#1A2030")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#0D1117")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 3 — Tahmin
# ══════════════════════════════════════════════════════════════════════════════

def render_step3():
    _section_hdr("Fiyat Tahmini")

    trainer = st.session_state.get("trainer")
    fe      = st.session_state.get("fe")
    y_all   = st.session_state.get("y_all")

    if trainer is None or fe is None:
        st.warning("Önce modeli eğit (Adım 2).")
        return

    _show_training_results()
    st.markdown("<br>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown("""
        <div style="
            font-family:'Syne',sans-serif;
            font-size:0.82rem;
            font-weight:700;
            color:#7A8FA6;
            letter-spacing:0.12em;
            text-transform:uppercase;
            margin-bottom:14px;
            display:flex;
            align-items:center;
            gap:8px;
        "><span style="width:2px;height:14px;background:#F59E0B;display:inline-block;border-radius:2px;"></span>
        ARAÇ ÖZELLİKLERİ</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        year = c1.number_input("Yıl", min_value=1990, max_value=2026, value=2021, step=1)
        km   = c2.number_input("Kilometre", min_value=0, max_value=600_000,
                                value=50_000, step=1_000)

        known_pkts = (
            sorted(list(fe.label_encoders["paket"].classes_))
            if "paket" in fe.label_encoders else []
        )
        if known_pkts:
            paket = st.selectbox("Paket / Donanım", options=["(Belirtilmemiş)"] + known_pkts)
            paket = None if paket == "(Belirtilmemiş)" else paket
        else:
            paket = st.text_input("Paket / Donanım (opsiyonel)", placeholder="ör: 1.5 Hybrid Veloce")

        st.markdown("""
        <div style="
            font-family:'Syne',sans-serif;
            font-size:0.82rem;
            font-weight:700;
            color:#7A8FA6;
            letter-spacing:0.12em;
            text-transform:uppercase;
            margin: 20px 0 12px 0;
            display:flex;
            align-items:center;
            gap:8px;
        "><span style="width:2px;height:14px;background:#2DD4BF;display:inline-block;border-radius:2px;"></span>
        KONDİSYON BİLGİSİ <span class="cond-tag">Opsiyonel</span></div>
        """, unsafe_allow_html=True)

        c3, c4, c5 = st.columns(3)
        errors        = c3.number_input("Hata",    min_value=0, max_value=50, value=0, step=1)
        repaints      = c4.number_input("Boya",    min_value=0, max_value=50, value=0, step=1)
        changed_parts = c5.number_input("Değişen", min_value=0, max_value=50, value=0, step=1)

        heavy_damage_bool = st.checkbox(
            "Ağır Hasar Kaydı var",
            value=False,
            help="Araçta ağır hasar kaydı (total loss veya ağır yapısal hasar) varsa işaretle.",
        )
        heavy_damage = 1 if heavy_damage_bool else 0

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("FİYAT TAHMİNİ YAP", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            user_input = {
                "year":          int(year),
                "km":            float(km),
                "model":         st.session_state.get("model_query", "").title(),
                "paket":         paket or "Standart",
                "errors":        float(errors),
                "repaints":      float(repaints),
                "changed_parts": float(changed_parts),
                "heavy_damage":  float(heavy_damage),
            }

            try:
                X_input  = fe.transform(user_input)
                raw_pred = float(trainer.best_model.predict(X_input)[0])
                predicted = max(raw_pred, 100_000.0)

                rmse   = trainer.test_metrics.get("rmse", predicted * 0.15)
                margin = 1.96 * rmse
                lower  = max(predicted - margin, 100_000.0)
                upper  = predicted + margin

                st.session_state["last_pred"] = {
                    "predicted": predicted,
                    "lower":     lower,
                    "upper":     upper,
                }
            except Exception as e:
                st.error(f"Tahmin hatası: {e}")

        pred = st.session_state.get("last_pred")
        if pred:
            _render_prediction_card(pred, y_all)
        else:
            st.markdown("""
            <div style="
                height:200px;
                display:flex;
                flex-direction:column;
                align-items:center;
                justify-content:center;
                border: 1px dashed #1E2A3A;
                border-radius:10px;
                color:#3A4860;
                font-size:0.85rem;
                text-align:center;
                letter-spacing:0.06em;
            ">
                <div style="font-size:2rem;margin-bottom:10px;">🏎</div>
                Araç özelliklerini gir<br>ve tahmini başlat
            </div>
            """, unsafe_allow_html=True)


def _render_prediction_card(pred: dict, y_all: pd.Series):
    price   = pred["predicted"]
    lower   = pred["lower"]
    upper   = pred["upper"]
    trainer = st.session_state.get("trainer")

    mape       = trainer.test_metrics.get("mape", 0) if trainer else 0
    confidence = max(0, 100 - mape)

    st.markdown(f"""
    <div class="price-result">
        <div class="price-eyebrow">Tahmini Satış Fiyatı</div>
        <div class="price-main">{_fmt_tl(price)}</div>
        <div class="price-range">
            %95 güven aralığı<br>
            <b>{_fmt_tl(lower)}</b> &nbsp;—&nbsp; <b>{_fmt_tl(upper)}</b>
        </div>
        <div style="
            margin-top:16px;
            font-family:'JetBrains Mono',monospace;
            font-size:0.65rem;
            color:#3D5068;
            letter-spacing:0.1em;
            text-transform:uppercase;
        ">DOĞRULUK {confidence:.0f}% &nbsp;·&nbsp; MAPE {mape:.1f}% &nbsp;·&nbsp; {trainer.best_model_name.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    if y_all is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        _plot_distribution_inline(y_all, price)


def _plot_distribution_inline(y: pd.Series, predicted_price: float):
    import matplotlib.transforms as mtransforms

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    # Histogram
    n, bins, _ = ax.hist(y / 1_000, bins=22, color="#1E3A5A", edgecolor="none", alpha=0.9)

    # Tahminin düştüğü bin'i amber renkle vurgula
    pred_k = predicted_price / 1_000
    for i in range(len(n)):
        if bins[i] <= pred_k < bins[i + 1]:
            ax.bar(
                (bins[i] + bins[i + 1]) / 2,
                n[i],
                width=(bins[i + 1] - bins[i]) * 0.95,
                color="#F0A500",
                alpha=0.85,
                edgecolor="none",
                zorder=3,
            )
            break

    # X ekseninin hemen altında büyük amber üçgen işareti (nokta gibi)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        pred_k, -0.06,
        marker="^", color="#F0A500", markersize=12,
        transform=trans, clip_on=False, zorder=5,
        label=f"Tahmininiz: {pred_k:.0f}K TL",
    )

    ax.set_xlabel("Fiyat (Bin TL)", fontsize=8, color="#8894A4")
    ax.set_ylabel("İlan Sayısı",   fontsize=8, color="#8894A4")
    ax.tick_params(colors="#8894A4", labelsize=7.5)
    legend = ax.legend(fontsize=8, framealpha=0, loc="upper left")
    for text in legend.get_texts():
        text.set_color("#F0A500")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#0D1117")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANA LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _render_hero()
    _render_step_indicator()

    step = st.session_state["step"]

    # ── Adım 1 ──
    render_step1()

    if step >= 2:
        st.markdown("---")
        render_step2()

    if step >= 3:
        st.markdown("---")
        render_step3()

    # Footer
    st.markdown("""
    <div style="
        margin-top:48px;
        padding-top:16px;
        border-top:1px solid #1C2535;
        font-family:'JetBrains Mono',monospace;
        font-size:0.65rem;
        color:#7A8FA6;
        text-align:center;
        letter-spacing:0.1em;
        text-transform:uppercase;
    ">
        veriler arabam.com'dan gerçek zamanlı çekilir &nbsp;·&nbsp;
        tahminler istatistiksel model çıktısıdır, kesin fiyat değildir
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
