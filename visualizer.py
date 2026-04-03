# visualizer.py — Feature importance ve SHAP görselleştirmeleri

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Headless ortam (GUI penceresi açılmaz)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import PLOTS_DIR, FEATURE_DISPLAY_NAMES

logger = logging.getLogger(__name__)

os.makedirs(PLOTS_DIR, exist_ok=True)

# Türkçe karakter için font ayarı (DejaVu unicode destekler)
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "figure.dpi":    120,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# Renk paleti
BAR_COLOR    = "#2E86AB"
ACCENT_COLOR = "#E84855"


def _turkish_label(feature: str) -> str:
    """Feature adını Türkçe etiketle."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature)


# ===== 1. FEATURE IMPORTANCE BARH =====

def plot_feature_importance(
    fi_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_k: int = 15,
    save_path: str = None,
) -> str:
    """
    LightGBM / RandomForest built-in feature_importances_ değerlerini
    yatay bar chart ile görselleştirir.

    Parametreler:
        fi_df      : 'feature' ve 'importance' sütunları içeren DataFrame
        top_k      : Gösterilecek max feature sayısı
        save_path  : None ise PLOTS_DIR/feature_importance.png'e kaydeder

    Döner:
        Kaydedilen dosya yolu
    """
    if fi_df.empty:
        logger.warning("Feature importance verisi boş, grafik oluşturulmadı.")
        return ""

    # Üst K feature
    df = fi_df.head(top_k).copy()
    df = df.sort_values("importance", ascending=True)   # barh için aşağıdan yukarı

    # Türkçe etiketler
    df["label"] = df["feature"].apply(_turkish_label)

    # Normalize (0–100)
    max_val = df["importance"].max()
    df["importance_norm"] = (df["importance"] / max_val * 100) if max_val > 0 else df["importance"]

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.5)))

    bars = ax.barh(
        df["label"],
        df["importance_norm"],
        color=BAR_COLOR,
        edgecolor="white",
        height=0.6,
    )

    # Değer etiketleri
    for bar, val in zip(bars, df["importance_norm"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center", ha="left",
            fontsize=9, color="#333333",
        )

    ax.set_xlabel("Önem Skoru (normalize, maks=100)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_xlim(0, 115)

    plt.tight_layout()

    out = save_path or os.path.join(PLOTS_DIR, "feature_importance.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance grafiği kaydedildi: {out}")
    return out


# ===== 2. SHAP SUMMARY PLOT =====

def plot_shap_summary(
    model,
    X_sample: pd.DataFrame,
    feature_names: list,
    save_path: str = None,
) -> str:
    """
    SHAP beeswarm summary plot — her feature'ın model çıktısına
    ne yönde ve ne kadar etki ettiğini gösterir.

    Parametreler:
        model        : Eğitilmiş LightGBM veya RandomForest modeli
        X_sample     : Gösterim için kullanılacak örnek DataFrame (max 500 satır)
        feature_names: Feature isimleri listesi
        save_path    : Kayıt yolu

    Döner:
        Kaydedilen dosya yolu (SHAP yüklü değilse boş string)
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP yüklü değil. `pip install shap` ile yükleyin.")
        return ""

    # Büyük dataset'te subsample
    if len(X_sample) > 500:
        X_sample = X_sample.sample(500, random_state=42)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Türkçe sütun isimlerini kullan
        X_display = X_sample.rename(
            columns={f: _turkish_label(f) for f in feature_names}
        )

        fig, ax = plt.subplots(figsize=(9, max(5, len(feature_names) * 0.45)))
        shap.summary_plot(
            shap_values, X_display,
            show=False, plot_size=None,
        )
        ax = plt.gca()
        ax.set_title("SHAP — Feature Etki Analizi", fontsize=13, fontweight="bold")

        out = save_path or os.path.join(PLOTS_DIR, "shap_summary.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary grafiği kaydedildi: {out}")
        return out

    except Exception as e:
        logger.warning(f"SHAP grafiği oluşturulamadı: {e}")
        return ""


# ===== 3. GERÇEK vs TAHMİN SCATTER =====

def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: str = None,
) -> str:
    """
    Gerçek fiyat vs. tahmin edilen fiyat scatter plot.
    İdeal tahmin = 45° diyagonal çizgi.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.5, s=25, color=BAR_COLOR, edgecolors="none")

    # İdeal çizgi
    combined = np.concatenate([y_true.values, y_pred])
    mn, mx = combined.min(), combined.max()
    ax.plot([mn, mx], [mn, mx], color=ACCENT_COLOR, linestyle="--", linewidth=1.5, label="İdeal")

    ax.set_xlabel("Gerçek Fiyat (TL)", fontsize=10)
    ax.set_ylabel("Tahmin Edilen Fiyat (TL)", fontsize=10)
    ax.set_title("Gerçek Fiyat vs. Tahmin", fontsize=13, fontweight="bold")

    # Eksen formatı: Türk lirası
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M" if x >= 1_000_000 else f"{x/1_000:.0f}K")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ax.legend(fontsize=9)
    plt.tight_layout()

    out = save_path or os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Actual vs Predicted grafiği kaydedildi: {out}")
    return out


# ===== 4. FİYAT DAĞILIMI =====

def plot_price_distribution(
    y: pd.Series,
    predicted_price: float = None,
    save_path: str = None,
) -> str:
    """
    İlan fiyatlarının histogramını çizer.
    Opsiyonel olarak tahmin edilen fiyatı dikey çizgi ile işaretler.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(y / 1_000, bins=30, color=BAR_COLOR, edgecolor="white", alpha=0.85)

    if predicted_price is not None:
        ax.axvline(
            predicted_price / 1_000,
            color=ACCENT_COLOR,
            linestyle="--",
            linewidth=2,
            label=f"Tahmininiz: {predicted_price/1_000:.0f}K TL",
        )
        ax.legend(fontsize=10)

    ax.set_xlabel("Fiyat (Bin TL)", fontsize=10)
    ax.set_ylabel("İlan Sayısı", fontsize=10)
    ax.set_title("Piyasa Fiyat Dağılımı", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = save_path or os.path.join(PLOTS_DIR, "price_distribution.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Fiyat dağılım grafiği kaydedildi: {out}")
    return out


# ===== 5. TOPLU GRAFİK ÜRETİMİ =====

def generate_all_plots(
    trainer,          # ModelTrainer instance
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_all: pd.Series,
    predicted_price: float = None,
) -> list:
    """
    Tüm grafikleri tek seferde üretir.
    Döner: Oluşturulan dosya yolları listesi
    """
    paths = []

    # 1. Feature Importance
    fi_df = trainer.get_feature_importances()
    if not fi_df.empty:
        p = plot_feature_importance(fi_df, title="Feature Önem Sıralaması")
        if p:
            paths.append(p)

    # 2. Actual vs Predicted
    y_pred = trainer.best_model.predict(X_test)
    p = plot_actual_vs_predicted(y_test, y_pred)
    if p:
        paths.append(p)

    # 3. Fiyat dağılımı
    p = plot_price_distribution(y_all, predicted_price=predicted_price)
    if p:
        paths.append(p)

    # 4. SHAP (opsiyonel)
    p = plot_shap_summary(
        trainer.best_model,
        X_test,
        trainer.feature_names,
    )
    if p:
        paths.append(p)

    print(f"\n{'='*50}")
    print(f"  Oluşturulan grafikler ({len(paths)} adet):")
    for path in paths:
        print(f"    → {path}")
    print(f"{'='*50}\n")

    return paths
