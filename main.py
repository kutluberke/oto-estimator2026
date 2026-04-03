#!/usr/bin/env python3
# main.py — Arabam.com Fiyat Tahmin Modeli — Ana giriş noktası
#
# Kullanım örnekleri:
#   python main.py --scrape --marka alfa-romeo --model tonale
#   python main.py --train
#   python main.py --predict
#   python main.py --all --marka volkswagen --model golf
#   python main.py --all --marka bmw --model 3-serisi

import os
import sys
import logging
import argparse
import pandas as pd

# ===== LOGLAMA KURULUMU (main.py yüklenince) =====
from config import LOGS_DIR, DATA_DIR, MODEL_DIR, PLOTS_DIR

os.makedirs(LOGS_DIR,  exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ===== ADIM 1: SCRAPING =====

def run_scrape(marka: str, model_query: str) -> pd.DataFrame:
    from scraper import scrape_listings
    from data_cleaner import clean_data

    print(f"\n🔍  {marka.upper()} {model_query.upper()} için ilanlar çekiliyor...")

    raw_df = scrape_listings(marka, model_query, save=True)

    if raw_df.empty:
        logger.error("Hiç ilan çekilemedi. Marka/model adlarını kontrol edin.")
        print(
            "\n❌ Hiç ilan bulunamadı!\n"
            "   Kontrol et:\n"
            "   • --marka parametresi URL formatında mı? (ör: alfa-romeo)\n"
            "   • --model parametresi URL formatında mı? (ör: tonale)\n"
            "   • arabam.com'da bu araç var mı?\n"
        )
        sys.exit(1)

    print(f"   Ham veri: {len(raw_df)} ilan")

    clean_df = clean_data(raw_df, save=True)

    print(f"   Temiz veri: {len(clean_df)} ilan  → data/cleaned_data.csv")
    return clean_df


# ===== ADIM 2: EĞİTİM =====

def run_train(df: pd.DataFrame = None) -> tuple:
    from feature_engineer import FeatureEngineer
    from model_trainer import ModelTrainer
    from config import CLEANED_DATA_PATH, ENGINEER_PATH

    if df is None:
        if not os.path.exists(CLEANED_DATA_PATH):
            logger.error(f"Temiz veri bulunamadı: {CLEANED_DATA_PATH}")
            print("\n❌ Önce --scrape çalıştır!\n")
            sys.exit(1)
        df = pd.read_csv(CLEANED_DATA_PATH)

    print(f"\n🤖  Model eğitimi başlıyor ({len(df)} ilan)...")

    # Feature engineering
    fe = FeatureEngineer()
    X, y, features = fe.fit_transform(df)
    fe.save(ENGINEER_PATH)

    # Model eğitimi
    trainer = ModelTrainer()
    trainer.train(X, y, features)
    trainer.save()

    # Grafikler
    from visualizer import generate_all_plots
    generate_all_plots(
        trainer,
        trainer.X_test,
        trainer.y_test,
        y_all=y,
    )

    print(f"\n✅  Eğitim tamamlandı!")
    print(f"   Model    → models/model.pkl")
    print(f"   Encoder  → models/feature_engineer.pkl")
    print(f"   Grafikler → plots/")

    return trainer, fe, y


# ===== ADIM 3: TAHMİN =====

def run_predict(y_all: pd.Series = None) -> None:
    from predictor import PricePredictor, interactive_predict
    from config import MODEL_PATH, ENGINEER_PATH, FEATURE_INFO_PATH

    for path in [MODEL_PATH, ENGINEER_PATH, FEATURE_INFO_PATH]:
        if not os.path.exists(path):
            print(f"\n❌ Gerekli dosya bulunamadı: {path}")
            print("   Önce --train çalıştır!\n")
            sys.exit(1)

    predictor = PricePredictor()

    # Eğer y_all verilmediyse CSV'den oku (dağılım grafiği için)
    if y_all is None:
        from config import CLEANED_DATA_PATH
        if os.path.exists(CLEANED_DATA_PATH):
            df = pd.read_csv(CLEANED_DATA_PATH)
            y_all = df["price"] if "price" in df.columns else None

    interactive_predict(predictor, y_all=y_all)


# ===== ARG PARSER =====

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="arabam.com İkinci El Araç Fiyat Tahmin Modeli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py --all --marka alfa-romeo --model tonale
  python main.py --scrape --marka volkswagen --model golf
  python main.py --train
  python main.py --predict
  python main.py --all --marka bmw --model 3-serisi

Marka/Model URL formatı (arabam.com'daki gibi, küçük harf + tire):
  alfa-romeo  tonale
  volkswagen  golf
  bmw         3-serisi
  toyota      corolla
  honda       civic
        """,
    )

    parser.add_argument(
        "--scrape",
        action="store_true",
        help="arabam.com'dan ilan çek ve temizle",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Temiz veriden model eğit",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Eğitilmiş modelle interaktif tahmin yap",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape → Train → Predict adımlarını sırayla çalıştır",
    )
    parser.add_argument(
        "--marka",
        type=str,
        default="alfa-romeo",
        metavar="MARKA",
        help="URL formatında marka (ör: alfa-romeo)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tonale",
        metavar="MODEL",
        help="URL formatında model (ör: tonale)",
    )

    return parser


# ===== MAIN =====

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # En az bir flag gerekli
    if not any([args.scrape, args.train, args.predict, args.all]):
        parser.print_help()
        sys.exit(0)

    # --all = scrape + train + predict
    if args.all:
        args.scrape  = True
        args.train   = True
        args.predict = True

    clean_df = None
    trainer  = None
    y_all    = None

    if args.scrape:
        clean_df = run_scrape(args.marka, args.model)

    if args.train:
        result = run_train(df=clean_df)
        if result:
            trainer, _, y_all = result

    if args.predict:
        run_predict(y_all=y_all)


if __name__ == "__main__":
    main()
