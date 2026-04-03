# predictor.py — Eğitilmiş modeli yükle ve interaktif CLI tahmin arayüzü

import json
import logging
import numpy as np
import pandas as pd

from config import MODEL_PATH, ENGINEER_PATH, FEATURE_INFO_PATH

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Kaydedilmiş model + feature engineer yükler,
    kullanıcı girişini işleyip fiyat tahmini üretir.
    """

    def __init__(
        self,
        model_path:    str = MODEL_PATH,
        engineer_path: str = ENGINEER_PATH,
        info_path:     str = FEATURE_INFO_PATH,
    ):
        import joblib
        self.model    = joblib.load(model_path)
        self.engineer = joblib.load(engineer_path)

        with open(info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

        self.train_rmse    = self.info.get("train_rmse", 0)
        self.best_model_nm = self.info.get("best_model", "model")
        logger.info(f"Predictor yüklendi. Model: {self.best_model_nm}")

    def predict(self, user_input: dict) -> dict:
        """
        Araç özelliklerini alır, fiyat tahmini üretir.

        Parametre:
            user_input : {
                'year': 2020, 'km': 50000,
                'model': 'Tonale', 'paket': '1.5 Edizione',
                'location': 'İstanbul',
                'errors': 0, 'repaints': 1, 'changed_parts': 0,
                'tramer': 0   # opsiyonel
            }

        Dönüş:
            {
                'predicted_price': 1_250_000.0,
                'lower_95':          950_000.0,
                'upper_95':        1_550_000.0,
                'model_used': 'LightGBM'
            }
        """
        try:
            X = self.engineer.transform(user_input)
            raw_pred = float(self.model.predict(X)[0])

            # Negatif veya saçma düşük fiyat guard
            predicted = max(raw_pred, 100_000.0)

            # %95 güven aralığı (RMSE bazlı basit yaklaşım)
            margin  = 1.96 * self.train_rmse if self.train_rmse > 0 else predicted * 0.15
            lower   = max(predicted - margin, 100_000.0)
            upper   = predicted + margin

            return {
                "predicted_price": predicted,
                "lower_95":        lower,
                "upper_95":        upper,
                "model_used":      self.best_model_nm,
            }

        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return {"error": str(e)}

    def get_known_locations(self) -> list:
        """Modelin bildiği il isimlerini döner."""
        return list(
            self.info.get("label_encoders", {}).get("location", {})
        )

    def get_known_models(self) -> list:
        """Modelin bildiği araç modellerini döner."""
        return list(
            self.info.get("label_encoders", {}).get("model", {})
        )


# ===== INPUT DOĞRULAMA =====

def _validate_input(data: dict) -> list:
    """Kullanıcı girişindeki sorunları listeler."""
    issues = []

    year = data.get("year")
    if year is None or not (1990 <= int(year) <= 2026):
        issues.append("Yıl 1990–2026 arasında olmalı.")

    km = data.get("km")
    if km is None or not (0 <= float(km) <= 600_000):
        issues.append("Kilometre 0–600.000 arasında olmalı.")

    if not data.get("location"):
        issues.append("Konum (il) boş olamaz.")

    return issues


# ===== FORMAT YARDIMCILARI =====

def _fmt_tl(amount: float) -> str:
    """1250000.0 → '1.250.000 TL'"""
    return f"{amount:,.0f} TL".replace(",", ".")


def _confidence_label(rmse: float, predicted: float) -> str:
    """RMSE / fiyat oranına göre güven etiketi döner."""
    if predicted == 0:
        return "belirsiz"
    ratio = rmse / predicted
    if ratio < 0.08:
        return "Yüksek"
    elif ratio < 0.18:
        return "Orta"
    else:
        return "Düşük"


# ===== İNTERAKTİF CLI =====

def interactive_predict(predictor: PricePredictor, y_all: pd.Series = None) -> None:
    """
    Döngüsel CLI arayüzü.
    Kullanıcı araç özelliklerini girer, model fiyat tahmini verir.
    """
    from visualizer import plot_price_distribution

    print("\n" + "=" * 58)
    print("  ARABAM.COM FİYAT TAHMİN MODELİ")
    print(f"  (Model: {predictor.best_model_nm})")
    print("=" * 58)

    known_locs = predictor.get_known_locations()

    while True:
        print("\nAraç özelliklerini girin  [Enter = boş bırak / varsayılan]")
        print("-" * 40)

        try:
            # --- Zorunlu alanlar ---
            year_str = input("  Yıl           (ör: 2021) : ").strip()
            year = int(year_str) if year_str else None
            if year is None:
                print("  ⚠  Yıl zorunlu, tekrar deneyin.")
                continue

            km_str = input("  Kilometre     (ör: 45000): ").strip()
            km = float(km_str.replace(".", "").replace(",", "")) if km_str else None
            if km is None:
                print("  ⚠  Kilometre zorunlu, tekrar deneyin.")
                continue

            location = input("  Konum / İl    (ör: İstanbul): ").strip()
            if not location:
                print("  ⚠  Konum zorunlu, tekrar deneyin.")
                continue

            # --- Opsiyonel alanlar ---
            paket    = input("  Paket / Trim  (ör: 1.5 Edizione) [Enter=boş]: ").strip() or None
            errors_s = input("  Hata sayısı   (ör: 0) [Enter=0]: ").strip()
            repaint_s= input("  Boya sayısı   (ör: 1) [Enter=0]: ").strip()
            changed_s= input("  Değişen parça (ör: 0) [Enter=0]: ").strip()
            tramer_s = input("  Tramer        (0=Yok / 1=Var) [Enter=bilinmiyor]: ").strip()

            user_input = {
                "year":          year,
                "km":            km,
                "location":      location,
                "paket":         paket,
                "errors":        int(errors_s)   if errors_s   else 0,
                "repaints":      int(repaint_s)  if repaint_s  else 0,
                "changed_parts": int(changed_s)  if changed_s  else 0,
            }
            if tramer_s in ("0", "1"):
                user_input["tramer"] = int(tramer_s)

            # Validasyon
            issues = _validate_input(user_input)
            if issues:
                for iss in issues:
                    print(f"  ⚠  {iss}")
                continue

            # Tahmin
            result = predictor.predict(user_input)

            if "error" in result:
                print(f"\n  ❌ Tahmin hatası: {result['error']}")
            else:
                price  = result["predicted_price"]
                lower  = result["lower_95"]
                upper  = result["upper_95"]
                conf   = _confidence_label(predictor.train_rmse, price)

                print(f"\n{'─'*50}")
                print(f"  TAHMİN SONUCU")
                print(f"{'─'*50}")
                print(f"  Tahmini Fiyat   : {_fmt_tl(price)}")
                print(f"  Alt Sınır (95%) : {_fmt_tl(lower)}")
                print(f"  Üst Sınır (95%) : {_fmt_tl(upper)}")
                print(f"  Güven Düzeyi    : {conf}")
                print(f"{'─'*50}")

                # Fiyat dağılım grafiği (eğer veri varsa)
                if y_all is not None:
                    plot_price_distribution(y_all, predicted_price=price)
                    print("  → Fiyat dağılım grafiği güncellendi: plots/price_distribution.png")

        except KeyboardInterrupt:
            print("\n\nÇıkılıyor...")
            break
        except ValueError as e:
            print(f"  ⚠  Geçersiz değer: {e}")
            continue

        again = input("\n  Başka tahmin yapmak ister misin? (e/h): ").strip().lower()
        if again != "e":
            print("Görüşürüz!\n")
            break


# ===== TEST =====
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = PricePredictor()
    interactive_predict(predictor)
