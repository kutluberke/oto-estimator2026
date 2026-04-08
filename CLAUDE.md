# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

arabam.com ikinci el araç fiyat tahmin uygulaması. Belirli bir marka/model için ilanları scrape eder, veriyi temizler, LightGBM modeli eğitir ve interaktif fiyat tahmini sunar. İki arayüz vardır: CLI (`main.py`) ve Streamlit web uygulaması (`app.py`).

## Kurulum & Çalıştırma

```bash
pip install -r requirements.txt

# CLI — üç adım sırasıyla veya tek komutla
python main.py --scrape --marka alfa-romeo --model tonale
python main.py --train
python main.py --predict
python main.py --all --marka volkswagen --model golf

# Streamlit arayüzü
streamlit run app.py
```

Marka/model parametreleri arabam.com URL formatında olmalı (küçük harf, tire-ayrılmış): `alfa-romeo`, `3-serisi`, `tiguan-allspace`.

## Mimari & Veri Akışı

Pipeline üç aşamadan oluşur ve her aşama öncekinin çıktısını kullanır:

```
scraper.py → data_cleaner.py → feature_engineer.py → model_trainer.py → predictor.py
```

**scraper.py** — `scrape_listings(marka, model_query)`:
- Liste sayfalarını sırayla, detay sayfalarını paralel ThreadPoolExecutor (5 worker) ile çeker.
- Her ilan için `detail_url` üzerinden hata/boya/değişen/ağır hasar verisi toplar.
- Çıktı: `data/raw_listings.csv`

**data_cleaner.py** — `clean_data(df)`:
- Fiyat/km/yıl aykırı değerleri `config.py::DATA_BOUNDS` ile filtreler.
- Çıktı: `data/cleaned_data.csv`

**feature_engineer.py** — `FeatureEngineer`:
- `fit_transform()` eğitimde, `transform()` tahmin anında çağrılır.
- LabelEncoder (model, paket) + StandardScaler uygular. Obje `models/feature_engineer.pkl` olarak kaydedilir.
- Feature sırası `FINAL_FEATURES` listesinde sabit — değiştirilirse model/encoder uyumsuzluğu oluşur.

**model_trainer.py** — `ModelTrainer`:
- LightGBM, RandomForest ve Ridge modellerini cross-validation ile karşılaştırır, en iyisini seçer.
- Dataset büyüklüğüne göre LightGBM hiperparametreleri otomatik seçilir (`config.py::LGBM_PARAMS_SMALL/MEDIUM/LARGE`).
- Çıktılar: `models/model.pkl`, `models/feature_info.json`

**predictor.py** — `PricePredictor`:
- Kaydedilmiş model + encoder'ı yükler, tek araç için tahmin yapar.

**app.py** — Streamlit arayüzü:
- `scrape_listings` ve `ModelTrainer`'ı doğrudan import eder; `main.py`'yi kullanmaz.
- Scraping için `progress_callback` parametresi ile canlı ilerleme gösterimi yapar.

## Yapılandırma (config.py)

Tüm ayarlar `config.py`'de merkezi olarak tanımlıdır:
- `DATA_BOUNDS` — fiyat/km/yıl geçerli aralıkları ve `rare_location_threshold`
- `SCRAPE_CONFIG` — delay, timeout, max_pages, max_retries
- `LGBM_PARAMS_*` — dataset büyüklüğüne göre hiperparametreler
- Dosya yolları (DATA_DIR, MODEL_DIR, PLOTS_DIR, LOGS_DIR) ve türevleri

## Dizin Yapısı

```
data/        — raw_listings.csv, cleaned_data.csv (scrape sonrası oluşur)
models/      — model.pkl, feature_engineer.pkl, feature_info.json (train sonrası oluşur)
plots/       — görselleştirme çıktıları
logs/        — app.log, scrape_log.txt
```

## Önemli Notlar

- `config.py`'deki `DATA_BOUNDS` dict'i scraper, cleaner ve feature engineer tarafından ortaklaşa kullanılır — bir bound değiştirilirse tüm pipeline etkilenir.
- arabam.com HTML yapısı sütun indeksleri yorumlu olarak `scraper.py` başında belgelenmiştir; site yapısı değişirse bu indeksler güncellenmeli.
- Rate limiting: liste sayfaları sequential (1.5–3s delay), detay sayfaları paralel fakat `_rate_lock` ile sıralı HTTP isteği yapar.
- `packages.txt` Streamlit Cloud deployment için sistem bağımlılıklarını içerir.
