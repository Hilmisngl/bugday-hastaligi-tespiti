# ─────────────────────────────────────────────────────────────
# config.py — Buğday Hastalığı Tespiti Proje Konfigürasyonu
# SUBÜ EEM · Yapay Zekaya Giriş 2025-2026
# ─────────────────────────────────────────────────────────────

# ── Görüntü & Model ──────────────────────────────────────────
IMG_SIZE    = 224          # EfficientNetB0 için standart boyut
BATCH_SIZE  = 32
SEED        = 42

# ── Eğitim Fazları ───────────────────────────────────────────
PHASE1_EPOCHS   = 15      # Başlık katmanları eğitimi
PHASE2_EPOCHS   = 20      # Fine-tuning
PHASE1_LR       = 1e-3
PHASE2_LR       = 1e-5
FINETUNE_LAYERS = 30      # Son kaç katman açılacak

# ── Veri Bölümü ──────────────────────────────────────────────
VALIDATION_SPLIT = 0.20   # %80 train / %20 validation

# ── Callback Parametreleri ───────────────────────────────────
EARLY_STOP_PATIENCE     = 5
REDUCE_LR_PATIENCE      = 3
REDUCE_LR_FACTOR        = 0.3

# ── Sınıf Adları (Dataninja Dataset) ─────────────────────────
CLASS_NAMES = [
    "Healthy",
    "Yellow Rust",
    "Brown Rust",
    "Septoria",
    "Powdery Mildew",
    "Loose Smut",
    "Stem Rust",
]

# ── Türkçe Hastalık Bilgileri ─────────────────────────────────
DISEASE_INFO = {
    "Healthy": {
        "tr_name":    "Sağlıklı Buğday",
        "pathogen":   "—",
        "symptoms":   "Herhangi bir hastalık belirtisi gözlemlenmedi.",
        "treatment":  "Rutin bakım ve sulama yeterlidir.",
    },
    "Yellow Rust": {
        "tr_name":    "Sarı Pas",
        "pathogen":   "Puccinia striiformis",
        "symptoms":   "Yapraklarda sarı-turuncu renkte çizgi şeklinde pustüller.",
        "treatment":  "Triazol grubu fungisit (propikonazol, tebukonazol) erken dönemde uygulanmalıdır.",
    },
    "Brown Rust": {
        "tr_name":    "Kahverengi Pas",
        "pathogen":   "Puccinia triticina",
        "symptoms":   "Yaprak yüzeyinde turuncu-kahverengi yuvarlak pustüller.",
        "treatment":  "Propikonazol veya trifloksistrobin içerikli fungisit uygulanmalıdır.",
    },
    "Septoria": {
        "tr_name":    "Septorya Yaprak Lekesi",
        "pathogen":   "Septoria tritici",
        "symptoms":   "Sarı kenarlı, içi kahverengi nekrotik lekeler; lekeler üzerinde siyah piknidler.",
        "treatment":  "Azoksistrobin veya kresoksim-metil içerikli fungisit, yağışlı dönem öncesi uygulanmalı.",
    },
    "Powdery Mildew": {
        "tr_name":    "Külleme",
        "pathogen":   "Blumeria graminis f. sp. tritici",
        "symptoms":   "Yaprak ve saplarda beyaz-gri pudra görünümünde misel örtüsü.",
        "treatment":  "Kükürt içerikli veya triadimenol fungisit erken dönemde etkilidir.",
    },
    "Loose Smut": {
        "tr_name":    "Rastık (Sürmeli)",
        "pathogen":   "Ustilago tritici",
        "symptoms":   "Başakların tamamı siyah-mor toz (spor kitlesi) ile kaplanır.",
        "treatment":  "Ekim öncesi tohumluk ilaçlaması (karboksin + thiram) zorunludur.",
    },
    "Stem Rust": {
        "tr_name":    "Sap Pası (Kara Pas)",
        "pathogen":   "Puccinia graminis f. sp. tritici",
        "symptoms":   "Yaprak ve saplarda turuncu-kırmızı uzun pustüller; geç dönemde siyahlaşır.",
        "treatment":  "Dayanıklı çeşit seçimi ve triazol fungisit kombinasyonu önerilir.",
    },
}

# ── Dosya Yolları ─────────────────────────────────────────────
MODEL_SAVE_PATH     = "models/bugday_model.h5"
CLASS_NAMES_PATH    = "models/class_names.json"
DATASET_DIR         = "data/wheat_dataset/"
