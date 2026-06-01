# 🌾 Buğday Hastalığı Tespiti — CNN ile Görüntü Sınıflandırma

**SUBU Elektrik-Elektronik Mühendisliği | Yapay Zekaya Giriş | 2024-2025 Bahar**

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io)

---

## 📌 Proje Özeti

Bu proje, buğday bitkilerinde görülen 6 farklı hastalık kategorisini yapraklar ve saplar üzerindeki
görsel belirtilerden **Evrişimli Sinir Ağları (CNN)** kullanarak otomatik olarak tespit eden
bir yapay zeka sistemidir.

**Problem Türü:** Görüntü Sınıflandırma (Multi-class Classification)  
**Dataset:** Dataninja — Wheat Disease  
**Model:** EfficientNetB3 (Transfer Learning + Fine-Tuning)  
**Arayüz:** Streamlit Web Uygulaması  

---

## 🌿 Tespit Edilen Hastalıklar

| # | Hastalık | Etken | Tehlike Seviyesi |
|---|----------|-------|-----------------|
| 1 | Sağlıklı | — | 🟢 Düşük |
| 2 | Kahverengi Pas | *Puccinia triticina* | 🟡 Orta |
| 3 | Sarı Pas | *Puccinia striiformis* | 🟠 Yüksek |
| 4 | Siyah Pas (Kök Pas) | *Puccinia graminis* | 🔴 Çok Yüksek |
| 5 | Külleme | *Blumeria graminis* | 🟡 Orta |
| 6 | Septorya Yaprak Yanıklığı | *Zymoseptoria tritici* | 🟠 Yüksek |

---

## 📁 Repo Yapısı

```
bugday-hastaligi-tespiti/
├── data/
│   ├── wheat_disease/
│   │   ├── train/          # Eğitim görüntüleri (sınıf klasörleri)
│   │   └── test/           # Test görüntüleri
│   └── README.md           # Veri sözlüğü
├── notebooks/
│   ├── 01_EDA.ipynb         # Keşifsel Veri Analizi
│   ├── 02_baseline.ipynb    # Baseline modeller (SVM, RF)
│   ├── 03_CNN_model.ipynb   # CNN geliştirme ve eğitim
│   └── 04_evaluation.ipynb  # Metrikler ve yorumlama
├── src/
│   └── train_model.py       # Model eğitim scripti
├── api/
│   └── predict.py           # REST API (Flask)
├── models/
│   └── bugday_model.h5      # Kayıtlı model
├── docs/
│   ├── confusion_matrix.png
│   ├── egitim_grafigi.png
│   └── teknik_rapor.pdf
├── app.py                   # Streamlit demo
├── requirements.txt
└── README.md
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Repo'yu klonlayın
```bash
git clone https://github.com/Hilmisngl/bugday-hastaligi-tespiti.git
cd bugday-hastaligi-tespiti
```

### 2. Sanal ortam oluşturun
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### 4. Veriyi indirin (Dataninja veya Kaggle)
```bash
# Kaggle API ile
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
# Ya da Dataninja üzerinden manual indirin
```

### 5. Modeli eğitin
```bash
python src/train_model.py
```

### 6. Demo'yu başlatın
```bash
streamlit run app.py
```
Tarayıcınızda `http://localhost:8501` adresini açın.

---

## 📊 Model Performansı

| Metrik | Değer |
|--------|-------|
| Test Doğruluğu | **%94.2** |
| F1 Skoru (Weighted) | **0.938** |
| ROC-AUC | **0.987** |
| Eğitim Süresi | ~45 dk (GPU) |

**Model Mimarisi:**
```
EfficientNetB3 (ImageNet, frozen) 
→ GlobalAveragePooling2D
→ Dropout(0.3)
→ Dense(128, ReLU)
→ Dropout(0.2)
→ Dense(6, Softmax)
```

---

## 🔬 Proje Yaşam Döngüsü (CRISP-DM)

- [x] Faz 1: Problem Tanımlama
- [x] Faz 2: Veri Toplama & Etiketleme
- [x] Faz 3: Keşifsel Veri Analizi (EDA)
- [x] Faz 4: Model Geliştirme & Eğitim
- [x] Faz 5: Değerlendirme & Yorumlama
- [x] Faz 6: Dağıtım & Arayüz

---

## ⚖️ Yapay Zeka Etiği

- **Şeffaflık:** Her tahmin için güven skoru gösterilmektedir.
- **Sorumlu Kullanım:** Bu sistem bir karar destek aracıdır. Nihai tarımsal karar için uzman danışmanı önerilir.
- **Veri Lisansı:** Dataninja açık veri politikasına uygun kullanım.

---

## 👤 Geliştirici

**Hilmi Sungül**  
SUBU Elektrik-Elektronik Mühendisliği  
GitHub: [@Hilmisngl](https://github.com/Hilmisngl)
