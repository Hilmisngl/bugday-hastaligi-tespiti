# 🌾 Buğday Hastalığı Tespiti

**SUBÜ EEM · Yapay Zekaya Giriş 2025-2026 Bahar Dönemi Dönem Projesi**

> Derin öğrenme tabanlı buğday hastalığı görüntü sınıflandırma sistemi.  
> EfficientNetB0 + Transfer Learning + Gradio Web Arayüzü

---

## 👥 Proje Ekibi

| İsim | Rol |
|------|-----|
| Hilmi Şengül | Proje Lideri, Model Geliştirme |
| Cevdet Emre Oruç | Veri & Demo Geliştirme |

---

## 📋 Proje Özeti

Bu proje, buğday bitkilerindeki hastalıkları fotoğraflardan otomatik olarak tespit eden bir yapay zeka sistemi geliştirmeyi amaçlamaktadır. Dataninja platformundan elde edilen buğday hastalığı görüntü veri seti kullanılarak EfficientNetB0 mimarisi üzerinde transfer öğrenimi uygulanmıştır.

**Tespit Edilen Hastalıklar:**
- 🟡 Sarı Pas (Puccinia striiformis)
- 🟤 Kahverengi Pas (Puccinia triticina)
- 🟠 Sap Pası (Puccinia graminis)
- ⚪ Külleme (Blumeria graminis)
- 🔵 Septorya Yaprak Lekesi
- ⚫ Rastık (Ustilago tritici)
- ✅ Sağlıklı

---

## 📁 Repo Yapısı

```
bugday-hastaligi-tespiti/
├── notebooks/          # Ana Jupyter notebook (Colab)
│   └── Buğday hastalık tespit.ipynb
├── src/                # Modüler Python kodları
├── data/               # Veri seti bağlantıları ve örnekler
├── docs/               # Teknik rapor ve görseller
├── api/                # Gradio demo uygulaması
├── README.md
└── requirements.txt
```

---

## 🚀 Kurulum ve Çalıştırma

### Google Colab (Önerilen)

1. [Notebook'u Colab'da aç](https://colab.research.google.com/github/Hilmisngl/bugday-hastaligi-tespiti/blob/main/notebooks/Bu%C4%9Fday%20hastal%C4%B1k%20tespit.ipynb)
2. Runtime → Change Runtime Type → **GPU (T4)**
3. Hücreleri sırayla çalıştır
4. Son hücrede Gradio `share=True` linki otomatik oluşur

### Lokal Kurulum

```bash
git clone https://github.com/Hilmisngl/bugday-hastaligi-tespiti.git
cd bugday-hastaligi-tespiti
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## 🧠 Model Mimarisi

| Parametre | Değer |
|-----------|-------|
| Base Model | EfficientNetB0 (ImageNet) |
| Görüntü Boyutu | 224 × 224 px |
| Faz 1 LR | 1e-3 (başlık eğitimi) |
| Faz 2 LR | 1e-5 (fine-tuning) |
| Val Accuracy | %93.1 |
| Macro F1 | 0.928 |

---

## 📊 Sonuçlar

| Model | Val Acc | F1 |
|-------|---------|----|
| Logistic Regression | %51.3 | 0.491 |
| Random Forest | %67.8 | 0.665 |
| VGG16 | %87.4 | 0.869 |
| ResNet50 | %90.2 | 0.897 |
| **EfficientNetB0 (Bizim)** | **%93.1** | **0.928** |

---

## ⚠️ Kullanım Notu

Bu uygulama yalnızca eğitim amaçlıdır. Gerçek tarımsal kararlar için uzman görüşü alınız.

---

## 📚 Dataset

- **Kaynak:** Dataninja — Buğday Hastalıkları Görüntü Analizi
- **Proje Kılavuzu:** Dataset #15
- **Sınıf Sayısı:** 7
- **Toplam Görüntü:** ~6480

---

*SUBÜ Elektrik-Elektronik Mühendisliği · 2025-2026*
