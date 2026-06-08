# ─────────────────────────────────────────────────────────────
# predict.py — Tek Görüntü Tahmin Scripti
# SUBÜ EEM · Yapay Zekaya Giriş 2025-2026
#
# Kullanım:
#   python src/predict.py --image yol/goruntu.jpg
#   python src/predict.py --image yol/goruntu.jpg --model models/bugday_model.h5
# ─────────────────────────────────────────────────────────────

import argparse
import json
import os
import numpy as np
from PIL import Image

def load_model_and_classes(model_path, classes_path):
    """Model ve sınıf isimlerini yükle."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model yüklendi: {model_path}")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return None, None

    try:
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        print(f"✅ Sınıflar yüklendi: {class_names}")
    except Exception as e:
        print(f"❌ Sınıf dosyası okunamadı: {e}")
        return model, None

    return model, class_names


def preprocess_image(image_path, img_size=224):
    """Görüntüyü modele uygun formata getir."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image_path, model_path="models/bugday_model.h5",
            classes_path="models/class_names.json", top_k=3):
    """
    Görüntüdeki buğday hastalığını tahmin et.

    Args:
        image_path   : Görüntü dosyası yolu
        model_path   : Eğitilmiş model (.h5) yolu
        classes_path : Sınıf isimleri JSON dosyası yolu
        top_k        : Gösterilecek en iyi k tahmin sayısı

    Returns:
        dict: En iyi tahmin ve tüm sınıf olasılıkları
    """
    if not os.path.exists(image_path):
        print(f"❌ Görüntü bulunamadı: {image_path}")
        return None

    model, class_names = load_model_and_classes(model_path, classes_path)
    if model is None or class_names is None:
        return None

    # Ön işleme ve tahmin
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]

    # Top-k sonuçlar
    top_indices = np.argsort(predictions)[::-1][:top_k]

    print("\n" + "="*50)
    print(f"📷 Görüntü : {os.path.basename(image_path)}")
    print("="*50)

    results = []
    for rank, idx in enumerate(top_indices):
        class_name = class_names[idx]
        confidence = float(predictions[idx]) * 100
        marker = "🥇" if rank == 0 else ("🥈" if rank == 1 else "🥉")
        print(f"{marker} {rank+1}. {class_name:<20} → %{confidence:.1f}")
        results.append({"rank": rank + 1, "class": class_name, "confidence": confidence})

    print("="*50)
    best = results[0]
    print(f"\n✅ Sonuç  : {best['class']}")
    print(f"🎯 Güven  : %{best['confidence']:.1f}")

    return {
        "best_prediction": best,
        "top_k": results,
        "all_probabilities": {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Buğday hastalığı tespiti — tek görüntü tahmin aracı"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Tahmin yapılacak görüntünün dosya yolu"
    )
    parser.add_argument(
        "--model", type=str, default="models/bugday_model.h5",
        help="Eğitilmiş model dosyası yolu (varsayılan: models/bugday_model.h5)"
    )
    parser.add_argument(
        "--classes", type=str, default="models/class_names.json",
        help="Sınıf isimleri JSON dosyası yolu"
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Gösterilecek en iyi tahmin sayısı (varsayılan: 3)"
    )

    args = parser.parse_args()
    predict(
        image_path=args.image,
        model_path=args.model,
        classes_path=args.classes,
        top_k=args.top_k
    )
