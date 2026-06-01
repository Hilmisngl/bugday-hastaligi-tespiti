"""
Buğday Hastalığı Tespiti - CNN Model Eğitim Scripti
SUBU EEF | Yapay Zekaya Giriş | 2024-2025 Bahar

Kullanım:
    python src/train_model.py

Gereksinimler:
    pip install tensorflow scikit-learn matplotlib seaborn
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Konfigürasyon ───────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 6
DATA_DIR = "data/wheat_disease"
MODEL_SAVE_PATH = "models/bugday_model.h5"

SINIF_ADLARI = [
    "Sağlıklı",
    "Kahverengi Pas",
    "Sarı Pas",
    "Siyah Pas",
    "Külleme",
    "Septorya Yaprak Yanıklığı"
]

# ─── Veri Yükleme & Augmentation ─────────────────────────────────────────────
def veri_hazirla(data_dir, img_size=224, batch_size=32):
    """Veriyi yükler, train/val/test split yapar."""
    
    # Augmentation (sadece eğitim için)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


# ─── Model Mimarisi ───────────────────────────────────────────────────────────
def model_olustur(num_classes=6, img_size=224):
    """
    Transfer Learning: EfficientNetB3 + Custom Head
    
    Mimari:
        EfficientNetB3 (ImageNet weights, frozen)
        → GlobalAveragePooling2D
        → Dropout(0.3)
        → Dense(128, ReLU)
        → Dropout(0.2)
        → Dense(num_classes, Softmax)
    """
    # Base model (ilk eğitimde dondurulmuş)
    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False  # Feature extraction aşaması
    
    # Custom head
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model


# ─── Eğitim ───────────────────────────────────────────────────────────────────
def model_egit(train_gen, val_gen, num_classes, epochs, learning_rate):
    model, base_model = model_olustur(num_classes)
    
    # Faz 1: Sadece head eğitimi
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_accuracy"
        )
    ]
    
    print("─── Faz 1: Feature Extraction ───")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Faz 2: Fine-tuning (son 30 katman serbest bırakılır)
    print("\n─── Faz 2: Fine-Tuning ───")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history1, history2


# ─── Değerlendirme ────────────────────────────────────────────────────────────
def model_degerlendir(model, test_gen, sinif_adlari):
    """Confusion matrix, classification report üretir."""
    
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes
    
    print("\n─── Classification Report ───")
    print(classification_report(y_true, y_pred, target_names=sinif_adlari))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=sinif_adlari,
        yticklabels=sinif_adlari,
        cmap="Greens"
    )
    plt.title("Confusion Matrix — Buğday Hastalığı Tespiti")
    plt.ylabel("Gerçek")
    plt.xlabel("Tahmin")
    plt.tight_layout()
    plt.savefig("docs/confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix 'docs/confusion_matrix.png' olarak kaydedildi.")


# ─── Eğitim Geçmişi Grafiği ──────────────────────────────────────────────────
def egitim_grafigi_ciz(history1, history2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    epochs1 = range(1, len(history1.history["accuracy"]) + 1)
    epochs2 = range(len(epochs1) + 1, len(epochs1) + len(history2.history["accuracy"]) + 1)
    
    axes[0].plot(epochs1, history1.history["accuracy"], "b-o", label="Eğitim (Faz 1)")
    axes[0].plot(epochs1, history1.history["val_accuracy"], "b--", label="Doğrulama (Faz 1)")
    axes[0].plot(epochs2, history2.history["accuracy"], "g-o", label="Eğitim (Faz 2)")
    axes[0].plot(epochs2, history2.history["val_accuracy"], "g--", label="Doğrulama (Faz 2)")
    axes[0].set_title("Model Doğruluğu")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Loss
    axes[1].plot(epochs1, history1.history["loss"], "b-o", label="Eğitim (Faz 1)")
    axes[1].plot(epochs1, history1.history["val_loss"], "b--", label="Doğrulama (Faz 1)")
    axes[1].plot(epochs2, history2.history["loss"], "g-o", label="Eğitim (Faz 2)")
    axes[1].plot(epochs2, history2.history["val_loss"], "g--", label="Doğrulama (Faz 2)")
    axes[1].set_title("Model Kaybı")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("docs/egitim_grafigi.png", dpi=150)
    plt.show()
    print("Eğitim grafiği 'docs/egitim_grafigi.png' olarak kaydedildi.")


# ─── Ana Akış ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    print("Veri hazırlanıyor...")
    train_gen, val_gen, test_gen = veri_hazirla(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    
    print(f"Eğitim örnekleri: {train_gen.samples}")
    print(f"Doğrulama örnekleri: {val_gen.samples}")
    print(f"Test örnekleri: {test_gen.samples}")
    print(f"Sınıflar: {train_gen.class_indices}")
    
    print("\nModel oluşturuluyor ve eğitiliyor...")
    model, history1, history2 = model_egit(train_gen, val_gen, NUM_CLASSES, EPOCHS, LEARNING_RATE)
    
    print("\nModel değerlendiriliyor...")
    model_degerlendir(model, test_gen, SINIF_ADLARI)
    
    egitim_grafigi_ciz(history1, history2)
    
    print(f"\nModel '{MODEL_SAVE_PATH}' olarak kaydedildi.")
    print("Eğitim tamamlandı!")
