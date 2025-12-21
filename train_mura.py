import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# =========================
# SAFE CONFIG 
# =========================
DATASET_DIR = "MURA-v1.1"     # folder containing MURA-v1.1
IMG_SIZE = 128               
BATCH_SIZE = 1               
EPOCHS = 3                   
MAX_TRAIN = 2000             
MAX_VAL = 500                

# =========================
# LOAD CSV PATHS
# =========================
def load_image_paths(csv_file):
    df = pd.read_csv(csv_file, header=None)
    return df[0].values

train_paths = load_image_paths(
    os.path.join(DATASET_DIR, "train_image_paths.csv")
)[:MAX_TRAIN]

val_paths = load_image_paths(
    os.path.join(DATASET_DIR, "valid_image_paths.csv")
)[:MAX_VAL]

# =========================
# LOAD IMAGES (RAM SAFE)
# =========================
def load_images(paths):
    images = []
    labels = []

    for rel_path in tqdm(paths):
        full_path = rel_path  # CSV already includes MURA-v1.1
        label = 1 if "positive" in rel_path.lower() else 0

        try:
            img = Image.open(full_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img, dtype=np.float32)

            img = tf.keras.applications.efficientnet.preprocess_input(img)

            images.append(img)
            labels.append(label)
        except Exception as e:
            print("Failed:", full_path)

    return np.array(images), np.array(labels)

print("📥 Loading training data...")
X_train, y_train = load_images(train_paths)

print("📥 Loading validation data...")
X_val, y_val = load_images(val_paths)

print("Train samples:", len(X_train))
print("Val samples:", len(X_val))

if len(X_train) == 0 or len(X_val) == 0:
    raise RuntimeError("❌ No images loaded. Check dataset paths.")

# =========================
# CLASS WEIGHTS
# =========================
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
total = neg + pos

class_weights = {
    0: total / (2 * neg),
    1: total / (2 * pos)
}

print("Class weights:", class_weights)

# =========================
# TF DATASETS
# =========================
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(500).batch(BATCH_SIZE).prefetch(1)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(1)

# =========================
# MODEL (FROZEN BACKBONE)
# =========================
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# =========================
# CHECKPOINT
# =========================
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)

# =========================
# TRAIN
# =========================
print("🚀 Starting SAFE local training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint_cb]
)

# =========================
# EVALUATE
# =========================
loss, acc, auc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation AUC: {auc:.4f}")

print("✅ Training finished safely. best_model.keras saved.")
