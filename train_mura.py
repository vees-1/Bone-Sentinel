import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = "MURA-v1.1"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5

# -------------------------
# LOAD IMAGE PATHS
# -------------------------
def load_image_paths(csv_file):
    df = pd.read_csv(csv_file, header=None)
    return df[0].values

train_paths = load_image_paths(
    os.path.join(DATASET_DIR, "train_image_paths.csv")
)

val_paths = load_image_paths(
    os.path.join(DATASET_DIR, "valid_image_paths.csv")
)

# -------------------------
# IMAGE LOADER
# -------------------------
def load_images(paths):
    images = []
    labels = []

    for path in tqdm(paths):
        img_path = path  # FIXED

        label = 1 if "positive" in path.lower() else 0

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img) / 255.0

            images.append(img)
            labels.append(label)
        except Exception as e:
            print("Failed:", img_path, e)

    return np.array(images), np.array(labels)
# -------------------------
# LOAD DATA
# -------------------------


print("📥 Loading training images...")
X_train, y_train = load_images(train_paths)

print("📥 Loading validation images...")
X_val, y_val = load_images(val_paths)

print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# -------------------------
# TF DATASETS
# -------------------------
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------
# MODEL (TRANSFER LEARNING)
# -------------------------
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
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# TRAIN
# -------------------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -------------------------
# SAVE MODEL
# -------------------------
model.save("best_model.keras")
print("✅ Saved best_model.keras")