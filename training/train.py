import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all logs, 1=filter INFO, 2=filter INFO+WARNING, 3=only errors

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import tensorflow as tf

# CPU-friendly settings for low-end laptops
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 8

# -----------------------
# 1) Load datasets
# -----------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=True
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Save class names for API mapping later
with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------
# 2) Augmentation (lighter for better validation stability)
# -----------------------
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomZoom(0.05),
], name="augmentation")

# -----------------------
# 3) Build model (EfficientNetB0)
# -----------------------
base = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Stage 1: freeze base
base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augment(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# -----------------------
# 4) Callbacks (stable)
# -----------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    ),
]

# -----------------------
# 5) Stage 1 training (frozen base)
# -----------------------
print("\n=== Stage 1: Train top layers (base frozen) ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)

# -----------------------
# 6) Stage 2 fine-tuning (unfreeze last 30 layers)
# -----------------------
print("\n=== Stage 2: Fine-tune last layers (lower LR) ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=callbacks
)

print("\n✅ Training finished.")
print(f"✅ Best model saved at: {MODEL_DIR / 'best_model.keras'}")
print(f"✅ Class names saved at: {MODEL_DIR / 'class_names.json'}")
