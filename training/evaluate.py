from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.keras"
TEST_DIR = ROOT / "data" / "test"

IMG_SIZE = (224, 224)
BATCH = 8

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False,
    label_mode="categorical"
)

loss, acc = model.evaluate(test_ds)
print("\n✅ TEST accuracy:", acc)
print("✅ TEST loss:", loss)
print("Class order:", test_ds.class_names)
