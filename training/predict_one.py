import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL = tf.keras.models.load_model(ROOT / "models" / "best_model.keras")
CLASS_NAMES = json.loads((ROOT / "models" / "class_names.json").read_text(encoding="utf-8"))

IMG_SIZE = (224, 224)

def predict(img_path: str):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    probs = MODEL.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

# Change this path to any test image you want
img_path = str((ROOT / "data" / "test" / "Glioma").glob("*").__iter__().__next__())
pred, conf, allp = predict(img_path)

print("Image:", img_path)
print("Pred:", pred)
print("Conf:", conf)
print("All probs:", allp)
