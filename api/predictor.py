import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

class Predictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_DIR / "best_model.keras")
        self.class_names = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
        self.img_size = (224, 224)

    def predict(self, img: Image.Image):
        img = img.convert("RGB").resize(self.img_size)
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.efficientnet.preprocess_input(x)

        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        pred_class = self.class_names[idx]
        confidence = float(probs[idx])
        all_probs = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        return pred_class, confidence, all_probs
