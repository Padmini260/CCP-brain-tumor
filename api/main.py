from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from .predictor import Predictor

app = FastAPI(title="Brain Tumor Classifier API")
predictor = Predictor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Brain Tumor Classifier API is running. Visit /docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    pred_class, confidence, all_probs = predictor.predict(img)

    verdict = pred_class if confidence >= 0.60 else "Uncertain"

    return {
        "verdict": verdict,
        "predicted_class": pred_class,
        "confidence": confidence,
        "all_probs": all_probs
    }
