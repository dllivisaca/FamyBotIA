from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

CATALOG_URL = "https://app.famysaludec.com/chatbot/catalogo-servicios"

vectorizer = joblib.load(MODEL_DIR / "vectorizer_famybot_v1.pkl")
classifier = joblib.load(MODEL_DIR / "classifier_famybot_v1.pkl")

app = FastAPI(title="FamyBot IA API", version="1.0.0")


class PredictRequest(BaseModel):
    texto: str

class SearchRequest(BaseModel):
    texto: str

@app.get("/")
def home():
    return {"status": "ok", "message": "FamyBot IA API activa"}


@app.post("/predict")
def predict(request: PredictRequest):
    texto = request.texto.strip()

    if not texto:
        return {
            "intencion": "desconocido",
            "texto": texto
        }

    X = vectorizer.transform([texto])
    intencion = classifier.predict(X)[0]

    return {
        "texto": texto,
        "intencion": intencion
    }


@app.get("/catalog")
def get_catalog():
    response = requests.get(CATALOG_URL, timeout=15)
    response.raise_for_status()
    return response.json()