from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
import requests
import unicodedata

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

CATALOG_URL = "https://app.famysaludec.com/chatbot/catalogo-servicios"
INTENCIONES_CATALOGO = {
    "cotizar_servicio",
    "consulta_servicios",
    "consulta_especialidades",
}
PALABRAS_IGNORADAS_CATALOGO = {
    "a",
    "de",
    "del",
    "el",
    "en",
    "especialidad",
    "especialidades",
    "hay",
    "hacen",
    "hacer",
    "informacion",
    "la",
    "las",
    "los",
    "me",
    "para",
    "por",
    "precio",
    "puede",
    "puedes",
    "realiza",
    "realizan",
    "realizar",
    "servicio",
    "servicios",
    "sobre",
    "tiene",
    "tienen",
    "un",
    "una",
    "valor",
}

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


def obtener_catalogo():
    try:
        response = requests.get(CATALOG_URL, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="No se pudo obtener el catalogo de servicios",
        ) from exc


def normalizar_texto(texto):
    texto = str(texto or "").strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(caracter for caracter in texto if unicodedata.category(caracter) != "Mn")


def buscar_servicios(texto_busqueda):
    catalogo = obtener_catalogo()
    texto = normalizar_texto(texto_busqueda)
    resultados = []

    if not texto:
        return resultados

    areas = catalogo.get("areas", []) if isinstance(catalogo, dict) else []

    for area in areas:
        servicios = area.get("services", []) if isinstance(area, dict) else []
        nombre_area = area.get("title") or area.get("name")
        categoria_area = area.get("category")

        for servicio in servicios:
            campos = [
                servicio.get("title"),
                servicio.get("name"),
                servicio.get("excerpt"),
                servicio.get("description"),
                nombre_area,
                categoria_area,
                servicio.get("area"),
                servicio.get("category"),
            ]
            texto_servicio = " ".join(normalizar_texto(campo) for campo in campos if campo)

            if texto in texto_servicio:
                resultados.append({
                    "id": servicio.get("id"),
                    "nombre": servicio.get("title") or servicio.get("name"),
                    "area": nombre_area,
                    "precio": servicio.get("price") or servicio.get("precio"),
                    "precio_promocion": (
                        servicio.get("promotion_price")
                        or servicio.get("precio_promocion")
                    ),
                    "presencial": bool(servicio.get("presencial")),
                    "virtual": bool(servicio.get("virtual")),
                })

                if len(resultados) >= 20:
                    return resultados

    return resultados


def preparar_consulta_catalogo(texto):
    palabras = []

    for palabra in normalizar_texto(texto).split():
        if palabra in PALABRAS_IGNORADAS_CATALOGO:
            continue
        if palabra.endswith("s") and len(palabra) > 4:
            palabra = palabra[:-1]
        palabras.append(palabra)

    return " ".join(palabras) or texto


@app.get("/catalog")
def get_catalog():
    return obtener_catalogo()


@app.post("/search-service")
def search_service(request: SearchRequest):
    texto = request.texto.strip()
    resultados = buscar_servicios(texto)

    return {
        "texto": texto,
        "total": len(resultados),
        "resultados": resultados,
    }


@app.post("/ask-catalog")
def ask_catalog(request: SearchRequest):
    texto = request.texto.strip()
    resultados = buscar_servicios(request.texto)
    total = len(resultados)

    if total == 0:
        accion = "sin_resultados"
        mensaje = (
            "No encontré servicios relacionados con tu consulta. Puedes escribir "
            "el nombre del servicio de otra forma o solicitar ayuda con un asesor."
        )
    elif total == 1:
        accion = "respuesta_directa"
        servicio = resultados[0]
        mensaje = (
            f"El servicio {servicio.get('nombre')} pertenece al área "
            f"{servicio.get('area')} y tiene un valor de ${servicio.get('precio')}."
        )
    else:
        accion = "listar_opciones"
        mensaje = (
            f"Encontré {total} opciones relacionadas con tu consulta. Puedes "
            "revisar la lista y responder con el nombre del servicio que deseas "
            "consultar."
        )

    return {
        "texto": texto,
        "total": total,
        "accion": accion,
        "mensaje": mensaje,
        "resultados": resultados,
    }


@app.post("/chat")
def chat(request: SearchRequest):
    texto = request.texto.strip()
    prediccion = predict(PredictRequest(texto=texto))
    intencion = prediccion["intencion"]

    if intencion in INTENCIONES_CATALOGO:
        consulta_catalogo = preparar_consulta_catalogo(texto)
        respuesta_catalogo = ask_catalog(SearchRequest(texto=consulta_catalogo))
        return {
            "texto": texto,
            "intencion": intencion,
            "accion": respuesta_catalogo["accion"],
            "mensaje": respuesta_catalogo["mensaje"],
            "total": respuesta_catalogo["total"],
            "resultados": respuesta_catalogo["resultados"],
        }

    return {
        "texto": texto,
        "intencion": intencion,
        "accion": "pendiente",
        "mensaje": "Esta intenci\u00f3n a\u00fan no est\u00e1 implementada.",
    }
