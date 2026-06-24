from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from difflib import SequenceMatcher
import json
import joblib
import os
from pathlib import Path
import requests
import traceback
from typing import Optional
import unicodedata
import re

from api.services.embedding_search import (
    buscar_servicios_semanticos,
    busqueda_semantica_disponible,
    estado_embeddings,
    obtener_embedding_model as obtener_embedding_model_semantico,
)
from api.services.service_index import construir_texto_indexable

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except Exception:
    rapidfuzz_fuzz = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

CATALOG_URL = "https://app.famysaludec.com/chatbot/catalogo-servicios"
FAMYBOT_IA_API_KEY = os.environ.get("FAMYBOT_IA_API_KEY", "").strip()
APP_CODE_VERSION = "2026-06-22-mixed-intent-v3"
MIN_CONF_ACCION_FLUJO = 0.55
MIN_RATIO_FUZZY_INTENCION = 0.86
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SCORE_SEMANTICO_CATALOGO = 0.62
MIN_SCORE_SEMANTICO_CON_EXACTO_FUZZY = 0.82
CONSULTAS_AMBIGUAS_SEMANTICAS = {
    "eco",
    "ecografia",
    "electro",
    "resonancia",
    "tomografia",
}
INTENCIONES_CATALOGO = {
    "cotizar_servicio",
    "consulta_servicios",
    "consulta_especialidades",
}
RESPUESTAS_SIMPLES = {
    "saludo": {
        "accion": "respuesta_simple",
        "mensaje": "¡Hola! Soy FamyBot IA. Puedo ayudarte con información sobre servicios, precios, horarios, ubicación y opciones de atención.",
    },
    "agradecimiento": {
        "accion": "respuesta_simple",
        "mensaje": "¡Con gusto! Estoy aquí para ayudarte.",
    },
    "consultar_horario": {
        "accion": "respuesta_simple",
        "mensaje": "Nuestro horario de atención es de lunes a viernes de 7:30 a 17:30 y sábados de 8:00 a 12:30.",
    },
    "consultar_ubicacion": {
        "accion": "respuesta_simple",
        "mensaje": "Nos encontramos en Quisquís 1109 y José Mascote, Guayaquil. Puedes abrir la ubicación en Google Maps o ver el croquis usando los botones disponibles.",
    },
    "desconocido": {
        "accion": "fallback",
        "mensaje": "No estoy seguro de haber entendido tu consulta. Puedes escribir una nueva consulta o usar el botón Menú principal.",
    },
}
MENSAJE_UBICACION_BOTONES = (
    "Estamos ubicados en Quisquís 1109 y José Mascote, Guayaquil.\n\n"
    "Si lo prefieres, puedes usar los botones para abrir la ubicación "
    "en Google Maps o ver el croquis de referencia."
)
MENSAJE_UBICACION_CON_BOTONES = RESPUESTAS_SIMPLES["consultar_ubicacion"]["mensaje"]
ACCIONES_FLUJO = {
    "agendar_cita": {
        "accion": "iniciar_agendamiento",
        "mensaje": "Te ayudo a iniciar el agendamiento de tu cita.",
    },
    "hablar_asesor": {
        "accion": "derivar_asesor",
        "mensaje": "Te puedo comunicar con un asesor para recibir atención personalizada.",
    },
    "consultar_resultados": {
        "accion": "solicitar_resultados",
        "mensaje": "Te ayudo a iniciar la solicitud de resultados.",
    },
    "proveedores": {
        "accion": "abrir_proveedores",
        "mensaje": "Te ayudo con las opciones para proveedores.",
    },
    "alianzas": {
        "accion": "abrir_alianzas",
        "mensaje": "Te ayudo con las opciones de alianzas estratégicas.",
    },
    "trabajo": {
        "accion": "abrir_trabajo",
        "mensaje": "Te ayudo con la información para trabajar con nosotros.",
    },
    "consultar_promociones": {
        "accion": "mostrar_promociones",
        "mensaje": "Te muestro las promociones disponibles.",
    },
}
PALABRAS_SALUDO_PURO = {
    "bna",
    "bnas",
    "bno",
    "bnos",
    "buen",
    "buena",
    "buenas",
    "buenos",
    "dia",
    "dias",
    "hola",
    "holaa",
    "holaaa",
    "holaaaa",
    "noche",
    "noches",
    "ola",
    "saludo",
    "saludos",
    "tarde",
    "tardes",
}
PALABRAS_FUZZY_INTENCION = {
    "consultar_ubicacion": {
        "ubicacion",
        "ubicaciones",
        "ubicado",
        "ubicados",
    },
    "consultar_horario": {
        "horario",
        "horarios",
    },
    "agendar_cita": {
        "agendar",
        "ajendar",
    },
    "proveedores": {
        "proveedor",
        "proveedores",
    },
    "alianzas": {
        "alianza",
        "alianzas",
        "aliansa",
    },
}
PALABRAS_RELLENO_INTENCION = PALABRAS_SALUDO_PURO | {
    "ayuda",
    "ayudas",
    "informacion",
    "necesito",
    "quiero",
    "quisiera",
    "saber",
}
PALABRAS_COMERCIALES_CATALOGO = {
    "consulta",
    "costo",
    "cuanto",
    "cuesta",
    "dispone",
    "disponen",
    "hacen",
    "realizan",
    "servicio",
    "servicios",
    "tiene",
    "tienen",
    "valor",
    "precio",
}
PALABRAS_COTIZACION = {
    "costo",
    "cuesta",
    "precio",
    "salen",
    "valor",
    "vale",
}
FRASES_COTIZACION = {
    "cuanto cuesta",
    "cuanto sale",
    "cuanto salen",
    "que cuesta",
    "que vale",
    "costo d",
    "precio de",
    "valor de",
}
PALABRAS_TRABAJO = {
    "contratando",
    "contratar",
    "curriculum",
    "curriculo",
    "cv",
    "laboral",
    "trabajar",
    "trabajo",
    "vacante",
}
FRASES_TRABAJO = {
    "asistente dental",
    "busca de trabajo",
    "buscando una oportunidad laboral",
    "correo de recursos humanos",
    "correo para enviar",
    "enviar mi cv",
    "enviar mi hoja de vida",
    "hoja de vida",
    "medico general",
    "oferta laboral",
    "personal requieren",
    "recursos humanos",
    "requieren de terapeuta",
    "soy enfermera",
    "soy medico",
    "terapeuta respiratorio",
}
PALABRAS_CONSULTA_SERVICIOS = {
    "caries",
    "examen",
    "examenes",
    "hacen",
    "limpieza",
    "quitan",
    "realizan",
    "realizar",
    "servicio",
    "tapar",
    "tiene",
    "tienen",
}
ESPECIALIDADES_CONOCIDAS = {
    "cardiologia",
    "dermatologia",
    "gastroenterologia",
    "gastroenterologo",
    "neumologia",
    "nutricion",
    "oftalmologia",
    "pediatria",
    "traumatologia",
    "urologia",
}
TERMINOS_MEDICOS_CLAROS = {
    "audiometria",
    "blanqueamiento",
    "calce",
    "calces",
    "caries",
    "cordal",
    "cordales",
    "doppler",
    "ecocardiograma",
    "ecografia",
    "eco",
    "electrocardiogrma",
    "electrocardiograma",
    "electromiografia",
    "endodoncia",
    "espirometria",
    "embarazo",
    "extraccion",
    "hepatobiliar",
    "hcg",
    "holter",
    "lavado",
    "limpieza",
    "mamografia",
    "mamas",
    "morfológica",
    "morfologica",
    "molar",
    "molares",
    "muela",
    "neumólogia",
    "neumologia",
    "odontologia",
    "placa",
    "prostata",
    "protesis",
    "profilaxis",
    "rayos",
    "radiografia",
    "recanalizacion",
    "renal",
    "resonancia",
    "restauracion",
    "rx",
    "sedacion",
    "trompas",
    "tubarica",
    "falopio",
    "nitroso",
    "oxido",
    "tomografia",
    "torax",
    "transvaginal",
}
PALABRAS_IGNORADAS_COMERCIAL = PALABRAS_SALUDO_PURO | {
    "al",
    "con",
    "consulta",
    "costo",
    "cual",
    "cuanto",
    "cuesta",
    "d",
    "de",
    "del",
    "dia",
    "disculpe",
    "dispone",
    "disponen",
    "el",
    "exacta",
    "es",
    "favor",
    "gracias",
    "guayaquil",
    "hacerme",
    "indicar",
    "la",
    "las",
    "lo",
    "los",
    "m",
    "mucha",
    "muchas",
    "orden",
    "precio",
    "por",
    "q",
    "que",
    "salen",
    "servicio",
    "servicios",
    "tambien",
    "tiene",
    "tienen",
    "uds",
    "usted",
    "un",
    "una",
    "valor",
    "vale",
    "y",
    "si",
}
MODIFICADORES_COMERCIALES_NO_SERVICIO = {
    "atencion",
    "atender",
    "atienden",
    "ayuda",
    "ayudan",
    "ayudar",
    "ayuden",
    "consulta",
    "costo",
    "cuesta",
    "directamente",
    "examen",
    "examenes",
    "favor",
    "gracias",
    "horario",
    "horarios",
    "indicar",
    "necesario",
    "orden",
    "precio",
    "prueba",
    "servicio",
    "servicios",
    "valor",
    "vale",
}
MODIFICADORES_ENTIDAD_CATALOGO = MODIFICADORES_COMERCIALES_NO_SERVICIO | {
    "agendar",
    "cita",
    "como",
    "direccion",
    "donde",
    "estan",
    "mi",
    "necesito",
    "puedo",
    "quiero",
    "quisiera",
    "reservar",
    "saber",
    "sacar",
    "turno",
    "ubicacion",
    "ubicaciones",
    "ubicado",
    "ubicados",
}
PALABRAS_HORARIO_CONSULTA = {
    "atencion",
    "atienden",
    "horario",
    "horarios",
    "sabado",
}
PALABRAS_UBICACION_CONSULTA = {
    "direccion",
    "donde",
    "ubicacion",
    "ubicaciones",
    "ubicado",
    "ubicados",
}
PALABRAS_AGENDAMIENTO_CONSULTA = {
    "ajendar",
    "agendar",
    "cita",
    "reservar",
    "turno",
}
NORMALIZACIONES_COMERCIALES_CONTROLADAS = (
    (r"\bprueba\s+de\s+holter\b", "holter"),
    (r"\bprueba\s+holter\b", "holter"),
    (r"\bexamen\s+de\s+holter\b", "holter"),
    (r"\becocardiogrma\b", "ecocardiograma"),
    (r"\belectrocardiogrma\b", "electrocardiograma"),
    (r"\bneumólogia\b", "neumologia"),
    (r"\bneumologia\b", "neumologia"),
)
CONECTORES_SERVICIOS = {
    "ademas",
    "dispone",
    "disponen",
    "e",
    "o",
    "tambien",
    "y",
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
    "misma",
    "mismo",
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
    "uds",
    "usted",
    "un",
    "una",
    "valor",
}
SINONIMOS_CATALOGO = {
    "ecografia venosa": "doppler venoso",
    "ecografía venosa": "doppler venoso",
    "lavado de oido": "lavado oidos",
    "consulta con especialista": "consulta de valoracion con el especialista",
    "consulta de valoracion": "consulta de valoracion con el especialista",
    "valoracion con especialista": "consulta de valoracion con el especialista",
    "resonancia de columna completa": "resonancia columna total",
    "resonancia magnetica de columna completa": "resonancia columna total",
    "rm columna completa": "resonancia columna total",
    "r.m. columna completa": "resonancia columna total",
    "lavado de oído": "lavado oidos",
    "prueba de embarazo": "prueba hcg",
    "test de embarazo": "prueba hcg",
    "examen de embarazo": "prueba hcg",
    "embarazo en sangre": "prueba hcg",
    "embarazo sangre": "prueba hcg",
    "prueba de embarazo de sangre": "prueba hcg",
    "pruebas de embarazo de sangre": "prueba hcg",
    "examen de embarazo en sangre": "prueba hcg",
    "beta hcg": "prueba hcg",
    "bhcg": "prueba hcg",
    "b hcg": "prueba hcg",
    "hcg embarazo": "prueba hcg",
    "gonadotropina corionica": "prueba hcg",
    "eco transvaginal": "ecografia endovaginal",
    "ecografia transvaginal": "ecografia endovaginal",
    "ultrasonido transvaginal": "ecografia endovaginal",
    "protesis dental": "protesis removible acrilica",
    "protesis": "protesis removible acrilica",
    "protesis removible": "protesis removible acrilica",
    "dientes postizos": "protesis removible acrilica",
    "sacar tercer molar": "extraccion de tercer molar",
    "sacar un tercer molar": "extraccion de tercer molar",
    "extraccion tercer molar": "extraccion de tercer molar",
    "muela del juicio": "extraccion de tercer molar",
    "cordal": "extraccion de tercer molar",
    "tercer molar": "extraccion de tercer molar",
    "terceros molares": "extraccion de tercer molar",
    "calce de muela": "restauracion",
    "calce dental": "restauracion",
    "calce de diente": "restauracion",
    "restauracion dental": "restauracion",
    "resina": "restauracion",
    "extraccion de muela": "extraccion dental",
    "extraccion muela": "extraccion dental",
    "sacar muela": "extraccion dental",
    "limpieza dental": "profilaxis",
    "ecografia abdominal completa": "ecografia abdominal",
    "quitan caries": "restauracion",
    "tapar caries": "restauracion",
    "caries": "restauracion",
    "operaciones": "cirugia",
    "operacion": "cirugia",
    "resonancia cerebral con contraste": "resonancia craneo simple contrastada",
    "resonancia cerebral contrastada": "resonancia craneo simple contrastada",
    "examenes de audiometria": "audiometria",
    "examen de audiometria": "audiometria",
    "rayos x": "radiografia",
    "rx": "radiografia",
    "placa": "radiografia",
    "eco morfologico": "ecografia morfologica",
    "eco morfológico": "ecografia morfologica",
    "eco 20 semanas": "ecografia morfologica",
    "eco de las 20 semanas": "ecografia morfologica",
    "eco de las 20 semanas de embarazo": "ecografia morfologica",
    "ecografia morfologica": "ecografia morfologica",
    "ecografía morfológica": "ecografia morfologica",
    "morfologica": "ecografia morfologica",
    "morfológica": "ecografia morfologica",
    "gas de la risa": "sedacion consciente oxido nitroso",
    "oxido nitroso": "sedacion consciente oxido nitroso",
    "óxido nitroso": "sedacion consciente oxido nitroso",
    "sedacion consciente": "sedacion consciente",
    "recanalizacion de trompas": "recanalizacion tubarica trompas de falopio",
    "recanalización de trompas": "recanalizacion tubarica trompas de falopio",
    "recanalizacion tubarica": "recanalizacion tubarica",
    "recanalización tubárica": "recanalizacion tubarica",
    "trompas de falopio": "trompas de falopio",
    "3er molar": "extraccion de tercer molar",
    "3er molares": "extraccion de tercer molar",
    "cordales": "extraccion de tercer molar",
    "examenes en las mamas": "mamas",
    "examen de mamas": "mamas",
    "resonancia contrastada abdominal": "resonancia abdomen simple contrastado",
    "resonancia abdominal contrastada": "resonancia abdomen simple contrastado",
    "tratamiento de conducto": "endodoncia",
    "calce": "restauracion",
    "calces": "restauracion",
    "blancamiento dental": "blanqueamiento dental",
    "gastroenterologo": "gastroenterologia",
    "rm": "resonancia magnetica",
    "r.m.": "resonancia magnetica",
}

CORRECCIONES_TEXTO = (
    (r"\bh+ola+\b", "hola"),
    (r"\bsiasen(?=\w)", "si hacen "),
    (r"\bsiasen\b", "si hacen"),
    (r"\bsirealisan(?=\w)", "si realizan "),
    (r"\bsirealisan\b", "si realizan"),
    (r"\brealisan\b", "realizan"),
    (r"\bajendar\b", "agendar"),
    (r"\basen\b", "hacen"),
    (r"\becocardiogrma\b", "ecocardiograma"),
    (r"\belectrocardiogrma\b", "electrocardiograma"),
    (r"\bextracion\b", "extraccion"),
    (r"\bblancamiento\b", "blanqueamiento"),
    (r"\benal\b", "renal"),
    (r"\bneumologia\b", "neumologia"),
    (r"\bprotesi\b", "protesis"),
    (r"\br\.?\s*m\.?\b", "resonancia magnetica"),
)

vectorizer = None
classifier = None
model_load_error = None
model_version = None
embedding_model = None
embedding_model_error = None


def cargar_modelo_intenciones():
    global vectorizer
    global classifier
    global model_load_error
    global model_version

    try:
        vectorizer = joblib.load(MODEL_DIR / "vectorizer_famybot_v2.pkl")
        classifier = joblib.load(MODEL_DIR / "classifier_famybot_v2.pkl")
        model_load_error = None
        model_version = "v2"
        print("FamyBot IA: modelo de intenciones activo: v2")
        return
    except Exception as exc_v2:
        print(f"FamyBot IA: error cargando modelo v2: {exc_v2}")

    try:
        vectorizer = joblib.load(MODEL_DIR / "vectorizer_famybot_v1.pkl")
        classifier = joblib.load(MODEL_DIR / "classifier_famybot_v1.pkl")
        model_load_error = None
        model_version = "v1_fallback"
        print("FamyBot IA: modelo de intenciones activo: v1_fallback")
    except Exception as exc_v1:
        vectorizer = None
        classifier = None
        model_load_error = str(exc_v1)
        model_version = None
        print(f"FamyBot IA: error cargando modelo fallback v1: {exc_v1}")


cargar_modelo_intenciones()

app = FastAPI(title="FamyBot IA API", version="1.0.0")

if not FAMYBOT_IA_API_KEY:
    print("[AUTH] warning: FAMYBOT_IA_API_KEY no configurada; API sin bloqueo")


class PredictRequest(BaseModel):
    texto: str

class SearchRequest(BaseModel):
    texto: str


def validar_api_key(
    request: Request,
    x_famybot_ia_key: Optional[str] = Header(
        default=None,
        alias="X-FamyBot-IA-Key",
    ),
):
    if not FAMYBOT_IA_API_KEY:
        return True

    if x_famybot_ia_key == FAMYBOT_IA_API_KEY:
        print(f"[AUTH] request autorizado: {request.url.path}")
        return True

    print(f"[AUTH] request rechazado: {request.url.path}")
    raise HTTPException(status_code=401, detail="unauthorized")


def obtener_embedding_model():
    global embedding_model
    global embedding_model_error

    if embedding_model is not None:
        return embedding_model

    try:
        embedding_model = obtener_embedding_model_semantico()
        embedding_model_error = None
        return embedding_model
    except Exception as exc:
        embedding_model_error = str(exc)
        raise


def calcular_similitud_coseno(vector_a, vector_b):
    valores_a = [float(valor) for valor in vector_a]
    valores_b = [float(valor) for valor in vector_b]
    producto = sum(a * b for a, b in zip(valores_a, valores_b))
    norma_a = sum(a * a for a in valores_a) ** 0.5
    norma_b = sum(b * b for b in valores_b) ** 0.5

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return producto / (norma_a * norma_b)

@app.get("/")
def home():
    return {"status": "ok", "message": "FamyBot IA API activa"}


@app.get("/health")
def health():
    catalog_ok = False
    services_count = 0

    try:
        from api.services.famysalud_api import obtener_servicios_normalizados

        servicios_normalizados = obtener_servicios_normalizados(obtener_catalogo())
        services_count = len(servicios_normalizados.get("servicios", []))
        catalog_ok = True
    except Exception:
        catalog_ok = False

    embeddings_status = estado_embeddings()

    return {
        "status": "ok",
        "model_loaded": vectorizer is not None and classifier is not None,
        "model_version": model_version,
        "app_code_version": APP_CODE_VERSION,
        "catalog_ok": catalog_ok,
        "services_count": services_count,
        "embeddings_enabled": embeddings_status["embeddings_enabled"],
        "sentence_transformers_available": embeddings_status[
            "sentence_transformers_available"
        ],
        "semantic_index_available": embeddings_status["semantic_index_available"],
        "fallback_search_enabled": True,
    }


@app.get("/diagnostics")
def diagnostics(_auth: bool = Depends(validar_api_key)):
    return health()


@app.get("/embedding-health")
def embedding_health(_auth: bool = Depends(validar_api_key)):
    embeddings_status = estado_embeddings()

    if not embeddings_status["sentence_transformers_available"]:
        return {
            "status": "disabled",
            "model_loaded": False,
            "message": "sentence-transformers no esta instalado; se usara busqueda lexical/fuzzy",
            **embeddings_status,
        }

    try:
        model = obtener_embedding_model()
        frases = [
            "ecografia abdominal",
            "ultrasonido de abdomen",
        ]
        embeddings = model.encode(frases)
        similarity = calcular_similitud_coseno(embeddings[0], embeddings[1])

        return {
            "status": "ok",
            "model_loaded": True,
            "embedding_dimension": len(embeddings[0]),
            "similarity": round(float(similarity), 4),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }


@app.get("/debug-entity-extractor")
def debug_entity_extractor(
    texto: str = "",
    _auth: bool = Depends(validar_api_key),
):
    entidades = extraer_entidades_consulta_catalogo(texto)

    return {
        "texto": texto,
        "model_version": model_version,
        "entidades": entidades,
        "usar_entidades_catalogo": consulta_catalogo_mixta_extraida(entidades),
        "has_extraer_entidades": callable(globals().get("extraer_entidades_consulta_catalogo")),
        "has_enriquecer_mensaje": callable(globals().get("enriquecer_mensaje_catalogo_con_flags")),
    }


@app.get("/debug-chat-flow")
def debug_chat_flow(
    texto: str = "",
    _auth: bool = Depends(validar_api_key),
):
    texto = texto.strip()
    prediccion = predecir_intencion(texto)
    intencion = str(prediccion["intencion"])
    entidades = extraer_entidades_consulta_catalogo(texto)
    usar_entidades_catalogo = consulta_catalogo_mixta_extraida(entidades)
    entro_branch_entidades = False
    total_catalogo = None
    accion_final = None
    search_mode_final = "none"

    if usar_entidades_catalogo:
        entro_branch_entidades = True
        respuesta_catalogo = buscar_catalogo_por_entidades(texto, entidades)
        total_catalogo = respuesta_catalogo["total"]

        if total_catalogo > 0:
            respuesta_catalogo = enriquecer_mensaje_catalogo_con_flags(
                respuesta_catalogo,
                entidades,
            )
            accion_final = respuesta_catalogo["accion"]
            search_mode_final = obtener_search_mode_busqueda(respuesta_catalogo)
        else:
            accion_final = "sin_resultados_catalogo"
            search_mode_final = obtener_search_mode_busqueda(respuesta_catalogo)

    entro_acciones_flujo = accion_final is None and intencion in ACCIONES_FLUJO

    if accion_final is None and entro_acciones_flujo:
        accion_final = ACCIONES_FLUJO[intencion]["accion"]

    return {
        "texto": texto,
        "app_code_version": APP_CODE_VERSION,
        "intencion_predicha": intencion,
        "usar_entidades_catalogo": usar_entidades_catalogo,
        "entidades": entidades,
        "entro_branch_entidades": entro_branch_entidades,
        "total_catalogo": total_catalogo,
        "entro_acciones_flujo": entro_acciones_flujo,
        "accion_final": accion_final,
        "search_mode_final": search_mode_final,
    }


@app.get("/debug-service-search")
def debug_service_search(
    texto: str = "",
    _auth: bool = Depends(validar_api_key),
):
    def resumir_top5(busqueda):
        return [
            {
                "id": resultado.get("id"),
                "nombre": resultado.get("nombre"),
                "area": resultado.get("area"),
                "precio": resultado.get("precio"),
                "score": resultado.get("score"),
            }
            for resultado in busqueda.get("resultados", [])[:5]
        ]

    entidades = extraer_entidades_consulta_catalogo(texto)
    embeddings_status = estado_embeddings()
    respuesta = {
        "texto": texto,
        "embeddings_available": embeddings_status["embeddings_enabled"],
        "sentence_transformers_available": embeddings_status[
            "sentence_transformers_available"
        ],
        "semantic_index_available": embeddings_status["semantic_index_available"],
        "fallback_search_enabled": True,
        "entidades": entidades,
        "buscar_servicios_total": None,
        "buscar_servicios_search_mode": "none",
        "buscar_servicios_top5": [],
        "buscar_catalogo_por_entidades_total": None,
        "buscar_catalogo_por_entidades_search_mode": "none",
        "buscar_catalogo_por_entidades_top5": [],
        "error": None,
    }

    try:
        busqueda_servicios = buscar_servicios(texto)
        respuesta["buscar_servicios_total"] = busqueda_servicios["total"]
        respuesta["buscar_servicios_search_mode"] = obtener_search_mode_busqueda(
            busqueda_servicios
        )
        respuesta["buscar_servicios_top5"] = resumir_top5(busqueda_servicios)

        busqueda_entidades = buscar_catalogo_por_entidades(texto, entidades)
        respuesta["buscar_catalogo_por_entidades_total"] = busqueda_entidades["total"]
        respuesta["buscar_catalogo_por_entidades_search_mode"] = (
            obtener_search_mode_busqueda(busqueda_entidades)
        )
        respuesta["buscar_catalogo_por_entidades_top5"] = resumir_top5(busqueda_entidades)
    except Exception as exc:
        respuesta["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    return respuesta


@app.get("/debug-catalog-status")
def debug_catalog_status(_auth: bool = Depends(validar_api_key)):
    cache_dir = BASE_DIR / "api" / "cache"
    cache_index_path = cache_dir / "service_index.json"
    cache_embeddings_path = cache_dir / "service_embeddings.npy"
    respuesta = {
        "catalog_ok": False,
        "updated_at": None,
        "areas_count": 0,
        "services_count": 0,
        "resonancias_count": 0,
        "sample_resonancias": [],
        "cache_dir": str(cache_dir),
        "cache_dir_exists": cache_dir.exists(),
        "cache_dir_writable": os.access(cache_dir, os.W_OK) if cache_dir.exists() else False,
        "cache_dir_listing": (
            sorted(item.name for item in cache_dir.iterdir())
            if cache_dir.exists()
            else []
        ),
        "index_path": str(cache_index_path),
        "embeddings_path": str(cache_embeddings_path),
        "cache_index_exists": cache_index_path.exists(),
        "cache_embeddings_exists": cache_embeddings_path.exists(),
        "cache_index_updated_at": None,
        "cache_documents_count": 0,
        "cache_contains_resonancia": False,
        "error": None,
    }

    try:
        from api.services.famysalud_api import obtener_servicios_normalizados

        catalogo = obtener_catalogo()
        servicios_normalizados = obtener_servicios_normalizados(catalogo)
        servicios = servicios_normalizados.get("servicios", [])
        areas = catalogo.get("areas", []) if isinstance(catalogo, dict) else []
        muestras = []

        for servicio in servicios:
            texto_servicio = normalizar_texto(
                " ".join(
                    str(valor)
                    for valor in (
                        servicio.get("nombre"),
                        servicio.get("area"),
                        servicio.get("slug"),
                        servicio.get("excerpt"),
                        servicio.get("description"),
                    )
                    if valor
                )
            )
            if any(
                termino in texto_servicio
                for termino in ("resonancia", "craneo", "cerebro")
            ):
                muestras.append({
                    "id": servicio.get("id"),
                    "nombre": servicio.get("nombre"),
                    "area": servicio.get("area"),
                    "precio": servicio.get("precio"),
                })

        respuesta.update({
            "catalog_ok": True,
            "updated_at": servicios_normalizados.get("updated_at"),
            "areas_count": len(areas),
            "services_count": len(servicios),
            "resonancias_count": sum(
                1
                for servicio in servicios
                if normalizar_texto(servicio.get("area")) == "resonancias"
            ),
            "sample_resonancias": muestras[:10],
        })

        if cache_index_path.exists():
            with cache_index_path.open("r", encoding="utf-8") as archivo:
                cache_index = json.load(archivo)

            documentos = cache_index.get("documentos", [])
            respuesta["cache_index_updated_at"] = cache_index.get("updated_at")
            respuesta["cache_documents_count"] = len(documentos)
            respuesta["cache_contains_resonancia"] = any(
                any(
                    termino in normalizar_texto(documento.get("texto"))
                    for termino in ("resonancia", "craneo", "cerebro")
                )
                for documento in documentos
            )
    except Exception as exc:
        respuesta["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    return respuesta


@app.get("/debug-build-service-index")
def debug_build_service_index(_auth: bool = Depends(validar_api_key)):
    cache_dir = BASE_DIR / "api" / "cache"
    cache_index_path = cache_dir / "service_index.json"
    cache_embeddings_path = cache_dir / "service_embeddings.npy"
    respuesta = {
        "ok": False,
        "total_documents": 0,
        "embeddings_shape": None,
        "cache_index_exists_after": False,
        "cache_embeddings_exists_after": False,
        "cache_dir_writable": os.access(cache_dir, os.W_OK) if cache_dir.exists() else False,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }

    try:
        from api.services import embedding_search

        indice = embedding_search.construir_indice_semantico(
            sinonimos_catalogo=SINONIMOS_CATALOGO,
            force_refresh=True,
        )
        embeddings = indice.get("embeddings")
        respuesta.update({
            "ok": True,
            "total_documents": len(indice.get("documentos", [])),
            "embeddings_shape": (
                str(getattr(embeddings, "shape", None))
                if embeddings is not None
                else None
            ),
            "cache_index_exists_after": cache_index_path.exists(),
            "cache_embeddings_exists_after": cache_embeddings_path.exists(),
            "cache_dir_writable": os.access(cache_dir, os.W_OK) if cache_dir.exists() else False,
        })
    except Exception as exc:
        respuesta.update({
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "cache_index_exists_after": cache_index_path.exists(),
            "cache_embeddings_exists_after": cache_embeddings_path.exists(),
            "cache_dir_writable": os.access(cache_dir, os.W_OK) if cache_dir.exists() else False,
        })

    return respuesta


@app.get("/debug-clear-service-cache")
def debug_clear_service_cache(_auth: bool = Depends(validar_api_key)):
    cache_paths = [
        BASE_DIR / "api" / "cache" / "service_index.json",
        BASE_DIR / "api" / "cache" / "service_embeddings.npy",
    ]
    respuesta = {
        "deleted": [],
        "missing": [],
        "error": None,
    }

    try:
        for cache_path in cache_paths:
            if cache_path.exists():
                cache_path.unlink()
                respuesta["deleted"].append(str(cache_path))
            else:
                respuesta["missing"].append(str(cache_path))

        try:
            from api.services import embedding_search

            embedding_search.semantic_index = None
        except Exception:
            pass
    except Exception as exc:
        respuesta["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    return respuesta


@app.get("/semantic-search-test")
def semantic_search_test(
    texto: str = "",
    _auth: bool = Depends(validar_api_key),
):
    try:
        busqueda = buscar_servicios_semanticos(
            texto,
            top_k=10,
            sinonimos_catalogo=SINONIMOS_CATALOGO,
        )

        if busqueda["total"] == 0:
            consulta_catalogo = preparar_consulta_catalogo(texto)
            fallback = buscar_servicios_fuzzy(consulta_catalogo)
            return {
                "status": "fallback",
                "texto": texto,
                "total": fallback["total"],
                "total_real": fallback["total_real"],
                "total_conocido": fallback["total_conocido"],
                "resultados": fallback["resultados"],
            }

        return {
            "status": "ok",
            "texto": texto,
            "total": busqueda["total"],
            "total_real": busqueda["total_real"],
            "total_conocido": busqueda["total_conocido"],
            "resultados": busqueda["resultados"],
        }
    except Exception as exc:
        try:
            consulta_catalogo = preparar_consulta_catalogo(texto)
            fallback = buscar_servicios_fuzzy(consulta_catalogo)
            return {
                "status": "fallback",
                "texto": texto,
                "message": str(exc),
                "total": fallback["total"],
                "total_real": fallback["total_real"],
                "total_conocido": fallback["total_conocido"],
                "resultados": fallback["resultados"],
            }
        except Exception as fallback_exc:
            return {
                "status": "error",
                "message": str(exc),
                "fallback_message": str(fallback_exc),
            }


@app.post("/predict")
def predict(request: PredictRequest, _auth: bool = Depends(validar_api_key)):
    if vectorizer is None or classifier is None:
        return {
            "texto": request.texto.strip(),
            "intencion": "desconocido",
            "error": "modelo_no_cargado",
        }

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


def predecir_intencion(texto):
    texto_original = texto.strip()
    texto = aplicar_correcciones_texto(texto_original)

    if not texto:
        return {
            "texto": texto_original,
            "intencion": "desconocido",
            "confianza": None,
        }

    intencion_regla = detectar_intencion_defensiva(texto)
    if intencion_regla:
        return {
            "texto": texto_original,
            "intencion": intencion_regla,
            "confianza": 1.0,
        }

    intencion_saludo = detectar_saludo_puro(texto)
    if intencion_saludo:
        return {
            "texto": texto_original,
            "intencion": intencion_saludo,
            "confianza": 1.0,
        }

    if vectorizer is None or classifier is None:
        return {
            "texto": texto_original,
            "intencion": "desconocido",
            "confianza": None,
            "error": "modelo_no_cargado",
        }

    X = vectorizer.transform([texto])
    intencion = classifier.predict(X)[0]
    confianza = None

    try:
        decision = classifier.decision_function(X)
        puntajes = decision[0] if hasattr(decision[0], "__iter__") else decision
        puntaje = max(puntajes)
        confianza = round(1 / (1 + pow(2.718281828, -float(puntaje))), 4)
    except Exception:
        confianza = None

    return {
        "texto": texto_original,
        "intencion": intencion,
        "confianza": confianza,
    }


def obtener_catalogo():
    try:
        response = requests.get(CATALOG_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(f"FamyBot IA: error obteniendo catalogo: {exc}")
        raise HTTPException(
            status_code=502,
            detail="No se pudo obtener el catalogo de servicios",
        ) from exc


def normalizar_texto(texto):
    texto = str(texto or "").strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(caracter for caracter in texto if unicodedata.category(caracter) != "Mn")
    return aplicar_correcciones_texto(texto)


def aplicar_correcciones_texto(texto):
    texto = str(texto or "").strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(caracter for caracter in texto if unicodedata.category(caracter) != "Mn")

    for patron, reemplazo in CORRECCIONES_TEXTO:
        texto = re.sub(patron, reemplazo, texto)

    return re.sub(r"\s+", " ", texto).strip()


def obtener_palabras_normalizadas(texto):
    return re.findall(r"[a-z0-9]+", normalizar_texto(texto))


def obtener_palabras_clave_intencion(texto):
    return [
        palabra
        for palabra in obtener_palabras_normalizadas(texto)
        if palabra not in PALABRAS_RELLENO_INTENCION
    ]


def detectar_saludo_puro(texto):
    palabras = obtener_palabras_normalizadas(texto)

    if not palabras:
        return None

    if texto_tiene_contexto_no_saludo(texto):
        return None

    if all(palabra in PALABRAS_SALUDO_PURO for palabra in palabras):
        return "saludo"

    return None


def texto_tiene_contexto_no_saludo(texto):
    texto_normalizado = normalizar_texto(texto)
    palabras = set(obtener_palabras_normalizadas(texto))

    return (
        contiene_indicador_trabajo(texto_normalizado, palabras)
        or contiene_indicador_cotizacion(texto_normalizado, palabras)
        or contiene_indicador_servicio(texto_normalizado, palabras)
        or contiene_indicador_agendamiento(palabras)
    )


def contiene_indicador_trabajo(texto_normalizado, palabras):
    if any(frase in texto_normalizado for frase in FRASES_TRABAJO):
        return True

    if PALABRAS_TRABAJO & palabras:
        return True

    if "correo" in palabras and {"curriculo", "cv", "trabajo", "laboral"} & palabras:
        return True

    return False


def contiene_indicador_cotizacion(texto_normalizado, palabras):
    return (
        any(frase in texto_normalizado for frase in FRASES_COTIZACION)
        or bool(PALABRAS_COTIZACION & palabras)
    )


def contiene_indicador_servicio(texto_normalizado, palabras):
    if PALABRAS_CONSULTA_SERVICIOS & palabras:
        return True

    if ESPECIALIDADES_CONOCIDAS & palabras:
        return True

    return any(
        termino in texto_normalizado
        for termino in (
            "ecografia",
            "electrocardiograma",
            "electromiografia",
            "mamografia",
            "protesis",
            "radiografia",
            "resonancia",
            "tomografia",
            "tercer molar",
            "cordal",
            "cordales",
        )
    )


def contiene_indicador_agendamiento(palabras):
    return bool({"agendar", "cita", "reservar", "turno"} & palabras)


def es_atencion_a_domicilio(texto):
    return "atencion a domicilio" in normalizar_texto(texto)


def es_solicitud_asesor(texto):
    texto_normalizado = normalizar_texto(texto)
    palabras = set(obtener_palabras_normalizadas(texto))

    if es_atencion_a_domicilio(texto_normalizado):
        return True

    frases_asesor = (
        "hablar con asesor",
        "hablar con un asesor",
        "hablar con alguien",
        "atencion al cliente",
        "ayuda de una persona",
        "atender un asesor",
        "atienda un asesor",
        "comunicar con asesor",
        "comunicarme con asesor",
        "comunicarme con un asesor",
    )
    if any(frase in texto_normalizado for frase in frases_asesor):
        return True

    if "asesor" in palabras and bool(
        palabras
        & {
            "asesor",
            "atender",
            "atienda",
            "atencion",
            "ayuda",
            "hablar",
            "necesito",
            "quiero",
        }
    ):
        return True

    return False


def detectar_intencion_defensiva(texto):
    texto_normalizado = normalizar_texto(texto)
    palabras = set(obtener_palabras_normalizadas(texto))

    if contiene_indicador_trabajo(texto_normalizado, palabras):
        return "trabajo"

    if "atencion a domicilio" in texto_normalizado:
        return "hablar_asesor"

    if es_solicitud_asesor(texto_normalizado):
        return "hablar_asesor"

    if contiene_indicador_cotizacion(texto_normalizado, palabras):
        return "cotizar_servicio"

    if contiene_indicador_agendamiento(palabras) or "sacar cita" in texto_normalizado:
        return "agendar_cita"

    if ESPECIALIDADES_CONOCIDAS & palabras:
        return "consulta_especialidades"

    if contiene_indicador_servicio(texto_normalizado, palabras):
        return "consulta_servicios"

    return None


def palabra_coincide_fuzzy(palabra, referencia):
    if palabra == referencia:
        return True

    if len(palabra) < 5 or len(referencia) < 5:
        return False

    if abs(len(palabra) - len(referencia)) > 2:
        return False

    return SequenceMatcher(None, palabra, referencia).ratio() >= MIN_RATIO_FUZZY_INTENCION


def detectar_intencion_frecuente_fuzzy(texto):
    palabras = obtener_palabras_clave_intencion(texto)

    if len(palabras) != 1:
        return None

    palabra = palabras[0]

    for intencion, referencias in PALABRAS_FUZZY_INTENCION.items():
        if any(palabra_coincide_fuzzy(palabra, referencia) for referencia in referencias):
            return intencion

    return None


def es_respuesta_numerica_sin_estado(texto):
    return bool(re.fullmatch(r"\s*\d+\s*", str(texto or "")))


def preparar_consulta_sin_relleno(texto):
    palabras = obtener_palabras_clave_intencion(texto)
    return " ".join(palabras) or texto


def aplicar_sinonimos_catalogo(texto):
    texto_normalizado = normalizar_texto(texto)

    for frase, termino_catalogo in SINONIMOS_CATALOGO.items():
        frase_normalizada = normalizar_texto(frase)
        if len(frase_normalizada) <= 3:
            if re.search(rf"\b{re.escape(frase_normalizada)}\b", texto_normalizado):
                return termino_catalogo
            continue
        if frase_normalizada in texto_normalizado:
            return termino_catalogo

    return texto


def aplico_sinonimo_catalogo(texto):
    return normalizar_texto(aplicar_sinonimos_catalogo(texto)) != normalizar_texto(texto)


EXPANSIONES_LEXICALES_CATALOGO = {
    "cerebral": ["craneo", "cerebro", "encefalo"],
    "cerebro": ["craneo", "cerebral", "encefalo"],
    "craneal": ["craneo", "cerebral"],
    "magnetica": ["resonancia"],
    "rm": ["resonancia"],
    "pierna": ["piernas", "miembros", "inferiores", "venoso"],
    "piernas": ["pierna", "miembros", "inferiores", "venoso"],
    "doppler": ["vascular", "venoso", "arterial"],
    "eco": ["ecografia", "ultrasonido"],
    "ultrasonido": ["ecografia", "eco"],
}


def ratio_texto_fuzzy(texto_a, texto_b):
    if rapidfuzz_fuzz is not None:
        return float(rapidfuzz_fuzz.partial_ratio(texto_a, texto_b)) / 100

    return SequenceMatcher(None, texto_a, texto_b).ratio()


def expandir_consulta_lexical(texto):
    textos = [normalizar_texto(texto)]
    texto_sinonimo = normalizar_texto(aplicar_sinonimos_catalogo(texto))

    if texto_sinonimo and texto_sinonimo not in textos:
        textos.append(texto_sinonimo)

    tokens = []
    for texto_base in list(textos):
        for token in obtener_tokens_utiles(texto_base):
            tokens.append(token)
            tokens.extend(EXPANSIONES_LEXICALES_CATALOGO.get(token, []))

    if tokens:
        textos.append(" ".join(tokens))

    return " ".join(texto for texto in textos if texto).strip()


def calcular_score_servicio_lexical(texto_busqueda, tokens_busqueda, servicio):
    texto_indexable = normalizar_texto(
        construir_texto_indexable(servicio, SINONIMOS_CATALOGO)
    )
    nombre = normalizar_texto(servicio.get("nombre"))
    area = normalizar_texto(servicio.get("area"))
    categoria = normalizar_texto(servicio.get("categoria"))
    slug = normalizar_texto(servicio.get("slug"))
    texto_prioritario = " ".join(
        parte for parte in (nombre, area, categoria, slug) if parte
    )
    tokens_servicio = obtener_tokens_servicio(texto_indexable)

    if not tokens_busqueda:
        return 0.0

    coincidencias = [
        token
        for token in tokens_busqueda
        if token_coincide(token, tokens_servicio, texto_indexable)
    ]
    cobertura = len(set(coincidencias)) / max(len(set(tokens_busqueda)), 1)
    score = cobertura * 0.62

    if texto_busqueda and texto_busqueda in texto_indexable:
        score += 0.25
    if texto_busqueda and texto_busqueda in texto_prioritario:
        score += 0.2

    coincidencias_prioritarias = [
        token
        for token in set(tokens_busqueda)
        if token in texto_prioritario
    ]
    score += min(len(coincidencias_prioritarias) * 0.08, 0.24)

    score += ratio_texto_fuzzy(texto_busqueda, texto_prioritario or texto_indexable) * 0.18

    return min(score, 1.0)


def resultado_servicio_catalogo(servicio):
    return {
        "id": servicio.get("id"),
        "nombre": servicio.get("nombre"),
        "area": servicio.get("area"),
        "precio": servicio.get("precio"),
        "precio_promocion": servicio.get("precio_promocion"),
        "presencial": bool(servicio.get("presencial")),
        "virtual": bool(servicio.get("virtual")),
    }


def buscar_servicios_fuzzy(texto_busqueda):
    from api.services.famysalud_api import obtener_servicios_normalizados

    catalogo = obtener_catalogo()
    texto_expandido = expandir_consulta_lexical(texto_busqueda)
    texto = normalizar_texto(texto_expandido)
    tokens_busqueda = obtener_tokens_utiles(texto_expandido)
    candidatos = []
    resultados = []
    total_real = 0
    total_conocido = True

    if not texto or not tokens_busqueda:
        return {
            "total": total_real,
            "total_real": total_real,
            "total_conocido": total_conocido,
            "resultados": resultados,
        }

    servicios = obtener_servicios_normalizados(catalogo).get("servicios", [])

    for servicio in servicios:
        score = calcular_score_servicio_lexical(texto, tokens_busqueda, servicio)

        if score < 0.45:
            continue

        resultado = resultado_servicio_catalogo(servicio)
        resultado["score"] = round(score, 4)
        candidatos.append(resultado)

    candidatos.sort(key=lambda resultado: resultado.get("score", 0), reverse=True)
    total_real = len(candidatos)
    resultados = candidatos[:20]

    return {
        "total": total_real,
        "total_real": total_real,
        "total_conocido": total_conocido,
        "resultados": resultados,
    }


def limpiar_resultado_semantico(resultado):
    return {
        "id": resultado.get("id"),
        "nombre": resultado.get("nombre"),
        "area": resultado.get("area"),
        "precio": resultado.get("precio"),
        "precio_promocion": resultado.get("precio_promocion"),
        "presencial": bool(resultado.get("presencial")),
        "virtual": bool(resultado.get("virtual")),
    }


def limpiar_busqueda_semantica(busqueda):
    resultados = [
        limpiar_resultado_semantico(resultado)
        for resultado in busqueda.get("resultados", [])
    ]

    return {
        "total": len(resultados),
        "total_real": len(resultados),
        "total_conocido": True,
        "resultados": resultados,
    }


def obtener_score_semantico_maximo(busqueda):
    scores = [
        float(resultado.get("score") or resultado.get("final_score") or 0)
        for resultado in busqueda.get("resultados", [])
    ]
    return max(scores) if scores else 0.0


def es_consulta_semantica_ambigua(texto):
    tokens = obtener_tokens_utiles(texto)
    return len(tokens) == 1 and tokens[0] in CONSULTAS_AMBIGUAS_SEMANTICAS


def fuzzy_tiene_coincidencia_exacta(texto_busqueda, busqueda_fuzzy):
    tokens_consulta = set(obtener_tokens_utiles(texto_busqueda))

    if not tokens_consulta:
        return False

    for resultado in busqueda_fuzzy.get("resultados", [])[:5]:
        tokens_resultado = obtener_tokens_servicio(
            normalizar_texto(
                " ".join(
                    str(valor)
                    for valor in (
                        resultado.get("nombre"),
                        resultado.get("area"),
                    )
                    if valor
                )
            )
        )

        if tokens_consulta.issubset(tokens_resultado):
            return True

    return False


def debe_evaluar_fuzzy_exacto(texto_busqueda):
    tokens = obtener_tokens_utiles(texto_busqueda)
    return 0 < len(tokens) <= 3


def log_busqueda_catalogo(origen, texto, total, detalle=None):
    mensaje = f"FamyBot IA: catalog_search={origen} texto={texto!r} total={total}"
    if detalle:
        mensaje = f"{mensaje} {detalle}"
    print(mensaje)


def agregar_search_mode(busqueda, search_mode):
    busqueda = dict(busqueda)
    busqueda["_search_mode"] = search_mode or "none"
    return busqueda


def obtener_search_mode_busqueda(busqueda):
    if not isinstance(busqueda, dict):
        return "none"
    return busqueda.get("_search_mode") or "none"


def combinar_search_modes(search_modes):
    modos = [modo for modo in search_modes if modo and modo != "none"]

    if not modos:
        return "none"

    if len(set(modos)) == 1:
        return modos[0]

    for modo_prioritario in (
        "semantic_fallback",
        "fuzzy_exact_override",
        "semantic",
        "lexical_fuzzy",
        "token_match",
        "fuzzy",
    ):
        if modo_prioritario in modos:
            return modo_prioritario

    return "none"


def agregar_intencion(intenciones, intencion):
    if intencion and intencion not in intenciones:
        intenciones.append(intencion)


def tiene_servicio_medico_claro(texto, entidades=None):
    entidades = entidades or extraer_entidades_consulta_catalogo(texto)
    texto_normalizado = normalizar_texto(aplicar_sinonimos_catalogo(texto))
    palabras = set(obtener_palabras_normalizadas(texto_normalizado))

    if entidades.get("has_specialty") or ESPECIALIDADES_CONOCIDAS & palabras:
        return True

    if TERMINOS_MEDICOS_CLAROS & palabras:
        return True

    if any(termino in texto_normalizado for termino in TERMINOS_MEDICOS_CLAROS):
        return True

    for servicio in entidades.get("services", []) or []:
        tokens_servicio = set(obtener_tokens_utiles(servicio))
        if tokens_servicio & (TERMINOS_MEDICOS_CLAROS | ESPECIALIDADES_CONOCIDAS):
            return True

    return False


def es_consulta_ubicacion_pura(entidades):
    return (
        entidades.get("asks_location")
        and not entidades.get("asks_price")
        and not entidades.get("asks_schedule")
        and not entidades.get("asks_booking")
    )


def es_consulta_horario_pura(entidades):
    return (
        entidades.get("asks_schedule")
        and not entidades.get("asks_price")
        and not entidades.get("asks_location")
        and not entidades.get("asks_booking")
    )


def es_agendamiento_puro(entidades):
    return (
        entidades.get("asks_booking")
        and not entidades.get("asks_price")
        and not entidades.get("asks_location")
        and not entidades.get("asks_schedule")
    )


def detectar_intenciones_multiples(texto, intencion_principal=None, entidades=None):
    entidades = entidades or extraer_entidades_consulta_catalogo(texto)
    intenciones = []
    tiene_servicio = tiene_servicio_medico_claro(texto, entidades)

    if intencion_principal == "saludo" and not any(
        entidades.get(clave)
        for clave in ("asks_price", "asks_location", "asks_schedule", "asks_booking")
    ):
        return ["saludo"]

    if intencion_principal == "trabajo":
        return ["trabajo"]

    if intencion_principal == "hablar_asesor":
        return ["hablar_asesor"]

    if tiene_servicio:
        agregar_intencion(intenciones, "consulta_servicios")
    if entidades.get("asks_price"):
        agregar_intencion(intenciones, "cotizar_servicio")
    if entidades.get("asks_location"):
        agregar_intencion(intenciones, "consultar_ubicacion")
    if entidades.get("asks_schedule"):
        agregar_intencion(intenciones, "consultar_horario")
    if entidades.get("asks_booking"):
        agregar_intencion(intenciones, "agendar_cita")
    if entidades.get("has_specialty"):
        agregar_intencion(intenciones, "consulta_especialidades")

    if intencion_principal == "consulta_servicios":
        if tiene_servicio:
            agregar_intencion(intenciones, intencion_principal)
    elif intencion_principal in INTENCIONES_CATALOGO:
        agregar_intencion(intenciones, intencion_principal)
    elif intencion_principal in RESPUESTAS_SIMPLES or intencion_principal in ACCIONES_FLUJO:
        agregar_intencion(intenciones, intencion_principal)

    return intenciones or [intencion_principal or "desconocido"]


def determinar_intencion_principal_catalogo(intencion_predicha, entidades):
    if entidades.get("services"):
        return "consulta_servicios"
    if entidades.get("asks_price"):
        return "cotizar_servicio"
    if intencion_predicha in INTENCIONES_CATALOGO:
        return intencion_predicha
    return "consulta_servicios"


def construir_respuesta_flags_simples(texto, intencion, confianza, entidades):
    bloques = []
    intencion_flags = intencion
    texto_normalizado = normalizar_texto(texto)
    pregunta_donde_agendar = bool(
        entidades.get("asks_location")
        and entidades.get("asks_booking")
        and "donde" in texto_normalizado
        and (
            "agendar" in texto_normalizado
            or "ajendar" in texto_normalizado
            or "reservar" in texto_normalizado
        )
    )

    if entidades.get("asks_location"):
        bloques.append(RESPUESTAS_SIMPLES["consultar_ubicacion"]["mensaje"])
        intencion_flags = "agendar_cita" if pregunta_donde_agendar else "consultar_ubicacion"
    if entidades.get("asks_schedule"):
        bloques.append(RESPUESTAS_SIMPLES["consultar_horario"]["mensaje"])
        if not entidades.get("asks_location"):
            intencion_flags = "consultar_horario"
    if entidades.get("asks_booking"):
        bloques.append(ACCIONES_FLUJO["agendar_cita"]["mensaje"])
        if not entidades.get("asks_location") and not entidades.get("asks_schedule"):
            intencion_flags = "agendar_cita"

    if not bloques:
        return None

    accion = "respuesta_simple"
    if entidades.get("asks_booking") and len(bloques) == 1:
        accion = ACCIONES_FLUJO["agendar_cita"]["accion"]

    return construir_respuesta_chat({
        "texto": texto,
        "intencion": intencion_flags,
        "confianza": confianza,
        "accion": accion,
        "mensaje": "\n\n".join(bloques),
    }, entidades=entidades)


def construir_respuesta_cotizacion_generica(texto, confianza, entidades):
    bloques = [
        (
            "Para indicarte un valor exacto, escribe el nombre de la especialidad, "
            "examen o servicio que necesitas."
        )
    ]

    if entidades.get("asks_location"):
        bloques.append(RESPUESTAS_SIMPLES["consultar_ubicacion"]["mensaje"])

    return construir_respuesta_chat({
        "texto": texto,
        "intencion": "cotizar_servicio",
        "confianza": confianza,
        "accion": "consulta_ubicacion" if entidades.get("asks_location") else "listar_opciones",
        "mensaje": "\n\n".join(bloques),
        "total": 0,
        "total_real": 0,
        "total_conocido": True,
        "resultados": [],
    }, entidades=entidades)


def construir_respuesta_chat(respuesta, search_mode=None, entidades=None):
    respuesta = dict(respuesta)
    entidades = entidades or extraer_entidades_consulta_catalogo(respuesta.get("texto", ""))
    respuesta["model_version"] = model_version
    respuesta["search_mode"] = search_mode or respuesta.get("search_mode") or "none"
    respuesta["intenciones_detectadas"] = respuesta.get(
        "intenciones_detectadas"
    ) or detectar_intenciones_multiples(
        respuesta.get("texto", ""),
        respuesta.get("intencion"),
        entidades,
    )
    if entidades.get("asks_location") or respuesta.get("intencion") == "consultar_ubicacion":
        respuesta["incluir_botones_ubicacion"] = True
        mensaje = str(respuesta.get("mensaje") or "")
        if MENSAJE_UBICACION_CON_BOTONES in mensaje:
            respuesta["mensaje"] = mensaje.replace(
                MENSAJE_UBICACION_CON_BOTONES,
                MENSAJE_UBICACION_BOTONES,
            ).strip()
    return respuesta


def buscar_servicios(texto_busqueda):
    if es_consulta_semantica_ambigua(texto_busqueda):
        busqueda_fuzzy = buscar_servicios_fuzzy(texto_busqueda)
        log_busqueda_catalogo(
            "lexical_fuzzy",
            texto_busqueda,
            busqueda_fuzzy["total"],
            "ambigua",
        )
        return agregar_search_mode(busqueda_fuzzy, "lexical_fuzzy")

    if not busqueda_semantica_disponible():
        busqueda_fuzzy = buscar_servicios_fuzzy(texto_busqueda)
        log_busqueda_catalogo(
            "lexical_fuzzy",
            texto_busqueda,
            busqueda_fuzzy["total"],
            "semantic_disabled",
        )
        return agregar_search_mode(busqueda_fuzzy, "lexical_fuzzy")

    try:
        busqueda_semantica = buscar_servicios_semanticos(
            texto_busqueda,
            top_k=20,
            sinonimos_catalogo=SINONIMOS_CATALOGO,
        )
        score_semantico = obtener_score_semantico_maximo(busqueda_semantica)

        if busqueda_semantica["total"] == 0 or score_semantico < MIN_SCORE_SEMANTICO_CATALOGO:
            busqueda_fuzzy = buscar_servicios_fuzzy(texto_busqueda)
            log_busqueda_catalogo(
                "semantic_with_fuzzy_fallback",
                texto_busqueda,
                busqueda_fuzzy["total"],
                f"score={score_semantico:.4f}",
            )
            return agregar_search_mode(busqueda_fuzzy, "semantic_fallback")

        if (
            score_semantico < MIN_SCORE_SEMANTICO_CON_EXACTO_FUZZY
            or debe_evaluar_fuzzy_exacto(texto_busqueda)
        ):
            busqueda_fuzzy = buscar_servicios_fuzzy(texto_busqueda)
            if (
                busqueda_fuzzy["total"] > 0
                and fuzzy_tiene_coincidencia_exacta(texto_busqueda, busqueda_fuzzy)
                and busqueda_fuzzy["total"] <= busqueda_semantica["total"]
                and (
                    score_semantico < MIN_SCORE_SEMANTICO_CON_EXACTO_FUZZY
                    or busqueda_fuzzy["total"] <= 5
                )
            ):
                log_busqueda_catalogo(
                    "fuzzy_exact_override",
                    texto_busqueda,
                    busqueda_fuzzy["total"],
                    f"score={score_semantico:.4f}",
                )
                return agregar_search_mode(busqueda_fuzzy, "fuzzy_exact_override")

        busqueda_limpia = limpiar_busqueda_semantica(busqueda_semantica)
        log_busqueda_catalogo(
            "semantic",
            texto_busqueda,
            busqueda_limpia["total"],
            f"score={score_semantico:.4f}",
        )
        return agregar_search_mode(busqueda_limpia, "semantic")
    except Exception as exc:
        busqueda_fuzzy = buscar_servicios_fuzzy(texto_busqueda)
        log_busqueda_catalogo(
            "semantic_with_fuzzy_fallback",
            texto_busqueda,
            busqueda_fuzzy["total"],
            f"error={exc}",
        )
        return agregar_search_mode(busqueda_fuzzy, "semantic_fallback")


def es_palabra_ignorada(palabra):
    if palabra in PALABRAS_IGNORADAS_CATALOGO:
        return True

    if len(palabra) <= 4:
        return False

    return any(
        SequenceMatcher(None, palabra, ignorada).ratio() >= 0.86
        for ignorada in PALABRAS_IGNORADAS_CATALOGO
        if len(ignorada) > 4
    )


def normalizar_token_catalogo(palabra):
    if palabra.endswith("s") and len(palabra) > 4:
        return palabra[:-1]
    return palabra


def obtener_tokens_utiles(texto):
    tokens = []

    for palabra in re.findall(r"[a-z0-9]+", normalizar_texto(texto)):
        if es_palabra_ignorada(palabra):
            continue
        tokens.append(normalizar_token_catalogo(palabra))

    return tokens


def obtener_tokens_servicio(texto):
    return {
        normalizar_token_catalogo(palabra)
        for palabra in re.findall(r"[a-z0-9]+", texto)
        if not es_palabra_ignorada(palabra)
    }


def token_coincide(token_busqueda, tokens_servicio, texto_servicio):
    if token_busqueda in texto_servicio or token_busqueda in tokens_servicio:
        return True

    if len(token_busqueda) <= 4:
        return False

    return any(
        abs(len(token_busqueda) - len(token_servicio)) <= 2
        and SequenceMatcher(None, token_busqueda, token_servicio).ratio() >= 0.84
        for token_servicio in tokens_servicio
    )


def servicio_coincide(texto_busqueda, tokens_busqueda, texto_servicio, tokens_servicio):
    if texto_busqueda in texto_servicio:
        return True

    return all(
        token_coincide(token, tokens_servicio, texto_servicio)
        for token in tokens_busqueda
    )


def preparar_consulta_catalogo(texto):
    texto_catalogo = aplicar_sinonimos_catalogo(texto)
    palabras = obtener_tokens_utiles(texto_catalogo)
    return " ".join(palabras) or texto_catalogo


def es_consulta_catalogo_comercial(texto):
    texto_normalizado = normalizar_texto(texto)
    palabras = obtener_palabras_normalizadas(texto)

    return (
        "cuanto cuesta" in texto_normalizado
        or any(palabra in PALABRAS_COMERCIALES_CATALOGO for palabra in palabras)
    )


def es_consulta_precio_y_ubicacion(texto):
    texto_normalizado = normalizar_texto(texto)
    palabras = set(obtener_palabras_normalizadas(texto))

    pide_consulta = "consulta" in palabras
    pide_precio = (
        "cuanto cuesta" in texto_normalizado
        or any(palabra in PALABRAS_COMERCIALES_CATALOGO for palabra in palabras)
    )
    pide_ubicacion = bool(
        {
            "direccion",
            "donde",
            "ubicacion",
            "ubicaciones",
            "ubicado",
            "ubicados",
        }
        & palabras
    )

    return pide_consulta and pide_precio and pide_ubicacion


def aplicar_normalizaciones_comerciales_controladas(texto):
    texto_normalizado = normalizar_texto(aplicar_sinonimos_catalogo(texto))

    for patron, reemplazo in NORMALIZACIONES_COMERCIALES_CONTROLADAS:
        texto_normalizado = re.sub(patron, reemplazo, texto_normalizado)

    return re.sub(r"\s+", " ", texto_normalizado).strip()


def obtener_tokens_comerciales_limpios(texto):
    texto_catalogo = aplicar_normalizaciones_comerciales_controladas(texto)
    tokens = []

    for palabra in obtener_palabras_normalizadas(texto_catalogo):
        if palabra in PALABRAS_IGNORADAS_COMERCIAL:
            continue
        if es_palabra_ignorada(palabra):
            continue
        tokens.append(normalizar_token_catalogo(palabra))

    tokens_fuertes = [
        token
        for token in tokens
        if token not in MODIFICADORES_COMERCIALES_NO_SERVICIO
    ]

    return tokens_fuertes or tokens


def obtener_tokens_fuertes_comerciales(texto):
    return [
        token
        for token in obtener_tokens_comerciales_limpios(texto)
        if token not in MODIFICADORES_COMERCIALES_NO_SERVICIO
    ]


def preparar_consulta_comercial_catalogo(texto):
    texto_catalogo = aplicar_normalizaciones_comerciales_controladas(texto)
    tokens = obtener_tokens_comerciales_limpios(texto_catalogo)
    return " ".join(tokens) or texto_catalogo


def limpiar_segmento_entidad_catalogo(segmento):
    tokens = []

    for palabra in obtener_palabras_normalizadas(segmento):
        if palabra in PALABRAS_IGNORADAS_COMERCIAL:
            continue
        if es_palabra_ignorada(palabra):
            continue
        tokens.append(normalizar_token_catalogo(palabra))

    tokens_fuertes = [
        token
        for token in tokens
        if token not in MODIFICADORES_ENTIDAD_CATALOGO
    ]

    if tokens and not tokens_fuertes:
        return ""

    return " ".join(tokens_fuertes).strip()


def extraer_entidades_consulta_catalogo(texto):
    texto_original_normalizado = normalizar_texto(texto)
    palabras_originales = set(obtener_palabras_normalizadas(texto))
    texto_normalizado = aplicar_normalizaciones_comerciales_controladas(texto)
    palabras = set(obtener_palabras_normalizadas(texto_normalizado))
    pide_precio = (
        any(frase in texto_original_normalizado for frase in FRASES_COTIZACION)
        or any(frase in texto_normalizado for frase in FRASES_COTIZACION)
        or bool(PALABRAS_COTIZACION & palabras_originales)
        or bool(PALABRAS_COTIZACION & palabras)
        or "en que precio" in texto_original_normalizado
        or "q vale" in texto_original_normalizado
        or "que vale" in texto_original_normalizado
    )
    pide_horario = bool(
        PALABRAS_HORARIO_CONSULTA & (palabras_originales | palabras)
    )
    pide_ubicacion = bool(
        PALABRAS_UBICACION_CONSULTA & (palabras_originales | palabras)
    )
    pide_agendamiento = (
        bool(PALABRAS_AGENDAMIENTO_CONSULTA & (palabras_originales | palabras))
        or "sacar cita" in texto_original_normalizado
        or "sacar cita" in texto_normalizado
    )
    tiene_especialidad = bool(ESPECIALIDADES_CONOCIDAS & (palabras_originales | palabras))

    entidades = {
        "services": [],
        "asks_price": pide_precio,
        "asks_schedule": pide_horario,
        "asks_location": pide_ubicacion,
        "asks_booking": pide_agendamiento,
        "has_specialty": tiene_especialidad,
    }
    servicios_vistos = set()
    tokens = re.findall(r"[a-z0-9]+|[,.]", texto_normalizado)
    segmento = []

    for token in tokens:
        if token in {",", "."} or token in CONECTORES_SERVICIOS:
            if segmento:
                servicio = limpiar_segmento_entidad_catalogo(" ".join(segmento))
                if servicio and servicio not in servicios_vistos:
                    servicios_vistos.add(servicio)
                    entidades["services"].append(servicio)
                segmento = []
            continue
        segmento.append(token)

    if segmento:
        servicio = limpiar_segmento_entidad_catalogo(" ".join(segmento))
        if servicio and servicio not in servicios_vistos:
            servicios_vistos.add(servicio)
            entidades["services"].append(servicio)

    return entidades


def consulta_catalogo_mixta_extraida(entidades):
    if len(entidades["services"]) > 1:
        return True

    tiene_flags_mixtos = any(
        entidades[clave]
        for clave in (
            "asks_schedule",
            "asks_location",
            "asks_booking",
        )
    )

    return bool(entidades["services"] and tiene_flags_mixtos)


def tiene_conectores_servicios(texto):
    if "," in texto or "." in texto:
        return True

    return any(
        palabra in CONECTORES_SERVICIOS
        for palabra in obtener_palabras_normalizadas(texto)
    )


def termino_comercial_util(termino):
    return any(len(token) > 2 for token in obtener_tokens_fuertes_comerciales(termino))


def iterar_segmentos_comerciales(texto):
    texto_normalizado = aplicar_normalizaciones_comerciales_controladas(texto)
    tokens = re.findall(r"[a-z0-9]+|[,.]", texto_normalizado)
    actual = []

    for token in tokens:
        if token in {",", "."} or token in CONECTORES_SERVICIOS:
            if actual:
                yield " ".join(actual)
                actual = []
            continue
        actual.append(token)

    if actual:
        yield " ".join(actual)


def obtener_terminos_comerciales_individuales(texto):
    terminos = []

    for segmento in iterar_segmentos_comerciales(texto):
        termino = preparar_consulta_comercial_catalogo(segmento)
        if termino_comercial_util(termino):
            terminos.append(termino)

    return terminos


def buscar_catalogo_por_entidades(texto, entidades):
    consulta_catalogo = preparar_consulta_comercial_catalogo(texto)
    busquedas = []

    for servicio in entidades["services"]:
        busqueda_servicio = filtrar_busqueda_por_tokens_exactos(
            buscar_servicios(servicio),
            servicio,
        )
        if busqueda_servicio["total"] > 0:
            busquedas.append(busqueda_servicio)

    if not busquedas:
        busqueda = filtrar_busqueda_por_tokens_exactos(
            buscar_servicios(consulta_catalogo),
            consulta_catalogo,
        )
        return construir_respuesta_busqueda_catalogo(consulta_catalogo, busqueda)

    print(
        "FamyBot IA: query_entity_extractor "
        f"services={entidades['services']} "
        f"price={entidades['asks_price']} "
        f"schedule={entidades['asks_schedule']} "
        f"location={entidades['asks_location']} "
        f"booking={entidades['asks_booking']}"
    )

    return construir_respuesta_busqueda_catalogo(
        consulta_catalogo,
        combinar_busquedas_catalogo(busquedas),
    )


def combinar_busquedas_catalogo(busquedas):
    resultados = []
    vistos = set()
    total_real = 0
    total_conocido = True
    search_modes = []

    for busqueda in busquedas:
        total_real += busqueda["total_real"]
        total_conocido = total_conocido and busqueda["total_conocido"]
        search_mode = obtener_search_mode_busqueda(busqueda)
        if search_mode and search_mode != "none":
            search_modes.append(search_mode)

        for resultado in busqueda["resultados"]:
            clave = resultado.get("id") or (
                resultado.get("nombre"),
                resultado.get("area"),
            )

            if clave in vistos:
                continue

            vistos.add(clave)
            resultados.append(resultado)

            if len(resultados) >= 20:
                break

    return {
        "total": len(vistos),
        "total_real": total_real,
        "total_conocido": total_conocido,
        "resultados": resultados,
        "_search_mode": combinar_search_modes(search_modes),
    }


def filtrar_busqueda_por_tokens_exactos(busqueda, consulta):
    tokens_consulta = set(obtener_tokens_utiles(consulta))

    if not tokens_consulta or not busqueda["resultados"]:
        return busqueda

    if len(tokens_consulta) != 1:
        return busqueda

    resultados_exactos = []

    for resultado in busqueda["resultados"]:
        tokens_resultado = obtener_tokens_servicio(
            normalizar_texto(
                " ".join(
                    str(valor)
                    for valor in (
                        resultado.get("nombre"),
                        resultado.get("area"),
                    )
                    if valor
                )
            )
        )

        if tokens_consulta.issubset(tokens_resultado):
            resultados_exactos.append(resultado)

    if not resultados_exactos:
        return busqueda

    return {
        "total": len(resultados_exactos),
        "total_real": len(resultados_exactos),
        "total_conocido": busqueda["total_conocido"],
        "resultados": resultados_exactos,
        "_search_mode": obtener_search_mode_busqueda(busqueda),
    }


def filtrar_eco_morfologica_embarazo(busqueda, texto):
    texto_normalizado = normalizar_texto(texto)

    if "20 semana" not in texto_normalizado or "embarazo" not in texto_normalizado:
        return busqueda

    resultados = [
        resultado
        for resultado in busqueda.get("resultados", [])
        if normalizar_texto(resultado.get("nombre")) == "ecografia morfologica"
    ]

    if not resultados:
        return busqueda

    return {
        "total": len(resultados),
        "total_real": len(resultados),
        "total_conocido": busqueda["total_conocido"],
        "resultados": resultados,
        "_search_mode": obtener_search_mode_busqueda(busqueda),
    }


def construir_mensaje_catalogo(total, total_conocido, resultados):
    if total == 0:
        return (
            "No encontré servicios relacionados con tu consulta. Puedes escribir "
            "el nombre completo del servicio o escribir una nueva consulta."
        )

    if total == 1:
        servicio = resultados[0]
        return (
            f"El servicio {servicio.get('nombre')} pertenece al área "
            f"{servicio.get('area')} y tiene un valor de ${servicio.get('precio')} "
            "en efectivo o transferencia."
        )

    if total > 10 and total_conocido:
        return (
            f"Encontré {total} opciones relacionadas con tu consulta. "
            "Te muestro las primeras 10. Puedes responder con el nombre completo "
            "del servicio o escribir una nueva consulta."
        )

    if total > 10:
        return (
            "Encontré varias opciones relacionadas con tu consulta. "
            "Te muestro las primeras 10. Puedes responder con el nombre completo "
            "del servicio o escribir una nueva consulta."
        )

    return (
        f"Encontré {total} opciones relacionadas con tu consulta. "
        "Puedes responder con el nombre completo del servicio o escribir "
        "una nueva consulta."
    )


def construir_bloque_precio_catalogo(total):
    if total == 1:
        return "El valor indicado corresponde a pago en efectivo o transferencia."

    return (
        "Los valores mostrados corresponden a pago en efectivo o transferencia. "
        "Puedes responder con el nombre completo del servicio o escribir una nueva consulta."
    )


def construir_bloque_agendamiento_catalogo(total, resultados):
    return "Si deseas agendar tu cita, responde \"agendar\" para comenzar el proceso."


def construir_respuesta_busqueda_catalogo(texto, busqueda):
    resultados = busqueda["resultados"]
    total = busqueda["total"]
    total_conocido = busqueda["total_conocido"]

    if total == 0:
        accion = "sin_resultados_catalogo"
    elif total == 1:
        accion = "respuesta_directa"
    else:
        accion = "listar_opciones"

    mensaje = construir_mensaje_catalogo(total, total_conocido, resultados)

    return {
        "texto": texto,
        "total": total,
        "total_real": busqueda["total_real"],
        "total_conocido": total_conocido,
        "accion": accion,
        "mensaje": mensaje,
        "resultados": resultados,
        "_search_mode": obtener_search_mode_busqueda(busqueda),
    }


def enriquecer_mensaje_catalogo_con_flags(respuesta_catalogo, entidades):
    respuesta_catalogo = dict(respuesta_catalogo)

    if respuesta_catalogo.get("total", 0) <= 0:
        return respuesta_catalogo

    mensaje = str(respuesta_catalogo.get("mensaje") or "")
    partes_adicionales = []
    total = int(respuesta_catalogo.get("total") or 0)
    resultados = respuesta_catalogo.get("resultados") or []
    extras = (
        ("asks_price", construir_bloque_precio_catalogo(total)),
        ("asks_location", RESPUESTAS_SIMPLES["consultar_ubicacion"]["mensaje"]),
        ("asks_schedule", RESPUESTAS_SIMPLES["consultar_horario"]["mensaje"]),
        ("asks_booking", construir_bloque_agendamiento_catalogo(total, resultados)),
    )
    mensaje_normalizado = normalizar_texto(mensaje)

    for flag, texto_extra in extras:
        if not entidades.get(flag):
            continue
        if normalizar_texto(texto_extra) in mensaje_normalizado:
            continue
        partes_adicionales.append(texto_extra)

    if partes_adicionales:
        respuesta_catalogo["mensaje"] = "\n\n".join([mensaje, *partes_adicionales])

    return respuesta_catalogo


def buscar_catalogo_comercial(texto):
    consulta_catalogo = preparar_consulta_comercial_catalogo(texto)
    busqueda = filtrar_busqueda_por_tokens_exactos(
        buscar_servicios(consulta_catalogo),
        consulta_catalogo,
    )
    busqueda = filtrar_eco_morfologica_embarazo(busqueda, texto)

    if not tiene_conectores_servicios(texto):
        return construir_respuesta_busqueda_catalogo(consulta_catalogo, busqueda)

    busquedas = []

    for termino in obtener_terminos_comerciales_individuales(texto):
        busqueda_termino = filtrar_busqueda_por_tokens_exactos(
            buscar_servicios(termino),
            termino,
        )
        if busqueda_termino["total"] > 0:
            busquedas.append(busqueda_termino)

    if not busquedas:
        return construir_respuesta_busqueda_catalogo(consulta_catalogo, busqueda)

    return construir_respuesta_busqueda_catalogo(
        consulta_catalogo,
        combinar_busquedas_catalogo(busquedas),
    )


def construir_respuesta_catalogo(
    texto,
    intencion,
    confianza,
    respuesta_catalogo,
    entidades=None,
):
    return construir_respuesta_chat({
        "texto": texto,
        "intencion": intencion,
        "confianza": confianza,
        "accion": respuesta_catalogo["accion"],
        "mensaje": respuesta_catalogo["mensaje"],
        "total": respuesta_catalogo["total"],
        "total_real": respuesta_catalogo["total_real"],
        "total_conocido": respuesta_catalogo["total_conocido"],
        "resultados": respuesta_catalogo["resultados"],
    }, respuesta_catalogo.get("_search_mode"), entidades)


@app.get("/catalog")
def get_catalog(_auth: bool = Depends(validar_api_key)):
    return obtener_catalogo()


@app.post("/search-service")
def search_service(request: SearchRequest, _auth: bool = Depends(validar_api_key)):
    texto = request.texto.strip()
    consulta_catalogo = preparar_consulta_catalogo(texto)
    busqueda = buscar_servicios(consulta_catalogo)

    return {
        "texto": texto,
        "total": busqueda["total"],
        "total_real": busqueda["total_real"],
        "total_conocido": busqueda["total_conocido"],
        "resultados": busqueda["resultados"],
    }


@app.post("/ask-catalog")
def ask_catalog(request: SearchRequest, _auth: bool = Depends(validar_api_key)):
    texto = request.texto.strip()
    consulta_catalogo = preparar_consulta_catalogo(texto)
    busqueda = buscar_servicios(consulta_catalogo)
    resultados = busqueda["resultados"]
    total = busqueda["total"]
    total_conocido = busqueda["total_conocido"]

    if total == 0:
        accion = "sin_resultados_catalogo"
    elif total == 1:
        accion = "respuesta_directa"
    else:
        accion = "listar_opciones"

    mensaje = construir_mensaje_catalogo(total, total_conocido, resultados)

    return {
        "texto": texto,
        "total": total,
        "total_real": busqueda["total_real"],
        "total_conocido": total_conocido,
        "accion": accion,
        "mensaje": mensaje,
        "resultados": resultados,
    }


@app.post("/chat")
def chat(request: SearchRequest, _auth: bool = Depends(validar_api_key)):
    texto = request.texto.strip()

    if es_respuesta_numerica_sin_estado(texto):
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": "desconocido",
            "intenciones_detectadas": ["desconocido"],
            "confianza": None,
            "accion": "seleccion_sin_contexto",
            "mensaje": (
                "No pude identificar el servicio seleccionado. Puedes escribir "
                "el nombre completo del servicio o responder \"agendar\" para iniciar "
                "el proceso de agendamiento."
            ),
        })

    prediccion = predecir_intencion(texto)
    intencion = str(prediccion["intencion"])
    confianza = prediccion["confianza"]
    print(f"FamyBot IA: intencion_original={intencion} confianza={confianza}")

    if prediccion.get("error") == "modelo_no_cargado":
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": intencion,
            "confianza": confianza,
            "accion": "fallback",
            "mensaje": "En este momento estoy iniciando mis servicios. Puedes intentar de nuevo en unos segundos o solicitar ayuda con un asesor.",
            "error": "modelo_no_cargado",
        })

    entidades_catalogo = extraer_entidades_consulta_catalogo(texto)
    usar_entidades_catalogo = consulta_catalogo_mixta_extraida(entidades_catalogo)
    tiene_servicio_medico = tiene_servicio_medico_claro(texto, entidades_catalogo)

    if es_solicitud_asesor(texto):
        accion_flujo = ACCIONES_FLUJO["hablar_asesor"]
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": "hablar_asesor",
            "confianza": confianza,
            "accion": accion_flujo["accion"],
            "mensaje": accion_flujo["mensaje"],
        }, entidades=entidades_catalogo)

    if intencion == "trabajo":
        accion_flujo = ACCIONES_FLUJO["trabajo"]
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": "trabajo",
            "confianza": confianza,
            "accion": accion_flujo["accion"],
            "mensaje": accion_flujo["mensaje"],
        }, entidades=entidades_catalogo)

    if es_consulta_ubicacion_pura(entidades_catalogo) and not tiene_servicio_medico:
        respuesta_simple = RESPUESTAS_SIMPLES["consultar_ubicacion"]
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": "consultar_ubicacion",
            "confianza": confianza,
            "accion": respuesta_simple["accion"],
            "mensaje": respuesta_simple["mensaje"],
        }, entidades=entidades_catalogo)

    if es_consulta_horario_pura(entidades_catalogo) and not tiene_servicio_medico:
        respuesta_simple = RESPUESTAS_SIMPLES["consultar_horario"]
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": "consultar_horario",
            "confianza": confianza,
            "accion": respuesta_simple["accion"],
            "mensaje": respuesta_simple["mensaje"],
        }, entidades=entidades_catalogo)

    if es_agendamiento_puro(entidades_catalogo) and not tiene_servicio_medico:
        accion_flujo = ACCIONES_FLUJO["agendar_cita"]
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": "agendar_cita",
            "confianza": confianza,
            "accion": accion_flujo["accion"],
            "mensaje": accion_flujo["mensaje"],
        }, entidades=entidades_catalogo)

    if (
        entidades_catalogo.get("asks_price")
        and not tiene_servicio_medico
        and not entidades_catalogo.get("services")
    ):
        return construir_respuesta_cotizacion_generica(
            texto,
            confianza,
            entidades_catalogo,
        )

    if usar_entidades_catalogo:
        respuesta_catalogo = buscar_catalogo_por_entidades(texto, entidades_catalogo)
        print(f"FamyBot IA: total_catalogo_entidades={respuesta_catalogo['total']}")

        if respuesta_catalogo["total"] > 0:
            respuesta_catalogo = enriquecer_mensaje_catalogo_con_flags(
                respuesta_catalogo,
                entidades_catalogo,
            )
            return construir_respuesta_catalogo(
                texto,
                "consulta_servicios",
                confianza,
                respuesta_catalogo,
                entidades_catalogo,
            )

        return construir_respuesta_catalogo(
            texto,
            "consulta_servicios",
            confianza,
            respuesta_catalogo,
            entidades_catalogo,
        )

    flags_simples = [
        entidades_catalogo.get("asks_location"),
        entidades_catalogo.get("asks_schedule"),
        entidades_catalogo.get("asks_booking"),
    ]
    if not entidades_catalogo.get("services") and sum(bool(flag) for flag in flags_simples) > 1:
        respuesta_flags = construir_respuesta_flags_simples(
            texto,
            intencion,
            confianza,
            entidades_catalogo,
        )
        if respuesta_flags is not None:
            return respuesta_flags

    if es_consulta_precio_y_ubicacion(texto):
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": "cotizar_servicio",
            "confianza": confianza,
            "accion": "consulta_ubicacion",
            "mensaje": (
                "Para indicarte el valor exacto de la consulta, por favor dime "
                "la especialidad o área que necesitas. Como referencia, la "
                "consulta de Medicina General tiene un valor de $15 en efectivo "
                "o transferencia. Nos encontramos en Quisquís 1109 y José Mascote, Guayaquil."
            ),
            "total": 0,
            "total_real": 0,
            "total_conocido": True,
            "resultados": [],
        }, entidades=entidades_catalogo)

    if es_consulta_catalogo_comercial(texto):
        if usar_entidades_catalogo:
            respuesta_catalogo = buscar_catalogo_por_entidades(texto, entidades_catalogo)
        else:
            respuesta_catalogo = buscar_catalogo_comercial(texto)
        respuesta_catalogo = enriquecer_mensaje_catalogo_con_flags(
            respuesta_catalogo,
            entidades_catalogo,
        )
        print(f"FamyBot IA: total_catalogo_comercial={respuesta_catalogo['total']}")
        intencion_catalogo = determinar_intencion_principal_catalogo(
            intencion,
            entidades_catalogo,
        )
        return construir_respuesta_catalogo(
            texto,
            intencion_catalogo,
            confianza,
            respuesta_catalogo,
            entidades_catalogo,
        )

    if aplico_sinonimo_catalogo(texto):
        consulta_catalogo = preparar_consulta_catalogo(texto)
        busqueda = filtrar_busqueda_por_tokens_exactos(
            buscar_servicios(consulta_catalogo),
            consulta_catalogo,
        )
        busqueda = filtrar_eco_morfologica_embarazo(busqueda, texto)
        respuesta_catalogo = construir_respuesta_busqueda_catalogo(
            consulta_catalogo,
            busqueda,
        )
        print(f"FamyBot IA: total_catalogo_sinonimo={respuesta_catalogo['total']}")

        if respuesta_catalogo["total"] > 0:
            intencion_catalogo = determinar_intencion_principal_catalogo(
                intencion,
                entidades_catalogo,
            )
            respuesta_catalogo = enriquecer_mensaje_catalogo_con_flags(
                respuesta_catalogo,
                entidades_catalogo,
            )
            return construir_respuesta_catalogo(
                texto,
                intencion_catalogo,
                confianza,
                respuesta_catalogo,
                entidades_catalogo,
            )

    if intencion in INTENCIONES_CATALOGO:
        if usar_entidades_catalogo or (
            entidades_catalogo.get("services") and tiene_servicio_medico
        ):
            respuesta_catalogo = buscar_catalogo_por_entidades(texto, entidades_catalogo)
        else:
            consulta_catalogo = preparar_consulta_catalogo(texto)
            busqueda = buscar_servicios(consulta_catalogo)
            respuesta_catalogo = construir_respuesta_busqueda_catalogo(
                consulta_catalogo,
                busqueda,
            )
        respuesta_catalogo = enriquecer_mensaje_catalogo_con_flags(
            respuesta_catalogo,
            entidades_catalogo,
        )
        print(f"FamyBot IA: total_catalogo={respuesta_catalogo['total']}")
        intencion_catalogo = determinar_intencion_principal_catalogo(
            intencion,
            entidades_catalogo,
        )
        return construir_respuesta_catalogo(
            texto,
            intencion_catalogo,
            confianza,
            respuesta_catalogo,
            entidades_catalogo,
        )

    if intencion in ACCIONES_FLUJO:
        if entidades_catalogo.get("services"):
            respuesta_catalogo = buscar_catalogo_por_entidades(texto, entidades_catalogo)
            respuesta_catalogo = enriquecer_mensaje_catalogo_con_flags(
                respuesta_catalogo,
                entidades_catalogo,
            )
            print(f"FamyBot IA: total_catalogo_pre_flujo={respuesta_catalogo['total']}")
            return construir_respuesta_catalogo(
                texto,
                "consulta_servicios",
                confianza,
                respuesta_catalogo,
                entidades_catalogo,
            )

        if confianza is None or confianza < MIN_CONF_ACCION_FLUJO:
            print(
                "FamyBot IA: accion_flujo_bloqueada_por_baja_confianza="
                f"{intencion}"
            )
            intencion_fuzzy = detectar_intencion_frecuente_fuzzy(texto)

            if intencion_fuzzy in ACCIONES_FLUJO:
                accion_flujo = ACCIONES_FLUJO[intencion_fuzzy]
                return construir_respuesta_chat({
                    "texto": texto,
                    "intencion": intencion_fuzzy,
                    "confianza": confianza,
                    "accion": accion_flujo["accion"],
                    "mensaje": accion_flujo["mensaje"],
                })

            if intencion_fuzzy in RESPUESTAS_SIMPLES:
                respuesta_simple = RESPUESTAS_SIMPLES[intencion_fuzzy]
                return construir_respuesta_chat({
                    "texto": texto,
                    "intencion": intencion_fuzzy,
                    "confianza": confianza,
                    "accion": respuesta_simple["accion"],
                    "mensaje": respuesta_simple["mensaje"],
                })

            consulta_catalogo = preparar_consulta_catalogo(texto)
            busqueda = buscar_servicios(consulta_catalogo)
            respuesta_catalogo = construir_respuesta_busqueda_catalogo(
                consulta_catalogo,
                busqueda,
            )
            print(f"FamyBot IA: total_catalogo={respuesta_catalogo['total']}")

            if respuesta_catalogo["total"] > 0:
                intencion_catalogo = determinar_intencion_principal_catalogo(
                    intencion,
                    entidades_catalogo,
                )
                return construir_respuesta_catalogo(
                    texto,
                    intencion_catalogo,
                    confianza,
                    respuesta_catalogo,
                    entidades_catalogo,
                )

            respuesta_simple = RESPUESTAS_SIMPLES["desconocido"]
            return construir_respuesta_chat({
                "texto": texto,
                "intencion": intencion,
                "confianza": confianza,
                "accion": respuesta_simple["accion"],
                "mensaje": respuesta_simple["mensaje"],
            })

        accion_flujo = ACCIONES_FLUJO[intencion]
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": intencion,
            "confianza": confianza,
            "accion": accion_flujo["accion"],
            "mensaje": accion_flujo["mensaje"],
        })

    if intencion == "saludo" and not detectar_saludo_puro(texto):
        intencion_fuzzy = detectar_intencion_frecuente_fuzzy(texto)

        if intencion_fuzzy in ACCIONES_FLUJO:
            accion_flujo = ACCIONES_FLUJO[intencion_fuzzy]
            return construir_respuesta_chat({
                "texto": texto,
                "intencion": intencion_fuzzy,
                "confianza": confianza,
                "accion": accion_flujo["accion"],
                "mensaje": accion_flujo["mensaje"],
            })

        if intencion_fuzzy in RESPUESTAS_SIMPLES:
            respuesta_simple = RESPUESTAS_SIMPLES[intencion_fuzzy]
            return construir_respuesta_chat({
                "texto": texto,
                "intencion": intencion_fuzzy,
                "confianza": confianza,
                "accion": respuesta_simple["accion"],
                "mensaje": respuesta_simple["mensaje"],
            })

        consulta_catalogo = preparar_consulta_sin_relleno(texto)
        busqueda = buscar_servicios(consulta_catalogo)
        respuesta_catalogo = construir_respuesta_busqueda_catalogo(
            consulta_catalogo,
            busqueda,
        )
        print(f"FamyBot IA: total_catalogo={respuesta_catalogo['total']}")

        if respuesta_catalogo["total"] > 0:
            return construir_respuesta_catalogo(
                texto,
                "consulta_servicios",
                confianza,
                respuesta_catalogo,
                entidades_catalogo,
            )

    if intencion in RESPUESTAS_SIMPLES:
        respuesta_simple = RESPUESTAS_SIMPLES[intencion]
        return construir_respuesta_chat({
            "texto": texto,
            "intencion": intencion,
            "confianza": confianza,
            "accion": respuesta_simple["accion"],
            "mensaje": respuesta_simple["mensaje"],
        })

    return construir_respuesta_chat({
        "texto": texto,
        "intencion": intencion,
        "confianza": confianza,
        "accion": "pendiente",
        "mensaje": "Esta intenci\u00f3n a\u00fan no est\u00e1 implementada.",
    })
