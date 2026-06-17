from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from difflib import SequenceMatcher
import joblib
import os
from pathlib import Path
import requests
from typing import Optional
import unicodedata
import re

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

CATALOG_URL = "https://app.famysaludec.com/chatbot/catalogo-servicios"
FAMYBOT_IA_API_KEY = os.environ.get("FAMYBOT_IA_API_KEY", "").strip()
MIN_CONF_ACCION_FLUJO = 0.55
MIN_RATIO_FUZZY_INTENCION = 0.86
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
        "mensaje": "Nos encontramos en Quisquís 1109 y José Mascote, Guayaquil. Si lo prefieres, puedes usar los botones de navegación para ver el croquis o abrir la ubicación directamente en Google Maps.",
    },
    "desconocido": {
        "accion": "fallback",
        "mensaje": "No estoy seguro de haber entendido tu consulta. Puedes escribirla de otra forma o elegir una opción del menú principal.",
    },
}
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
PALABRAS_IGNORADAS_COMERCIAL = PALABRAS_SALUDO_PURO | {
    "al",
    "con",
    "consulta",
    "costo",
    "cual",
    "cuanto",
    "cuesta",
    "de",
    "del",
    "dia",
    "disculpe",
    "dispone",
    "disponen",
    "el",
    "es",
    "la",
    "las",
    "lo",
    "los",
    "precio",
    "q",
    "que",
    "salen",
    "servicio",
    "servicios",
    "tambien",
    "tiene",
    "tienen",
    "un",
    "una",
    "valor",
    "vale",
    "y",
    "si",
}
CONECTORES_SERVICIOS = {
    "ademas",
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
    "lavado de oído": "lavado oidos",
    "prueba de embarazo": "prueba hcg",
    "test de embarazo": "prueba hcg",
    "examen de embarazo": "prueba hcg",
    "embarazo en sangre": "prueba hcg",
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
    "protesis removible": "protesis removible acrilica",
    "dientes postizos": "protesis removible acrilica",
    "sacar tercer molar": "extraccion de tercer molar",
    "sacar un tercer molar": "extraccion de tercer molar",
    "extraccion tercer molar": "extraccion de tercer molar",
    "muela del juicio": "extraccion de tercer molar",
    "cordal": "extraccion de tercer molar",
    "calce de muela": "restauracion",
    "calce dental": "restauracion",
    "calce de diente": "restauracion",
    "restauracion dental": "restauracion",
    "extraccion de muela": "extraccion dental",
    "extraccion muela": "extraccion dental",
    "sacar muela": "extraccion dental",
}

vectorizer = None
classifier = None
model_load_error = None

try:
    vectorizer = joblib.load(MODEL_DIR / "vectorizer_famybot_v1.pkl")
    classifier = joblib.load(MODEL_DIR / "classifier_famybot_v1.pkl")
    print("FamyBot IA: modelo cargado correctamente")
except Exception as exc:
    model_load_error = str(exc)
    print(f"FamyBot IA: error cargando modelo: {exc}")

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

@app.get("/")
def home():
    return {"status": "ok", "message": "FamyBot IA API activa"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": vectorizer is not None and classifier is not None,
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
    texto = texto.strip()

    if not texto:
        return {
            "texto": texto,
            "intencion": "desconocido",
            "confianza": None,
        }

    intencion_saludo = detectar_saludo_puro(texto)
    if intencion_saludo:
        return {
            "texto": texto,
            "intencion": intencion_saludo,
            "confianza": 1.0,
        }

    if vectorizer is None or classifier is None:
        return {
            "texto": texto,
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
        "texto": texto,
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
    return "".join(caracter for caracter in texto if unicodedata.category(caracter) != "Mn")


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

    if all(palabra in PALABRAS_SALUDO_PURO for palabra in palabras):
        return "saludo"

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


def preparar_consulta_sin_relleno(texto):
    palabras = obtener_palabras_clave_intencion(texto)
    return " ".join(palabras) or texto


def aplicar_sinonimos_catalogo(texto):
    texto_normalizado = normalizar_texto(texto)

    for frase, termino_catalogo in SINONIMOS_CATALOGO.items():
        if normalizar_texto(frase) in texto_normalizado:
            return termino_catalogo

    return texto


def buscar_servicios(texto_busqueda):
    catalogo = obtener_catalogo()
    texto = normalizar_texto(texto_busqueda)
    tokens_busqueda = obtener_tokens_utiles(texto_busqueda)
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
            tokens_servicio = obtener_tokens_servicio(texto_servicio)

            if servicio_coincide(texto, tokens_busqueda, texto_servicio, tokens_servicio):
                total_real += 1

                if len(resultados) < 20:
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

    return {
        "total": total_real,
        "total_real": total_real,
        "total_conocido": total_conocido,
        "resultados": resultados,
    }


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


def preparar_consulta_comercial_catalogo(texto):
    texto_catalogo = aplicar_sinonimos_catalogo(texto)
    palabras = []

    for palabra in obtener_palabras_normalizadas(texto_catalogo):
        if palabra in PALABRAS_IGNORADAS_COMERCIAL:
            continue
        if es_palabra_ignorada(palabra):
            continue
        palabras.append(normalizar_token_catalogo(palabra))

    return " ".join(palabras) or texto_catalogo


def tiene_conectores_servicios(texto):
    if "," in texto:
        return True

    return any(
        palabra in CONECTORES_SERVICIOS
        for palabra in obtener_palabras_normalizadas(texto)
    )


def termino_comercial_util(termino):
    tokens = []

    for palabra in obtener_palabras_normalizadas(termino):
        if palabra in PALABRAS_IGNORADAS_COMERCIAL:
            continue
        if es_palabra_ignorada(palabra):
            continue
        tokens.append(normalizar_token_catalogo(palabra))

    return any(len(token) > 2 for token in tokens)


def obtener_terminos_comerciales_individuales(texto):
    terminos = []
    actual = []

    for palabra in obtener_palabras_normalizadas(texto):
        if palabra in CONECTORES_SERVICIOS:
            if actual:
                termino = preparar_consulta_comercial_catalogo(" ".join(actual))
                if termino_comercial_util(termino):
                    terminos.append(termino)
                actual = []
            continue
        actual.append(palabra)

    if actual:
        termino = preparar_consulta_comercial_catalogo(" ".join(actual))
        if termino_comercial_util(termino):
            terminos.append(termino)

    return terminos


def combinar_busquedas_catalogo(busquedas):
    resultados = []
    vistos = set()
    total_real = 0
    total_conocido = True

    for busqueda in busquedas:
        total_real += busqueda["total_real"]
        total_conocido = total_conocido and busqueda["total_conocido"]

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
    }


def construir_respuesta_busqueda_catalogo(texto, busqueda):
    resultados = busqueda["resultados"]
    total = busqueda["total"]
    total_conocido = busqueda["total_conocido"]

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
            f"{servicio.get('area')} y tiene un valor de ${servicio.get('precio')} "
            "en efectivo o transferencia."
        )
    else:
        accion = "listar_opciones"
        if total > 10 and total_conocido:
            mensaje = (
                f"Encontré {total} opciones relacionadas con tu consulta. "
                "Te muestro las primeras 10. Puedes responder con el nombre "
                "del servicio que deseas consultar."
            )
        elif total > 10:
            mensaje = (
                "Encontré varias opciones relacionadas con tu consulta. "
                "Te muestro las primeras 10. Puedes responder con el nombre "
                "del servicio que deseas consultar."
            )
        else:
            mensaje = (
                f"Encontré {total} opciones relacionadas con tu consulta. Puedes "
                "revisar la lista y responder con el nombre del servicio que deseas "
                "consultar."
            )

    return {
        "texto": texto,
        "total": total,
        "total_real": busqueda["total_real"],
        "total_conocido": total_conocido,
        "accion": accion,
        "mensaje": mensaje,
        "resultados": resultados,
    }


def buscar_catalogo_comercial(texto):
    consulta_catalogo = preparar_consulta_comercial_catalogo(texto)
    busqueda = filtrar_busqueda_por_tokens_exactos(
        buscar_servicios(consulta_catalogo),
        consulta_catalogo,
    )

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


def construir_respuesta_catalogo(texto, intencion, confianza, respuesta_catalogo):
    return {
        "texto": texto,
        "intencion": intencion,
        "confianza": confianza,
        "accion": respuesta_catalogo["accion"],
        "mensaje": respuesta_catalogo["mensaje"],
        "total": respuesta_catalogo["total"],
        "total_real": respuesta_catalogo["total_real"],
        "total_conocido": respuesta_catalogo["total_conocido"],
        "resultados": respuesta_catalogo["resultados"],
    }


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
            f"{servicio.get('area')} y tiene un valor de ${servicio.get('precio')} "
            "en efectivo o transferencia."
        )
    else:
        accion = "listar_opciones"
        if total > 10 and total_conocido:
            mensaje = (
                f"Encontré {total} opciones relacionadas con tu consulta. "
                "Te muestro las primeras 10. Puedes responder con el nombre "
                "del servicio que deseas consultar."
            )
        elif total > 10:
            mensaje = (
                "Encontré varias opciones relacionadas con tu consulta. "
                "Te muestro las primeras 10. Puedes responder con el nombre "
                "del servicio que deseas consultar."
            )
        else:
            mensaje = (
                f"Encontré {total} opciones relacionadas con tu consulta. Puedes "
                "revisar la lista y responder con el nombre del servicio que deseas "
                "consultar."
            )

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
    prediccion = predecir_intencion(texto)
    intencion = prediccion["intencion"]
    confianza = prediccion["confianza"]
    print(f"FamyBot IA: intencion_original={intencion} confianza={confianza}")

    if prediccion.get("error") == "modelo_no_cargado":
        return {
            "texto": texto,
            "intencion": intencion,
            "confianza": confianza,
            "accion": "fallback",
            "mensaje": "En este momento estoy iniciando mis servicios. Puedes intentar de nuevo en unos segundos o solicitar ayuda con un asesor.",
            "error": "modelo_no_cargado",
        }

    if es_consulta_precio_y_ubicacion(texto):
        return {
            "texto": texto,
            "intencion": "consulta_servicios",
            "confianza": confianza,
            "accion": "consulta_ubicacion",
            "mensaje": (
                "Para indicarte el valor exacto de la consulta, por favor dime "
                "la especialidad o área que necesitas. Como referencia, la "
                "consulta de Medicina General tiene un valor de $15 en efectivo "
                "o transferencia. También puedo compartirte nuestra ubicación."
            ),
            "total": 0,
            "total_real": 0,
            "total_conocido": True,
            "resultados": [],
        }

    if es_consulta_catalogo_comercial(texto):
        respuesta_catalogo = buscar_catalogo_comercial(texto)
        print(f"FamyBot IA: total_catalogo_comercial={respuesta_catalogo['total']}")
        intencion_catalogo = intencion if intencion in INTENCIONES_CATALOGO else "consulta_servicios"
        return construir_respuesta_catalogo(
            texto,
            intencion_catalogo,
            confianza,
            respuesta_catalogo,
        )

    if intencion in INTENCIONES_CATALOGO:
        consulta_catalogo = preparar_consulta_catalogo(texto)
        respuesta_catalogo = ask_catalog(SearchRequest(texto=consulta_catalogo))
        print(f"FamyBot IA: total_catalogo={respuesta_catalogo['total']}")
        return construir_respuesta_catalogo(texto, intencion, confianza, respuesta_catalogo)

    if intencion in ACCIONES_FLUJO:
        if confianza is None or confianza < MIN_CONF_ACCION_FLUJO:
            print(
                "FamyBot IA: accion_flujo_bloqueada_por_baja_confianza="
                f"{intencion}"
            )
            intencion_fuzzy = detectar_intencion_frecuente_fuzzy(texto)

            if intencion_fuzzy in ACCIONES_FLUJO:
                accion_flujo = ACCIONES_FLUJO[intencion_fuzzy]
                return {
                    "texto": texto,
                    "intencion": intencion_fuzzy,
                    "confianza": confianza,
                    "accion": accion_flujo["accion"],
                    "mensaje": accion_flujo["mensaje"],
                }

            if intencion_fuzzy in RESPUESTAS_SIMPLES:
                respuesta_simple = RESPUESTAS_SIMPLES[intencion_fuzzy]
                return {
                    "texto": texto,
                    "intencion": intencion_fuzzy,
                    "confianza": confianza,
                    "accion": respuesta_simple["accion"],
                    "mensaje": respuesta_simple["mensaje"],
                }

            consulta_catalogo = preparar_consulta_catalogo(texto)
            respuesta_catalogo = ask_catalog(SearchRequest(texto=consulta_catalogo))
            print(f"FamyBot IA: total_catalogo={respuesta_catalogo['total']}")

            if respuesta_catalogo["total"] > 0:
                return construir_respuesta_catalogo(
                    texto,
                    intencion,
                    confianza,
                    respuesta_catalogo,
                )

            respuesta_simple = RESPUESTAS_SIMPLES["desconocido"]
            return {
                "texto": texto,
                "intencion": intencion,
                "confianza": confianza,
                "accion": respuesta_simple["accion"],
                "mensaje": respuesta_simple["mensaje"],
            }

        accion_flujo = ACCIONES_FLUJO[intencion]
        return {
            "texto": texto,
            "intencion": intencion,
            "confianza": confianza,
            "accion": accion_flujo["accion"],
            "mensaje": accion_flujo["mensaje"],
        }

    if intencion == "saludo" and not detectar_saludo_puro(texto):
        intencion_fuzzy = detectar_intencion_frecuente_fuzzy(texto)

        if intencion_fuzzy in ACCIONES_FLUJO:
            accion_flujo = ACCIONES_FLUJO[intencion_fuzzy]
            return {
                "texto": texto,
                "intencion": intencion_fuzzy,
                "confianza": confianza,
                "accion": accion_flujo["accion"],
                "mensaje": accion_flujo["mensaje"],
            }

        if intencion_fuzzy in RESPUESTAS_SIMPLES:
            respuesta_simple = RESPUESTAS_SIMPLES[intencion_fuzzy]
            return {
                "texto": texto,
                "intencion": intencion_fuzzy,
                "confianza": confianza,
                "accion": respuesta_simple["accion"],
                "mensaje": respuesta_simple["mensaje"],
            }

        consulta_catalogo = preparar_consulta_sin_relleno(texto)
        respuesta_catalogo = ask_catalog(SearchRequest(texto=consulta_catalogo))
        print(f"FamyBot IA: total_catalogo={respuesta_catalogo['total']}")

        if respuesta_catalogo["total"] > 0:
            return construir_respuesta_catalogo(
                texto,
                "consulta_servicios",
                confianza,
                respuesta_catalogo,
            )

    if intencion in RESPUESTAS_SIMPLES:
        respuesta_simple = RESPUESTAS_SIMPLES[intencion]
        return {
            "texto": texto,
            "intencion": intencion,
            "confianza": confianza,
            "accion": respuesta_simple["accion"],
            "mensaje": respuesta_simple["mensaje"],
        }

    return {
        "texto": texto,
        "intencion": intencion,
        "confianza": confianza,
        "accion": "pendiente",
        "mensaje": "Esta intenci\u00f3n a\u00fan no est\u00e1 implementada.",
    }
