import json
import importlib.util
from pathlib import Path
import re
import unicodedata

import numpy as np

from api.services.famysalud_api import obtener_servicios_normalizados
from api.services.service_index import construir_documentos_servicios


BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "api" / "cache"
INDEX_PATH = CACHE_DIR / "service_index.json"
EMBEDDINGS_PATH = CACHE_DIR / "service_embeddings.npy"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_VERSION = 3
DEFAULT_TOP_K = 10
MIN_SIMILARITY = 0.25
MIN_FINAL_SCORE = 0.25

embedding_model = None
semantic_index = None
sentence_transformers_available = None
embedding_model_error = None


def sentence_transformers_disponible():
    global sentence_transformers_available

    if sentence_transformers_available is not None:
        return sentence_transformers_available

    sentence_transformers_available = (
        importlib.util.find_spec("sentence_transformers") is not None
    )

    return sentence_transformers_available


def obtener_embedding_model():
    global embedding_model
    global embedding_model_error

    if embedding_model is not None:
        return embedding_model

    try:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(MODEL_NAME)
        embedding_model_error = None
        return embedding_model
    except Exception as exc:
        embedding_model_error = str(exc)
        raise


def indice_semantico_en_cache_disponible():
    return INDEX_PATH.exists() and EMBEDDINGS_PATH.exists()


def busqueda_semantica_disponible():
    return sentence_transformers_disponible() and indice_semantico_en_cache_disponible()


def estado_embeddings():
    return {
        "embeddings_enabled": sentence_transformers_disponible(),
        "sentence_transformers_available": sentence_transformers_disponible(),
        "semantic_index_available": indice_semantico_en_cache_disponible(),
        "model_loaded": embedding_model is not None,
        "model_name": MODEL_NAME,
        "error": embedding_model_error,
    }


EXPANSIONES_CONSULTA = {
    "eco": ["ecografia"],
    "ultrasonido": ["ecografia"],
    "higado": ["hepatobiliar", "hepatico"],
    "vesicula": ["biliar", "hepatobiliar"],
    "muela del juicio": ["tercer molar"],
    "tratamiento de conducto": ["endodoncia"],
    "calce": ["restauracion"],
    "calces": ["restauracion"],
}

CORRECCIONES_TEXTO = (
    (r"\bsiasen(?=\w)", "si hacen "),
    (r"\bsiasen\b", "si hacen"),
    (r"\bsirealisan(?=\w)", "si realizan "),
    (r"\bsirealisan\b", "si realizan"),
    (r"\brealisan\b", "realizan"),
    (r"\basen\b", "hacen"),
    (r"\becocardiogrma\b", "ecocardiograma"),
    (r"\bextracion\b", "extraccion"),
    (r"\bblancamiento\b", "blanqueamiento"),
    (r"\benal\b", "renal"),
    (r"\bprotesi\b", "protesis"),
)

STOPWORDS_CONSULTA = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "el",
    "en",
    "la",
    "las",
    "los",
    "para",
    "por",
    "un",
    "una",
}


def normalizar_texto_simple(texto):
    texto = str(texto or "").strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    for patron, reemplazo in CORRECCIONES_TEXTO:
        texto = re.sub(patron, reemplazo, texto)

    return re.sub(r"\s+", " ", texto).strip()


def obtener_tokens(texto):
    tokens = []

    for token in re.findall(r"[a-z0-9]+", normalizar_texto_simple(texto)):
        if token in STOPWORDS_CONSULTA:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)

    return tokens


def expandir_consulta(texto, sinonimos_catalogo=None):
    sinonimos_catalogo = sinonimos_catalogo or {}
    texto_normalizado = normalizar_texto_simple(texto)
    expansiones = []

    for frase, termino_catalogo in sinonimos_catalogo.items():
        if normalizar_texto_simple(frase) in texto_normalizado:
            expansiones.append(termino_catalogo)

    for frase, terminos in EXPANSIONES_CONSULTA.items():
        if normalizar_texto_simple(frase) in texto_normalizado:
            expansiones.extend(terminos)

    if not expansiones:
        return texto

    return f"{texto}. {' '.join(sorted(set(expansiones)))}"


def obtener_senales_consulta(texto):
    texto_normalizado = normalizar_texto_simple(texto)
    texto_expandido = expandir_consulta(texto)
    tokens = set(obtener_tokens(texto_expandido))

    return {
        "texto_normalizado": texto_normalizado,
        "texto_expandido": texto_expandido,
        "tokens": tokens,
        "busca_ecografia": bool({"eco", "ecografia", "ultrasonido"} & tokens),
        "busca_hepatobiliar": bool(
            {"higado", "hepatobiliar", "hepatico", "vesicula", "biliar"} & tokens
        ),
        "busca_resonancia": "resonancia" in tokens,
        "busca_tomografia": "tomografia" in tokens,
        "busca_odontologia": bool({"muela", "molar", "cordal", "cordales"} & tokens),
    }


def calcular_score_lexico(documento, senales):
    tokens_consulta = senales["tokens"]
    if not tokens_consulta:
        return 0.0

    metadata = documento["metadata"]
    nombre = normalizar_texto_simple(metadata.get("nombre"))
    area = normalizar_texto_simple(metadata.get("area"))
    slug = normalizar_texto_simple(metadata.get("slug"))
    texto = normalizar_texto_simple(documento.get("texto"))
    tokens_nombre = set(obtener_tokens(nombre))
    tokens_area = set(obtener_tokens(area))
    tokens_slug = set(obtener_tokens(slug))
    tokens_documento = set(obtener_tokens(texto))
    coincidencias = tokens_consulta & tokens_documento
    score = len(coincidencias) / max(len(tokens_consulta), 1)

    for token in tokens_consulta:
        if len(token) <= 3:
            continue
        if token in tokens_nombre:
            score += 0.12
        if token in tokens_slug:
            score += 0.06
        if token in tokens_area:
            score += 0.06

    return min(score, 1.0)


def calcular_boost_area(documento, senales):
    area = normalizar_texto_simple(documento["metadata"].get("area"))
    nombre = normalizar_texto_simple(documento["metadata"].get("nombre"))
    texto = normalizar_texto_simple(documento.get("texto"))
    boost = 0.0

    if senales["busca_ecografia"]:
        if area == "ecografias":
            boost += 0.18
        elif not nombre.startswith("eco"):
            boost -= 0.18

    if senales["busca_resonancia"]:
        if area == "resonancias":
            boost += 0.18
        else:
            boost -= 0.12

    if senales["busca_tomografia"]:
        if area == "tomografias":
            boost += 0.18
        else:
            boost -= 0.12

    if senales["busca_odontologia"]:
        if area == "odontologia":
            boost += 0.14
        else:
            boost -= 0.10

    if senales["busca_hepatobiliar"]:
        if any(
            termino in texto
            for termino in ("higado", "hepatobiliar", "hepatico", "vesicula", "biliar")
        ):
            boost += 0.16
        else:
            boost -= 0.06

    return boost


def calcular_score_final(semantic_score, lexical_score, area_boost):
    score = (semantic_score * 0.70) + (lexical_score * 0.30) + area_boost
    return max(0.0, min(score, 1.0))


def cargar_cache(updated_at):
    if not INDEX_PATH.exists() or not EMBEDDINGS_PATH.exists():
        return None

    with INDEX_PATH.open("r", encoding="utf-8") as archivo:
        index_data = json.load(archivo)

    if index_data.get("updated_at") != updated_at:
        return None

    if index_data.get("index_version") != INDEX_VERSION:
        return None

    embeddings = np.load(EMBEDDINGS_PATH)
    return {
        "updated_at": updated_at,
        "documentos": index_data.get("documentos", []),
        "embeddings": embeddings,
    }


def guardar_cache(updated_at, documentos, embeddings):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with INDEX_PATH.open("w", encoding="utf-8") as archivo:
        json.dump(
            {
                "updated_at": updated_at,
                "index_version": INDEX_VERSION,
                "model_name": MODEL_NAME,
                "documentos": documentos,
            },
            archivo,
            ensure_ascii=False,
        )

    np.save(EMBEDDINGS_PATH, embeddings)


def construir_indice_semantico(sinonimos_catalogo=None, force_refresh=False):
    global semantic_index

    if not sentence_transformers_disponible():
        raise RuntimeError(
            "sentence-transformers no esta instalado; indice semantico deshabilitado"
        )

    catalogo = obtener_servicios_normalizados()
    updated_at = catalogo.get("updated_at")

    if not force_refresh and semantic_index is not None:
        if semantic_index.get("updated_at") == updated_at:
            return semantic_index

    if not force_refresh:
        cache = cargar_cache(updated_at)
        if cache is not None:
            semantic_index = cache
            return semantic_index

    documentos = construir_documentos_servicios(
        catalogo.get("servicios", []),
        sinonimos_catalogo,
    )
    textos = [documento["texto"] for documento in documentos]
    model = obtener_embedding_model()
    embeddings = model.encode(textos, convert_to_numpy=True, normalize_embeddings=True)

    semantic_index = {
        "updated_at": updated_at,
        "documentos": documentos,
        "embeddings": embeddings,
    }
    guardar_cache(updated_at, documentos, embeddings)
    return semantic_index


def buscar_servicios_semanticos(texto, top_k=DEFAULT_TOP_K, sinonimos_catalogo=None):
    texto = str(texto or "").strip()
    if not texto:
        return {
            "total": 0,
            "total_real": 0,
            "total_conocido": True,
            "resultados": [],
        }

    if not sentence_transformers_disponible():
        return {
            "total": 0,
            "total_real": 0,
            "total_conocido": True,
            "resultados": [],
            "_semantic_available": False,
            "_semantic_error": "sentence-transformers no esta instalado",
        }

    index = construir_indice_semantico(sinonimos_catalogo=sinonimos_catalogo)
    documentos = index["documentos"]
    embeddings = index["embeddings"]

    if not documentos:
        return {
            "total": 0,
            "total_real": 0,
            "total_conocido": True,
            "resultados": [],
        }

    model = obtener_embedding_model()
    texto_expandido = expandir_consulta(texto, sinonimos_catalogo)
    senales = obtener_senales_consulta(texto_expandido)
    query_embedding = model.encode(
        [texto_expandido],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    scores = embeddings @ query_embedding
    candidatos = []

    for indice, semantic_score in enumerate(scores):
        semantic_score = float(semantic_score)
        if semantic_score < MIN_SIMILARITY:
            continue

        documento = documentos[int(indice)]
        lexical_score = calcular_score_lexico(documento, senales)
        area_boost = calcular_boost_area(documento, senales)
        final_score = calcular_score_final(semantic_score, lexical_score, area_boost)

        if final_score < MIN_FINAL_SCORE:
            continue

        candidatos.append(
            {
                "indice": int(indice),
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "final_score": final_score,
            }
        )

    candidatos.sort(key=lambda item: item["final_score"], reverse=True)
    resultados = []

    for candidato in candidatos[: min(int(top_k), len(candidatos))]:
        indice = candidato["indice"]
        metadata = dict(documentos[int(indice)]["metadata"])
        metadata["semantic_score"] = round(candidato["semantic_score"], 4)
        metadata["lexical_score"] = round(candidato["lexical_score"], 4)
        metadata["final_score"] = round(candidato["final_score"], 4)
        metadata["score"] = metadata["final_score"]
        resultados.append(metadata)

    return {
        "total": len(resultados),
        "total_real": len(resultados),
        "total_conocido": True,
        "resultados": resultados,
    }
