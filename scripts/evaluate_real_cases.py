import argparse
import json
import sys
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CASES_PATH = BASE_DIR / "tests" / "real_cases_v1.json"
DEFAULT_REPORT_PATH = BASE_DIR / "reports" / "evaluation_report.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api import app  # noqa: E402


def normalizar(texto):
    texto = str(texto or "").strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )


def cargar_casos(path):
    with path.open("r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    if isinstance(data, dict):
        return data.get("cases", [])

    return data


def cachear_catalogo():
    catalogo = app.obtener_catalogo()
    app.obtener_catalogo = lambda: catalogo

    try:
        from api.services import embedding_search
        from api.services import famysalud_api

        servicios = famysalud_api.obtener_servicios_normalizados(catalogo)
        embedding_search.obtener_servicios_normalizados = lambda: servicios
    except Exception:
        pass


def obtener_intenciones_esperadas(caso):
    if caso.get("expected_intents"):
        return caso["expected_intents"]

    expected_intent = caso.get("expected_intent")
    return [expected_intent] if expected_intent else []


def obtener_servicios_esperados(caso):
    if caso.get("expected_services"):
        return caso["expected_services"]

    expected_service = caso.get("expected_service")
    return [expected_service] if expected_service else []


def servicio_en_resultado(resultado, esperado):
    esperado_normalizado = normalizar(esperado)
    texto_resultado = " ".join(
        str(valor)
        for valor in (
            resultado.get("nombre"),
            resultado.get("area"),
        )
        if valor is not None
    )
    resultado_normalizado = normalizar(texto_resultado)

    if esperado_normalizado in resultado_normalizado:
        return True

    return any(
        regla_match_servicio(esperado_normalizado, resultado_normalizado)
        for regla_match_servicio in (
            match_tercer_molar,
            match_resonancia_cerebral_contraste,
            match_columna_completa,
            match_calce,
            match_protesis,
            match_mamografia,
            match_craneo_cerebro,
        )
    )


def match_tercer_molar(esperado, resultado):
    return (
        ("tercer molar" in esperado or "muela del juicio" in esperado)
        and "tercer molar" in resultado
    )


def match_resonancia_cerebral_contraste(esperado, resultado):
    pide_cerebral = any(
        termino in esperado
        for termino in ("resonancia cerebral", "cerebral", "craneo", "cerebro")
    )
    pide_contraste = "contraste" in esperado or "contrast" in esperado

    if not pide_cerebral:
        return False

    if not any(termino in resultado for termino in ("craneo", "cerebro")):
        return False

    if pide_contraste:
        return "contrast" in resultado

    return True


def match_columna_completa(esperado, resultado):
    return (
        any(termino in esperado for termino in ("columna completa", "columna total"))
        and "columna" in resultado
    )


def match_calce(esperado, resultado):
    return (
        any(termino in esperado for termino in ("calce", "restauracion"))
        and any(termino in resultado for termino in ("restauracion", "calce"))
    )


def match_protesis(esperado, resultado):
    return "protesis" in esperado and "protesis" in resultado


def match_mamografia(esperado, resultado):
    return "mamografia" in esperado and "mamografia" in resultado


def match_craneo_cerebro(esperado, resultado):
    return (
        any(termino in esperado for termino in ("craneo", "cerebro"))
        and any(termino in resultado for termino in ("craneo", "cerebro"))
    )


def rank_servicio(resultados, servicios_esperados):
    if not servicios_esperados:
        return None

    for indice, resultado in enumerate(resultados, start=1):
        if any(servicio_en_resultado(resultado, esperado) for esperado in servicios_esperados):
            return indice

    return None


def resumir_resultados(resultados, limite=5):
    return [
        {
            "id": resultado.get("id"),
            "nombre": resultado.get("nombre"),
            "area": resultado.get("area"),
            "precio": resultado.get("precio"),
        }
        for resultado in resultados[:limite]
    ]


def evaluar_caso(caso):
    texto = caso["text"]
    intenciones_esperadas = obtener_intenciones_esperadas(caso)
    servicios_esperados = obtener_servicios_esperados(caso)

    prediccion = app.predecir_intencion(texto)
    predicted_intent = prediccion.get("intencion")
    intent_ok = predicted_intent in intenciones_esperadas if intenciones_esperadas else None

    busqueda = None
    resultados = []
    service_rank = None

    if servicios_esperados:
        consulta_catalogo = app.preparar_consulta_catalogo(texto)
        busqueda = app.buscar_servicios(consulta_catalogo)
        resultados = busqueda.get("resultados", []) or []
        service_rank = rank_servicio(resultados, servicios_esperados)

    return {
        "text": texto,
        "expected_intents": intenciones_esperadas,
        "predicted_intent": predicted_intent,
        "intent_ok": intent_ok,
        "expected_services": servicios_esperados,
        "service_rank": service_rank,
        "top_services": resumir_resultados(resultados),
        "search_total": busqueda.get("total") if busqueda else None,
    }


def calcular_metricas(evaluaciones):
    total = len(evaluaciones)
    intent_evaluados = [
        evaluacion
        for evaluacion in evaluaciones
        if evaluacion["intent_ok"] is not None
    ]
    intent_aciertos = sum(1 for evaluacion in intent_evaluados if evaluacion["intent_ok"])
    intent_errores = [
        evaluacion
        for evaluacion in intent_evaluados
        if not evaluacion["intent_ok"]
    ]

    service_evaluados = [
        evaluacion
        for evaluacion in evaluaciones
        if evaluacion["expected_services"]
    ]
    top1 = sum(1 for evaluacion in service_evaluados if evaluacion["service_rank"] == 1)
    top3 = sum(
        1
        for evaluacion in service_evaluados
        if evaluacion["service_rank"] is not None and evaluacion["service_rank"] <= 3
    )
    top5 = sum(
        1
        for evaluacion in service_evaluados
        if evaluacion["service_rank"] is not None and evaluacion["service_rank"] <= 5
    )

    return {
        "total_cases": total,
        "intent_cases": len(intent_evaluados),
        "intent_correct": intent_aciertos,
        "intent_errors": len(intent_errores),
        "intent_accuracy": round(intent_aciertos / len(intent_evaluados), 4)
        if intent_evaluados
        else None,
        "service_cases": len(service_evaluados),
        "service_top1_correct": top1,
        "service_top3_correct": top3,
        "service_top5_correct": top5,
        "service_top1_accuracy": round(top1 / len(service_evaluados), 4)
        if service_evaluados
        else None,
        "service_top3_accuracy": round(top3 / len(service_evaluados), 4)
        if service_evaluados
        else None,
        "service_top5_accuracy": round(top5 / len(service_evaluados), 4)
        if service_evaluados
        else None,
    }


def construir_errores(evaluaciones):
    errores = []

    for evaluacion in evaluaciones:
        intent_error = evaluacion["intent_ok"] is False
        service_error = (
            evaluacion["expected_services"]
            and (
                evaluacion["service_rank"] is None
                or evaluacion["service_rank"] > 5
            )
        )

        if not intent_error and not service_error:
            continue

        errores.append({
            "text": evaluacion["text"],
            "expected_intent": (
                evaluacion["expected_intents"][0]
                if len(evaluacion["expected_intents"]) == 1
                else None
            ),
            "expected_intents": evaluacion["expected_intents"],
            "predicted_intent": evaluacion["predicted_intent"],
            "expected_service": (
                evaluacion["expected_services"][0]
                if len(evaluacion["expected_services"]) == 1
                else None
            ),
            "expected_services": evaluacion["expected_services"],
            "service_rank": evaluacion["service_rank"],
            "top_services": evaluacion["top_services"],
        })

    return errores


def main():
    parser = argparse.ArgumentParser(
        description="Evalua casos reales de FamyBot IA."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    casos = cargar_casos(args.cases)
    if not casos:
        print(f"No hay casos para evaluar en {args.cases}")
        return 1

    cachear_catalogo()
    evaluaciones = [evaluar_caso(caso) for caso in casos]
    metricas = calcular_metricas(evaluaciones)
    errores = construir_errores(evaluaciones)
    reporte = {
        **metricas,
        "errors": errores,
        "evaluations": evaluaciones,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, ensure_ascii=False, indent=2)

    print(
        "Resumen: "
        f"total={metricas['total_cases']} "
        f"intent_accuracy={metricas['intent_accuracy']} "
        f"service_top1={metricas['service_top1_accuracy']} "
        f"service_top3={metricas['service_top3_accuracy']} "
        f"service_top5={metricas['service_top5_accuracy']}"
    )
    print(f"Reporte: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
