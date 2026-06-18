import argparse
import json
import sys
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CASES_PATH = BASE_DIR / "tests" / "manual_cases_famybot_ia.json"

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
        return data.get("casos", [])

    return data


def cachear_catalogo():
    catalogo = app.obtener_catalogo()
    app.obtener_catalogo = lambda: catalogo


def resultado_contiene(respuesta, esperado):
    esperado_normalizado = normalizar(esperado)

    for resultado in respuesta.get("resultados", []) or []:
        texto_resultado = " ".join(
            str(valor)
            for valor in (
                resultado.get("nombre"),
                resultado.get("area"),
                resultado.get("precio"),
            )
            if valor is not None
        )

        if esperado_normalizado in normalizar(texto_resultado):
            return True

    return False


def total_respuesta(respuesta):
    total = respuesta.get("total")
    if isinstance(total, int):
        return total
    return len(respuesta.get("resultados", []) or [])


def validar_caso(caso, respuesta):
    errores = []
    accion = respuesta.get("accion")
    total = total_respuesta(respuesta)
    resultados = respuesta.get("resultados", []) or []

    debe_encontrar = caso.get("debe_encontrar")
    if debe_encontrar is True and (total <= 0 or not resultados):
        errores.append("esperaba resultados, pero no encontró ninguno")
    elif debe_encontrar is False and (total > 0 or resultados):
        errores.append(f"esperaba cero resultados, pero encontró {total}")

    accion_esperada = caso.get("accion_esperada")
    if accion_esperada and accion != accion_esperada:
        errores.append(
            f"accion esperada={accion_esperada!r}, accion actual={accion!r}"
        )

    acciones_no_permitidas = caso.get("acciones_no_permitidas") or []
    if accion in acciones_no_permitidas:
        errores.append(f"accion no permitida={accion!r}")

    contiene_resultado = caso.get("contiene_resultado")
    if contiene_resultado and not resultado_contiene(respuesta, contiene_resultado):
        errores.append(
            f"ningún resultado contiene {contiene_resultado!r}"
        )

    total_minimo = caso.get("total_minimo")
    if total_minimo is not None and total < int(total_minimo):
        errores.append(f"total mínimo={total_minimo}, total actual={total}")

    return errores


def ejecutar_caso(indice, caso):
    texto = caso["texto"]
    respuesta = app.chat(app.SearchRequest(texto=texto))
    errores = validar_caso(caso, respuesta)
    estado = "PASS" if not errores else "FAIL"
    total = total_respuesta(respuesta)

    print(
        f"[{estado}] {indice:02d}. {texto} "
        f"-> accion={respuesta.get('accion')} total={total}"
    )

    if errores:
        for error in errores:
            print(f"       - {error}")
        mensaje = respuesta.get("mensaje")
        if mensaje:
            print(f"       mensaje: {mensaje}")
        for resultado in (respuesta.get("resultados", []) or [])[:5]:
            print(
                "       resultado: "
                f"{resultado.get('nombre')} | {resultado.get('area')} | "
                f"${resultado.get('precio')}"
            )

    return not errores


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta casos manuales reales contra FamyBot IA."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Ruta al archivo JSON de casos.",
    )
    args = parser.parse_args()

    casos = cargar_casos(args.cases)
    if not casos:
        print(f"No hay casos para ejecutar en {args.cases}")
        return 1

    cachear_catalogo()

    aprobados = 0
    for indice, caso in enumerate(casos, start=1):
        if ejecutar_caso(indice, caso):
            aprobados += 1

    total = len(casos)
    fallidos = total - aprobados
    print(f"\nResumen: total={total} pass={aprobados} fail={fallidos}")
    return 0 if fallidos == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
