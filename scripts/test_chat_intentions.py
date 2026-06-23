import argparse
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CASES_PATH = BASE_DIR / "tests" / "real_patient_chat_intentions.json"
DEFAULT_REPORT_PATH = BASE_DIR / "reports" / "chat_intentions_report.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def load_cases(path):
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("cases", payload if isinstance(payload, list) else [])


def load_failed_indexes(path):
    if not path or not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    failed = payload.get("fallos", [])
    indexes = set()
    for failure in failed:
        index = failure.get("index")
        if isinstance(index, int):
            indexes.add(index)
    return indexes


def normalize_intents(intents):
    if intents is None:
        return []
    return sorted(str(intent) for intent in intents)


def cache_catalog(app):
    catalog = app.obtener_catalogo()
    app.obtener_catalogo = lambda: catalog
    original_buscar_servicios = app.buscar_servicios
    search_cache = {}

    def cached_buscar_servicios(text):
        key = str(text or "")
        if key not in search_cache:
            search_cache[key] = copy.deepcopy(original_buscar_servicios(text))
        return copy.deepcopy(search_cache[key])

    app.buscar_servicios = cached_buscar_servicios

    try:
        from api.services import embedding_search
        from api.services import famysalud_api

        normalized = famysalud_api.obtener_servicios_normalizados(catalog)
        embedding_search.obtener_servicios_normalizados = lambda: normalized
    except Exception:
        pass


def local_chat(text, disable_semantic=True):
    from api import app

    if disable_semantic:
        app.busqueda_semantica_disponible = lambda: False

    if not getattr(local_chat, "_catalog_cached", False):
        try:
            cache_catalog(app)
            local_chat._catalog_cached = True
        except Exception:
            local_chat._catalog_cached = False

    with contextlib.redirect_stdout(io.StringIO()):
        return app.chat(app.SearchRequest(texto=text))


def remote_chat(base_url, text, api_key=None, timeout=20):
    headers = {}
    if api_key:
        headers["X-FamyBot-IA-Key"] = api_key

    response = requests.post(
        f"{base_url.rstrip('/')}/chat",
        json={"texto": text},
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def evaluate_case(case, response):
    expected_intent = case["intencion_esperada"]
    expected_intents = normalize_intents(case["intenciones_detectadas_esperadas"])
    expected_action = case["accion_esperada"]

    actual_intent = str(response.get("intencion"))
    actual_intents = normalize_intents(response.get("intenciones_detectadas"))
    actual_action = response.get("accion")

    diffs = []
    if actual_intent != expected_intent:
        diffs.append({
            "field": "intencion",
            "expected": expected_intent,
            "actual": actual_intent,
        })
    if actual_intents != expected_intents:
        diffs.append({
            "field": "intenciones_detectadas",
            "expected": expected_intents,
            "actual": actual_intents,
        })
    if actual_action != expected_action:
        diffs.append({
            "field": "accion",
            "expected": expected_action,
            "actual": actual_action,
        })

    return diffs


def intents_contain_expected(expected, actual):
    expected_set = set(normalize_intents(expected))
    actual_set = set(normalize_intents(actual))
    return expected_set.issubset(actual_set)


def expected_snapshot(case):
    return {
        "intencion": case.get("intencion_esperada"),
        "intenciones_detectadas": normalize_intents(
            case.get("intenciones_detectadas_esperadas")
        ),
        "accion": case.get("accion_esperada"),
    }


def actual_snapshot(response):
    if not response:
        return None

    return {
        "intencion": response.get("intencion"),
        "intenciones_detectadas": normalize_intents(
            response.get("intenciones_detectadas")
        ),
        "accion": response.get("accion"),
    }


def select_cases(cases, offset=1, limit=None, category=None, only_failed=False, report_path=None):
    offset = max(int(offset or 1), 1)
    failed_indexes = load_failed_indexes(report_path) if only_failed else None
    selected = []

    for original_index, case in enumerate(cases, start=1):
        if original_index < offset:
            continue
        if category and case.get("categoria_fuente") != category:
            continue
        if failed_indexes is not None and original_index not in failed_indexes:
            continue
        selected.append((original_index, case))
        if limit and len(selected) >= int(limit):
            break

    return selected


def list_categories(cases):
    counts = {}
    for case in cases:
        category = case.get("categoria_fuente") or "(sin_categoria)"
        counts[category] = counts.get(category, 0) + 1
    for category, count in sorted(counts.items()):
        print(f"{category}: {count}")


def run_cases(selected_cases, base_url=None, api_key=None, fail_fast=False, disable_semantic=True):
    results = []

    for index, case in selected_cases:
        text = case["texto"]
        try:
            if base_url:
                response = remote_chat(base_url, text, api_key=api_key)
            else:
                response = local_chat(text, disable_semantic=disable_semantic)
            diffs = evaluate_case(case, response)
            ok = not diffs
        except Exception as exc:
            response = None
            diffs = [{
                "field": "exception",
                "expected": "respuesta valida",
                "actual": f"{type(exc).__name__}: {exc}",
            }]
            ok = False

        result = {
            "index": index,
            "ok": ok,
            "texto": text,
            "categoria_fuente": case.get("categoria_fuente"),
            "variant_type": case.get("variant_type"),
            "esperado": expected_snapshot(case),
            "obtenido": actual_snapshot(response),
            "diffs": diffs,
            "response": response,
        }
        results.append(result)

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {index:03d}. {text}")
        if diffs:
            for diff in diffs:
                print(
                    f"       {diff['field']}: "
                    f"esperado={diff['expected']!r} obtenido={diff['actual']!r}"
                )
            if response:
                print(
                    "       obtenido resumen: "
                    f"intencion={response.get('intencion')!r} "
                    f"intenciones={response.get('intenciones_detectadas')!r} "
                    f"accion={response.get('accion')!r}"
                )

        if fail_fast and not ok:
            break

    return results


def print_summary(results):
    total = len(results)
    passed = sum(1 for result in results if result["ok"])
    failed = total - passed
    accuracy = passed / total if total else 0.0
    metrics = calculate_metrics(results)
    print("\nReporte:")
    print(f"  total_ejecutados: {total}")
    print(f"  aprobados: {passed}")
    print(f"  fallidos: {failed}")
    print(f"  accuracy_general: {accuracy:.4f}")
    print(f"  accuracy_intencion: {metrics['accuracy_intencion']:.4f}")
    print(f"  accuracy_accion: {metrics['accuracy_accion']:.4f}")
    print(
        "  accuracy_intenciones_detectadas_exacta: "
        f"{metrics['accuracy_intenciones_detectadas_exacta']:.4f}"
    )
    print(
        "  accuracy_intenciones_detectadas_contiene_esperadas: "
        f"{metrics['accuracy_intenciones_detectadas_contiene_esperadas']:.4f}"
    )
    print(f"  fallos_por_categoria: {metrics['fallos_por_categoria_fuente']}")
    print(f"  fallos_por_campo: {metrics['fallos_por_campo']}")

    if failed:
        print("\nDiferencias:")
        for result in results:
            if result["ok"]:
                continue
            print(f"- Caso {result['index']}: {result['texto']}")
            for diff in result["diffs"]:
                print(
                    f"  {diff['field']}: "
                    f"esperado={diff['expected']!r} obtenido={diff['actual']!r}"
                )


def increment_counter(counter, key):
    counter[key] = counter.get(key, 0) + 1


def calculate_metrics(results):
    total = len(results)
    intent_ok = 0
    action_ok = 0
    intents_exact_ok = 0
    intents_contains_ok = 0
    confusion = {}
    failures_by_category = {}
    failures_by_field = {}
    failures_by_action_pair = {}
    failures_by_intent_pair = {}

    for result in results:
        expected = result.get("esperado") or {}
        actual = result.get("obtenido") or {}
        expected_intent = expected.get("intencion")
        actual_intent = actual.get("intencion")

        if actual_intent == expected_intent:
            intent_ok += 1
        if actual.get("accion") == expected.get("accion"):
            action_ok += 1
        if normalize_intents(actual.get("intenciones_detectadas")) == normalize_intents(
            expected.get("intenciones_detectadas")
        ):
            intents_exact_ok += 1
        if intents_contain_expected(
            expected.get("intenciones_detectadas"),
            actual.get("intenciones_detectadas"),
        ):
            intents_contains_ok += 1

        confusion.setdefault(str(expected_intent), {})
        increment_counter(confusion[str(expected_intent)], str(actual_intent))

        if not result["ok"]:
            increment_counter(
                failures_by_category,
                result.get("categoria_fuente") or "(sin_categoria)",
            )
            increment_counter(
                failures_by_action_pair,
                f"{expected.get('accion')} -> {actual.get('accion')}",
            )
            increment_counter(
                failures_by_intent_pair,
                f"{expected_intent} -> {actual_intent}",
            )
            for diff in result.get("diffs", []):
                increment_counter(failures_by_field, diff.get("field", "unknown"))

    divisor = total or 1
    return {
        "accuracy_intencion": round(intent_ok / divisor, 4),
        "accuracy_accion": round(action_ok / divisor, 4),
        "accuracy_intenciones_detectadas_exacta": round(intents_exact_ok / divisor, 4),
        "accuracy_intenciones_detectadas_contiene_esperadas": round(
            intents_contains_ok / divisor,
            4,
        ),
        "matriz_confusion_intencion": confusion,
        "fallos_por_categoria_fuente": dict(sorted(failures_by_category.items())),
        "fallos_por_campo": dict(sorted(failures_by_field.items())),
        "fallos_por_accion_esperada_vs_obtenida": dict(
            sorted(failures_by_action_pair.items())
        ),
        "fallos_por_intencion_esperada_vs_obtenida": dict(
            sorted(failures_by_intent_pair.items())
        ),
    }


def build_failure_summary(failures, limit=20):
    summary = []
    for failure in failures[:limit]:
        summary.append({
            "index": failure["index"],
            "categoria_fuente": failure.get("categoria_fuente"),
            "texto": failure["texto"],
            "esperado": failure.get("esperado"),
            "obtenido": failure.get("obtenido"),
            "campos": [
                diff.get("field")
                for diff in failure.get("diferencias", [])
            ],
        })
    return summary


def build_report(results, filters):
    total = len(results)
    passed = sum(1 for result in results if result["ok"])
    failed = total - passed
    failures = []

    for result in results:
        if result["ok"]:
            continue
        failures.append({
            "index": result["index"],
            "texto": result["texto"],
            "categoria_fuente": result.get("categoria_fuente"),
            "variant_type": result.get("variant_type"),
            "esperado": result.get("esperado"),
            "obtenido": result.get("obtenido"),
            "diferencias": result.get("diffs", []),
        })

    metrics = calculate_metrics(results)
    return {
        "total_ejecutados": total,
        "aprobados": passed,
        "fallidos": failed,
        "accuracy_general": round(passed / total, 4) if total else 0.0,
        **metrics,
        "filters": filters,
        "top_20_fallos_resumidos": build_failure_summary(failures, 20),
        "fallos": failures,
        "resultados": [
            {
                "index": result["index"],
                "ok": result["ok"],
                "texto": result["texto"],
                "categoria_fuente": result.get("categoria_fuente"),
                "variant_type": result.get("variant_type"),
                "esperado": result.get("esperado"),
                "obtenido": result.get("obtenido"),
                "diferencias": result.get("diffs", []),
            }
            for result in results
        ],
    }


def save_report(report, path):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nReporte JSON guardado en: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Valida /chat con mensajes reales de pacientes."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--url", help="URL base de la API. Si se omite, usa app.chat local.")
    parser.add_argument("--base-url", help="Alias de --url.")
    parser.add_argument("--api-key", help="Valor para X-FamyBot-IA-Key al usar --url.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument("--category", help="Filtra por categoria_fuente.")
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Lista categorias disponibles y termina sin ejecutar pruebas.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Repite solo casos fallidos del reporte indicado por --output.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-semantic",
        action="store_true",
        help="No fuerza fallback lexical/fuzzy en modo local.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if not cases:
        print(f"No hay casos en {args.cases}")
        return 1

    if args.list_categories:
        list_categories(cases)
        return 0

    base_url = args.base_url or args.url
    selected_cases = select_cases(
        cases,
        offset=args.offset,
        limit=args.limit,
        category=args.category,
        only_failed=args.only_failed,
        report_path=args.output,
    )
    if not selected_cases:
        print("No hay casos para ejecutar con los filtros indicados.")
        return 1

    results = run_cases(
        selected_cases,
        base_url=base_url,
        api_key=args.api_key,
        fail_fast=args.fail_fast,
        disable_semantic=not args.allow_semantic,
    )
    print_summary(results)
    report = build_report(
        results,
        {
            "cases": str(args.cases),
            "offset": args.offset,
            "limit": args.limit,
            "category": args.category,
            "only_failed": args.only_failed,
            "base_url": base_url,
            "allow_semantic": args.allow_semantic,
        },
    )
    save_report(report, args.output)
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
