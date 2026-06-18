import csv
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent.parent
SYNTHETIC_DATASET = BASE_DIR / "dataset" / "dataset_famybot_sintetico_v1.csv"
REAL_DATASET = BASE_DIR / "dataset" / "dataset_famybot_real_v1.csv"
REAL_CASES = BASE_DIR / "tests" / "real_cases_v1.json"
MODEL_DIR = BASE_DIR / "model"
REPORT_PATH = BASE_DIR / "reports" / "training_report_v2.json"
VECTORIZER_PATH = MODEL_DIR / "vectorizer_famybot_v2.pkl"
CLASSIFIER_PATH = MODEL_DIR / "classifier_famybot_v2.pkl"
RANDOM_STATE = 42


def cargar_csv(path):
    registros = []

    with path.open("r", encoding="utf-8-sig", newline="") as archivo:
        reader = csv.DictReader(archivo)
        for fila in reader:
            texto = str(fila.get("texto") or "").strip()
            intencion = str(fila.get("intencion") or "").strip()

            if not texto or not intencion:
                continue

            registros.append({
                "texto": texto,
                "intencion": intencion,
                "fuente": path.name,
            })

    return registros


def cargar_real_cases(path):
    with path.open("r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    casos = data.get("cases", []) if isinstance(data, dict) else data
    registros = []

    for caso in casos:
        texto = str(caso.get("text") or "").strip()
        intencion = caso.get("expected_intent")

        if not intencion and caso.get("expected_intents"):
            intencion = caso["expected_intents"][0]

        intencion = str(intencion or "").strip()

        if not texto or not intencion:
            continue

        registros.append({
            "texto": texto,
            "intencion": intencion,
            "fuente": path.name,
        })

    return registros


def cargar_dataset():
    registros = []
    registros.extend(cargar_csv(SYNTHETIC_DATASET))
    registros.extend(cargar_csv(REAL_DATASET))
    registros.extend(cargar_real_cases(REAL_CASES))

    vistos = set()
    deduplicados = []
    for registro in registros:
        clave = (registro["texto"], registro["intencion"])
        if clave in vistos:
            continue
        vistos.add(clave)
        deduplicados.append(registro)

    return deduplicados


def puede_estratificar(y, test_size=0.2):
    conteo = Counter(y)
    if len(conteo) < 2:
        return False

    if min(conteo.values()) < 2:
        return False

    test_count = max(1, int(round(len(y) * test_size)))
    return test_count >= len(conteo)


def dividir_dataset(textos, etiquetas):
    estratificar = puede_estratificar(etiquetas)
    stratify = etiquetas if estratificar else None

    return (*train_test_split(
        textos,
        etiquetas,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    ), estratificar)


def entrenar(textos_train, y_train):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(textos_train)
    classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    return vectorizer, classifier


def evaluar(vectorizer, classifier, textos_test, y_test):
    X_test = vectorizer.transform(textos_test)
    predicciones = classifier.predict(X_test)
    labels = sorted(set(y_test) | set(predicciones))

    return {
        "accuracy": round(float(accuracy_score(y_test, predicciones)), 4),
        "classification_report": classification_report(
            y_test,
            predicciones,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, predicciones, labels=labels).tolist(),
        },
    }


def main():
    registros = cargar_dataset()
    if not registros:
        print("No hay registros para entrenar.")
        return 1

    textos = [registro["texto"] for registro in registros]
    etiquetas = [registro["intencion"] for registro in registros]
    distribucion = dict(sorted(Counter(etiquetas).items()))

    textos_train, textos_test, y_train, y_test, estratificado = dividir_dataset(
        textos,
        etiquetas,
    )

    vectorizer, classifier = entrenar(textos_train, y_train)
    metricas = evaluar(vectorizer, classifier, textos_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)

    reporte = {
        "version": "v2",
        "total_registros": len(registros),
        "train_registros": len(textos_train),
        "test_registros": len(textos_test),
        "split_estratificado": estratificado,
        "distribucion_intenciones": distribucion,
        "accuracy": metricas["accuracy"],
        "classification_report": metricas["classification_report"],
        "confusion_matrix": metricas["confusion_matrix"],
        "artefactos": {
            "vectorizer": str(VECTORIZER_PATH),
            "classifier": str(CLASSIFIER_PATH),
        },
    }

    with REPORT_PATH.open("w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, ensure_ascii=False, indent=2)

    print(
        "Entrenamiento v2: "
        f"total={len(registros)} "
        f"train={len(textos_train)} "
        f"test={len(textos_test)} "
        f"accuracy={metricas['accuracy']} "
        f"estratificado={estratificado}"
    )
    print(f"Vectorizer: {VECTORIZER_PATH}")
    print(f"Classifier: {CLASSIFIER_PATH}")
    print(f"Reporte: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
