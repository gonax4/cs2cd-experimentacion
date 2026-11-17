"""Evaluación externa exploratoria de XGBoost: SOLO VALIDACION.

Selecciona umbral maximizando F1 (igual al protocolo original).
No abre TEST ni modifica métricas históricas de otros clasificadores.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

VALIDACION = Path("datos/procesado/caracteristicas_validacion.csv")
PREDICTORES = Path("salidas/regresion_logistica/predictores_finales_72.txt")
MODELO = Path("modelos/xgboost_best_pipeline.joblib")
SALIDA = Path("salidas/xgboost/validacion")


def evaluar(y, prob, umbral):
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "umbral": float(umbral),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "especificidad": float(tn / (tn + fp)) if tn + fp else None,
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    }


def main():
    for ruta in (VALIDACION, PREDICTORES, MODELO):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta {ruta}. Consulta README.md")
    datos = pd.read_csv(VALIDACION)
    if set(datos["particion"].dropna().unique()) != {"VALIDACION"}:
        raise ValueError("Este análisis admite solo VALIDACION externa")
    nombres = [s.strip() for s in PREDICTORES.read_text(
        encoding="utf-8").splitlines() if s.strip()]
    if len(nombres) != 72 or len(set(nombres)) != 72:
        raise ValueError("Se esperaban 72 predictores congelados")
    if set(nombres) - set(datos.columns):
        raise ValueError("Predictores ausentes")
    X = datos[nombres].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Hay NaN o infinitos en los predictores")
    y = datos["etiqueta"].astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Las dos clases deben estar presentes")
    modelo = joblib.load(MODELO)
    prob = modelo.predict_proba(X)[:, 1]
    umbrales = np.round(np.arange(0.10, 0.91, 0.01), 2)
    tabla = pd.DataFrame([evaluar(y, prob, u) for u in umbrales])
    # Mismas reglas de desempate del script 8 histórico:
    # mayor F1, menos falsos positivos, mayor especificidad, umbral menor.
    seleccion = tabla.sort_values(
        ["f1", "falsos_positivos", "especificidad", "umbral"],
        ascending=[False, True, False, True],
    ).iloc[0].to_dict()
    seleccion["roc_auc"] = float(roc_auc_score(y, prob))
    seleccion["pr_auc"] = float(average_precision_score(y, prob))
    seleccion["modelo"] = "XGBoost"
    seleccion["conjunto"] = "VALIDACION"
    seleccion["caracter"] = "Exploratorio, no prueba final independiente"
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "metricas_por_umbral.csv", index=False)
    etiquetas = (prob >= seleccion["umbral"]).astype(int)
    pd.DataFrame({
        "clave": datos["clave"],
        "partida_id": datos["partida_id"],
        "etiqueta_real": y,
        "probabilidad_xgb": prob,
        "prediccion_xgb": etiquetas,
        "falso_negativo": (y == 1) & (etiquetas == 0),
    }).to_csv(SALIDA / "predicciones_validacion.csv", index=False)
    (SALIDA / "metricas_seleccionadas.json").write_text(
        json.dumps(seleccion, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("XGBoost — VALIDACION externa")
    print(json.dumps(seleccion, ensure_ascii=False, indent=2))
    print("No se abrió TEST ni se modificaron sus resultados originales.")


if __name__ == "__main__":
    main()
