"""Comparación descriptiva de XGBoost a recall equivalente (solo TRAIN).

Emplea probabilidades fuera de fold del experimento 17.
NO reentrena, NO consulta VALIDACION o TEST y NO redefine el modelo final.
Los umbrales se calculan post hoc con las mismas predicciones evaluadas:
solo sirven para analizar el intercambio precision-recall en TRAIN.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

CARPETA = Path("salidas/xgboost/ponderacion_train")
ENTRADA = CARPETA / "predicciones_oof_train.csv"
SALIDA_TABLA = CARPETA / "comparacion_igual_recall.csv"
SALIDA_DIFERENCIAS = CARPETA / "diferencias_igual_recall.csv"
SALIDA_GRAFICO = CARPETA / "comparacion_igual_recall.png"
SALIDA_NOTAS = CARPETA / "notas_igual_recall.json"

OBJETIVOS_RECALL = (0.60, 0.70, 0.73)
VARIANTES = (
    ("sin_ponderacion", "probabilidad_sin_ponderacion", "Sin ponderación"),
    ("con_ponderacion", "probabilidad_con_ponderacion", "Con ponderación"),
)


def leer():
    if not ENTRADA.is_file():
        raise FileNotFoundError(
            f"Falta {ENTRADA}. Ejecuta primero "
            "python 17_comparar_ponderacion_xgboost_train.py"
        )
    datos = pd.read_csv(ENTRADA)
    requeridas = {"indice_fila", "fold", "etiqueta_real"}
    requeridas.update(col for _, col, _ in VARIANTES)
    if requeridas - set(datos.columns):
        raise ValueError("Faltan columnas de probabilidades fuera de fold")
    if datos.empty or datos["indice_fila"].duplicated().any():
        raise ValueError("Índices de ventanas duplicados o tabla vacía")
    indices = datos["indice_fila"].to_numpy()
    if not np.array_equal(np.sort(indices), np.arange(len(datos))):
        raise ValueError("Los índices de las ventanas no coinciden")
    if (datos["fold"].isna().any()
            or set(datos["fold"].unique()) != {1, 2, 3, 4, 5}):
        raise ValueError("El archivo debe contener los cinco folds agrupados")
    if datos["etiqueta_real"].isna().any():
        raise ValueError("Faltan etiquetas reales")
    y = datos["etiqueta_real"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Las etiquetas deben ser 0 y 1")
    for _, columna, _ in VARIANTES:
        probabilidades = datos[columna].to_numpy(dtype=float)
        if (not np.isfinite(probabilidades).all()
                or ((probabilidades < 0) | (probabilidades > 1)).any()):
            raise ValueError(f"Probabilidades inválidas en {columna}")
    return datos, y


def evaluar(y, prob, objetivo, nombre):
    # precision[:-1] y recall[:-1] corresponden a thresholds.
    # Elegimos el umbral MÁS ALTO que todavía alcanza el recall objetivo.
    # Así cada modelo predice el mínimo número de positivos necesario
    # para alcanzar al menos ese recall, incluidos empates en probabilidad.
    precision, recall, thresholds = precision_recall_curve(y, prob)
    elegibles = np.flatnonzero(recall[:-1] >= objetivo)
    if not len(elegibles):
        raise ValueError(f"No se puede alcanzar recall={objetivo}")
    idx = int(elegibles[-1])
    umbral = float(thresholds[idx])
    pred = (prob >= umbral).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    if rec < objetivo - 1e-12:
        raise RuntimeError("El umbral elegido no cumple el recall objetivo")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "recall_objetivo": objetivo,
        "configuracion": nombre,
        "umbral_descriptivo": umbral,
        "recall_observado": rec,
        "precision": prec,
        "f1": f1,
        "verdaderos_positivos": tp,
        "falsos_negativos": fn,
        "falsos_positivos": fp,
        "verdaderos_negativos": tn,
        "tasa_falsos_positivos": fp / (fp + tn),
    }


def main():
    datos, y = leer()
    filas = []
    for objetivo in OBJETIVOS_RECALL:
        for nombre, columna, _ in VARIANTES:
            prob = datos[columna].to_numpy(dtype=float)
            filas.append(evaluar(y, prob, objetivo, nombre))

    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA_TABLA, index=False)
    diferencias = []
    for objetivo in OBJETIVOS_RECALL:
        par = tabla.loc[tabla["recall_objetivo"] == objetivo].set_index(
            "configuracion"
        )
        sin_p = par.loc["sin_ponderacion"]
        con_p = par.loc["con_ponderacion"]
        diferencias.append({
            "recall_objetivo": objetivo,
            "recall_sin": float(sin_p["recall_observado"]),
            "recall_con": float(con_p["recall_observado"]),
            "diferencia_recall_pp_con_menos_sin": (
                float(con_p["recall_observado"] - sin_p["recall_observado"]) * 100
            ),
            "precision_sin": float(sin_p["precision"]),
            "precision_con": float(con_p["precision"]),
            "diferencia_precision_pp_con_menos_sin": (
                float(con_p["precision"] - sin_p["precision"]) * 100
            ),
            "fp_sin": int(sin_p["falsos_positivos"]),
            "fp_con": int(con_p["falsos_positivos"]),
            "diferencia_fp_con_menos_sin": (
                int(con_p["falsos_positivos"] - sin_p["falsos_positivos"])
            ),
            "umbral_sin": float(sin_p["umbral_descriptivo"]),
            "umbral_con": float(con_p["umbral_descriptivo"]),
        })
    comparacion = pd.DataFrame(diferencias)
    comparacion.to_csv(SALIDA_DIFERENCIAS, index=False)

    # Gráfico centrado en el costo de los falsos positivos a recall comparable.
    x = np.arange(len(OBJETIVOS_RECALL))
    ancho = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.5))
    valores = []
    for nombre, _, etiqueta in VARIANTES:
        orden = tabla.loc[tabla["configuracion"] == nombre].sort_values(
            "recall_objetivo"
        )
        valores.append(orden["falsos_positivos"].to_numpy())
    barras_1 = ax.bar(x - ancho / 2, valores[0], ancho, label="Sin ponderación")
    barras_2 = ax.bar(x + ancho / 2, valores[1], ancho, label="Con ponderación")
    ax.bar_label(barras_1, padding=3, fontsize=9)
    ax.bar_label(barras_2, padding=3, fontsize=9)
    ax.set(
        title="Falsos positivos a recall objetivo equivalente (TRAIN, OOF)",
        xlabel="Recall mínimo objetivo",
        ylabel="Número de falsos positivos",
        xticks=x,
        xticklabels=[f"{v:.0%}" for v in OBJETIVOS_RECALL],
    )
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5, 0.015,
        "Umbrales calculados post hoc. Análisis descriptivo, NO validación independiente.",
        fontsize=9, color="dimgray", ha="center",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(SALIDA_GRAFICO, dpi=220, bbox_inches="tight")
    plt.close(fig)

    SALIDA_NOTAS.write_text(
        json.dumps({
            "datos": "Predicciones fuera de fold de TRAIN (script 17)",
            "validacion_externa": "No se utiliza VALIDACION ni TEST",
            "objetivos_recall": OBJETIVOS_RECALL,
            "criterio_umbral": (
                "Mayor umbral que logra recall >= objetivo, "
                "incluyendo empates en las probabilidades."
            ),
            "observacion": (
                "Los recalls alcanzados pueden diferir ligeramente por la "
                "naturaleza discreta de las ventanas y los empates."
            ),
            "limitacion": (
                "Umbrales derivados de las mismas predicciones con las que se "
                "calculan las métricas. Las comparaciones son descriptivas y "
                "no prueban mejoras en datos independientes. La selección "
                "original de hiperparámetros también usó TRAIN."
            ),
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Comparación de ponderación a igual recall objetivo (TRAIN, OOF)")
    print(tabla[
        ["recall_objetivo", "configuracion", "umbral_descriptivo",
         "recall_observado", "precision", "f1",
         "falsos_positivos", "falsos_negativos"]
    ].to_string(index=False))
    print("\nDiferencia CON menos SIN ponderación:")
    print(comparacion[
        ["recall_objetivo", "diferencia_precision_pp_con_menos_sin",
         "diferencia_fp_con_menos_sin"]
    ].to_string(index=False))
    print("\nArchivos generados en:", CARPETA)
    print("No se reentrenó, no se accedió a VALIDACION ni TEST.")
    print("Los umbrales son descriptivos, no una selección independiente.")


if __name__ == "__main__":
    main()
