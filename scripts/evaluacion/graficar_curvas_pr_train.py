"""Curvas precision-recall de XGBoost con y sin ponderación.

Utiliza únicamente predicciones FUERA DE FOLD generadas por el script 17
sobre TRAIN. No entrena modelos, elige umbrales ni abre VALIDACION o TEST.
Los resultados son exploratorios, con hiperparámetros fijados previamente.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

CARPETA = Path("salidas/xgboost/ponderacion_train")
ENTRADA = CARPETA / "predicciones_oof_train.csv"
SALIDA_GRAFICO = CARPETA / "curvas_precision_recall.png"
SALIDA_RESUMEN = CARPETA / "resumen_curvas_pr.csv"
UMBRAL_FIJO = 0.50

VARIANTES = [
    ("sin_ponderacion", "probabilidad_sin_ponderacion", "Sin ponderación"),
    ("con_ponderacion", "probabilidad_con_ponderacion", "Con ponderación"),
]


def leer_y_verificar():
    if not ENTRADA.is_file():
        raise FileNotFoundError(
            f"No existe {ENTRADA}. Ejecuta primero "
            "python scripts/modelos/comparar_ponderacion_xgboost_train.py"
        )
    df = pd.read_csv(ENTRADA)
    columnas = {"indice_fila", "fold", "etiqueta_real"}
    columnas.update(col for _, col, _ in VARIANTES)
    if columnas - set(df.columns):
        raise ValueError("Faltan columnas de predicciones fuera de fold")
    if df.empty or df["indice_fila"].duplicated().any():
        raise ValueError("Hay filas vacías o duplicadas")
    if not np.array_equal(
        np.sort(df["indice_fila"].to_numpy()), np.arange(len(df))
    ):
        raise ValueError("Los índices de las ventanas no coinciden")
    if set(df["fold"].dropna().unique()) != {1, 2, 3, 4, 5}:
        raise ValueError("Se esperaban cinco folds agrupados")
    y = df["etiqueta_real"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Se esperaban etiquetas reales 0 y 1")
    for _, columna, _ in VARIANTES:
        prob = df[columna].to_numpy(dtype=float)
        if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
            raise ValueError(f"Probabilidades inválidas en {columna}")
    return df, y


def main():
    datos, y = leer_y_verificar()
    prevalencia = float(y.mean())
    resumen = []
    fig, ax = plt.subplots(figsize=(9, 6))

    for nombre, columna, etiqueta in VARIANTES:
        prob = datos[columna].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y, prob)
        ap = float(average_precision_score(y, prob))
        pred = (prob >= UMBRAL_FIJO).astype(int)
        p50 = float(precision_score(y, pred, zero_division=0))
        r50 = float(recall_score(y, pred, zero_division=0))
        f50 = float(f1_score(y, pred, zero_division=0))
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        resumen.append({
            "configuracion": nombre,
            "n_ventanas": len(y),
            "n_positivas": int(y.sum()),
            "prevalencia_positiva": prevalencia,
            "average_precision_oof": ap,
            "umbral_fijo": UMBRAL_FIJO,
            "precision_umbral_050": p50,
            "recall_umbral_050": r50,
            "f1_umbral_050": f50,
            "falsos_positivos_umbral_050": fp,
            "falsos_negativos_umbral_050": fn,
        })
        ax.plot(
            recall, precision, linewidth=2.0,
            label=f"{etiqueta} (AP={ap:.3f})",
        )
        ax.scatter(
            [r50], [p50], marker="o", s=60, zorder=5,
            label=f"{etiqueta}, umbral 0.50",
        )

    ax.axhline(
        prevalencia, linestyle="--", linewidth=1.4, color="gray",
        label=f"Prevalencia positiva ({prevalencia:.3f})",
    )
    ax.set(
        title="Curvas precisión-recall de XGBoost (TRAIN, fuera de fold)",
        xlabel="Recall (sensibilidad)",
        ylabel="Precisión",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    ax.grid(alpha=0.22)
    ax.legend(loc="lower left", fontsize=9)
    fig.text(
        0.5, 0.01,
        "Cinco folds agrupados por partida. Comparación exploratoria; "
        "no se utilizó VALIDACION ni TEST.",
        fontsize=9, ha="center", color="dimgray",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(SALIDA_GRAFICO, dpi=220, bbox_inches="tight")
    plt.close(fig)

    resultado = pd.DataFrame(resumen)
    resultado.to_csv(SALIDA_RESUMEN, index=False)
    print("Curvas PR: XGBoost con y sin ponderación (solo TRAIN)")
    print(resultado[
        ["configuracion", "average_precision_oof",
         "precision_umbral_050", "recall_umbral_050", "f1_umbral_050",
         "falsos_positivos_umbral_050", "falsos_negativos_umbral_050"]
    ].to_string(index=False))
    print(f"Gráfico: {SALIDA_GRAFICO}")
    print(f"Tabla: {SALIDA_RESUMEN}")
    print(
        "Interpretación: AP mide el ordenamiento a lo largo de umbrales; "
        "el punto 0.50 muestra una decisión operativa. "
        "No se eligió un nuevo umbral ni se utilizó TEST."
    )


if __name__ == "__main__":
    main()
