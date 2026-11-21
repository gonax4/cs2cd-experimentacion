import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

RUTA_TEST = Path("datos/procesado/caracteristicas_test.csv")
RUTA_PREDICTORES = Path("salidas/regresion_logistica/predictores_finales_72.txt")
RUTA_MODELO_RL = Path("modelos/regresion_logistica_best_pipeline.joblib")
RUTA_MODELO_RF = Path("modelos/random_forest_best_pipeline.joblib")
RUTA_UMBRALES = Path("salidas/seleccion_umbral/umbrales_seleccionados.json")

CARPETA_SALIDA = Path("salidas/evaluacion_final_test")
MARCADOR = CARPETA_SALIDA / "TEST_EVALUADO.txt"


def evaluar_modelo(nombre, modelo, X, y, umbral):
    prob = modelo.predict_proba(X)[:, 1]
    pred = (prob >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    especificidad = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    metricas = {
        "modelo": nombre,
        "umbral_fijo": float(umbral),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "especificidad": float(especificidad),
        "roc_auc": float(
            roc_auc_score(y, prob)
        ),
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    }

    return prob, pred, metricas


def guardar_matriz(nombre_archivo, titulo, y, pred):
    matriz = confusion_matrix(
        y, pred, labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(matriz)

    ax.set_title(titulo)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(
        ["No tramposo (0)", "Tramposo (1)"]
    )
    ax.set_yticklabels(
        ["No tramposo (0)", "Tramposo (1)"]
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(matriz[i, j]),
                ha="center", va="center"
            )

    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(
        CARPETA_SALIDA / nombre_archivo,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)


print("EVALUACIÓN FINAL ÚNICA SOBRE TEST")
print("=================================\n")

if MARCADOR.exists():
    raise RuntimeError(
        "TEST ya fue evaluado anteriormente. "
        "Existe salidas/evaluacion_final_test/TEST_EVALUADO.txt. "
        "El script no se ejecutará de nuevo automáticamente."
    )

for ruta in [
    RUTA_TEST,
    RUTA_PREDICTORES,
    RUTA_MODELO_RL,
    RUTA_MODELO_RF,
    RUTA_UMBRALES,
]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo requerido: {ruta}"
        )

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

test = pd.read_csv(RUTA_TEST)

if set(test["particion"].dropna().unique()) != {"TEST"}:
    raise ValueError(
        "El archivo contiene observaciones fuera de TEST."
    )

with open(RUTA_PREDICTORES, "r", encoding="utf-8") as f:
    predictores = [
        linea.strip()
        for linea in f
        if linea.strip()
    ]

if len(predictores) != 72:
    raise ValueError(
        f"Se esperaban 72 predictores y se encontraron "
        f"{len(predictores)}."
    )

faltantes = [
    c for c in predictores
    if c not in test.columns
]

if faltantes:
    raise ValueError(
        "Faltan predictores en TEST: "
        + ", ".join(faltantes)
    )

X_test = test[predictores].copy()
y_test = test["etiqueta"].astype(int).copy()

if X_test.isna().any().any():
    raise ValueError(
        "TEST contiene valores faltantes."
    )

if not np.isfinite(
    X_test.to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "TEST contiene valores infinitos."
    )

modelo_rl = joblib.load(RUTA_MODELO_RL)
modelo_rf = joblib.load(RUTA_MODELO_RF)

with open(RUTA_UMBRALES, "r", encoding="utf-8") as f:
    datos_umbrales = json.load(f)

umbral_rl = float(
    datos_umbrales["regresion_logistica"]["umbral"]
)
umbral_rf = float(
    datos_umbrales["random_forest"]["umbral"]
)

print(f"Filas TEST: {len(test):,}")
print(f"Partidas TEST: {test['partida_id'].nunique():,}")
print(f"Predictores congelados: {len(predictores)}")
print(f"Umbral RL congelado: {umbral_rl:.2f}")
print(f"Umbral RF congelado: {umbral_rf:.2f}")
print(
    "\nNo se modificarán modelos, características ni "
    "umbrales a partir de los resultados de TEST.\n"
)

prob_rl, pred_rl, metricas_rl = evaluar_modelo(
    "Regresion Logistica",
    modelo_rl,
    X_test,
    y_test,
    umbral_rl,
)

prob_rf, pred_rf, metricas_rf = evaluar_modelo(
    "Random Forest",
    modelo_rf,
    X_test,
    y_test,
    umbral_rf,
)

metricas = pd.DataFrame([
    metricas_rl,
    metricas_rf,
])

metricas.to_csv(
    CARPETA_SALIDA / "metricas_finales_test.csv",
    index=False
)

pd.DataFrame({
    "clave": test["clave"],
    "partida_id": test["partida_id"],
    "etiqueta_real": y_test,
    "probabilidad_rl": prob_rl,
    "prediccion_rl": pred_rl,
    "probabilidad_rf": prob_rf,
    "prediccion_rf": pred_rf,
}).to_csv(
    CARPETA_SALIDA / "predicciones_test.csv",
    index=False
)

guardar_matriz(
    "matriz_confusion_rl.png",
    f"Regresión Logística - TEST (umbral={umbral_rl:.2f})",
    y_test,
    pred_rl,
)

guardar_matriz(
    "matriz_confusion_rf.png",
    f"Random Forest - TEST (umbral={umbral_rf:.2f})",
    y_test,
    pred_rf,
)

fpr_rl, tpr_rl, _ = roc_curve(
    y_test, prob_rl
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test, prob_rf
)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    fpr_rl,
    tpr_rl,
    label=(
        "Regresión Logística "
        f"(AUC={metricas_rl['roc_auc']:.3f})"
    )
)

ax.plot(
    fpr_rf,
    tpr_rf,
    label=(
        "Random Forest "
        f"(AUC={metricas_rf['roc_auc']:.3f})"
    )
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Azar"
)

ax.set_xlabel(
    "Tasa de falsos positivos"
)
ax.set_ylabel(
    "Tasa de verdaderos positivos"
)
ax.set_title(
    "Curvas ROC - Evaluación final TEST"
)
ax.legend()
fig.tight_layout()

fig.savefig(
    CARPETA_SALIDA / "curvas_roc_test.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

with open(
    CARPETA_SALIDA / "metricas_finales_test.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "regresion_logistica": metricas_rl,
            "random_forest": metricas_rf,
            "criterio_principal": "F1",
            "nota_metodologica": (
                "TEST fue utilizado únicamente después de "
                "congelar predictores, hiperparámetros y "
                "umbrales. Sus resultados no se utilizarán "
                "para modificar el procedimiento."
            ),
        },
        f,
        ensure_ascii=False,
        indent=2
    )

MARCADOR.write_text(
    (
        "TEST evaluado con predictores, modelos y umbrales "
        "previamente congelados.\n"
        f"Umbral RL: {umbral_rl:.2f}\n"
        f"Umbral RF: {umbral_rf:.2f}\n"
        "No modificar el experimento basándose en TEST.\n"
    ),
    encoding="utf-8"
)

print("EVALUACIÓN FINAL COMPLETADA")
print("===========================\n")

for _, fila in metricas.iterrows():
    print(fila["modelo"])
    print("-" * len(fila["modelo"]))
    print(f"Umbral: {fila['umbral_fijo']:.2f}")
    print(f"Accuracy: {fila['accuracy']:.6f}")
    print(f"Precision: {fila['precision']:.6f}")
    print(f"Recall: {fila['recall']:.6f}")
    print(f"F1: {fila['f1']:.6f}")
    print(
        f"Especificidad: "
        f"{fila['especificidad']:.6f}"
    )
    print(f"ROC-AUC: {fila['roc_auc']:.6f}")
    print(
        f"TN: {int(fila['verdaderos_negativos'])}"
    )
    print(
        f"FP: {int(fila['falsos_positivos'])}"
    )
    print(
        f"FN: {int(fila['falsos_negativos'])}"
    )
    print(
        f"TP: {int(fila['verdaderos_positivos'])}"
    )
    print()

print("Resultados guardados en:")
print(CARPETA_SALIDA)

print(
    "\nTEST queda cerrado. No se deben ajustar "
    "características, hiperparámetros ni umbrales "
    "a partir de estos resultados."
)
