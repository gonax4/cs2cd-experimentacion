import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


RUTA_VALIDACION = Path(
    "datos/procesado/caracteristicas_validacion.csv"
)

RUTA_PREDICTORES = Path(
    "salidas/regresion_logistica/predictores_finales_72.txt"
)

RUTA_MODELO_RL = Path(
    "modelos/regresion_logistica_best_pipeline.joblib"
)

RUTA_MODELO_RF = Path(
    "modelos/random_forest_best_pipeline.joblib"
)

CARPETA_SALIDA = Path(
    "salidas/seleccion_umbral"
)

UMBRALES = np.round(
    np.arange(0.10, 0.91, 0.01),
    2
)


def evaluar_umbral(y_true, probabilidades, umbral):
    y_pred = (
        probabilidades >= umbral
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    especificidad = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "umbral": float(umbral),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),
        "especificidad": float(
            especificidad
        ),
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
        "predicciones_positivas": int(
            (y_pred == 1).sum()
        ),
        "predicciones_negativas": int(
            (y_pred == 0).sum()
        ),
    }


def seleccionar_umbral(tabla):
    max_f1 = tabla["f1"].max()

    candidatos = tabla[
        np.isclose(
            tabla["f1"],
            max_f1,
            rtol=0,
            atol=1e-12
        )
    ].copy()

    min_fp = candidatos[
        "falsos_positivos"
    ].min()

    candidatos = candidatos[
        candidatos["falsos_positivos"]
        == min_fp
    ].copy()

    max_especificidad = candidatos[
        "especificidad"
    ].max()

    candidatos = candidatos[
        np.isclose(
            candidatos["especificidad"],
            max_especificidad,
            rtol=0,
            atol=1e-12
        )
    ].copy()

    empate_residual = len(candidatos) > 1

    candidatos = candidatos.sort_values(
        by="umbral",
        ascending=True
    )

    seleccionado = candidatos.iloc[0]

    return seleccionado, candidatos, empate_residual


def evaluar_modelo(
    modelo,
    X_validacion,
    y_validacion,
):
    probabilidades = modelo.predict_proba(
        X_validacion
    )[:, 1]

    resultados = []

    for umbral in UMBRALES:
        resultados.append(
            evaluar_umbral(
                y_validacion,
                probabilidades,
                umbral
            )
        )

    tabla = pd.DataFrame(
        resultados
    )

    seleccionado, empates, empate_residual = (
        seleccionar_umbral(tabla)
    )

    return (
        probabilidades,
        tabla,
        seleccionado,
        empates,
        empate_residual,
    )


print("SELECCIÓN DE UMBRAL EN VALIDACIÓN EXTERNA")
print("=========================================\n")

rutas_requeridas = [
    RUTA_VALIDACION,
    RUTA_PREDICTORES,
    RUTA_MODELO_RL,
    RUTA_MODELO_RF,
]

for ruta in rutas_requeridas:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo requerido: {ruta}"
        )

CARPETA_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

validacion = pd.read_csv(
    RUTA_VALIDACION
)

if set(
    validacion["particion"]
    .dropna()
    .unique()
) != {"VALIDACION"}:
    raise ValueError(
        "El archivo contiene observaciones que no pertenecen "
        "únicamente a VALIDACION."
    )

with open(
    RUTA_PREDICTORES,
    "r",
    encoding="utf-8"
) as f:
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
    if c not in validacion.columns
]

if faltantes:
    raise ValueError(
        "Faltan predictores en VALIDACION: "
        + ", ".join(faltantes)
    )

X_val = validacion[
    predictores
].copy()

y_val = validacion[
    "etiqueta"
].astype(int).copy()

if X_val.isna().any().any():
    raise ValueError(
        "VALIDACION contiene valores faltantes en los "
        "predictores congelados."
    )

if not np.isfinite(
    X_val.to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "VALIDACION contiene infinitos en los predictores."
    )

modelo_rl = joblib.load(
    RUTA_MODELO_RL
)

modelo_rf = joblib.load(
    RUTA_MODELO_RF
)

print(f"Filas VALIDACION: {len(validacion):,}")
print(f"Partidas VALIDACION: {validacion['partida_id'].nunique():,}")
print(f"Predictores congelados: {len(predictores)}")
print(f"Umbrales evaluados por modelo: {len(UMBRALES)}")
print("Rango: 0.10 a 0.90")
print("Paso: 0.01")

print("\nEvaluando Regresión Logística...")

(
    prob_rl,
    tabla_rl,
    seleccionado_rl,
    empates_rl,
    empate_residual_rl,
) = evaluar_modelo(
    modelo_rl,
    X_val,
    y_val,
)

print("Evaluando Random Forest...")

(
    prob_rf,
    tabla_rf,
    seleccionado_rf,
    empates_rf,
    empate_residual_rf,
) = evaluar_modelo(
    modelo_rf,
    X_val,
    y_val,
)

tabla_rl.to_csv(
    CARPETA_SALIDA
    / "umbrales_regresion_logistica.csv",
    index=False
)

tabla_rf.to_csv(
    CARPETA_SALIDA
    / "umbrales_random_forest.csv",
    index=False
)

pd.DataFrame({
    "clave": validacion["clave"],
    "partida_id": validacion["partida_id"],
    "etiqueta_real": y_val,
    "probabilidad_rl": prob_rl,
    "probabilidad_rf": prob_rf,
}).to_csv(
    CARPETA_SALIDA
    / "probabilidades_validacion.csv",
    index=False
)

if empate_residual_rl:
    empates_rl.to_csv(
        CARPETA_SALIDA
        / "empate_residual_rl.csv",
        index=False
    )

if empate_residual_rf:
    empates_rf.to_csv(
        CARPETA_SALIDA
        / "empate_residual_rf.csv",
        index=False
    )

resumen_modelos = pd.DataFrame([
    {
        "modelo": "Regresion Logistica",
        **seleccionado_rl.to_dict(),
        "empate_residual": bool(
            empate_residual_rl
        ),
    },
    {
        "modelo": "Random Forest",
        **seleccionado_rf.to_dict(),
        "empate_residual": bool(
            empate_residual_rf
        ),
    },
])

resumen_modelos.to_csv(
    CARPETA_SALIDA
    / "umbrales_seleccionados.csv",
    index=False
)

resumen_json = {
    "regresion_logistica": {
        "umbral": float(
            seleccionado_rl["umbral"]
        ),
        "f1_validacion": float(
            seleccionado_rl["f1"]
        ),
        "precision_validacion": float(
            seleccionado_rl["precision"]
        ),
        "recall_validacion": float(
            seleccionado_rl["recall"]
        ),
        "especificidad_validacion": float(
            seleccionado_rl["especificidad"]
        ),
        "falsos_positivos": int(
            seleccionado_rl["falsos_positivos"]
        ),
        "falsos_negativos": int(
            seleccionado_rl["falsos_negativos"]
        ),
        "empate_residual": bool(
            empate_residual_rl
        ),
    },
    "random_forest": {
        "umbral": float(
            seleccionado_rf["umbral"]
        ),
        "f1_validacion": float(
            seleccionado_rf["f1"]
        ),
        "precision_validacion": float(
            seleccionado_rf["precision"]
        ),
        "recall_validacion": float(
            seleccionado_rf["recall"]
        ),
        "especificidad_validacion": float(
            seleccionado_rf["especificidad"]
        ),
        "falsos_positivos": int(
            seleccionado_rf["falsos_positivos"]
        ),
        "falsos_negativos": int(
            seleccionado_rf["falsos_negativos"]
        ),
        "empate_residual": bool(
            empate_residual_rf
        ),
    },
    "reglas_seleccion": [
        "Mayor F1 en VALIDACION externa",
        "Si hay empate: menor numero de falsos positivos",
        "Si persiste: mayor especificidad",
        (
            "Si persiste un empate exacto despues de las "
            "reglas anteriores, se conserva el menor umbral "
            "solo como regla determinista de reproducibilidad."
        ),
    ],
    "rango_umbral": {
        "inicio": 0.10,
        "fin": 0.90,
        "paso": 0.01,
    },
}

with open(
    CARPETA_SALIDA
    / "umbrales_seleccionados.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        resumen_json,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n\nSELECCIÓN FINALIZADA")
print("====================")

print("\nREGRESIÓN LOGÍSTICA")
print("-------------------")
print(
    f"Umbral seleccionado: "
    f"{seleccionado_rl['umbral']:.2f}"
)
print(
    f"F1 VALIDACION: "
    f"{seleccionado_rl['f1']:.6f}"
)
print(
    f"Precision: "
    f"{seleccionado_rl['precision']:.6f}"
)
print(
    f"Recall: "
    f"{seleccionado_rl['recall']:.6f}"
)
print(
    f"Especificidad: "
    f"{seleccionado_rl['especificidad']:.6f}"
)
print(
    f"FP: {int(seleccionado_rl['falsos_positivos'])}"
)
print(
    f"FN: {int(seleccionado_rl['falsos_negativos'])}"
)
print(
    f"Empate residual: {empate_residual_rl}"
)

print("\nRANDOM FOREST")
print("-------------")
print(
    f"Umbral seleccionado: "
    f"{seleccionado_rf['umbral']:.2f}"
)
print(
    f"F1 VALIDACION: "
    f"{seleccionado_rf['f1']:.6f}"
)
print(
    f"Precision: "
    f"{seleccionado_rf['precision']:.6f}"
)
print(
    f"Recall: "
    f"{seleccionado_rf['recall']:.6f}"
)
print(
    f"Especificidad: "
    f"{seleccionado_rf['especificidad']:.6f}"
)
print(
    f"FP: {int(seleccionado_rf['falsos_positivos'])}"
)
print(
    f"FN: {int(seleccionado_rf['falsos_negativos'])}"
)
print(
    f"Empate residual: {empate_residual_rf}"
)

print("\nArchivos guardados en:")
print(CARPETA_SALIDA)

print(
    "\nIMPORTANTE: los umbrales quedan congelados después "
    "de esta ejecución."
)
print(
    "TEST todavía NO ha sido cargado ni utilizado."
)
