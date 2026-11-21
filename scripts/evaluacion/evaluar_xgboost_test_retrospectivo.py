"""Evaluación retrospectiva de XGBoost sobre TEST (solo una ejecución).

Se agrega al estudio original de RL y RF SIN volver a evaluarlos ni modificar
sus resultados históricos. La selección del umbral debe constar previamente en
VALIDACION y se usa SIN nuevos ajustes (0.66 en la ejecución registrada).

LIMITACIÓN: TEST ya se había observado para RL/RF antes de introducir XGBoost.
Esta comparación es retrospectiva/exploratoria, NO una evaluación independiente
ciega de los tres modelos. No usar TEST para modificar el método o elegir umbral.
"""
import __main__
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

# Para recuperar el estimador XGBoost guardado, incluso si fue serializado
# como __main__.XGBPonderado por el entrenamiento inicial.
DIRECTORIO_MODELOS = Path(__file__).resolve().parents[1] / "modelos"
if str(DIRECTORIO_MODELOS) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_MODELOS))
from xgb_estimador import XGBPonderado

TEST = Path("datos/procesado/caracteristicas_test.csv")
PREDICTORES = Path("salidas/regresion_logistica/predictores_finales_72.txt")
MODELO_XGB = Path("modelos/xgboost_best_pipeline.joblib")
VALIDACION_XGB = Path("salidas/xgboost/validacion/metricas_seleccionadas.json")
HISTORICO_RL_RF = Path("salidas/evaluacion_final_test/metricas_finales_test.csv")
MARCADOR_HISTORICO = Path("salidas/evaluacion_final_test/TEST_EVALUADO.txt")

SALIDA = Path("salidas/evaluacion_xgboost_test_retrospectiva")
MARCADOR_XGB = SALIDA / "TEST_XGBOOST_EVALUADO.txt"
NOMBRES_SALIDA = (
    "metricas_xgboost_test.csv",
    "metricas_xgboost_test.json",
    "predicciones_xgboost_test.csv",
    "matriz_confusion_xgboost_test.png",
    "comparacion_rl_rf_xgb_test_retrospectiva.csv",
)


def main():
    # Estas comprobaciones van ANTES de leer TEST.
    if not MARCADOR_HISTORICO.is_file():
        raise RuntimeError(
            "No se encontró el marcador de TEST histórico de RL/RF. "
            "Este script solo complementa esa evaluación ya realizada."
        )
    if MARCADOR_XGB.exists():
        raise RuntimeError(
            "Ya existe una evaluación de XGBoost en TEST. No se repetirá: "
            f"{MARCADOR_XGB}"
        )
    if SALIDA.exists():
        previos = [SALIDA / n for n in NOMBRES_SALIDA if (SALIDA / n).exists()]
        if previos:
            raise RuntimeError(
                "Hay archivos de una posible ejecución parcial en "
                f"{SALIDA}. Revísalos antes de repetir: {previos}"
            )
    for archivo in (TEST, PREDICTORES, MODELO_XGB, VALIDACION_XGB, HISTORICO_RL_RF):
        if not archivo.is_file():
            raise FileNotFoundError(f"Falta un archivo requerido: {archivo}")

    umbrales = json.loads(VALIDACION_XGB.read_text(encoding="utf-8"))
    if umbrales.get("conjunto") != "VALIDACION" or umbrales.get("modelo") != "XGBoost":
        raise ValueError("No se encontró una selección de XGBoost en VALIDACION")
    umbral = float(umbrales["umbral"])
    if not np.isfinite(umbral) or not (0.0 < umbral < 1.0):
        raise ValueError("Umbral previamente seleccionado inválido")
    # Esta versión corresponde al único modelo XGBoost ya evaluado en VALIDACION.
    # No reemplazar 0.66 por un valor elegido al observar TEST.
    if not np.isclose(umbral, 0.66, rtol=0, atol=1e-9):
        raise ValueError(
            f"Se esperaba el umbral 0.66 validado anteriormente, no {umbral}. "
            "Verificar procedencia antes de abrir TEST."
        )

    nombres = [s.strip() for s in PREDICTORES.read_text(
        encoding="utf-8").splitlines() if s.strip()]
    if len(nombres) != 72 or len(set(nombres)) != 72:
        raise ValueError("La lista congelada debe contener 72 predictores únicos")

    historico = pd.read_csv(HISTORICO_RL_RF)
    necesarios = {"modelo", "f1", "recall", "precision", "roc_auc", "umbral_fijo"}
    if not necesarios.issubset(historico.columns):
        raise ValueError("La tabla histórica de RL/RF no tiene las columnas esperadas")
    if len(historico) != 2 or set(historico["modelo"].astype(str)) != {
        "Regresion Logistica", "Random Forest"
    }:
        raise ValueError("La tabla histórica no contiene exactamente RL y RF")

    datos = pd.read_csv(TEST)
    if "particion" not in datos or datos["particion"].isna().any() or (
        set(datos["particion"].unique()) != {"TEST"}
    ):
        raise ValueError("Solo se admiten observaciones de TEST")
    for columna in ("clave", "partida_id", "etiqueta"):
        if columna not in datos or datos[columna].isna().any():
            raise ValueError(f"TEST no incluye metadatos válidos: {columna}")
    if set(nombres) - set(datos.columns):
        raise ValueError("Faltan predictores congelados en TEST")
    X = datos[nombres].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        raise ValueError("Hay NaN o infinitos en los predictores de TEST")
    y = datos["etiqueta"].astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("TEST debe tener ambas clases 0 y 1")

    __main__.XGBPonderado = XGBPonderado
    modelo = joblib.load(MODELO_XGB)
    if getattr(modelo, "n_features_in_", 72) != 72:
        raise ValueError("El modelo guardado no usa 72 predictores")
    prob = np.asarray(modelo.predict_proba(X)[:, 1], dtype=float)
    if len(prob) != len(y) or not np.isfinite(prob).all():
        raise ValueError("Probabilidades no válidas")
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("Probabilidades fuera del intervalo [0, 1]")
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(
        y, pred, labels=[0, 1]).ravel())
    resultado = {
        "modelo": "XGBoost",
        "umbral_fijo": umbral,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "especificidad": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "verdaderos_negativos": tn,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "verdaderos_positivos": tp,
    }

    # Los archivos históricos de TEST no se sobrescriben en ningún caso.
    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([resultado]).to_csv(
        SALIDA / "metricas_xgboost_test.csv", index=False)
    pd.DataFrame({
        "clave": datos["clave"],
        "partida_id": datos["partida_id"],
        "etiqueta_real": y,
        "probabilidad_xgboost": prob,
        "prediccion_xgboost": pred,
    }).to_csv(SALIDA / "predicciones_xgboost_test.csv", index=False)

    figura, eje = plt.subplots(figsize=(6, 5))
    matriz = np.array([[tn, fp], [fn, tp]])
    eje.imshow(matriz)
    eje.set_title(f"XGBoost - TEST retrospectivo (umbral {umbral:.2f})")
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_xticks((0, 1), labels=("No tramposo", "Tramposo"))
    eje.set_yticks((0, 1), labels=("No tramposo", "Tramposo"))
    for fila in range(2):
        for columna in range(2):
            eje.text(columna, fila, str(matriz[fila, columna]),
                     ha="center", va="center")
    figura.tight_layout()
    figura.savefig(SALIDA / "matriz_confusion_xgboost_test.png", dpi=200)
    plt.close(figura)

    combinado = pd.concat(
        [historico, pd.DataFrame([resultado])], ignore_index=True)
    combinado["tipo_evaluacion"] = [
        "TEST historico RL/RF", "TEST historico RL/RF",
        "TEST retrospectivo XGBoost (no independiente del desarrollo)",
    ]
    combinado.to_csv(
        SALIDA / "comparacion_rl_rf_xgb_test_retrospectiva.csv", index=False,
    )
    (SALIDA / "metricas_xgboost_test.json").write_text(
        json.dumps({
            "resultados": resultado,
            "umbral_procedencia": str(VALIDACION_XGB),
            "modelo_procedencia": str(MODELO_XGB),
            "conjunto": "TEST",
            "alcance": "Evaluación retrospectiva exploratoria, no independiente",
            "limitacion": (
                "TEST se había observado en el estudio de RL/RF antes de "
                "incorporar XGBoost. La comparación posterior no es una "
                "confirmación independiente de tres modelos. Ningún resultado "
                "de TEST debe utilizarse para reajustar el modelo."
            ),
        }, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    MARCADOR_XGB.write_text(
        "XGBoost evaluado retrospectivamente sobre TEST una sola vez.\n"
        "Umbral preseleccionado en VALIDACION: 0.66.\n"
        "Los resultados históricos de RL y RF no fueron modificados.\n"
        "No usar TEST para ajustar o elegir modelos.\n",
        encoding="utf-8",
    )
    print("\nEVALUACIÓN RETROSPECTIVA XGBOOST — TEST")
    print(pd.DataFrame([resultado]).to_string(index=False))
    print("\nComparación descriptiva (NO confirmación independiente):")
    print(combinado[
        ["modelo", "umbral_fijo", "precision", "recall", "f1",
         "falsos_positivos", "falsos_negativos", "roc_auc"]
    ].to_string(index=False))
    print(f"\nArchivos nuevos en: {SALIDA}")
    print("Los resultados históricos originales de RL/RF no se tocaron.")


if __name__ == "__main__":
    main()
