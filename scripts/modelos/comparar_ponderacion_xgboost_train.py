"""Comparación controlada de XGBoost con y sin ponderación, SOLO TRAIN.

Se mantienen los 72 predictores, los cinco folds agrupados por partida,
los hiperparámetros de XGBoost y el umbral de decisión de 0.50.
La única diferencia entre las dos variantes es scale_pos_weight.

Experimento exploratorio: la configuración base fue seleccionada previamente
mediante CV sobre TRAIN. No se abre VALIDACION ni TEST, ni se reemplaza el
modelo XGBoost ya entrenado.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgb_estimador import XGBPonderado


TRAIN = Path("datos/procesado/caracteristicas_train.csv")
PREDICTORES = Path("salidas/regresion_logistica/predictores_finales_72.txt")
MAPA_FOLDS = Path("salidas/regresion_logistica/mapa_partidas_folds_internos.csv")
CONFIG_XGB = Path("salidas/xgboost/mejor_configuracion_xgb.json")
SALIDA = Path("salidas/xgboost/ponderacion_train")
UMBRAL = 0.50


def cargar():
    for archivo in (TRAIN, PREDICTORES, MAPA_FOLDS, CONFIG_XGB):
        if not archivo.is_file():
            raise FileNotFoundError(f"Falta {archivo}")

    datos = pd.read_csv(TRAIN)
    if set(datos["particion"].dropna().unique()) != {"TRAIN"}:
        raise ValueError("El experimento solo puede utilizar TRAIN")

    nombres = [
        s.strip() for s in PREDICTORES.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    if len(nombres) != 72 or len(set(nombres)) != 72:
        raise ValueError("Se necesitan los 72 predictores únicos originales")
    if set(nombres) - set(datos.columns):
        raise ValueError("Faltan predictores en TRAIN")

    X = datos[nombres].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Se encontraron NaN o infinitos")
    y = datos["etiqueta"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Las dos clases deben estar presentes")

    mapa = pd.read_csv(MAPA_FOLDS)
    if not {"partida_id", "fold_validacion"}.issubset(mapa.columns):
        raise ValueError("Columnas ausentes en el mapa de folds")
    if (mapa["partida_id"].duplicated().any()
            or set(mapa["fold_validacion"]) != {1, 2, 3, 4, 5}):
        raise ValueError("El mapa de folds es inválido")
    asignacion = dict(zip(
        mapa["partida_id"].astype(str), mapa["fold_validacion"].astype(int)
    ))
    grupos = datos["partida_id"].astype(str)
    if set(grupos.unique()) != set(asignacion):
        raise ValueError("El mapa de folds no coincide con TRAIN")
    fold_por_fila = grupos.map(asignacion).to_numpy()
    folds = []
    for numero in range(1, 6):
        indices_tr = np.flatnonzero(fold_por_fila != numero)
        indices_val = np.flatnonzero(fold_por_fila == numero)
        if not len(indices_tr) or not len(indices_val):
            raise ValueError(f"Fold {numero} vacío")
        if set(grupos.iloc[indices_tr]) & set(grupos.iloc[indices_val]):
            raise ValueError("Fuga de información entre partidas")
        folds.append((numero, indices_tr, indices_val))

    configuracion = json.loads(CONFIG_XGB.read_text(encoding="utf-8"))
    parametros = {
        clave.removeprefix("modelo__"): valor
        for clave, valor in configuracion["best_params"].items()
    }
    parametros.pop("balancear", None)
    if set(parametros) - set(XGBPonderado().get_params()):
        raise ValueError("La configuración contiene parámetros desconocidos")
    return X, y, folds, parametros


def metricas(y, prob):
    pred = (prob >= UMBRAL).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {
        "ventanas": len(y),
        "positivas": tp + fn,
        "negativas": fp + tn,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def main():
    X, y, folds, parametros = cargar()
    filas_folds, filas_resumen = [], []
    probabilidades_oof = {}
    SALIDA.mkdir(parents=True, exist_ok=True)
    print(
        f"TRAIN: {len(y)} ventanas, 72 características, "
        "5 folds por partida. Umbral fijo: 0.50.",
        flush=True,
    )

    for balancear in (False, True):
        nombre = "con_ponderacion" if balancear else "sin_ponderacion"
        prob_oof = np.full(len(y), np.nan)
        for numero, indices_tr, indices_val in folds:
            modelo = XGBPonderado(**parametros, balancear=balancear)
            modelo.fit(X.iloc[indices_tr], y[indices_tr])
            prob = modelo.predict_proba(X.iloc[indices_val])[:, 1]
            prob_oof[indices_val] = prob

            resultado = metricas(y[indices_val], prob)
            filas_folds.append({
                "configuracion": nombre,
                "fold": numero,
                "scale_pos_weight_fit": float(modelo.scale_pos_weight_),
                **resultado,
            })
            print(
                f"{nombre} | fold {numero} | "
                f"peso={modelo.scale_pos_weight_:.3f} | "
                f"F1={resultado['f1']:.4f} | "
                f"recall={resultado['recall']:.4f} | "
                f"FP={resultado['fp']}",
                flush=True,
            )

        if not np.isfinite(prob_oof).all():
            raise RuntimeError("No se generaron predicciones para todas las filas")
        probabilidades_oof[nombre] = prob_oof.copy()

        # El resumen agregado utiliza exactamente una predicción fuera de fold
        # por ventana. Las medias por fold se calculan por separado.
        filas_resumen.append({
            "configuracion": nombre,
            "umbral": UMBRAL,
            **metricas(y, prob_oof),
        })

    # Conservar probabilidades fuera de fold para analizar curvas PR sin
    # repetir los 10 entrenamientos. Cada fila procede de un fold donde la
    # partida correspondiente no participó en el entrenamiento.
    fold_por_fila = np.zeros(len(y), dtype=int)
    for numero, _, indices_val in folds:
        if (fold_por_fila[indices_val] != 0).any():
            raise ValueError("Una ventana figura en varios folds")
        fold_por_fila[indices_val] = numero
    if (fold_por_fila == 0).any():
        raise ValueError("Hay ventanas sin fold")
    predicciones_oof = pd.DataFrame({
        "indice_fila": np.arange(len(y), dtype=int),
        "fold": fold_por_fila,
        "etiqueta_real": y,
        "probabilidad_sin_ponderacion": probabilidades_oof["sin_ponderacion"],
        "probabilidad_con_ponderacion": probabilidades_oof["con_ponderacion"],
    })
    predicciones_oof.to_csv(
        SALIDA / "predicciones_oof_train.csv", index=False
    )

    por_fold = pd.DataFrame(filas_folds)
    resumen = pd.DataFrame(filas_resumen)
    medias = por_fold.groupby("configuracion", as_index=False).agg(
        media_f1=("f1", "mean"),
        media_recall=("recall", "mean"),
        media_precision=("precision", "mean"),
        media_roc_auc=("roc_auc", "mean"),
        media_pr_auc=("pr_auc", "mean"),
        desviacion_f1=("f1", "std"),
        desviacion_recall=("recall", "std"),
    )
    resumen = resumen.merge(medias, on="configuracion")
    por_fold.to_csv(SALIDA / "metricas_por_fold.csv", index=False)
    resumen.to_csv(SALIDA / "comparacion_ponderacion.csv", index=False)

    notas = {
        "tipo": "Ablación de ponderación exploratoria, no confirmatoria",
        "datos": "Solo TRAIN. Ni VALIDACION ni TEST fueron abiertos",
        "unidad": "Ventana, folds separados por partida",
        "folds": 5,
        "predictores": 72,
        "umbral": UMBRAL,
        "hiperparametros_fijos": parametros,
        "versiones": ["scale_pos_weight=1.0", "scale_pos_weight=N_neg/N_pos"],
        "control_de_fuga": (
            "El peso se calcula solo con las etiquetas de entrenamiento "
            "de cada fold, nunca con las de validación del mismo fold."
        ),
        "limitacion": (
            "Los hiperparámetros fueron elegidos originalmente con una "
            "búsqueda que incluyó ponderación, por lo que el contraste "
            "aisla el peso dado ese conjunto fijo de hiperparámetros, "
            "pero no determina la mejor configuración posible sin ponderar."
        ),
    }
    (SALIDA / "notas_metodologicas.json").write_text(
        json.dumps(notas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nComparación fuera de fold en TRAIN:")
    print(
        resumen[
            ["configuracion", "media_f1", "media_recall",
             "media_precision", "tp", "fn", "fp", "tn"]
        ].to_string(index=False)
    )
    print("\nResultados guardados en:", SALIDA)
    print("Probabilidades OOF disponibles para 18_graficar_curvas_pr_train.py")
    print("No se modificó el modelo original. No se abrió VALIDACION ni TEST.")


if __name__ == "__main__":
    main()
