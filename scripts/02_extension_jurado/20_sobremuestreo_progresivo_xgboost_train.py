"""Experimento 20: sobremuestreo progresivo de la clase positiva en TRAIN.

ATENCION: Este procedimiento DUPLICA ejemplos positivos con reemplazo.
Es random oversampling, NO es aumento sintético de telemetría ni utiliza las
ventanas _aug* disponibles en el dataset original. No genera señales nuevas.

Evalúa exclusivamente por los cinco folds agrupados por partida guardados.
Aplica el sobremuestreo solo a la porción de entrenamiento de cada fold;
las ventanas de validación interna permanecen originales e intactas.
XGBoost se entrena sin scale_pos_weight para aislar el efecto del remuestreo.
Hiperparámetros fijos de la configuración del script 12. Umbral fijo 0.50.
No abre VALIDACION externa ni TEST y no reemplaza el modelo final guardado.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from xgb_estimador import XGBPonderado


SEMILLA = 42
RUTA_TRAIN = Path("datos/procesado/caracteristicas_train.csv")
RUTA_PRED = Path("salidas/regresion_logistica/predictores_finales_72.txt")
RUTA_FOLDS = Path("salidas/regresion_logistica/mapa_partidas_folds_internos.csv")
RUTA_CONFIG = Path("salidas/xgboost/mejor_configuracion_xgb.json")
SALIDA = Path("salidas/xgboost/sobremuestreo_train")
UMBRAL = 0.50

# Número de copias adicionales, expresado como fracción del número de
# positivos presentes en TRAIN del fold (0.5 = 50% de positivos adicionales).
NIVELES = (0.0, 0.5, 1.0, 2.0, 3.0)


def cargar():
    for ruta in (RUTA_TRAIN, RUTA_PRED, RUTA_FOLDS, RUTA_CONFIG):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta el archivo {ruta}")
    datos = pd.read_csv(RUTA_TRAIN)
    if set(datos["particion"].dropna().unique()) != {"TRAIN"}:
        raise ValueError("Solo se permite TRAIN")
    nombres = [
        s.strip() for s in RUTA_PRED.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    if len(nombres) != 72 or len(set(nombres)) != 72:
        raise ValueError("Se esperaban 72 predictores únicos")
    if set(nombres) - set(datos.columns):
        raise ValueError("Faltan predictores")
    X = datos[nombres].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Valores no finitos en TRAIN")
    y = datos["etiqueta"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Faltan clases 0 o 1")
    grupos = datos["partida_id"].astype(str)

    mapa = pd.read_csv(RUTA_FOLDS)
    if not {"partida_id", "fold_validacion"}.issubset(mapa.columns):
        raise ValueError("Mapa de folds incompleto")
    if (mapa["partida_id"].duplicated().any()
            or set(mapa["fold_validacion"].unique()) != set(range(1, 6))):
        raise ValueError("Mapa de folds inválido")
    asignacion = dict(zip(
        mapa["partida_id"].astype(str), mapa["fold_validacion"].astype(int)
    ))
    if set(grupos.unique()) != set(asignacion):
        raise ValueError("Las partidas no coinciden con el mapa de folds")
    folds_de_filas = grupos.map(asignacion).to_numpy()
    folds = []
    for n in range(1, 6):
        tr = np.flatnonzero(folds_de_filas != n)
        va = np.flatnonzero(folds_de_filas == n)
        if not len(tr) or not len(va):
            raise ValueError("Fold sin filas")
        if set(grupos.iloc[tr]) & set(grupos.iloc[va]):
            raise ValueError("Fuga de partidas entre entrenamiento y fold")
        if set(np.unique(y[tr])) != {0, 1}:
            raise ValueError("El entrenamiento de un fold necesita las dos clases")
        folds.append((n, tr, va))

    info = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
    params = {
        nombre.removeprefix("modelo__"): valor
        for nombre, valor in info["best_params"].items()
    }
    params["balancear"] = False
    if set(params) - set(XGBPonderado().get_params()):
        raise ValueError("Hiperparámetros desconocidos")
    return X, y, folds, params


def evaluar(y, prob):
    pred = (prob >= UMBRAL).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {
        "n": len(y),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def main():
    X, y, folds, params = cargar()
    SALIDA.mkdir(parents=True, exist_ok=True)
    probabilidades_oof = {
        nivel: np.full(len(y), np.nan) for nivel in NIVELES
    }
    filas = []
    print(
        f"TRAIN: {len(y)} ventanas, 72 predictores, "
        f"5 folds por partida, {len(NIVELES)} niveles. "
        "Sin ponderación adicional.",
        flush=True,
    )

    # Se genera un solo conjunto de copias por fold para que los niveles
    # altos incluyan los mismos ejemplos añadidos en niveles menores.
    for numero, indices_tr, indices_val in folds:
        positivos = indices_tr[y[indices_tr] == 1]
        negativos = indices_tr[y[indices_tr] == 0]
        rng = np.random.default_rng(SEMILLA + numero)
        maximo_extra = int(round(max(NIVELES) * len(positivos)))
        copias_disponibles = rng.choice(
            positivos, size=maximo_extra, replace=True
        )
        for nivel in NIVELES:
            n_extra = int(round(nivel * len(positivos)))
            indice_remuestreado = np.concatenate(
                [indices_tr, copias_disponibles[:n_extra]]
            )
            # Duplicados SOLO en datos de entrenamiento. Los indices de
            # validación interna son los originales del fold y disjuntos.
            if np.intersect1d(
                np.unique(indice_remuestreado), indices_val
            ).size:
                raise RuntimeError("Se mezclaron ejemplos de validación")

            model = XGBPonderado(**params)
            model.fit(X.iloc[indice_remuestreado], y[indice_remuestreado])
            prob = model.predict_proba(X.iloc[indices_val])[:, 1]
            probabilidades_oof[nivel][indices_val] = prob

            metricas = evaluar(y[indices_val], prob)
            registros = {
                "nivel_extra": nivel,
                "fold": numero,
                "positivos_originales_train_fold": len(positivos),
                "negativos_originales_train_fold": len(negativos),
                "positivos_adicionales": n_extra,
                "positivos_efectivos": len(positivos) + n_extra,
                "porcentaje_positivos_tras_sobremuestreo": (
                    (len(positivos) + n_extra) / len(indice_remuestreado) * 100
                ),
                "n_entrenamiento_efectivo": len(indice_remuestreado),
                **metricas,
            }
            filas.append(registros)
            print(
                f"fold {numero} | +{nivel:.0%} positivos | "
                f"F1={metricas['f1']:.4f} "
                f"recall={metricas['recall']:.4f} "
                f"precision={metricas['precision']:.4f} "
                f"FP={metricas['fp']}",
                flush=True,
            )

    por_fold = pd.DataFrame(filas)
    agregados = []
    for nivel in NIVELES:
        prob = probabilidades_oof[nivel]
        if not np.isfinite(prob).all():
            raise RuntimeError(f"Predicciones OOF incompletas, nivel {nivel}")
        agregados.append({"nivel_extra": nivel, **evaluar(y, prob)})

    resumen = pd.DataFrame(agregados)
    medias = por_fold.groupby("nivel_extra", as_index=False).agg(
        media_f1=("f1", "mean"),
        desviacion_f1=("f1", "std"),
        media_recall=("recall", "mean"),
        media_precision=("precision", "mean"),
        media_pr_auc=("pr_auc", "mean"),
        media_roc_auc=("roc_auc", "mean"),
        min_porcentaje_positivo_train=(
            "porcentaje_positivos_tras_sobremuestreo", "min"
        ),
        max_porcentaje_positivo_train=(
            "porcentaje_positivos_tras_sobremuestreo", "max"
        ),
    )
    resumen = resumen.merge(medias, on="nivel_extra")
    por_fold.to_csv(SALIDA / "metricas_por_fold.csv", index=False)
    resumen.to_csv(SALIDA / "comparacion_sobremuestreo.csv", index=False)

    (SALIDA / "notas_metodologicas.json").write_text(
        json.dumps({
            "tipo": "Sobremuestreo aleatorio progresivo (duplicación)",
            "alcance": "Solo TRAIN con validación cruzada agrupada por partida",
            "niveles_positivos_adicionales": NIVELES,
            "predictores": 72,
            "umbral": UMBRAL,
            "hiperparametros_fijos": params,
            "ponderacion_adicional": False,
            "semilla": SEMILLA,
            "restriccion": (
                "Solo se repiten muestras positivas del subconjunto de "
                "entrenamiento del fold; nunca las de su validación."
            ),
            "no_hace": (
                "No transforma series de ticks, no genera observaciones "
                "sintéticas y no utiliza muestras originales _aug*. "
                "Ni VALIDACION ni TEST fueron utilizados."
            ),
            "limitaciones": [
                "Duplicar ventanas no agrega información independiente.",
                "Ventanas de una misma partida pueden estar correlacionadas.",
                "Los hiperparámetros fijos se eligieron con una búsqueda "
                "previa sobre TRAIN e incluían ponderación como opción.",
                "La comparación es exploratoria, no confirmatoria.",
            ],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nResultados agregados fuera de fold:")
    print(resumen[
        ["nivel_extra", "media_f1", "media_recall",
         "media_precision", "pr_auc", "tp", "fn", "fp"]
    ].to_string(index=False))
    print("\nSalidas:", SALIDA)
    print("No se modificaron modelos. No se utilizó VALIDACION ni TEST.")


if __name__ == "__main__":
    main()
