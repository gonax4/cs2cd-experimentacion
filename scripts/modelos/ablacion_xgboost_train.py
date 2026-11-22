"""Estudio exploratorio de ablación con XGBoost SOLO sobre TRAIN.

HIPÓTESIS GENERADA TRAS OBSERVAR VALIDACION: Se sospecha que algunas
características contextuales se asocian con un recall desigual.
Este análisis es exploratorio, NO es una evaluación confirmatoria.

Usa los mismos cinco folds por partida; nunca lee VALIDACION ni TEST.
Mantiene hiperparámetros y umbral de 0.50 iguales para los cuatro escenarios.
No reemplaza el modelo guardado del script 12.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from xgb_estimador import XGBPonderado

RUTA_TRAIN = Path("datos/procesado/caracteristicas_train.csv")
RUTA_PRED = Path("salidas/regresion_logistica/predictores_finales_72.txt")
RUTA_FOLDS = Path("salidas/regresion_logistica/mapa_partidas_folds_internos.csv")
RUTA_CONFIG = Path("salidas/xgboost/mejor_configuracion_xgb.json")
SALIDA = Path("salidas/xgboost/ablacion_train")
UMBRAL = 0.50

# Se quitan también la proporción y el indicador del rifle para reducir
# la redundancia entre predictores, en lugar de eliminar una sola columna.
EXCLUSIONES = {
    "completo_72": [],
    "sin_wallbang": ["kill_wallbang"],
    "sin_rifle_semi": [
        "weapon_semi_rifle_kill",
        "weapon_semi_rifle_proporcion",
    ],
    "sin_ambos": [
        "kill_wallbang",
        "weapon_semi_rifle_kill",
        "weapon_semi_rifle_proporcion",
    ],
}


def preparar():
    for ruta in (RUTA_TRAIN, RUTA_PRED, RUTA_FOLDS, RUTA_CONFIG):
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe {ruta}. Consulta README.md")
    df = pd.read_csv(RUTA_TRAIN)
    if set(df["particion"].dropna().unique()) != {"TRAIN"}:
        raise ValueError("Solo se permite TRAIN")
    predictores = [
        n.strip() for n in RUTA_PRED.read_text(encoding="utf-8").splitlines()
        if n.strip()
    ]
    if len(predictores) != 72 or len(set(predictores)) != 72:
        raise ValueError("Se esperaban 72 predictores únicos")
    if set(predictores) - set(df.columns):
        raise ValueError("Faltan predictores en TRAIN")
    if set(df["etiqueta"].dropna().unique()) != {0, 1}:
        raise ValueError("Se esperaban las dos etiquetas 0 y 1")
    if not np.isfinite(
        df[predictores].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Faltantes o infinitos en TRAIN")

    mapa = pd.read_csv(RUTA_FOLDS)
    if (mapa["partida_id"].duplicated().any()
            or set(mapa["fold_validacion"].unique()) != {1, 2, 3, 4, 5}):
        raise ValueError("Mapa de folds inválido")
    asignacion = dict(zip(
        mapa["partida_id"].astype(str), mapa["fold_validacion"].astype(int)
    ))
    grupos = df["partida_id"].astype(str)
    if set(grupos.unique()) != set(asignacion):
        raise ValueError("El mapa y las partidas de TRAIN no coinciden")
    folds = []
    grupos_fold = grupos.map(asignacion).to_numpy()
    for numero in range(1, 6):
        tr = np.flatnonzero(grupos_fold != numero)
        val = np.flatnonzero(grupos_fold == numero)
        if len(tr) == 0 or len(val) == 0:
            raise ValueError(f"Fold {numero} vacío")
        if set(grupos.iloc[tr]) & set(grupos.iloc[val]):
            raise ValueError(f"Fuga de partidas en fold {numero}")
        folds.append((numero, tr, val))

    config = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
    params = {
        clave.removeprefix("modelo__"): valor
        for clave, valor in config["best_params"].items()
    }
    if set(params) - set(XGBPonderado().get_params()):
        raise ValueError("Parámetros XGBoost inesperados")
    return df, predictores, folds, params


def contar(y, pred):
    reales = np.asarray(y, dtype=int)
    estimadas = np.asarray(pred, dtype=int)
    tp = int(((reales == 1) & (estimadas == 1)).sum())
    fn = int(((reales == 1) & (estimadas == 0)).sum())
    fp = int(((reales == 0) & (estimadas == 1)).sum())
    tn = int(((reales == 0) & (estimadas == 0)).sum())
    return {
        "ventanas": len(reales), "positivas": tp + fn, "negativas": tn + fp,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "precision": precision_score(reales, estimadas, zero_division=0),
        "f1": f1_score(reales, estimadas, zero_division=0),
    }


def main():
    df, predictores, folds, params = preparar()
    y = df["etiqueta"].to_numpy(dtype=int)
    SALIDA.mkdir(parents=True, exist_ok=True)

    por_fold, subgrupos, globales = [], [], []
    for escenario, excluidas in EXCLUSIONES.items():
        columnas = [nombre for nombre in predictores if nombre not in excluidas]
        if len(columnas) != len(predictores) - len(excluidas):
            raise ValueError(f"Columnas excluidas no encontradas: {escenario}")
        X = df[columnas]
        pred_oof = np.full(len(df), -1, dtype=int)
        for fold, tr, va in folds:
            modelo = XGBPonderado(**params)
            modelo.fit(X.iloc[tr], y[tr])
            prob = modelo.predict_proba(X.iloc[va])[:, 1]
            pred = (prob >= UMBRAL).astype(int)
            pred_oof[va] = pred
            por_fold.append({
                "escenario": escenario, "fold": fold, "n_features": len(columnas),
                "scale_pos_weight_fit": modelo.scale_pos_weight_,
                **contar(y[va], pred),
            })
            print(
                f"{escenario} | fold {fold} | "
                f"F1={f1_score(y[va], pred, zero_division=0):.4f} "
                f"recall={recall_score(y[va], pred, zero_division=0):.4f}",
                flush=True,
            )
        if (pred_oof == -1).any():
            raise RuntimeError("Hay ventanas sin predicción OOF")
        globales.append({
            "escenario": escenario, "n_features": len(columnas),
            **contar(y, pred_oof),
        })
        # Definir subgrupos con atributos originales para todos los escenarios,
        # aunque el modelo de ese escenario los haya excluido.
        for wallbang in (0, 1):
            for rifle in (0, 1):
                mascara = (
                    (df["kill_wallbang"] == wallbang)
                    & (df["weapon_semi_rifle_kill"] == rifle)
                ).to_numpy()
                subgrupos.append({
                    "escenario": escenario,
                    "kill_wallbang": wallbang,
                    "weapon_semi_rifle_kill": rifle,
                    **contar(y[mascara], pred_oof[mascara]),
                })

    por_fold_df = pd.DataFrame(por_fold)
    resumen = pd.DataFrame(globales)
    medias = por_fold_df.groupby("escenario", as_index=False).agg(
        media_f1=("f1", "mean"),
        desv_f1=("f1", "std"),
        media_recall=("recall", "mean"),
        desv_recall=("recall", "std"),
    )
    resumen = resumen.merge(medias, on="escenario")
    por_fold_df.to_csv(SALIDA / "metricas_por_fold.csv", index=False)
    pd.DataFrame(subgrupos).to_csv(
        SALIDA / "metricas_oof_por_subgrupo.csv", index=False
    )
    resumen.to_csv(SALIDA / "comparacion_ablacion.csv", index=False)
    (SALIDA / "notas_metodologicas.json").write_text(
        json.dumps({
            "particion_utilizada": "TRAIN",
            "otros_conjuntos": "Ni VALIDACION ni TEST fueron abiertos",
            "caracter": (
                "Exploratorio: los subgrupos y variables se propusieron "
                "después de inspeccionar VALIDACION. No es confirmatorio."
            ),
            "unidad": "ventana, con folds separados por partida",
            "folds": 5,
            "umbral": UMBRAL,
            "hiperparametros": params,
            "exclusiones": EXCLUSIONES,
            "criterio": (
                "Comparar F1 y recall medio por fold, y falsos positivos "
                "por subgrupo en predicciones fuera de fold. No escoger "
                "un nuevo modelo final usando el TEST histórico."
            ),
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nComparación exploratoria TRAIN (predicciones fuera de fold):")
    print(resumen[["escenario", "n_features", "media_f1",
                   "media_recall", "f1", "recall", "fp", "fn"]]
          .to_string(index=False))
    print("\nNo se modificó el modelo guardado. No se abrió VALIDACION ni TEST.")


if __name__ == "__main__":
    main()
