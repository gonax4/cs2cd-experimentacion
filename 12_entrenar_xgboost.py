"""Tercer clasificador: XGBoost con los 72 predictores y los mismos folds.

No abre VALIDACION externa ni TEST. La selección se hace en TRAIN con
RandomizedSearchCV y los cinco folds por partida ya guardados.
El peso positivo se calcula solamente con la porción de entrenamiento
de cada fold mediante fit; se contrasta la opción sin ponderación.
"""
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


SEMILLA = 42
N_ITER = 20
RUTA_TRAIN = Path("datos/procesado/caracteristicas_train.csv")
RUTA_PREDICTORES = Path("salidas/regresion_logistica/predictores_finales_72.txt")
RUTA_FOLDS = Path("salidas/regresion_logistica/mapa_partidas_folds_internos.csv")
SALIDA = Path("salidas/xgboost")
RUTA_MODELO = Path("modelos/xgboost_best_pipeline.joblib")


class XGBPonderado(ClassifierMixin, BaseEstimator):
    """Calcula scale_pos_weight con cada conjunto entregado a fit, sin fuga.

    balancear=False equivale a scale_pos_weight=1.0.
    """

    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=1, reg_lambda=1.0, balancear=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.balancear = balancear

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=int)
        clases, cantidades = np.unique(y_array, return_counts=True)
        if not np.array_equal(clases, [0, 1]):
            raise ValueError("El entrenamiento requiere ambas clases 0 y 1")
        negativos = int(cantidades[0])
        positivos = int(cantidades[1])
        self.scale_pos_weight_ = (
            negativos / positivos if self.balancear else 1.0
        )
        self.modelo_ = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEMILLA,
            n_jobs=1,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight_,
        )
        self.modelo_.fit(X, y_array)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = self.modelo_.n_features_in_
        return self

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)

    def predict(self, X):
        return self.modelo_.predict(X)


def cargar_datos():
    for ruta in (RUTA_TRAIN, RUTA_PREDICTORES, RUTA_FOLDS):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta {ruta}. Consulta README.md")
    train = pd.read_csv(RUTA_TRAIN)
    if set(train["particion"].dropna().unique()) != {"TRAIN"}:
        raise ValueError("Solo pueden entrenarse filas de TRAIN")
    nombres = [s.strip() for s in RUTA_PREDICTORES.read_text(
        encoding="utf-8").splitlines() if s.strip()]
    if len(nombres) != 72 or len(set(nombres)) != 72:
        raise ValueError("Se necesitan 72 predictores únicos y congelados")
    if set(nombres) - set(train.columns):
        raise ValueError("Faltan predictores en TRAIN")
    X = train[nombres].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Existen NaN o infinitos en los predictores")
    y = train["etiqueta"].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Se esperaban etiquetas 0 y 1")
    return train, X, y, nombres


def reconstruir_folds(train):
    mapa = pd.read_csv(RUTA_FOLDS)
    if not {"partida_id", "fold_validacion"}.issubset(mapa.columns):
        raise ValueError("Mapa de folds incompleto")
    if mapa["partida_id"].duplicated().any():
        raise ValueError("Hay partidas duplicadas en el mapa")
    if set(mapa["fold_validacion"]) != set(range(1, 6)):
        raise ValueError("Se esperaban los folds 1 a 5")
    asignacion = dict(zip(mapa["partida_id"].astype(str),
                          mapa["fold_validacion"].astype(int)))
    grupos = train["partida_id"].astype(str)
    if set(grupos.unique()) != set(asignacion):
        raise ValueError("El mapa de folds no coincide con TRAIN")
    folds = []
    for numero in range(1, 6):
        val = np.flatnonzero(grupos.map(asignacion).to_numpy() == numero)
        tr = np.flatnonzero(grupos.map(asignacion).to_numpy() != numero)
        if not len(val) or not len(tr):
            raise ValueError(f"Fold {numero} vacío")
        if set(grupos.iloc[val]) & set(grupos.iloc[tr]):
            raise RuntimeError("Fuga entre partidas")
        folds.append((tr, val))
    return folds


def main():
    train, X, y, nombres = cargar_datos()
    folds = reconstruir_folds(train)
    SALIDA.mkdir(parents=True, exist_ok=True)
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)

    # Búsqueda moderada. Sin doble paralelismo para controlar memoria.
    espacio = {
        "modelo__n_estimators": [200, 400],
        "modelo__max_depth": [3, 5],
        "modelo__learning_rate": [0.03, 0.1],
        "modelo__subsample": [0.8, 1.0],
        "modelo__colsample_bytree": [0.8, 1.0],
        "modelo__min_child_weight": [1, 5],
        "modelo__balancear": [False, True],
    }
    pipeline = Pipeline([("modelo", XGBPonderado())])
    busqueda = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=espacio,
        n_iter=N_ITER,
        scoring={"f1": "f1", "recall": "recall",
                 "precision": "precision", "roc_auc": "roc_auc",
                 "pr_auc": "average_precision"},
        refit="f1",
        cv=folds,
        n_jobs=2,
        pre_dispatch=2,
        random_state=SEMILLA,
        return_train_score=False,
        error_score="raise",
        verbose=2,
    )
    print("Entrenando XGBoost sobre TRAIN, usando cinco folds agrupados.")
    print(f"Ventanas={len(X)} | Partidas={train['partida_id'].nunique()} "
          f"| Características={len(nombres)} | Ajustes={N_ITER * 5}")
    busqueda.fit(X, y)
    resultados = pd.DataFrame(busqueda.cv_results_).sort_values(
        by="rank_test_f1")
    resultados.to_csv(SALIDA / "resultados_randomizedsearch_xgb.csv", index=False)
    joblib.dump(busqueda.best_estimator_, RUTA_MODELO)
    configuracion = {
        "best_params": busqueda.best_params_,
        "best_mean_cv_f1": float(busqueda.best_score_),
        "scoring_principal": "f1 con umbral predeterminado 0.5",
        "folds": 5,
        "agrupacion": "partida_id",
        "predictores": len(nombres),
        "iteraciones_busqueda": N_ITER,
        "semilla": SEMILLA,
        "scale_pos_weight_ajustado_en_cada_fit": True,
        "scale_pos_weight_refit_train": (
            busqueda.best_estimator_.named_steps["modelo"].scale_pos_weight_
        ),
        "versiones": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "nota": "No se cargaron VALIDACION ni TEST. TEST histórico permanece cerrado.",
    }
    (SALIDA / "mejor_configuracion_xgb.json").write_text(
        json.dumps(configuracion, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nMejor configuración:", busqueda.best_params_)
    print("F1 media CV:", busqueda.best_score_)
    print("Modelo guardado:", RUTA_MODELO)
    print("Siguiente paso: 13_evaluar_xgboost_validacion.py")


if __name__ == "__main__":
    main()
