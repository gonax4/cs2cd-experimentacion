"""Estimador reusable de XGBoost para serialización estable con joblib.

Al guardar nuevas versiones, la clase queda identificada como
xgb_estimador.XGBPonderado en lugar de __main__.XGBPonderado.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

SEMILLA = 42


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

