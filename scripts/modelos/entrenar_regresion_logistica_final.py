import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEMILLA = 42

RUTA_TRAIN = Path("datos/procesado/caracteristicas_train.csv")
CARPETA_SALIDA = Path("salidas/regresion_logistica")
CARPETA_MODELOS = Path("modelos")

METADATOS = {
    "clave", "partida_id", "origen", "archivo_id",
    "jugador_id", "kill_id", "clase", "etiqueta", "particion"
}

CARACTERISTICAS_EXCLUIDAS = {
    "victim_health_min",
    "victim_health_kill",
}

COLUMNAS_BINARIAS = [
    "kill_headshot",
    "kill_through_smoke",
    "kill_wallbang",
    "kill_attacker_midair",
    "weapon_knife_kill",
    "weapon_auto_rifle_kill",
    "weapon_semi_rifle_kill",
    "weapon_pistol_kill",
    "weapon_grenade_kill",
    "weapon_smg_kill",
    "weapon_shotgun_kill",
]

GRID_PARAMETROS = {
    "modelo__C": [0.01, 0.1, 1, 10, 100],
    "modelo__class_weight": [None, "balanced"],
}


print("REGRESIÓN LOGÍSTICA - GRID SEARCH")
print("=================================\n")

if not RUTA_TRAIN.exists():
    raise FileNotFoundError(
        "No se encontró datos/procesado/caracteristicas_train.csv"
    )

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(RUTA_TRAIN)

if set(train["particion"].dropna().unique()) != {"TRAIN"}:
    raise ValueError(
        "El archivo contiene observaciones fuera de TRAIN."
    )

predictores_candidatos = [
    c for c in train.columns
    if c not in METADATOS
    and not c.startswith("diag_")
]

predictores_finales = [
    c for c in predictores_candidatos
    if c not in CARACTERISTICAS_EXCLUIDAS
]

if len(predictores_finales) != 72:
    raise ValueError(
        f"Se esperaban 72 predictores finales y se obtuvieron "
        f"{len(predictores_finales)}."
    )

for columna in COLUMNAS_BINARIAS:
    if columna not in predictores_finales:
        raise ValueError(
            f"Falta la columna binaria esperada: {columna}"
        )

columnas_continuas = [
    c for c in predictores_finales
    if c not in COLUMNAS_BINARIAS
]

X = train[predictores_finales].copy()
y = train["etiqueta"].astype(int).copy()
grupos = train["partida_id"].astype(str).copy()

X_num = X.apply(pd.to_numeric, errors="coerce")

if X_num.isna().any().any():
    raise ValueError(
        "Se encontraron valores faltantes o no numéricos."
    )

if not np.isfinite(
    X_num.to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "Se encontraron infinitos en los predictores."
    )

for columna in COLUMNAS_BINARIAS:
    valores = set(
        X[columna].astype(float).unique().tolist()
    )
    if not valores.issubset({0.0, 1.0}):
        raise ValueError(
            f"{columna} contiene valores fuera de 0/1: {valores}"
        )


# ============================================================
# CONGELAR VECTOR FINAL
# ============================================================

with open(
    CARPETA_SALIDA / "predictores_finales_72.txt",
    "w",
    encoding="utf-8"
) as f:
    for columna in predictores_finales:
        f.write(columna + "\n")

pd.DataFrame({
    "caracteristica": predictores_finales,
    "tratamiento_rl": [
        "sin_escalar"
        if c in COLUMNAS_BINARIAS
        else "StandardScaler"
        for c in predictores_finales
    ]
}).to_csv(
    CARPETA_SALIDA / "definicion_predictores_finales.csv",
    index=False
)

with open(
    CARPETA_SALIDA / "caracteristicas_excluidas_train.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "criterio": "varianza cero detectada exclusivamente en TRAIN",
            "caracteristicas": sorted(CARACTERISTICAS_EXCLUIDAS),
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FOLDS INTERNOS AGRUPADOS POR PARTIDA
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEMILLA
)

splits = list(
    cv.split(
        X,
        y,
        groups=grupos
    )
)

mapa_fold_validacion = []
resumen_folds = []

for fold, (idx_train, idx_val) in enumerate(splits, start=1):

    partidas_entrenamiento = set(
        train.iloc[idx_train]["partida_id"]
        .astype(str)
        .unique()
    )
    partidas_validacion = set(
        train.iloc[idx_val]["partida_id"]
        .astype(str)
        .unique()
    )

    if partidas_entrenamiento & partidas_validacion:
        raise RuntimeError(
            f"Se detectó fuga de partidas en el fold {fold}."
        )

    for partida in sorted(partidas_validacion):
        mapa_fold_validacion.append({
            "partida_id": partida,
            "fold_validacion": fold
        })

    y_tr = y.iloc[idx_train]
    y_va = y.iloc[idx_val]

    resumen_folds.append({
        "fold": fold,
        "partidas_train": len(partidas_entrenamiento),
        "partidas_validacion": len(partidas_validacion),
        "ventanas_train": len(idx_train),
        "ventanas_validacion": len(idx_val),
        "positivos_train": int((y_tr == 1).sum()),
        "negativos_train": int((y_tr == 0).sum()),
        "positivos_validacion": int((y_va == 1).sum()),
        "negativos_validacion": int((y_va == 0).sum()),
        "porcentaje_positivos_train": float(
            (y_tr == 1).mean() * 100
        ),
        "porcentaje_positivos_validacion": float(
            (y_va == 1).mean() * 100
        ),
    })

mapa_folds_df = pd.DataFrame(mapa_fold_validacion)
resumen_folds_df = pd.DataFrame(resumen_folds)

if mapa_folds_df["partida_id"].nunique() != train["partida_id"].nunique():
    raise RuntimeError(
        "No todas las partidas de TRAIN quedaron asignadas "
        "exactamente una vez como validación interna."
    )

if mapa_folds_df["partida_id"].duplicated().any():
    raise RuntimeError(
        "Alguna partida aparece en más de un fold de validación."
    )

mapa_folds_df.to_csv(
    CARPETA_SALIDA / "mapa_partidas_folds_internos.csv",
    index=False
)

resumen_folds_df.to_csv(
    CARPETA_SALIDA / "resumen_folds_internos.csv",
    index=False
)


# ============================================================
# PIPELINE
# ============================================================

# El StandardScaler está dentro del Pipeline.
# Así, en cada fold se ajusta únicamente con la parte
# de entrenamiento interna y no con su validación.
preprocesador = ColumnTransformer(
    transformers=[
        (
            "continuas",
            StandardScaler(),
            columnas_continuas
        ),
        (
            "binarias",
            "passthrough",
            COLUMNAS_BINARIAS
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

modelo = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
)

pipeline = Pipeline([
    ("preprocesamiento", preprocesador),
    ("modelo", modelo),
])


# ============================================================
# GRID SEARCH
# ============================================================

busqueda = GridSearchCV(
    estimator=pipeline,
    param_grid=GRID_PARAMETROS,
    scoring="f1",
    cv=splits,
    refit=True,
    n_jobs=-1,
    return_train_score=True,
    verbose=2,
)

print(f"Filas TRAIN: {len(train):,}")
print(f"Partidas TRAIN: {train['partida_id'].nunique():,}")
print(f"Predictores candidatos: {len(predictores_candidatos)}")
print("Características constantes excluidas: 2")
print(f"Predictores finales: {len(predictores_finales)}")
print(f"Continuas estandarizadas: {len(columnas_continuas)}")
print(f"Binarias/one-hot sin escalar: {len(COLUMNAS_BINARIAS)}")
print("Configuraciones: 10")
print("Folds: 5")
print("Ajustes CV esperados: 50")
print("\nIniciando GridSearchCV...\n")

busqueda.fit(X, y)


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

resultados_cv = pd.DataFrame(
    busqueda.cv_results_
).sort_values(
    by=["rank_test_score", "mean_test_score"],
    ascending=[True, False]
)

resultados_cv.to_csv(
    CARPETA_SALIDA / "resultados_gridsearch_rl.csv",
    index=False
)

joblib.dump(
    busqueda.best_estimator_,
    CARPETA_MODELOS / "regresion_logistica_best_pipeline.joblib"
)

with open(
    CARPETA_SALIDA / "mejor_configuracion_rl.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "best_params": busqueda.best_params_,
            "best_mean_cv_f1": float(busqueda.best_score_),
            "numero_predictores": len(predictores_finales),
            "numero_continuas": len(columnas_continuas),
            "numero_binarias": len(COLUMNAS_BINARIAS),
            "semilla": SEMILLA,
            "cv": (
                "StratifiedGroupKFold("
                "n_splits=5, shuffle=True, random_state=42)"
            ),
            "scoring": "f1",
            "solver": "lbfgs",
            "penalty": "l2",
            "max_iter": 1000,
        },
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    CARPETA_SALIDA / "versiones_entrenamiento.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# RESUMEN
# ============================================================

print("\n\nBÚSQUEDA FINALIZADA")
print("===================")

print("\nMejores hiperparámetros:")
print(busqueda.best_params_)

print("\nF1 medio de validación cruzada:")
print(f"{busqueda.best_score_:.6f}")

print("\nResumen de folds:")
print(
    resumen_folds_df.to_string(index=False)
)

print("\nTop configuraciones:")

columnas_mostrar = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "param_modelo__C",
    "param_modelo__class_weight",
]

print(
    resultados_cv[
        columnas_mostrar
    ].head(10).to_string(index=False)
)

print("\nPipeline guardado en:")
print(
    CARPETA_MODELOS
    / "regresion_logistica_best_pipeline.joblib"
)

print(
    "\nIMPORTANTE: VALIDACIÓN externa y TEST todavía NO han sido utilizados."
)
