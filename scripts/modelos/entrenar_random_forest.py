import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEMILLA = 42
N_ITER = 50
N_JOBS_BUSQUEDA = 4

RUTA_TRAIN = Path("datos/procesado/caracteristicas_train.csv")
RUTA_PREDICTORES = Path(
    "salidas/regresion_logistica/predictores_finales_72.txt"
)
RUTA_FOLDS = Path(
    "salidas/regresion_logistica/mapa_partidas_folds_internos.csv"
)

CARPETA_SALIDA = Path("salidas/random_forest")
CARPETA_MODELOS = Path("modelos")

ESPACIO_PARAMETROS = {
    "modelo__n_estimators": [200, 400, 600, 800],
    "modelo__max_depth": [10, 20, 30, None],
    "modelo__max_features": ["sqrt", "log2", 0.5],
    "modelo__min_samples_split": [2, 5, 10],
    "modelo__min_samples_leaf": [1, 2, 4],
    "modelo__class_weight": [None, "balanced"],
}


# ============================================================
# CARGA
# ============================================================

print("RANDOM FOREST - RANDOMIZED SEARCH")
print("=================================\n")

for ruta in [RUTA_TRAIN, RUTA_PREDICTORES, RUTA_FOLDS]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo requerido: {ruta}"
        )

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(RUTA_TRAIN)

if set(train["particion"].dropna().unique()) != {"TRAIN"}:
    raise ValueError(
        "El archivo contiene observaciones fuera de TRAIN."
    )

with open(
    RUTA_PREDICTORES,
    "r",
    encoding="utf-8"
) as f:
    predictores_finales = [
        linea.strip()
        for linea in f
        if linea.strip()
    ]

if len(predictores_finales) != 72:
    raise ValueError(
        f"Se esperaban 72 predictores congelados y se encontraron "
        f"{len(predictores_finales)}."
    )

faltantes = [
    c for c in predictores_finales
    if c not in train.columns
]

if faltantes:
    raise ValueError(
        "Faltan predictores en TRAIN: "
        + ", ".join(faltantes)
    )

X = train[predictores_finales].copy()
y = train["etiqueta"].astype(int).copy()

# Verificación de calidad antes de entrenar.
X_num = X.apply(pd.to_numeric, errors="coerce")

if X_num.isna().any().any():
    raise ValueError(
        "Se encontraron faltantes o valores no numéricos."
    )

if not np.isfinite(
    X_num.to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "Se encontraron valores infinitos."
    )


# ============================================================
# RECONSTRUIR EXACTAMENTE LOS MISMOS FOLDS DE RL
# ============================================================

mapa_folds = pd.read_csv(RUTA_FOLDS)

columnas_requeridas = {
    "partida_id",
    "fold_validacion",
}

if not columnas_requeridas.issubset(
    mapa_folds.columns
):
    raise ValueError(
        "El archivo de folds no contiene las columnas esperadas."
    )

if mapa_folds["partida_id"].duplicated().any():
    raise ValueError(
        "Alguna partida aparece más de una vez en el mapa de folds."
    )

mapa_dict = dict(
    zip(
        mapa_folds["partida_id"].astype(str),
        mapa_folds["fold_validacion"].astype(int),
    )
)

partidas_train = set(
    train["partida_id"]
    .astype(str)
    .unique()
)

partidas_mapa = set(
    mapa_dict.keys()
)

if partidas_train != partidas_mapa:
    faltan_en_mapa = sorted(
        partidas_train - partidas_mapa
    )

    sobran_en_mapa = sorted(
        partidas_mapa - partidas_train
    )

    raise ValueError(
        "El mapa de folds no coincide con las partidas de TRAIN. "
        f"Faltan en mapa: {len(faltan_en_mapa)}; "
        f"sobran en mapa: {len(sobran_en_mapa)}."
    )

splits = []
resumen_folds = []

partidas_fila = (
    train["partida_id"]
    .astype(str)
)

for fold in range(1, 6):

    mascara_val = (
        partidas_fila.map(mapa_dict)
        == fold
    )

    idx_val = np.flatnonzero(
        mascara_val.to_numpy()
    )

    idx_train = np.flatnonzero(
        (~mascara_val).to_numpy()
    )

    if len(idx_val) == 0:
        raise RuntimeError(
            f"El fold {fold} no contiene observaciones de validación."
        )

    partidas_tr = set(
        train.iloc[idx_train]["partida_id"]
        .astype(str)
        .unique()
    )

    partidas_va = set(
        train.iloc[idx_val]["partida_id"]
        .astype(str)
        .unique()
    )

    if partidas_tr & partidas_va:
        raise RuntimeError(
            f"Fuga de partidas en fold {fold}."
        )

    y_tr = y.iloc[idx_train]
    y_va = y.iloc[idx_val]

    resumen_folds.append({
        "fold": fold,
        "partidas_train": len(partidas_tr),
        "partidas_validacion": len(partidas_va),
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

    splits.append(
        (idx_train, idx_val)
    )

resumen_folds_df = pd.DataFrame(
    resumen_folds
)

resumen_folds_df.to_csv(
    CARPETA_SALIDA
    / "resumen_folds_reutilizados.csv",
    index=False
)


# ============================================================
# MODELO Y PIPELINE
# ============================================================

# No se aplica StandardScaler a Random Forest.
# Tampoco se usa imputación porque TRAIN no contiene faltantes.
modelo = RandomForestClassifier(
    bootstrap=True,
    random_state=SEMILLA,

    # El paralelismo se controla en RandomizedSearchCV.
    # Dejar n_jobs=1 aquí evita multiplicar procesos dentro de procesos.
    n_jobs=1,
)

pipeline = Pipeline([
    ("modelo", modelo),
])


# ============================================================
# RANDOMIZED SEARCH
# ============================================================

busqueda = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=ESPACIO_PARAMETROS,
    n_iter=N_ITER,
    scoring="f1",
    cv=splits,
    refit=True,
    random_state=SEMILLA,

    # Cuatro ajustes simultáneos: compromiso entre CPU y RAM
    # para el equipo de 16 GB utilizado en el experimento.
    n_jobs=N_JOBS_BUSQUEDA,
    pre_dispatch=N_JOBS_BUSQUEDA,

    return_train_score=True,
    verbose=2,
    error_score="raise",
)

total_combinaciones = (
    len(ESPACIO_PARAMETROS["modelo__n_estimators"])
    * len(ESPACIO_PARAMETROS["modelo__max_depth"])
    * len(ESPACIO_PARAMETROS["modelo__max_features"])
    * len(ESPACIO_PARAMETROS["modelo__min_samples_split"])
    * len(ESPACIO_PARAMETROS["modelo__min_samples_leaf"])
    * len(ESPACIO_PARAMETROS["modelo__class_weight"])
)

print(f"Filas TRAIN: {len(train):,}")
print(f"Partidas TRAIN: {train['partida_id'].nunique():,}")
print(f"Predictores finales congelados: {len(predictores_finales)}")
print(f"Combinaciones posibles: {total_combinaciones}")
print(f"Configuraciones aleatorias a evaluar: {N_ITER}")
print("Folds internos: 5")
print(f"Ajustes CV esperados: {N_ITER * 5}")
print("bootstrap: True")
print(f"random_state: {SEMILLA}")
print(f"Procesos paralelos de búsqueda: {N_JOBS_BUSQUEDA}")

print("\nIniciando RandomizedSearchCV...\n")

inicio = time.perf_counter()

busqueda.fit(
    X,
    y
)

fin = time.perf_counter()
duracion_segundos = fin - inicio


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
    CARPETA_SALIDA
    / "resultados_randomizedsearch_rf.csv",
    index=False
)

# Guardar las 50 configuraciones realmente muestreadas.
columnas_params = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "param_modelo__n_estimators",
    "param_modelo__max_depth",
    "param_modelo__max_features",
    "param_modelo__min_samples_split",
    "param_modelo__min_samples_leaf",
    "param_modelo__class_weight",
]

resultados_cv[
    columnas_params
].to_csv(
    CARPETA_SALIDA
    / "configuraciones_evaluadas_rf.csv",
    index=False
)

joblib.dump(
    busqueda.best_estimator_,
    CARPETA_MODELOS
    / "random_forest_best_pipeline.joblib"
)

mejor_configuracion = {
    "best_params": busqueda.best_params_,
    "best_mean_cv_f1": float(
        busqueda.best_score_
    ),
    "numero_predictores": len(
        predictores_finales
    ),
    "semilla": SEMILLA,
    "n_iter": N_ITER,
    "total_combinaciones_posibles": total_combinaciones,
    "cv": (
        "Mismos 5 folds internos guardados durante "
        "Regresión Logística, derivados originalmente de "
        "StratifiedGroupKFold(n_splits=5, shuffle=True, "
        "random_state=42)"
    ),
    "scoring": "f1",
    "bootstrap": True,
    "duracion_busqueda_segundos": float(
        duracion_segundos
    ),
}

with open(
    CARPETA_SALIDA
    / "mejor_configuracion_rf.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        mejor_configuracion,
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    CARPETA_SALIDA
    / "espacio_busqueda_rf.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [10, 20, 30, None],
            "max_features": ["sqrt", "log2", 0.5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced"],
            "bootstrap": True,
        },
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    CARPETA_SALIDA
    / "versiones_entrenamiento.json",
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
print(
    busqueda.best_params_
)

print("\nF1 medio de validación cruzada:")
print(
    f"{busqueda.best_score_:.6f}"
)

print("\nDuración registrada de la búsqueda:")
print(
    f"{duracion_segundos:.2f} segundos"
)

print("\nResumen de folds reutilizados:")
print(
    resumen_folds_df.to_string(
        index=False
    )
)

print("\nTop 10 configuraciones:")

print(
    resultados_cv[
        columnas_params
    ].head(10).to_string(
        index=False
    )
)

print("\nPipeline guardado en:")
print(
    CARPETA_MODELOS
    / "random_forest_best_pipeline.joblib"
)

print(
    "\nIMPORTANTE: VALIDACIÓN externa y TEST todavía NO han sido utilizados."
)
