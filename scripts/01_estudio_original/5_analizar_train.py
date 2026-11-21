import pandas as pd
import numpy as np
from pathlib import Path


RUTA_TRAIN = Path("datos/procesado/caracteristicas_train.csv")
CARPETA_SALIDA = Path("salidas/analisis_train")

METADATOS = {
    "clave",
    "partida_id",
    "origen",
    "archivo_id",
    "jugador_id",
    "kill_id",
    "clase",
    "etiqueta",
    "particion",
}

PREFIJO_DIAGNOSTICO = "diag_"


print("ANÁLISIS EXPLORATORIO DE TRAIN")
print("==============================\n")


if not RUTA_TRAIN.exists():
    raise FileNotFoundError(
        "No se encontró datos/procesado/caracteristicas_train.csv"
    )

CARPETA_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

train = pd.read_csv(RUTA_TRAIN)

print(f"Filas de TRAIN: {len(train):,}")
print(f"Columnas totales: {len(train.columns)}")


# ============================================================
# 1. Comprobaciones estructurales
# ============================================================

if "particion" not in train.columns:
    raise ValueError("Falta la columna 'particion'.")

particiones = train["particion"].dropna().unique().tolist()

if particiones != ["TRAIN"] and set(particiones) != {"TRAIN"}:
    raise ValueError(
        f"El archivo contiene particiones distintas de TRAIN: {particiones}"
    )

if "etiqueta" not in train.columns:
    raise ValueError("Falta la columna objetivo 'etiqueta'.")


# ============================================================
# 2. Separar metadatos, diagnósticos y predictores
# ============================================================

columnas_diagnostico = [
    c for c in train.columns
    if c.startswith(PREFIJO_DIAGNOSTICO)
]

predictores = [
    c for c in train.columns
    if c not in METADATOS
    and c not in columnas_diagnostico
]

print(f"Predictores candidatos: {len(predictores)}")
print(f"Columnas diagnósticas: {len(columnas_diagnostico)}")


with open(
    CARPETA_SALIDA / "lista_predictores.txt",
    "w",
    encoding="utf-8"
) as f:
    for columna in predictores:
        f.write(columna + "\n")


# ============================================================
# 3. Distribución de clases
# ============================================================

conteo_clases = (
    train["etiqueta"]
    .value_counts()
    .sort_index()
)

porcentaje_clases = (
    train["etiqueta"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

resumen_clases = pd.DataFrame({
    "etiqueta": conteo_clases.index,
    "cantidad": conteo_clases.values,
    "porcentaje": porcentaje_clases.values,
})

resumen_clases.to_csv(
    CARPETA_SALIDA / "distribucion_clases_train.csv",
    index=False
)


# ============================================================
# 4. Calidad de cada predictor
# ============================================================

registros = []

for columna in predictores:

    serie = pd.to_numeric(
        train[columna],
        errors="coerce"
    )

    valores = serie.to_numpy(dtype=np.float64)

    faltantes = int(serie.isna().sum())
    infinitos = int(np.isinf(valores).sum())

    finitos = serie[
        np.isfinite(valores)
    ]

    unicos = int(serie.nunique(dropna=True))

    registro = {
        "caracteristica": columna,
        "faltantes": faltantes,
        "infinitos": infinitos,
        "valores_unicos": unicos,
        "constante": unicos <= 1,
    }

    if len(finitos) > 0:
        registro.update({
            "min": float(finitos.min()),
            "q01": float(finitos.quantile(0.01)),
            "q25": float(finitos.quantile(0.25)),
            "mediana": float(finitos.quantile(0.50)),
            "media": float(finitos.mean()),
            "q75": float(finitos.quantile(0.75)),
            "q99": float(finitos.quantile(0.99)),
            "max": float(finitos.max()),
            "desv_std": float(finitos.std(ddof=0)),
        })
    else:
        registro.update({
            "min": np.nan,
            "q01": np.nan,
            "q25": np.nan,
            "mediana": np.nan,
            "media": np.nan,
            "q75": np.nan,
            "q99": np.nan,
            "max": np.nan,
            "desv_std": np.nan,
        })

    registros.append(registro)


calidad = pd.DataFrame(registros)

calidad.to_csv(
    CARPETA_SALIDA / "calidad_caracteristicas_train.csv",
    index=False
)


# ============================================================
# 5. Valores extremos por regla IQR
#    Solo diagnóstico: NO se eliminan automáticamente.
# ============================================================

extremos = []

for columna in predictores:

    serie = pd.to_numeric(
        train[columna],
        errors="coerce"
    )

    serie = serie.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if serie.nunique() <= 2:
        continue

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        cantidad_extremos = 0
        limite_inferior = q1
        limite_superior = q3
    else:
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        cantidad_extremos = int(
            (
                (serie < limite_inferior)
                | (serie > limite_superior)
            ).sum()
        )

    extremos.append({
        "caracteristica": columna,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "limite_inferior_iqr": float(limite_inferior),
        "limite_superior_iqr": float(limite_superior),
        "cantidad_fuera_iqr": cantidad_extremos,
        "porcentaje_fuera_iqr": (
            cantidad_extremos / len(serie) * 100
            if len(serie) > 0
            else 0
        ),
    })


pd.DataFrame(extremos).to_csv(
    CARPETA_SALIDA / "extremos_iqr_train.csv",
    index=False
)


# ============================================================
# 6. Comprobaciones de valores potencialmente imposibles
# ============================================================

problemas = []


def registrar_problema(columna, regla, mascara):
    cantidad = int(mascara.sum())

    if cantidad > 0:
        problemas.append({
            "caracteristica": columna,
            "regla": regla,
            "cantidad": cantidad,
            "porcentaje": cantidad / len(train) * 100,
        })


for columna in predictores:

    serie = pd.to_numeric(
        train[columna],
        errors="coerce"
    )

    # Proporciones: deben estar entre 0 y 1.
    if "proporcion" in columna:
        registrar_problema(
            columna,
            "Proporción fuera de [0, 1]",
            (serie < 0) | (serie > 1)
        )

    # Indicadores binarios de la kill.
    if columna.startswith("kill_") or columna.endswith("_kill"):
        if columna not in {
            "victim_health_kill",
            "victim_health_cambio_inicio_kill",
            "attacker_pitch_head_delta_kill",
            "attacker_yaw_head_delta_kill",
        }:
            registrar_problema(
                columna,
                "Indicador binario distinto de 0/1",
                ~serie.isin([0, 1])
                & serie.notna()
            )

    # Varianzas, rangos y conteos no pueden ser negativos.
    if (
        columna.endswith("_var")
        or columna.endswith("_rango")
        or columna.endswith("_conteo")
        or columna == "weapon_transiciones"
    ):
        registrar_problema(
            columna,
            "Valor negativo donde se esperaba >= 0",
            serie < 0
        )

    # Distancia 3D tampoco puede ser negativa.
    if columna.startswith("distancia_3d"):
        registrar_problema(
            columna,
            "Distancia negativa",
            serie < 0
        )


problemas_df = pd.DataFrame(
    problemas,
    columns=[
        "caracteristica",
        "regla",
        "cantidad",
        "porcentaje",
    ]
)

problemas_df.to_csv(
    CARPETA_SALIDA / "valores_potencialmente_imposibles_train.csv",
    index=False
)


# ============================================================
# 7. Diagnóstico de armas
# ============================================================

if columnas_diagnostico:

    diag_armas = train[columnas_diagnostico].describe().T

    diag_armas.to_csv(
        CARPETA_SALIDA / "diagnostico_armas_train.csv"
    )

    resumen_diag = []

    for columna in columnas_diagnostico:
        serie = pd.to_numeric(
            train[columna],
            errors="coerce"
        )

        resumen_diag.append({
            "diagnostico": columna,
            "filas_con_valor_mayor_0": int((serie > 0).sum()),
            "porcentaje_filas_mayor_0": float((serie > 0).mean() * 100),
            "min": float(serie.min()),
            "max": float(serie.max()),
            "media": float(serie.mean()),
        })

    pd.DataFrame(resumen_diag).to_csv(
        CARPETA_SALIDA / "resumen_diagnostico_armas_train.csv",
        index=False
    )


# ============================================================
# 8. Características constantes
# ============================================================

constantes = calidad[
    calidad["constante"]
].copy()

constantes.to_csv(
    CARPETA_SALIDA / "caracteristicas_constantes_train.csv",
    index=False
)


# ============================================================
# 9. Resumen general
# ============================================================

faltantes_totales = int(
    calidad["faltantes"].sum()
)

infinitos_totales = int(
    calidad["infinitos"].sum()
)

resumen_general = pd.DataFrame([
    {
        "filas_train": len(train),
        "columnas_totales": len(train.columns),
        "predictores_candidatos": len(predictores),
        "columnas_diagnostico": len(columnas_diagnostico),
        "faltantes_totales_predictores": faltantes_totales,
        "infinitos_totales_predictores": infinitos_totales,
        "caracteristicas_constantes": int(calidad["constante"].sum()),
        "problemas_valores_imposibles": len(problemas_df),
    }
])

resumen_general.to_csv(
    CARPETA_SALIDA / "resumen_general_train.csv",
    index=False
)


# ============================================================
# 10. Mostrar lo importante en consola
# ============================================================

print("\nDISTRIBUCIÓN DE CLASES")
print("======================")
print(
    resumen_clases.to_string(
        index=False
    )
)

print("\nCALIDAD GENERAL")
print("===============")

print(
    f"Valores faltantes en predictores: "
    f"{faltantes_totales:,}"
)

print(
    f"Valores infinitos en predictores: "
    f"{infinitos_totales:,}"
)

print(
    f"Características constantes: "
    f"{int(calidad['constante'].sum())}"
)

print(
    f"Reglas de valores potencialmente imposibles con incidencias: "
    f"{len(problemas_df)}"
)


if len(constantes) > 0:
    print("\nCARACTERÍSTICAS CONSTANTES")
    print("==========================")
    print(
        constantes[
            ["caracteristica", "valores_unicos"]
        ].to_string(index=False)
    )


if len(problemas_df) > 0:
    print("\nVALORES POTENCIALMENTE IMPOSIBLES")
    print("=================================")
    print(
        problemas_df.to_string(
            index=False
        )
    )


if columnas_diagnostico:
    print("\nDIAGNÓSTICO DE ARMAS")
    print("====================")

    resumen_diag_df = pd.read_csv(
        CARPETA_SALIDA
        / "resumen_diagnostico_armas_train.csv"
    )

    print(
        resumen_diag_df.to_string(
            index=False
        )
    )


print("\nARCHIVOS GENERADOS")
print("==================")

for ruta in sorted(
    CARPETA_SALIDA.iterdir()
):
    print(ruta)


print(
    "\nIMPORTANTE: este análisis no elimina valores extremos "
    "ni modifica características. Solo genera evidencia para "
    "decidir usando exclusivamente TRAIN."
)
