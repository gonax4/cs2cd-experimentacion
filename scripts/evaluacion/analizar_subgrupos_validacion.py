"""Diagnóstico descriptivo de XGBoost por subgrupos de VALIDACION.

No entrena, modifica umbrales, selecciona variables ni abre TEST.
La interpretación es exploratoria y no implica causalidad.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATOS = Path("datos/procesado/caracteristicas_validacion.csv")
PRED = Path("salidas/xgboost/validacion/predicciones_validacion.csv")
SALIDA = Path("salidas/xgboost/validacion")
INDICADORES = [
    "kill_wallbang",
    "weapon_semi_rifle_kill",
    "kill_headshot",
    "kill_through_smoke",
]


def cargar():
    for ruta in (DATOS, PRED):
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta}")
    datos, pred = pd.read_csv(DATOS), pd.read_csv(PRED)
    requeridos = {"clave", "partida_id", "particion", "etiqueta", *INDICADORES}
    if requeridos - set(datos.columns):
        raise ValueError("Faltan características o metadatos en VALIDACION")
    if {"clave", "partida_id", "etiqueta_real", "prediccion_xgb"} - set(pred.columns):
        raise ValueError("Las predicciones de XGBoost están incompletas")
    if set(datos["particion"].dropna().unique()) != {"VALIDACION"}:
        raise ValueError("Solo se permite VALIDACION externa")
    if (datos["clave"].duplicated().any() or pred["clave"].duplicated().any()
            or len(datos) != len(pred) or set(datos["clave"]) != set(pred["clave"])):
        raise ValueError("Las claves de VALIDACION no coinciden")
    tabla = datos.merge(
        pred[["clave", "partida_id", "etiqueta_real", "prediccion_xgb"]],
        on="clave", suffixes=("", "_pred"), validate="one_to_one",
    )
    if not (tabla["partida_id"] == tabla["partida_id_pred"]).all():
        raise ValueError("Partidas inconsistentes")
    if not (tabla["etiqueta"] == tabla["etiqueta_real"]).all():
        raise ValueError("Etiquetas inconsistentes")
    for nombre in ["etiqueta", "prediccion_xgb", *INDICADORES]:
        if tabla[nombre].isna().any() or not set(tabla[nombre].unique()).issubset({0, 1}):
            raise ValueError(f"{nombre} no es un indicador binario válido")
    return tabla


def resumen(grupo, indicador, valor):
    real = grupo["etiqueta"].to_numpy(dtype=int)
    predicho = grupo["prediccion_xgb"].to_numpy(dtype=int)
    tp = int(((real == 1) & (predicho == 1)).sum())
    fn = int(((real == 1) & (predicho == 0)).sum())
    fp = int(((real == 0) & (predicho == 1)).sum())
    tn = int(((real == 0) & (predicho == 0)).sum())
    pos, neg = tp + fn, tn + fp
    return {
        "indicador": indicador, "valor": valor,
        "ventanas": len(grupo), "positivas": pos, "negativas": neg,
        "verdaderos_positivos": tp, "falsos_negativos": fn,
        "falsos_positivos": fp, "verdaderos_negativos": tn,
        "recall_positivos": tp / pos if pos else np.nan,
        "tasa_falsos_negativos": fn / pos if pos else np.nan,
        "tasa_falsos_positivos": fp / neg if neg else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
    }


def main():
    tabla = cargar()
    filas = []
    for caracteristica in INDICADORES:
        for valor in (0, 1):
            filas.append(resumen(
                tabla.loc[tabla[caracteristica] == valor], caracteristica, str(valor)
            ))
    combinacion = "kill_wallbang + weapon_semi_rifle_kill"
    for wallbang in (0, 1):
        for arma in (0, 1):
            filtro = ((tabla["kill_wallbang"] == wallbang)
                      & (tabla["weapon_semi_rifle_kill"] == arma))
            filas.append(resumen(tabla.loc[filtro], combinacion, f"{wallbang}/{arma}"))
    reporte = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    reporte.to_csv(SALIDA / "desempeno_por_subgrupos.csv", index=False)
    print("Análisis descriptivo por subgrupos en VALIDACION")
    print(reporte[["indicador", "valor", "positivas", "falsos_negativos",
                   "recall_positivos", "negativas", "falsos_positivos"]]
          .to_string(index=False))
    print("No seleccionar variables a partir de VALIDACION. No se abrió TEST.")


if __name__ == "__main__":
    main()
