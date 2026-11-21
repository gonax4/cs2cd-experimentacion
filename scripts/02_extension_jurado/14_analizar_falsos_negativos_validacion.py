"""Analiza falsos negativos de XGBoost en VALIDACION externa.

Solo describe asociaciones dentro de las ventanas etiquetadas como positivas.
No obtiene importancias causales ni usa TEST.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATOS = Path("datos/procesado/caracteristicas_validacion.csv")
PREDICCIONES = Path("salidas/xgboost/validacion/predicciones_validacion.csv")
PREDICTORES = Path("salidas/regresion_logistica/predictores_finales_72.txt")
SALIDA = Path("salidas/xgboost/validacion")


def main():
    for ruta in (DATOS, PREDICCIONES, PREDICTORES):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta {ruta}")
    datos = pd.read_csv(DATOS)
    pred = pd.read_csv(PREDICCIONES)
    if set(datos["particion"].unique()) != {"VALIDACION"}:
        raise ValueError("Se requiere únicamente VALIDACION")
    if datos["clave"].duplicated().any() or pred["clave"].duplicated().any():
        raise ValueError("Existen claves duplicadas: no se puede unir 1 a 1")
    if len(datos) != len(pred) or set(datos["clave"]) != set(pred["clave"]):
        raise ValueError("Las predicciones no corresponden exactamente a VALIDACION")
    unidos = datos.merge(
        pred[["clave", "partida_id", "etiqueta_real",
              "prediccion_xgb", "probabilidad_xgb"]],
        on="clave", how="inner", suffixes=("", "_pred"), validate="one_to_one",
    )
    if not (unidos["partida_id"] == unidos["partida_id_pred"]).all():
        raise ValueError("Partidas inconsistentes entre predicciones y datos")
    if not (unidos["etiqueta"] == unidos["etiqueta_real"]).all():
        raise ValueError("Etiquetas inconsistentes")
    positivos = unidos.loc[unidos["etiqueta"] == 1].copy()
    positivos["falso_negativo"] = positivos["prediccion_xgb"] == 0
    if positivos.empty:
        raise ValueError("No hay ventanas positivas que analizar")

    partidas = positivos.groupby("partida_id", as_index=False).agg(
        ventanas_positivas=("clave", "size"),
        falsos_negativos=("falso_negativo", "sum"),
    )
    partidas["tasa_falsos_negativos"] = (
        partidas["falsos_negativos"] / partidas["ventanas_positivas"]
    )
    SALIDA.mkdir(parents=True, exist_ok=True)
    partidas.sort_values(
        ["falsos_negativos", "tasa_falsos_negativos"], ascending=False
    ).to_csv(SALIDA / "falsos_negativos_por_partida.csv", index=False)

    nombres = [s.strip() for s in PREDICTORES.read_text(
        encoding="utf-8").splitlines() if s.strip()]
    filas = []
    fn = positivos.loc[positivos["falso_negativo"]]
    tp = positivos.loc[~positivos["falso_negativo"]]
    for nombre in nombres:
        x_fn = pd.to_numeric(fn[nombre], errors="coerce")
        x_tp = pd.to_numeric(tp[nombre], errors="coerce")
        filas.append({
            "caracteristica": nombre,
            "mediana_falsos_negativos": (
                float(x_fn.median()) if x_fn.notna().any() else np.nan
            ),
            "mediana_verdaderos_positivos": (
                float(x_tp.median()) if x_tp.notna().any() else np.nan
            ),
            "n_fn": len(fn),
            "n_tp": len(tp),
        })
    pd.DataFrame(filas).to_csv(
        SALIDA / "caracteristicas_fn_vs_tp.csv", index=False
    )
    print(f"Positivos en VALIDACION: {len(positivos)}")
    print(f"Falsos negativos: {len(fn)}")
    print(f"Recall observado: {len(tp) / len(positivos):.4f}")
    print("Salidas descriptivas guardadas. No se utilizó TEST.")


if __name__ == "__main__":
    main()
