"""Diagnóstico descriptivo del desbalance sin abrir datos crudos ni volver a evaluar TEST.

Entradas ligeras ya publicadas en el repositorio:
  datos/procesado/resumen_particiones.csv
  salidas/regresion_logistica/resumen_folds_internos.csv
  salidas/evaluacion_final_test/metricas_finales_test.csv

Solo lee los resultados HISTÓRICOS de TEST; no realiza predicciones nuevas.
"""
import json
from pathlib import Path

import pandas as pd


RUTA_PARTICIONES = Path("datos/procesado/resumen_particiones.csv")
RUTA_FOLDS = Path("salidas/regresion_logistica/resumen_folds_internos.csv")
RUTA_RESULTADOS = Path("salidas/evaluacion_final_test/metricas_finales_test.csv")
SALIDA = Path("salidas/diagnostico_jurado")


def leer_csv(ruta, columnas):
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta el archivo: {ruta}")
    tabla = pd.read_csv(ruta)
    faltantes = set(columnas) - set(tabla.columns)
    if faltantes:
        raise ValueError(f"{ruta} no contiene {sorted(faltantes)}")
    return tabla


def distribucion_por_particion(tabla):
    resultado = tabla.copy()
    resultado["total"] = resultado["cheater"] + resultado["notcheater"]
    if (resultado["total"] <= 0).any():
        raise ValueError("Hay particiones sin ventanas")
    resultado["porcentaje_positivos"] = (
        100 * resultado["cheater"] / resultado["total"]
    )
    resultado["proporcion_negativos_por_positivo"] = (
        resultado["notcheater"] / resultado["cheater"].replace(0, float("nan"))
    )
    return resultado


def resumen_de_folds(tabla):
    if tabla["fold"].nunique() != 5:
        raise ValueError("Se esperaban cinco folds internos")
    resultado = tabla.copy()
    for lado in ("train", "validacion"):
        pos = resultado[f"positivos_{lado}"]
        neg = resultado[f"negativos_{lado}"]
        if ((pos + neg) <= 0).any():
            raise ValueError(f"Hay folds sin ventanas de {lado}")
        resultado[f"porcentaje_positivos_{lado}_verificado"] = (
            100 * pos / (pos + neg)
        )
    return resultado


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    particiones = distribucion_por_particion(
        leer_csv(RUTA_PARTICIONES, ["particion", "cheater", "notcheater"])
    )
    if set(particiones["particion"]) != {"TRAIN", "VALIDACION", "TEST"}:
        raise ValueError("Las tres particiones esperadas no están presentes")

    folds = resumen_de_folds(
        leer_csv(
            RUTA_FOLDS,
            ["fold", "positivos_train", "negativos_train",
             "positivos_validacion", "negativos_validacion"],
        )
    )
    originales = leer_csv(
        RUTA_RESULTADOS,
        ["modelo", "recall", "precision", "f1", "falsos_negativos",
         "verdaderos_positivos", "falsos_positivos"],
    )
    particiones.to_csv(SALIDA / "distribucion_particiones.csv", index=False)
    folds.to_csv(SALIDA / "distribucion_folds.csv", index=False)
    originales.to_csv(SALIDA / "metricas_test_historicas.csv", index=False)

    total_positivos_test = int(
        particiones.loc[particiones["particion"] == "TEST", "cheater"].iloc[0]
    )
    for fila in originales.itertuples():
        evaluados = int(fila.falsos_negativos + fila.verdaderos_positivos)
        if evaluados != total_positivos_test:
            raise ValueError(
                f"Los positivos históricos de {fila.modelo} no coinciden "
                "con el resumen original de TEST"
            )

    resumen = {
        "proposito": "Diagnóstico descriptivo de etiquetas y folds",
        "clase_positiva": "cheater=1",
        "unidad": "ventana etiquetada, no jugador identificado",
        "test": "Solo lectura de métricas históricas, sin nuevas predicciones",
        "porcentaje_positivos_train": float(
            particiones.loc[particiones["particion"] == "TRAIN",
                            "porcentaje_positivos"].iloc[0]
        ),
        "porcentaje_positivos_validacion": float(
            particiones.loc[particiones["particion"] == "VALIDACION",
                            "porcentaje_positivos"].iloc[0]
        ),
        "porcentaje_positivos_test_historico": float(
            particiones.loc[particiones["particion"] == "TEST",
                            "porcentaje_positivos"].iloc[0]
        ),
        "interpretacion": (
            "La desproporción se puede medir, pero estos conteos no "
            "demuestran que sea la causa del recall observado. "
            "Los folds agrupan por partida y sus proporciones pueden variar."
        ),
    }
    (SALIDA / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nDistribución de las particiones:")
    print(particiones[["particion", "cheater", "notcheater",
                       "porcentaje_positivos",
                       "proporcion_negativos_por_positivo"]].to_string(index=False))
    print("\nPorcentaje positivo en los cinco folds de validación interna:")
    print(folds[["fold", "porcentaje_positivos_validacion_verificado"]]
          .to_string(index=False))
    print("\nTEST: solo se copiaron métricas históricas, sin volver a evaluarlo.")
    print(f"\nResultados guardados en {SALIDA}")


if __name__ == "__main__":
    main()
