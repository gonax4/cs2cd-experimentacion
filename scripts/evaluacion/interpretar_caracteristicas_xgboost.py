"""Importancias nativas del XGBoost ya entrenado, sin abrir TEST ni VALIDACION.

Analiza reducción total de pérdida ("total_gain") de los splits del booster.
También guarda "gain" medio por split y frecuencia ("weight") por característica.
No reentrena, no hace predicciones y no sobrescribe resultados de RL/RF.

Uso desde la raíz del repositorio:
    python scripts/evaluacion/interpretar_modelos_y_resultados.py --xgboost
"""
import __main__
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Compatibilidad con joblib que serializó XGBPonderado como __main__.
DIRECTORIO_MODELOS = Path(__file__).resolve().parents[1] / "modelos"
if str(DIRECTORIO_MODELOS) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_MODELOS))
from xgb_estimador import XGBPonderado

PREDICTORES = Path("salidas/regresion_logistica/predictores_finales_72.txt")
MODELO = Path("modelos/xgboost_best_pipeline.joblib")
SALIDA = Path("salidas/interpretacion_modelos")
N_SALIDA = 20


def grupo_de(nombre):
    # Misma agrupación usada en interpretar_modelos_y_resultados.py para RF.
    if nombre.startswith("attacker_vel_"):
        return "Movimiento"
    if nombre.startswith((
        "attacker_pitch_delta_", "attacker_yaw_delta_",
        "attacker_pitch_head_delta_", "attacker_yaw_head_delta_",
    )):
        return "Apuntado y orientación"
    if nombre.startswith(("delta_x_", "delta_y_", "delta_z_", "distancia_3d_")):
        return "Relación espacial"
    if nombre.startswith("weapon_"):
        return "Armas"
    if nombre.startswith((
        "attacker_flashed_", "victim_noise_", "attacker_shot_",
        "kill_", "victim_health_",
    )):
        return "Combate y estados"
    return "Otros"


def alinear_nombres(booster, predictores):
    """Identifica correctamente los atributos usados por XGBoost al entrenar."""
    reales = booster.feature_names
    if reales is None:
        # XGBoost entrenado sin columnas nombradas: convención f0..f71.
        return {f"f{i}": nombre for i, nombre in enumerate(predictores)}
    if len(reales) != len(predictores) or len(set(reales)) != len(reales):
        raise RuntimeError("Los nombres internos del booster no incluyen 72 variables únicas")
    if reales == predictores:
        return {nombre: nombre for nombre in predictores}
    if reales == [f"f{i}" for i in range(len(predictores))]:
        return {f"f{i}": nombre for i, nombre in enumerate(predictores)}
    if set(reales) == set(predictores):
        # Mismo conjunto, aunque el booster tenga otro orden.
        return {nombre: nombre for nombre in reales}
    raise RuntimeError(
        "Las variables del modelo XGBoost no coinciden con los 72 "
        "predictores congelados. No es seguro interpretar las importancias."
    )


def main():
    for ruta in (PREDICTORES, MODELO):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta {ruta}")
    nombres = [linea.strip() for linea in PREDICTORES.read_text(
        encoding="utf-8").splitlines() if linea.strip()]
    if len(nombres) != 72 or len(set(nombres)) != 72:
        raise ValueError("Se requieren los 72 predictores congelados y únicos")

    __main__.XGBPonderado = XGBPonderado
    pipeline = joblib.load(MODELO)
    if "modelo" not in getattr(pipeline, "named_steps", {}):
        raise RuntimeError("El pipeline XGBoost no tiene la etapa 'modelo'")
    estimador = pipeline.named_steps["modelo"]
    if not isinstance(estimador, XGBPonderado) or not hasattr(estimador, "modelo_"):
        raise RuntimeError("No se encontró el XGBoost entrenado dentro de XGBPonderado")
    booster = estimador.modelo_.get_booster()
    mapa = alinear_nombres(booster, nombres)

    # get_score omite predictores nunca utilizados en los árboles: asignar 0.
    tipos = {
        "total_gain": booster.get_score(importance_type="total_gain"),
        "gain": booster.get_score(importance_type="gain"),
        "weight": booster.get_score(importance_type="weight"),
    }
    for tipo, valores in tipos.items():
        desconocidas = set(valores) - set(mapa)
        if desconocidas:
            raise RuntimeError(
                f"El booster usa predictores desconocidos en {tipo}: {desconocidas}"
            )

    resultado = []
    for clave, nombre in mapa.items():
        resultado.append({
            "caracteristica": nombre,
            "grupo": grupo_de(nombre),
            "ganancia_total": float(tipos["total_gain"].get(clave, 0.0)),
            "ganancia_media_por_split": float(tipos["gain"].get(clave, 0.0)),
            "cantidad_splits": int(tipos["weight"].get(clave, 0)),
        })
    tabla = pd.DataFrame(resultado)
    if not np.isfinite(
        tabla[["ganancia_total", "ganancia_media_por_split"]].to_numpy()
    ).all() or (tabla["ganancia_total"] < 0).any():
        raise ValueError("El modelo devolvió importancias inválidas")
    suma = float(tabla["ganancia_total"].sum())
    if suma <= 0:
        raise RuntimeError("No hay ganancia positiva que permita crear importancias")
    tabla["importancia_relativa"] = tabla["ganancia_total"] / suma
    tabla = tabla.sort_values(
        ["importancia_relativa", "caracteristica"],
        ascending=[False, True],
    ).reset_index(drop=True)
    if len(tabla) != 72 or set(tabla["caracteristica"]) != set(nombres):
        raise RuntimeError("No se recuperaron correctamente los 72 predictores")

    # Comparación POR GRUPOS, incluyendo tamaño del grupo: el total por sí
    # solo favorece a grupos con más características.
    grupos = (tabla.groupby("grupo", as_index=False)
              .agg(n_caracteristicas=("caracteristica", "size"),
                   importancia_total=("importancia_relativa", "sum"),
                   importancia_media=("importancia_relativa", "mean"),
                   n_splits=("cantidad_splits", "sum"))
              .sort_values("importancia_total", ascending=False))

    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "importancias_xgboost.csv", index=False)
    grupos.to_csv(SALIDA / "importancias_grupos_xgboost.csv", index=False)

    top = tabla.head(N_SALIDA).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    ax.barh(top["caracteristica"], top["importancia_relativa"])
    ax.set_title("XGBoost: 20 características por ganancia total relativa")
    ax.set_xlabel("Importancia relativa (total_gain normalizada)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(SALIDA / "top20_importancias_xgboost.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    notas = {
        "modelo": str(MODELO),
        "n_predictores": 72,
        "metodo": "total_gain (reducción total de función objetivo en los splits)",
        "normalizacion": "total_gain de cada predictor / suma global de total_gain",
        "otras_medidas": {
            "gain": "ganancia media por split en que aparece el predictor",
            "weight": "cantidad de divisiones que emplean el predictor",
        },
        "predictores_sin_splits": int((tabla["cantidad_splits"] == 0).sum()),
        "grupo": "Mismas reglas descriptivas de agrupación usadas con RF",
        "sin_reentrenamiento": True,
        "uso_de_test": False,
        "uso_de_validacion": False,
        "limitaciones": (
            "Importancia interna de árboles, no causal. Puede verse afectada "
            "por correlación entre predictores, el número de divisiones y el "
            "criterio de entrenamiento. total_gain no comparte escala ni "
            "significado con MDI de RF o coeficientes de RL; comparar rankings "
            "y grupos con cautela. No se analizaron interacciones ni SHAP."
        ),
    }
    (SALIDA / "notas_importancias_xgboost.json").write_text(
        json.dumps(notas, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("IMPORTANCIAS DE XGBOOST (MODELO YA ENTRENADO)")
    print("\nTop 10 características por total_gain relativa:")
    print(tabla[["caracteristica", "importancia_relativa", "cantidad_splits"]]
          .head(10).to_string(index=False))
    print("\nResumen de grupos:")
    print(grupos.to_string(index=False))
    print("\nResultados en:", SALIDA)
    print("No se reentrenó XGBoost ni se abrió VALIDACIÓN o TEST.")


if __name__ == "__main__":
    main()
