"""Auditoría estratificada de originales vs _aug0 en múltiples partidas de TRAIN.

Selecciona una ventana por partida para cada clase (aleatorio reproducible).
Compara los Parquet únicamente de las claves de TRAIN seleccionadas.
No reentrena ni evalúa modelos y no usa VALIDACION ni TEST.
La selección busca diversidad de partidas, no es una muestra probabilística
representativa de todas las ventanas del dataset.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

TRAIN = Path("datos/procesado/caracteristicas_train.csv")
SALIDA = Path("salidas/auditoria_aug_estratificada")
RUTA_21 = Path(__file__).with_name("21_auditar_aug_cs2cd.py")


def cargar_funcion_auditoria():
    """Reutilizar el cálculo del experimento 21 sin ejecutar su main."""
    spec = importlib.util.spec_from_file_location("auditoria_aug_21", RUTA_21)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el script 21")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.auditar_par


def seleccionar_claves(n_por_clase, semilla):
    if not TRAIN.is_file():
        raise FileNotFoundError(f"Falta {TRAIN}")
    df = pd.read_csv(
        TRAIN,
        usecols=["clave", "partida_id", "etiqueta"],
        dtype={"clave": str, "partida_id": str, "etiqueta": int},
    )
    if df.empty or df[["clave", "partida_id", "etiqueta"]].isna().any().any():
        raise ValueError("TRAIN está vacío o contiene metadatos nulos")
    if df["clave"].duplicated().any() or set(df["etiqueta"].unique()) != {0, 1}:
        raise ValueError("TRAIN debe tener claves únicas y ambas clases")
    rng = np.random.default_rng(semilla)
    seleccion = []
    for clase in (0, 1):
        clase_df = df.loc[df["etiqueta"] == clase]
        grupos = clase_df.groupby("partida_id", sort=True)
        partidas = np.array(sorted(grupos.groups), dtype=object)
        if len(partidas) < n_por_clase:
            raise ValueError(
                f"Clase {clase}: Solo hay {len(partidas)} partidas "
                f"y se solicitaron {n_por_clase} parejas distintas"
            )
        partidas_elegidas = rng.choice(partidas, size=n_por_clase, replace=False)
        for partida in partidas_elegidas:
            indices = grupos.groups[str(partida)]
            elegido = int(rng.choice(indices.to_numpy()))
            fila = df.loc[elegido]
            seleccion.append({
                "clave_original": str(fila["clave"]),
                "partida_id": str(fila["partida_id"]),
                "etiqueta": int(fila["etiqueta"]),
            })
    return pd.DataFrame(seleccion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--por-clase", type=int, default=20)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()
    if args.por_clase < 1:
        raise ValueError("--por-clase debe ser positivo")

    seleccion = seleccionar_claves(args.por_clase, args.semilla)
    objetivos = seleccion.set_index("clave_original").to_dict(orient="index")
    auditar_par = cargar_funcion_auditoria()
    SALIDA.mkdir(parents=True, exist_ok=True)
    seleccion.to_csv(SALIDA / "claves_seleccionadas.csv", index=False)

    # Únicamente retener los Parquet de las claves seleccionadas.
    # Cada pareja puede encontrarse en distinto orden durante el streaming.
    pendientes = {}
    auditados, errores = {}, []
    visitados = 0
    print(
        f"Seleccionadas {len(seleccion)} ventanas de TRAIN "
        f"({args.por_clase} por clase, de partidas distintas).",
        flush=True,
    )
    print("Recorriendo CS2CD por streaming, sin decodificar otros Parquet...", flush=True)
    dataset = load_dataset("CS2CD/Context_window_256", split="train", streaming=True)
    for ejemplo in dataset:
        visitados += 1
        clave = str(ejemplo["__key__"])
        es_original = clave in objetivos
        es_aug0 = clave.endswith("_aug0") and clave[:-5] in objetivos
        if es_original or es_aug0:
            padre = clave if es_original else clave[:-5]
            if padre not in auditados:
                paquete = pendientes.setdefault(padre, {})
                paquete["original" if es_original else "aug"] = ejemplo["parquet"]
                if {"original", "aug"}.issubset(paquete):
                    try:
                        fila = auditar_par(
                            paquete["original"], paquete["aug"], padre,
                            padre + "_aug0", objetivos[padre]["etiqueta"],
                        )
                        fila["partida_id"] = objetivos[padre]["partida_id"]
                        auditados[padre] = fila
                    except Exception as exc:
                        errores.append({"clave_original": padre, "error": str(exc)})
                        auditados[padre] = None
                    del pendientes[padre]
        if visitados % 20000 == 0:
            exitosos = sum(f is not None for f in auditados.values())
            print(
                f"Registros recorridos: {visitados:,} | "
                f"Parejas válidas: {exitosos}/{len(objetivos)}",
                flush=True,
            )
        if len(auditados) == len(objetivos):
            break

    exitos = [fila for fila in auditados.values() if fila is not None]
    columnas_originales = [
        "clave_original", "clave_aug", "etiqueta", "partida_id",
        "max_cambio_no_espacial", "max_cambio_coordenadas",
        "max_cambio_relativo", "max_cambio_28_caracteristicas_espaciales",
        "n_caracteristicas_espaciales_cambio_mayor_1e6",
        "n_caracteristicas_espaciales_cambio_mayor_1e3",
    ]
    tabla = pd.DataFrame(exitos)
    for nombre in columnas_originales:
        if nombre not in tabla.columns:
            tabla[nombre] = pd.Series(dtype="float64")
    tabla.to_csv(SALIDA / "pares_auditados_estratificados.csv", index=False)
    pd.DataFrame(
        errores, columns=["clave_original", "error"]
    ).to_csv(SALIDA / "errores_auditoria.csv", index=False)
    faltantes = seleccion.loc[
        ~seleccion["clave_original"].isin(auditados.keys())
    ]
    faltantes.to_csv(SALIDA / "parejas_no_encontradas.csv", index=False)

    por_clase = []
    for clase in (0, 1):
        t = tabla.loc[tabla["etiqueta"] == clase]
        por_clase.append({
            "clase": clase,
            "parejas": len(t),
            "partidas": int(t["partida_id"].nunique()),
            "max_cambio_no_espacial": (
                float(t["max_cambio_no_espacial"].max()) if len(t) else None
            ),
            "max_cambio_relativo": (
                float(t["max_cambio_relativo"].max()) if len(t) else None
            ),
            "max_cambio_espacial": (
                float(t["max_cambio_28_caracteristicas_espaciales"].max())
                if len(t) else None
            ),
            "n_pares_con_cambio_espacial_mayor_1e6": int(
                (t["n_caracteristicas_espaciales_cambio_mayor_1e6"] > 0).sum()
            ),
            "n_pares_con_cambio_espacial_mayor_1e3": int(
                (t["n_caracteristicas_espaciales_cambio_mayor_1e3"] > 0).sum()
            ),
        })
    pd.DataFrame(por_clase).to_csv(
        SALIDA / "resumen_por_clase.csv", index=False
    )
    notas = {
        "seleccion": "Una ventana aleatoria por partida, separada por clase",
        "semilla": args.semilla,
        "solicitadas_por_clase": args.por_clase,
        "auditadas": len(exitos),
        "errores": len(errores),
        "sin_pareja_encontrada": len(faltantes),
        "registros_recorridos": visitados,
        "solo_train": True,
        "criterio": "Compara original con variante _aug0 de la misma clave",
        "limitacion": (
            "Muestra estratificada para diversidad de partidas, no "
            "representativa de la distribución real de ventanas. "
            "Solo se compara _aug0. No se realiza entrenamiento, "
            "selección de umbral ni evaluación externa."
        ),
    }
    (SALIDA / "notas_metodologicas.json").write_text(
        json.dumps(notas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nRESUMEN POR CLASE")
    print(pd.DataFrame(por_clase).to_string(index=False))
    print(
        f"\nPares válidos: {len(exitos)}/{len(objetivos)}. "
        f"Errores: {len(errores)}. No encontrados: {len(faltantes)}."
    )
    print("Archivos guardados en:", SALIDA)
    print("No se entrenó ni se abrió VALIDACION o TEST.")
    if errores or len(faltantes):
        print("ADVERTENCIA: Revisar los archivos de errores y parejas faltantes.")


if __name__ == "__main__":
    main()
