"""Auditoría muestral de ventanas CS2CD originales y aumentadas (solo TRAIN).

Verifica qué cambia en los archivos Parquet originales y _augN.
Fuente del método: Pinkvinus/CS2_cheat_detection/DataAugmentation/dataaugmentation.py.
El ruido se aplica a las coordenadas, con recorte a [0, 1] y conversión float32.
Este procedimiento NO reentrena, ni utiliza VALIDACION ni TEST.
La muestra consecutiva del streaming no es estadísticamente representativa.
"""
import argparse
import json
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

TRAIN = Path("datos/procesado/caracteristicas_train.csv")
SALIDA = Path("salidas/auditoria_aug")
PATRON_AUG = re.compile(r"_aug\d+$")
COORDENADAS = {
    f"{jugador}_{eje}"
    for jugador in ("attacker", "victim")
    for eje in ("X", "Y", "Z")
}


def extraer_28_espaciales(df):
    """Replica las 28 características espaciales del script 4."""
    segmento = df.iloc[:224]
    series = {}
    diferencias = []
    for eje in ("X", "Y", "Z"):
        v = (segmento[f"victim_{eje}"].to_numpy(dtype=float)
             - segmento[f"attacker_{eje}"].to_numpy(dtype=float))
        series[f"delta_{eje.lower()}"] = v
        diferencias.append(v)
    series["distancia_3d"] = np.sqrt(sum(d * d for d in diferencias))
    salida = {}
    for nombre, valores in series.items():
        cambios = np.abs(np.diff(valores))
        salida.update({
            f"{nombre}_media": float(np.mean(valores)),
            f"{nombre}_var": float(np.var(valores, ddof=0)),
            f"{nombre}_min": float(np.min(valores)),
            f"{nombre}_max": float(np.max(valores)),
            f"{nombre}_rango": float(np.ptp(valores)),
            f"{nombre}_cambio_media_abs": float(np.mean(cambios)),
            f"{nombre}_cambio_max_abs": float(np.max(cambios)),
        })
    assert len(salida) == 28
    return salida


def auditar_par(orig_bytes, aug_bytes, clave_original, clave_aug, etiqueta):
    a = pd.read_parquet(BytesIO(orig_bytes))
    b = pd.read_parquet(BytesIO(aug_bytes))
    if a.shape != (256, 44) or b.shape != (256, 44):
        raise ValueError("Se esperaban 256 ticks y 44 columnas")
    if set(a.columns) != set(b.columns) or not COORDENADAS.issubset(a.columns):
        raise ValueError("Columnas diferentes o coordenadas ausentes")
    if a.iloc[223]["attacker_kill"] != 1 or b.iloc[223]["attacker_kill"] != 1:
        raise ValueError("No coincide el evento kill del tick 223")

    max_otros = 0.0
    for columna in sorted(set(a.columns) - COORDENADAS):
        x = a[columna].to_numpy(dtype=float)
        z = b[columna].to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError(f"Valores no finitos en {columna}")
        max_otros = max(max_otros, float(np.max(np.abs(z - x))))

    cambio_coords = 0.0
    cambio_relativo = 0.0
    for eje in ("X", "Y", "Z"):
        at_a = a[f"attacker_{eje}"].to_numpy(dtype=float)
        at_b = b[f"attacker_{eje}"].to_numpy(dtype=float)
        vi_a = a[f"victim_{eje}"].to_numpy(dtype=float)
        vi_b = b[f"victim_{eje}"].to_numpy(dtype=float)
        if not all(np.isfinite(x).all() for x in (at_a, at_b, vi_a, vi_b)):
            raise ValueError("Coordenadas no finitas")
        cambio_coords = max(cambio_coords, float(np.max(np.abs(at_b - at_a))),
                            float(np.max(np.abs(vi_b - vi_a))))
        cambio_relativo = max(
            cambio_relativo,
            float(np.max(np.abs((vi_b - at_b) - (vi_a - at_a)))),
        )

    f_original, f_aug = extraer_28_espaciales(a), extraer_28_espaciales(b)
    variaciones = [abs(f_aug[c] - f_original[c]) for c in f_original]
    return {
        "clave_original": clave_original,
        "clave_aug": clave_aug,
        "etiqueta": int(etiqueta),
        "max_cambio_no_espacial": max_otros,
        "no_espaciales_iguales_tolerancia_1e5": max_otros <= 1e-5,
        "max_cambio_coordenadas": cambio_coords,
        "max_cambio_relativo": cambio_relativo,
        "max_cambio_28_caracteristicas_espaciales": max(variaciones),
        "n_caracteristicas_espaciales_cambio_mayor_1e6": sum(v > 1e-6 for v in variaciones),
        "n_caracteristicas_espaciales_cambio_mayor_1e3": sum(v > 1e-3 for v in variaciones),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pares", type=int, default=20)
    parser.add_argument("--max-registros", type=int, default=50000)
    args = parser.parse_args()
    if args.pares < 1 or args.max_registros < 1:
        raise ValueError("Los límites deben ser positivos")
    if not TRAIN.is_file():
        raise FileNotFoundError(f"Falta {TRAIN}")
    tabla_train = pd.read_csv(TRAIN, usecols=["clave", "etiqueta"])
    if tabla_train["clave"].duplicated().any():
        raise ValueError("Claves duplicadas en TRAIN")
    etiquetas_train = dict(zip(tabla_train["clave"].astype(str),
                               tabla_train["etiqueta"].astype(int)))

    originales = {}
    auditados, errores = [], []
    visitados = 0
    print("Auditoría por streaming: solo se analizan pares pertenecientes a TRAIN.",
          flush=True)
    dataset = load_dataset("CS2CD/Context_window_256", split="train", streaming=True)
    for ejemplo in dataset:
        visitados += 1
        if visitados > args.max_registros or len(auditados) >= args.pares:
            break
        clave = str(ejemplo["__key__"])
        if PATRON_AUG.search(clave):
            padre = PATRON_AUG.sub("", clave)
            if padre in originales:
                try:
                    resultado = auditar_par(
                        originales.pop(padre), ejemplo["parquet"],
                        padre, clave, etiquetas_train[padre],
                    )
                    auditados.append(resultado)
                except Exception as exc:
                    errores.append({"clave_aug": clave, "error": str(exc)})
                if auditados and len(auditados) % 5 == 0:
                    print(f"Pares revisados: {len(auditados)}", flush=True)
        elif clave in etiquetas_train and len(originales) + len(auditados) < args.pares:
            originales[clave] = ejemplo["parquet"]

    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(auditados).to_csv(SALIDA / "pares_auditados.csv", index=False)
    pd.DataFrame(errores, columns=["clave_aug", "error"]).to_csv(
        SALIDA / "errores_auditoria.csv", index=False,
    )
    if not auditados:
        raise RuntimeError("No se encontraron pares auditables dentro del límite.")
    datos = pd.DataFrame(auditados)
    resumen = {
        "pares_auditados": len(datos),
        "registros_revisados": min(visitados, args.max_registros),
        "positivos_auditados": int((datos["etiqueta"] == 1).sum()),
        "negativos_auditados": int((datos["etiqueta"] == 0).sum()),
        "errores": len(errores),
        "max_cambio_no_espacial": float(datos["max_cambio_no_espacial"].max()),
        "max_cambio_relativo": float(datos["max_cambio_relativo"].max()),
        "max_cambio_espacial": float(
            datos["max_cambio_28_caracteristicas_espaciales"].max()),
        "pares_con_algun_cambio_espacial_mayor_0_001": int(
            (datos["n_caracteristicas_espaciales_cambio_mayor_1e3"] > 0).sum()),
        "no_espaciales_sin_cambio_importante": int(
            datos["no_espaciales_iguales_tolerancia_1e5"].sum()),
        "limitaciones": (
            "Muestra consecutiva del streaming, no representativa. "
            "Se comprueba una sola variante _aug por ventana original de TRAIN. "
            "El recorte [0,1] puede alterar distancias relativas. "
            "Ni VALIDACION ni TEST se utilizaron para entrenar o evaluar."
        ),
    }
    (SALIDA / "resumen_auditoria.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("\nAUDITORÍA DE AUG (SOLO TRAIN)")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print("\nResultados guardados en:", SALIDA)
    print("No se modificaron modelos ni particiones.")


if __name__ == "__main__":
    main()
