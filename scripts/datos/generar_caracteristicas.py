from datasets import load_dataset
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
import re

TICK_OBJETIVO = 223
LONGITUD_SEGMENTO = 224
LOTE_ESCRITURA = 500

RUTA_INDICE = Path("datos/procesado/indice_ventanas_originales.csv")
CARPETA_SALIDA = Path("datos/procesado")

ARCHIVOS_SALIDA = {
    "TRAIN": CARPETA_SALIDA / "caracteristicas_train.csv",
    "VALIDACION": CARPETA_SALIDA / "caracteristicas_validacion.csv",
    "TEST": CARPETA_SALIDA / "caracteristicas_test.csv",
}

PATRON_AUMENTO = re.compile(r"_aug\d+$")

COLUMNAS_ARMAS = [
    "attacker_weapon_knife",
    "attacker_weapon_auto_rifle",
    "attacker_weapon_semi_rifle",
    "attacker_weapon_pistol",
    "attacker_weapon_grenade",
    "attacker_weapon_smg",
    "attacker_weapon_shotgun",
]

def estadisticas_basicas(valores, prefijo):
    x = np.asarray(valores, dtype=np.float64)
    return {
        f"{prefijo}_media": float(np.mean(x)),
        f"{prefijo}_var": float(np.var(x, ddof=0)),
        f"{prefijo}_min": float(np.min(x)),
        f"{prefijo}_max": float(np.max(x)),
        f"{prefijo}_rango": float(np.max(x) - np.min(x)),
    }

def estadisticas_delta_angular(valores, prefijo):
    x = np.asarray(valores, dtype=np.float64)
    abs_x = np.abs(x)
    return {
        f"{prefijo}_media_abs": float(np.mean(abs_x)),
        f"{prefijo}_var": float(np.var(x, ddof=0)),
        f"{prefijo}_max_abs": float(np.max(abs_x)),
    }

def estadisticas_delta_cabeza(valores, prefijo):
    x = np.asarray(valores, dtype=np.float64)
    abs_x = np.abs(x)
    return {
        f"{prefijo}_media_abs": float(np.mean(abs_x)),
        f"{prefijo}_min_abs": float(np.min(abs_x)),
        f"{prefijo}_var": float(np.var(x, ddof=0)),
        f"{prefijo}_kill": float(x[-1]),
    }

def estadisticas_espaciales(valores, prefijo):
    x = np.asarray(valores, dtype=np.float64)
    cambios = np.abs(np.diff(x))
    resultado = {
        f"{prefijo}_media": float(np.mean(x)),
        f"{prefijo}_var": float(np.var(x, ddof=0)),
        f"{prefijo}_min": float(np.min(x)),
        f"{prefijo}_max": float(np.max(x)),
        f"{prefijo}_rango": float(np.max(x) - np.min(x)),
    }
    resultado[f"{prefijo}_cambio_media_abs"] = float(np.mean(cambios)) if len(cambios) else 0.0
    resultado[f"{prefijo}_cambio_max_abs"] = float(np.max(cambios)) if len(cambios) else 0.0
    return resultado

def estado_armas_por_tick(segmento):
    matriz = segmento[COLUMNAS_ARMAS].to_numpy(dtype=np.float64)
    activos = matriz > 0.5
    cantidad_activas = activos.sum(axis=1)
    estados = np.full(len(segmento), -1, dtype=int)
    exactamente_una = cantidad_activas == 1
    estados[exactamente_una] = np.argmax(activos[exactamente_una], axis=1)
    estados[cantidad_activas > 1] = -2
    return estados, cantidad_activas

def extraer_caracteristicas(df, metadata):
    if len(df) != 256:
        raise ValueError(f"Ventana {metadata['clave']} con {len(df)} filas, se esperaban 256.")
    if len(df.columns) != 44:
        raise ValueError(f"Ventana {metadata['clave']} con {len(df.columns)} columnas, se esperaban 44.")
    if df.loc[TICK_OBJETIVO, "attacker_kill"] != 1:
        raise ValueError(f"Ventana {metadata['clave']} sin kill objetivo en el tick 223.")

    segmento = df.iloc[:LONGITUD_SEGMENTO].copy()

    f = {
        "clave": metadata["clave"],
        "partida_id": metadata["partida_id"],
        "origen": metadata["origen"],
        "archivo_id": int(metadata["archivo_id"]),
        "jugador_id": metadata["jugador_id"],
        "kill_id": int(metadata["kill_id"]),
        "clase": metadata["clase"],
        "etiqueta": int(metadata["etiqueta"]),
        "particion": metadata["particion"],
    }

    f.update(estadisticas_basicas(segmento["attacker_vel"], "attacker_vel"))
    f.update(estadisticas_delta_angular(segmento["attacker_pitch_delta"], "attacker_pitch_delta"))
    f.update(estadisticas_delta_angular(segmento["attacker_yaw_delta"], "attacker_yaw_delta"))
    f.update(estadisticas_delta_cabeza(segmento["attacker_pitch_head_delta"], "attacker_pitch_head_delta"))
    f.update(estadisticas_delta_cabeza(segmento["attacker_yaw_head_delta"], "attacker_yaw_head_delta"))

    dx = segmento["victim_X"].to_numpy(dtype=np.float64) - segmento["attacker_X"].to_numpy(dtype=np.float64)
    dy = segmento["victim_Y"].to_numpy(dtype=np.float64) - segmento["attacker_Y"].to_numpy(dtype=np.float64)
    dz = segmento["victim_Z"].to_numpy(dtype=np.float64) - segmento["attacker_Z"].to_numpy(dtype=np.float64)
    distancia = np.sqrt(dx**2 + dy**2 + dz**2)

    f.update(estadisticas_espaciales(dx, "delta_x"))
    f.update(estadisticas_espaciales(dy, "delta_y"))
    f.update(estadisticas_espaciales(dz, "delta_z"))
    f.update(estadisticas_espaciales(distancia, "distancia_3d"))

    flashed = segmento["attacker_flashed"].to_numpy(dtype=np.float64) > 0
    ruido = segmento["victim_noise"].to_numpy(dtype=np.float64) > 0
    disparos = segmento["attacker_shot"].to_numpy(dtype=np.float64) > 0

    f["attacker_flashed_proporcion_activo"] = float(np.mean(flashed))
    f["victim_noise_proporcion_activo"] = float(np.mean(ruido))
    f["attacker_shot_conteo"] = int(np.sum(disparos))
    f["attacker_shot_proporcion"] = float(np.mean(disparos))

    fila_kill = segmento.iloc[-1]
    f["kill_headshot"] = int(fila_kill["is_kill_headshot"] > 0)
    f["kill_through_smoke"] = int(fila_kill["is_kill_through_smoke"] > 0)
    f["kill_wallbang"] = int(fila_kill["is_kill_wallbang"] > 0)
    f["kill_attacker_midair"] = int(fila_kill["attacker_midair"] > 0)

    for columna in COLUMNAS_ARMAS:
        nombre = columna.replace("attacker_weapon_", "")
        valores = segmento[columna].to_numpy(dtype=np.float64) > 0.5
        f[f"weapon_{nombre}_proporcion"] = float(np.mean(valores))
        f[f"weapon_{nombre}_kill"] = int(fila_kill[columna] > 0.5)

    estados_armas, cantidad_activas = estado_armas_por_tick(segmento)

    estado_anterior = estados_armas[:-1]
    estado_actual = estados_armas[1:]

    # Solo se considera una transición cuando ambos ticks
    # tienen exactamente una categoría de arma válida.
    pares_validos = (
        (estado_anterior >= 0)
        & (estado_actual >= 0)
    )

    f["weapon_transiciones"] = int(
        np.sum(
            pares_validos
            & (estado_anterior != estado_actual)
        )
    )

    f["diag_weapon_ticks_sin_categoria"] = int(
        np.sum(cantidad_activas == 0)
    )

    f["diag_weapon_ticks_multiples"] = int(
        np.sum(cantidad_activas > 1)
    )

    f["diag_weapon_categorias_activas_kill"] = int(
        cantidad_activas[-1]
    )

    salud = segmento["victim_health"].to_numpy(dtype=np.float64)
    f["victim_health_media"] = float(np.mean(salud))
    f["victim_health_min"] = float(np.min(salud))
    f["victim_health_kill"] = float(salud[-1])
    f["victim_health_cambio_inicio_kill"] = float(salud[-1] - salud[0])

    return f

def claves_ya_procesadas():
    procesadas = set()
    for ruta in ARCHIVOS_SALIDA.values():
        if ruta.exists() and ruta.stat().st_size > 0:
            for chunk in pd.read_csv(ruta, usecols=["clave"], chunksize=10000):
                procesadas.update(chunk["clave"].astype(str).tolist())
    return procesadas

def escribir_lote(particion, filas):
    if not filas:
        return
    ruta = ARCHIVOS_SALIDA[particion]
    existe = ruta.exists() and ruta.stat().st_size > 0
    pd.DataFrame(filas).to_csv(ruta, mode="a", header=not existe, index=False)

print("GENERACIÓN DE CARACTERÍSTICAS - CS2CD")
print("=====================================\n")

if not RUTA_INDICE.exists():
    raise FileNotFoundError(
        "No se encontró datos/procesado/indice_ventanas_originales.csv."
    )

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
indice = pd.read_csv(RUTA_INDICE)

columnas_indice_requeridas = {
    "clave", "origen", "archivo_id", "partida_id", "jugador_id",
    "kill_id", "clase", "etiqueta", "particion"
}
faltantes_indice = columnas_indice_requeridas - set(indice.columns)
if faltantes_indice:
    raise ValueError(
        "Faltan columnas en indice_ventanas_originales.csv: "
        + ", ".join(sorted(faltantes_indice))
    )

metadata_por_clave = indice.set_index("clave").to_dict(orient="index")
esperadas_por_particion = indice["particion"].value_counts().to_dict()

print("Ventanas esperadas:")
for particion in ["TRAIN", "VALIDACION", "TEST"]:
    print(f"  {particion}: {esperadas_por_particion.get(particion, 0):,}")

procesadas_previamente = claves_ya_procesadas()
if procesadas_previamente:
    print(
        f"\nSe encontraron {len(procesadas_previamente):,} ventanas ya procesadas. "
        "Se intentará reanudar."
    )

dataset = load_dataset(
    "CS2CD/Context_window_256",
    split="train",
    streaming=True
)

buffers = {"TRAIN": [], "VALIDACION": [], "TEST": []}
nuevas = {"TRAIN": 0, "VALIDACION": 0, "TEST": 0}
aumentadas_ignoradas = 0
originales_fuera_indice = 0
errores = []
visitadas_originales = 0

print("\nProcesando ventanas originales...\n")

for ejemplo in dataset:
    clave = ejemplo["__key__"]
    nombre = clave.split("/")[-1]

    if PATRON_AUMENTO.search(nombre):
        aumentadas_ignoradas += 1
        continue

    visitadas_originales += 1

    if clave in procesadas_previamente:
        continue

    metadata = metadata_por_clave.get(clave)
    if metadata is None:
        originales_fuera_indice += 1
        continue

    metadata = dict(metadata)
    metadata["clave"] = clave
    particion = metadata["particion"]

    try:
        df = pd.read_parquet(BytesIO(ejemplo["parquet"]))
        fila = extraer_caracteristicas(df, metadata)
        buffers[particion].append(fila)
        nuevas[particion] += 1
    except Exception as exc:
        errores.append({"clave": clave, "error": str(exc)})
        print(f"\nERROR en {clave}: {exc}")

    for nombre_particion in buffers:
        if len(buffers[nombre_particion]) >= LOTE_ESCRITURA:
            escribir_lote(nombre_particion, buffers[nombre_particion])
            buffers[nombre_particion].clear()

    total_nuevas = sum(nuevas.values())
    if total_nuevas > 0 and total_nuevas % 1000 == 0:
        print(f"\rNuevas ventanas procesadas: {total_nuevas:,}", end="")

for particion in buffers:
    escribir_lote(particion, buffers[particion])
    buffers[particion].clear()

print("\n\nRECORRIDO TERMINADO")
print("===================")
print(f"\nVentanas originales visitadas: {visitadas_originales:,}")
print(f"Muestras aumentadas ignoradas: {aumentadas_ignoradas:,}")
print(f"Originales fuera del índice: {originales_fuera_indice:,}")
print(f"Errores durante extracción: {len(errores):,}")

if errores:
    ruta_errores = Path("salidas/errores_generacion_caracteristicas.csv")
    ruta_errores.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(errores).to_csv(ruta_errores, index=False)
    print(f"Detalle de errores guardado en: {ruta_errores}")

resumen_final = []

print("\nARCHIVOS FINALES")
print("================")

for particion in ["TRAIN", "VALIDACION", "TEST"]:
    ruta = ARCHIVOS_SALIDA[particion]

    if not ruta.exists():
        cantidad = 0
        columnas = 0
    else:
        df_control = pd.read_csv(ruta)
        cantidad = len(df_control)
        columnas = len(df_control.columns)
        duplicadas = df_control["clave"].duplicated().sum()
        if duplicadas > 0:
            raise RuntimeError(f"{ruta} contiene {duplicadas} claves duplicadas.")

    esperado = int(esperadas_por_particion.get(particion, 0))
    coincide = cantidad == esperado

    resumen_final.append({
        "particion": particion,
        "filas_generadas": cantidad,
        "filas_esperadas": esperado,
        "columnas_totales": columnas,
        "coincide_con_indice": coincide,
    })

    print(
        f"{particion}: {cantidad:,}/{esperado:,} filas "
        f"- columnas: {columnas} - {'OK' if coincide else 'REVISAR'}"
    )

resumen_final = pd.DataFrame(resumen_final)
Path("salidas").mkdir(parents=True, exist_ok=True)
resumen_final.to_csv(
    "salidas/resumen_generacion_caracteristicas.csv",
    index=False
)

if (
    resumen_final["coincide_con_indice"].all()
    and len(errores) == 0
    and originales_fuera_indice == 0
):
    print("\nGENERACIÓN COMPLETA Y CONSISTENTE.")
    print(
        "Siguiente paso: analizar únicamente TRAIN antes de entrenar los modelos."
    )
else:
    print(
        "\nLa generación terminó, pero hay elementos por revisar "
        "antes de continuar con los modelos."
    )
