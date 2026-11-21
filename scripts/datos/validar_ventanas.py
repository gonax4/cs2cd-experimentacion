from datasets import load_dataset
import pandas as pd
from io import BytesIO
from pathlib import Path
import re


N_VENTANAS = 100
TICK_OBJETIVO = 223


print("VALIDACIÓN DE 100 VENTANAS ORIGINALES DE CS2CD")
print("================================================\n")


dataset = load_dataset(
    "CS2CD/Context_window_256",
    split="train",
    streaming=True
)

print("Dataset conectado.\n")


# Detecta muestras aumentadas como:
# ...kill_0_aug0
# ...kill_0_aug1
# etc.
patron_aumento = re.compile(r"_aug\d+$")


# Ejemplos esperados:
# cdata-cheater-file_0-Player_3-kill_0
# cdata-notcheater-file_0-Player_5-kill_2
# ncdata-notcheater-file_4-Player_2-kill_1
patron_nombre = re.compile(
    r"^(cdata|ncdata)-(cheater|notcheater)-"
    r"file_(\d+)-(.+)-kill_(\d+)$"
)


resultados = []

procesadas = 0
aumentadas_ignoradas = 0
nombres_no_reconocidos = 0


for ejemplo in dataset:

    if procesadas >= N_VENTANAS:
        break

    clave = ejemplo["__key__"]
    nombre = clave.split("/")[-1]


    # 1. Ignorar ventanas creadas mediante aumento de datos
    if patron_aumento.search(nombre):
        aumentadas_ignoradas += 1
        continue


    # 2. Interpretar el nombre de la ventana
    coincidencia = patron_nombre.match(nombre)

    if coincidencia is None:
        nombres_no_reconocidos += 1
        print("\nNombre no reconocido:", nombre)
        continue


    origen = coincidencia.group(1)
    clase = coincidencia.group(2)
    archivo_id = int(coincidencia.group(3))
    jugador_id = coincidencia.group(4)
    kill_id = int(coincidencia.group(5))

    partida_id = f"{origen}_file_{archivo_id}"

    etiqueta = 1 if clase == "cheater" else 0


    # 3. Abrir la ventana Parquet
    df = pd.read_parquet(
        BytesIO(ejemplo["parquet"])
    )


    # 4. Encontrar todos los ticks en los que hubo una eliminación
    ticks_kill = df.index[
        df["attacker_kill"] == 1
    ].tolist()

    cantidad_kill = len(ticks_kill)


    # 5. Comprobar que la eliminación objetivo está en el tick 223
    kill_en_tick_objetivo = (
        len(df) > TICK_OBJETIVO
        and df.loc[TICK_OBJETIVO, "attacker_kill"] == 1
    )


    # 6. Recortar desde el inicio hasta la eliminación objetivo
    if kill_en_tick_objetivo:

        tick_kill = TICK_OBJETIVO

        segmento = df.iloc[
            :TICK_OBJETIVO + 1
        ].copy()

        filas_hasta_kill = len(segmento)

    else:

        tick_kill = None
        filas_hasta_kill = None


    # 7. Comprobar la estructura general
    estructura_correcta = (
        len(df) == 256
        and len(df.columns) == 44
        and kill_en_tick_objetivo
    )


    # 8. Guardar resultado de esta ventana
    resultados.append({
        "ventana": procesadas,
        "origen": origen,
        "partida_id": partida_id,
        "jugador_id": jugador_id,
        "kill_id": kill_id,
        "clase": clase,
        "etiqueta": etiqueta,
        "filas_originales": len(df),
        "columnas": len(df.columns),
        "cantidad_attacker_kill": cantidad_kill,
        "ticks_con_kill": str(ticks_kill),
        "kill_en_tick_objetivo": kill_en_tick_objetivo,
        "tick_kill": tick_kill,
        "filas_hasta_kill": filas_hasta_kill,
        "estructura_correcta": estructura_correcta
    })


    procesadas += 1

    print(
        f"\rVentanas originales procesadas: "
        f"{procesadas}/{N_VENTANAS}",
        end=""
    )


# 9. Crear tabla con los resultados
resumen = pd.DataFrame(resultados)


Path("salidas").mkdir(exist_ok=True)

ruta_salida = Path(
    "salidas/validacion_100_ventanas_originales.csv"
)


resumen.to_csv(
    ruta_salida,
    index=False
)


# 10. Mostrar resumen
print("\n\nRESULTADOS")
print("==========")

print("\nVentanas originales analizadas:")
print(len(resumen))

print("\nMuestras aumentadas ignoradas:")
print(aumentadas_ignoradas)

print("\nNombres no reconocidos:")
print(nombres_no_reconocidos)

print("\nEstructura correcta:")
print(
    resumen["estructura_correcta"].value_counts(
        dropna=False
    )
)

print("\nEliminación presente en tick objetivo 223:")
print(
    resumen["kill_en_tick_objetivo"].value_counts(
        dropna=False
    )
)

print("\nCantidad de eliminaciones dentro de cada ventana:")
print(
    resumen["cantidad_attacker_kill"]
    .value_counts()
    .sort_index()
)

print("\nPosición de la eliminación objetivo:")
print(
    resumen["tick_kill"]
    .value_counts()
    .sort_index()
)

print("\nLongitud del segmento utilizado:")
print(
    resumen["filas_hasta_kill"]
    .value_counts()
    .sort_index()
)

print("\nClases encontradas:")
print(
    resumen["clase"].value_counts()
)

print("\nPartidas distintas:")
print(
    resumen["partida_id"].nunique()
)

print("\nPrimeras 10 ventanas:")
print(
    resumen.head(10).to_string(index=False)
)

print("\nVentanas con más de una eliminación:")
print(
    resumen[
        resumen["cantidad_attacker_kill"] > 1
    ][
        [
            "partida_id",
            "jugador_id",
            "kill_id",
            "cantidad_attacker_kill",
            "ticks_con_kill",
            "kill_en_tick_objetivo"
        ]
    ].head(20).to_string(index=False)
)

print("\nArchivo generado:")
print(ruta_salida)