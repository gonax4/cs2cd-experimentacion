from datasets import load_dataset
import pandas as pd
import numpy as np
from pathlib import Path
import re


SEMILLA = 42


print("CONSTRUCCIÓN DEL ÍNDICE Y PARTICIONES DE CS2CD")
print("================================================\n")


dataset = load_dataset(
    "CS2CD/Context_window_256",
    split="train",
    streaming=True
)


patron_aumento = re.compile(r"_aug\d+$")

patron_nombre = re.compile(
    r"^(cdata|ncdata)-(cheater|notcheater)-"
    r"file_(\d+)-(.+)-kill_(\d+)$"
)


registros = []

aumentadas_ignoradas = 0
nombres_no_reconocidos = 0


print("Recorriendo el dataset...")

for ejemplo in dataset:

    clave = ejemplo["__key__"]
    nombre = clave.split("/")[-1]


    # Excluir muestras creadas mediante aumento de datos
    if patron_aumento.search(nombre):
        aumentadas_ignoradas += 1
        continue


    coincidencia = patron_nombre.match(nombre)

    if coincidencia is None:

        nombres_no_reconocidos += 1

        print(
            "\nADVERTENCIA - nombre no reconocido:",
            nombre
        )

        continue


    origen = coincidencia.group(1)
    clase = coincidencia.group(2)
    archivo_id = int(coincidencia.group(3))
    jugador_id = coincidencia.group(4)
    kill_id = int(coincidencia.group(5))


    # cdata_file_0 y ncdata_file_0 son partidas diferentes.
    partida_id = f"{origen}_file_{archivo_id}"


    etiqueta = (
        1 if clase == "cheater"
        else 0
    )


    registros.append({
        "clave": clave,
        "origen": origen,
        "archivo_id": archivo_id,
        "partida_id": partida_id,
        "jugador_id": jugador_id,
        "kill_id": kill_id,
        "clase": clase,
        "etiqueta": etiqueta
    })


    if len(registros) % 5000 == 0:

        print(
            f"\rVentanas originales indexadas: "
            f"{len(registros):,}",
            end=""
        )


print("\n\nRecorrido terminado.")


# ------------------------------------------------
# Crear índice completo de ventanas originales
# ------------------------------------------------

indice = pd.DataFrame(registros)


print("\nVentanas originales encontradas:")
print(f"{len(indice):,}")

print("\nMuestras aumentadas ignoradas:")
print(f"{aumentadas_ignoradas:,}")

print("\nNombres no reconocidos:")
print(nombres_no_reconocidos)


if nombres_no_reconocidos > 0:
    raise RuntimeError(
        "Se encontraron nombres no reconocidos. "
        "No se crearán las particiones hasta revisarlos."
    )


# ------------------------------------------------
# Obtener las partidas únicas
# ------------------------------------------------

partidas = (
    indice[
        ["partida_id", "origen"]
    ]
    .drop_duplicates()
    .sort_values("partida_id")
    .reset_index(drop=True)
)


print("\nNúmero total de partidas:")
print(len(partidas))


print("\nPartidas según origen:")
print(
    partidas["origen"].value_counts()
)


# ------------------------------------------------
# Mezclar PARTIDAS, no ventanas
# ------------------------------------------------

rng = np.random.default_rng(SEMILLA)


ids_partidas = (
    partidas["partida_id"]
    .to_numpy()
    .copy()
)


rng.shuffle(ids_partidas)


total_partidas = len(ids_partidas)


n_train = round(
    total_partidas * 0.70
)

n_validacion = round(
    total_partidas * 0.15
)


partidas_train = set(
    ids_partidas[:n_train]
)

partidas_validacion = set(
    ids_partidas[
        n_train:
        n_train + n_validacion
    ]
)

partidas_test = set(
    ids_partidas[
        n_train + n_validacion:
    ]
)


# ------------------------------------------------
# Verificar que no haya fugas entre conjuntos
# ------------------------------------------------

assert partidas_train.isdisjoint(
    partidas_validacion
)

assert partidas_train.isdisjoint(
    partidas_test
)

assert partidas_validacion.isdisjoint(
    partidas_test
)


assert len(
    partidas_train
    | partidas_validacion
    | partidas_test
) == total_partidas


# ------------------------------------------------
# Asignar cada ventana a su partición
# ------------------------------------------------

def asignar_particion(partida_id):

    if partida_id in partidas_train:
        return "TRAIN"

    if partida_id in partidas_validacion:
        return "VALIDACION"

    if partida_id in partidas_test:
        return "TEST"

    raise ValueError(
        f"Partida sin asignar: {partida_id}"
    )


indice["particion"] = (
    indice["partida_id"]
    .apply(asignar_particion)
)


# ------------------------------------------------
# Crear carpeta de salida
# ------------------------------------------------

Path("datos/procesado").mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------
# Guardar índice completo
# ------------------------------------------------

indice.to_csv(
    "datos/procesado/"
    "indice_ventanas_originales.csv",
    index=False
)


# ------------------------------------------------
# Guardar IDs de las partidas
# ------------------------------------------------

pd.DataFrame({
    "partida_id": sorted(partidas_train)
}).to_csv(
    "datos/procesado/partidas_train.csv",
    index=False
)


pd.DataFrame({
    "partida_id": sorted(partidas_validacion)
}).to_csv(
    "datos/procesado/partidas_validacion.csv",
    index=False
)


pd.DataFrame({
    "partida_id": sorted(partidas_test)
}).to_csv(
    "datos/procesado/partidas_test.csv",
    index=False
)


# ------------------------------------------------
# Crear resumen
# ------------------------------------------------

resumen = (
    indice
    .groupby("particion")
    .agg(
        partidas=(
            "partida_id",
            "nunique"
        ),
        ventanas=(
            "partida_id",
            "size"
        ),
        cheater=(
            "etiqueta",
            lambda x: int(
                (x == 1).sum()
            )
        ),
        notcheater=(
            "etiqueta",
            lambda x: int(
                (x == 0).sum()
            )
        )
    )
)


orden = [
    "TRAIN",
    "VALIDACION",
    "TEST"
]

resumen = resumen.reindex(orden)


resumen["porcentaje_partidas"] = (
    resumen["partidas"]
    / total_partidas
    * 100
)


resumen["porcentaje_ventanas"] = (
    resumen["ventanas"]
    / len(indice)
    * 100
)


resumen.to_csv(
    "datos/procesado/"
    "resumen_particiones.csv"
)


# ------------------------------------------------
# Mostrar resultados
# ------------------------------------------------

print("\n\nPARTICIONES CREADAS")
print("===================")

print("\nSemilla:")
print(SEMILLA)

print("\nResumen:")
print(resumen)


print("\nComprobación de solapamiento:")

print(
    "TRAIN ∩ VALIDACIÓN:",
    len(
        partidas_train
        & partidas_validacion
    )
)

print(
    "TRAIN ∩ TEST:",
    len(
        partidas_train
        & partidas_test
    )
)

print(
    "VALIDACIÓN ∩ TEST:",
    len(
        partidas_validacion
        & partidas_test
    )
)


print("\nArchivos generados:")

print(
    "datos/procesado/"
    "indice_ventanas_originales.csv"
)

print(
    "datos/procesado/"
    "partidas_train.csv"
)

print(
    "datos/procesado/"
    "partidas_validacion.csv"
)

print(
    "datos/procesado/"
    "partidas_test.csv"
)

print(
    "datos/procesado/"
    "resumen_particiones.csv"
)