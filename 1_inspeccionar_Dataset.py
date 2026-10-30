from datasets import load_dataset
import pandas as pd
from io import BytesIO


print("Conectando con el dataset CS2CD...")

dataset = load_dataset(
    "CS2CD/Context_window_256",
    split="train",
    streaming=True
)

print("Dataset conectado correctamente.")


# Obtener una sola ventana
ejemplo = next(iter(dataset))

print("\nIdentificador de la ventana:")
print(ejemplo["__key__"])


# Los datos Parquet vienen como bytes.
# BytesIO permite tratarlos como si fueran un archivo.
archivo_parquet = BytesIO(ejemplo["parquet"])


# Convertimos el Parquet en una tabla de columnas y filas
df = pd.read_parquet(archivo_parquet)


print("\nDimensiones de la tabla:")
print(df.shape)


print("\nColumnas encontradas:")

for columna in df.columns:
    print("-", columna)


print("\nPrimeras 5 filas:")
print(df.head())

print("\nValores posibles de attacker_kill:")
print(df["attacker_kill"].unique())

print("\nCantidad de ticks donde attacker_kill = 1:")
print((df["attacker_kill"] == 1).sum())


ticks_kill = df.index[df["attacker_kill"] == 1].tolist()

print("\nTicks donde ocurre la eliminación:")
print(ticks_kill)


if len(ticks_kill) > 0:
    tick_kill = ticks_kill[0]

    print("\nPrimer tick de eliminación:")
    print(tick_kill)

    segmento = df.loc[:tick_kill].copy()

    print("\nTamaño del segmento desde el inicio hasta la eliminación:")
    print(segmento.shape)

    print("\nTicks cercanos a la eliminación:")

    inicio = max(0, tick_kill - 3)
    fin = min(len(df) - 1, tick_kill + 3)

    columnas_interes = [
        "attacker_shot",
        "attacker_kill",
        "is_kill_headshot",
        "is_kill_through_smoke",
        "is_kill_wallbang",
        "attacker_midair"
    ]

    print(df.loc[inicio:fin, columnas_interes])
else:
    print("\nADVERTENCIA: esta ventana no contiene attacker_kill = 1")