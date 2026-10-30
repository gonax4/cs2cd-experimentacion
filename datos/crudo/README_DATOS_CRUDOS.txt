README - DATOS CRUDOS (CS2CD)
=================================

Esta carpeta se conserva para documentar la procedencia de los datos crudos.
No contiene una copia descargada de los archivos Parquet originales.

Fuente de los datos:
https://huggingface.co/datasets/CS2CD/Context_window_256

Forma de acceso implementada:
Los scripts Python leen el dataset público mediante la biblioteca datasets de
Hugging Face, con load_dataset("CS2CD/Context_window_256", split="train",
streaming=True). Cada observación de telemetría se obtiene como un bloque de
bytes Parquet y se interpreta con pd.read_parquet(BytesIO(...)).

Importante: split="train" es el nombre del split original de Hugging Face; no
es lo mismo que TRAIN, VALIDACION y TEST, que se crean posteriormente agrupando
las observaciones por partida.

Procesamiento realizado:
- Se utilizaron 90,707 ventanas originales de 256 ticks x 44 variables.
- Se excluyeron 127,553 ventanas con aumento de datos (_aug...).
- Las particiones y el índice están en datos/procesado/.
- Las tablas de características están en datos/procesado/caracteristicas_*.csv.
- Los modelos y las evidencias de evaluación se encuentran en modelos/ y salidas/.

Scripts que permiten reproducir la ingesta y transformación:
1_inspeccionar_Dataset.py  -> inspección inicial de una ventana.
2_validar_ventanas.py     -> comprobación estructural preliminar (100 ventanas).
3_crear_particiones.py    -> índice y particiones por partida.
4_generar_caracteristicas.py -> extracción y procesamiento de las ventanas
                              originales; validaciones estructurales por ventana.

Este README NO sustituye los datos crudos originales. Para obtenerlos es
necesario acceder al dataset de Hugging Face mediante los scripts del proyecto.
Se requiere conexión a Internet durante la lectura por streaming.
