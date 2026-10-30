# CS2CD — Experimentación con Regresión Logística y Random Forest

Implementación de una investigación académica para **comparar modelos que identifican patrones asociados a jugadores etiquetados como tramposos en Counter-Strike 2** mediante datos de telemetría. Se evalúan **Regresión Logística** y **Random Forest**; una acción aislada (como un headshot) no prueba una trampa.

> **Avance 2.1 — Seminario de Investigación I.** Este repositorio contiene código fuente y evidencias ligeras. Los CSV procesados completos, los modelos entrenados y otros artefactos voluminosos se entregan por separado en la carpeta compartida del curso. Los Parquet originales no se almacenan localmente.

## Fuente de datos

Dataset: [CS2CD / Context_window_256 (Hugging Face)](https://huggingface.co/datasets/CS2CD/Context_window_256).

Los scripts usan `load_dataset("CS2CD/Context_window_256", split="train", streaming=True)` y `pd.read_parquet(BytesIO(...))`. Cada ventana contiene 256 ticks × 44 variables. El `train` de Hugging Face es un split de origen y **no** equivale al TRAIN experimental.

## Flujo de experimentación

| Paso | Script | Responsabilidad |
|---|---|---|
| 1 | `1_inspeccionar_Dataset.py` | Inspección de ventanas originales |
| 2 | `2_validar_ventanas.py` | Validación preliminar de 100 ventanas |
| 3 | `3_crear_particiones.py` | Exclusión de ventanas aumentadas y particiones por partida |
| 4 | `4_generar_caracteristicas.py` | Comprobación estructural global e ingeniería de características |
| 5 | `5_analizar_train.py` | Análisis exploratorio exclusivamente de TRAIN |
| 6 | `6_entrenar_regresion_logistica_final.py` | Regresión logística y GridSearchCV |
| 7 | `7_entrenar_random_forest.py` | Random Forest y RandomizedSearchCV |
| 8 | `8_seleccionar_umbrales_validacion.py` | Selección de umbral con VALIDACIÓN |
| 9 | `9_evaluacion_final_test.py` | Evaluación final con TEST |
| 10 | `10_interpretar_modelos_y_resultados.py` | Interpretación de coeficientes e importancias |

### Validación de ventanas

El script 2 revisó 100 ventanas como **validación preliminar**, no como validación exhaustiva del dataset. Durante la extracción, el script 4 comprobó que cada ventana original procesada tuviera 256 filas, 44 columnas y el evento `attacker_kill` en el tick 223. Se procesaron **90,707 ventanas originales** sin errores de extracción registrados. El script 2 incorpora otros diagnósticos que no deben confundirse con estas tres comprobaciones estructurales globales.

### Variables y particiones

Se excluyeron **127,553 ventanas aumentadas** y se conservaron **90,707 originales**; la partición por partida separó 556 partidas TRAIN, 119 VALIDACIÓN y 119 TEST. La ingeniería de características usa los ticks 0–223 (224 ticks) y genera **86 columnas**: nueve metadatos, 74 predictoras candidatas y tres de diagnóstico. El análisis de TRAIN excluyó dos predictoras constantes, dejando **72 predictores finales**.

## Resultados en TEST

| Modelo | Umbral fijado en VALIDACIÓN | F1 | Precisión | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Regresión Logística | 0.67 | 0.605918 | 0.649580 | 0.567756 | 0.843366 |
| Random Forest | 0.53 | **0.642485** | **0.707102** | **0.588689** | **0.854247** |

- Regresión Logística: cinco valores de `C` × dos opciones de `class_weight` × cinco folds = **50 ajustes**; mejor `C=1`, `class_weight=balanced`, F1 CV = **0.620078**.
- Random Forest: 864 combinaciones posibles, 50 seleccionadas con `RandomizedSearchCV` × cinco folds = **250 ajustes**; mejor 800 árboles, `max_depth=None`, `max_features=log2`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight=balanced`; F1 CV = **0.686682**.
- No se utilizó Optuna. Las importancias de características no prueban causalidad ni identifican trampas en un evento individual.

## Datos y reproducción

Los datos originales se obtienen por streaming. La entrega completa del aula virtual conserva los CSV de `datos/procesado`, modelos `.joblib` y evidencias pesadas; GitHub contiene el código y archivos ligeros. `demostracion_lectura_cs2cd.py` requiere los CSV procesados.

En Windows PowerShell, con una versión compatible de Python:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Los scripts 5–10 dependen de los artefactos del experimento completo; Random Forest requiere recursos considerables. **No volver a ejecutar la evaluación final de TEST** del script 9 por simple demostración: el resultado ya se calculó y tiene el marcador `TEST_EVALUADO.txt`.
