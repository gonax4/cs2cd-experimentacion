# CS2CD — Experimentación con Regresión Logística y Random Forest

Implementación de una investigación académica para **comparar modelos que identifican patrones asociados a jugadores etiquetados como tramposos en Counter-Strike 2** mediante datos de telemetría. Se evalúan **Regresión Logística** y **Random Forest**; no se afirma que una acción aislada (como un headshot) pruebe una trampa.

> **Avance 2.1 — Seminario de Investigación I.** Este repositorio contiene el **código fuente y las evidencias ligeras**. Los CSV procesados completos, modelos entrenados y otros artefactos voluminosos se entregan **por separado en la carpeta compartida del curso**. No se incluyen ni se sustituyen los Parquet originales.

## Fuente y acceso a datos

- Dataset público: [CS2CD / Context_window_256 (Hugging Face)](https://huggingface.co/datasets/CS2CD/Context_window_256).
- El acceso se implementa con `load_dataset("CS2CD/Context_window_256", split="train", streaming=True)` y la lectura de bytes Parquet con `pd.read_parquet(BytesIO(...))`.
- Cada ventana tiene **256 ticks × 44 variables**. El split original de Hugging Face (`train`) **no** representa la partición TRAIN del experimento.
- `datos/crudo/README_DATOS_CRUDOS.txt` y `fuentes/ORIGEN_DATASET_CS2CD.txt` documentan la procedencia. La carpeta `datos/crudo` no contiene una descarga permanente del conjunto original.

## Flujo de trabajo

| Orden | Script | Función |
|---|---|---|
| 1 | `1_inspeccionar_Dataset.py` | Inspección de la estructura de una ventana original. |
| 2 | `2_validar_ventanas.py` | Verificación **preliminar de 100 ventanas**, no de todo el conjunto. |
| 3 | `3_crear_particiones.py` | Exclusión de ventanas aumentadas e índice/particiones agrupadas por partida. |
| 4 | `4_generar_caracteristicas.py` | Comprobación estructural por ventana y extracción de atributos tabulares. |
| 5 | `5_analizar_train.py` | Análisis de calidad y selección de variables exclusivamente con TRAIN. |
| 6 | `6_entrenar_regresion_logistica_final.py` | `Pipeline` con escalado y `GridSearchCV` por partidas. |
| 7 | `7_entrenar_random_forest.py` | `RandomizedSearchCV` con los mismos folds agrupados. |
| 8 | `8_seleccionar_umbrales_validacion.py` | Umbral de decisión seleccionado con VALIDACIÓN. |
| 9 | `9_evaluacion_final_test.py` | Evaluación final con TEST y modelos/umbrales fijados. |
| 10 | `10_interpretar_modelos_y_resultados.py` | Importancias y coeficientes de modelos guardados. |

`demostracion_lectura_cs2cd.py` sirve para inspeccionar algunos campos del CSV de TRAIN de forma **solo lectura** (requiere los CSV de la entrega completa).

### Validación de ventanas: 100 frente a 90,707

El script 2 examinó **100 ventanas como prueba preliminar**. La función de extracción del script 4 comprueba en **cada una de las 90,707 ventanas originales procesadas** que existan 256 filas, 44 columnas y el evento objetivo `attacker_kill` en el tick 223. No debe confundirse esa verificación estructural global con el conjunto completo de diagnósticos preliminares del script 2. El resultado está registrado en `salidas/resumen_generacion_caracteristicas.csv`: se generaron todas las filas esperadas y no se registraron errores de extracción.

### Datos, selección y particiones

Se retuvieron **90,707 ventanas originales** y se excluyeron **127,553 aumentadas**. La partición por partida quedó distribuida en 556 partidas TRAIN, 119 VALIDACIÓN y 119 TEST. La ingeniería de atributos utiliza los ticks **0 a 223** y produce **86 columnas** por fila: 9 metadatos, 74 predictoras candidatas y 3 diagnósticas. El EDA de TRAIN encontró dos predictoras constantes (`victim_health_min` y `victim_health_kill`), que se excluyeron para conservar **72 predictores finales**. Los nombres están en `salidas/regresion_logistica/predictores_finales_72.txt`.

## Resultados obtenidos

Los resultados numéricos provienen de `salidas/evaluacion_final_test/metricas_finales_test.csv`, guardado en el proyecto original.

| Modelo | Umbral seleccionado en VALIDACIÓN | F1 en TEST | Precisión en TEST | Recall en TEST | ROC-AUC en TEST |
|---|---:|---:|---:|---:|---:|
| Regresión Logística | 0.67 | 0.605918 | 0.649580 | 0.567756 | 0.843366 |
| Random Forest | 0.53 | **0.642485** | **0.707102** | **0.588689** | **0.854247** |

- Regresión Logística: 10 combinaciones × 5 folds = **50 ajustes**, mejor `C=1`, `class_weight=balanced`; F1 CV ≈ **0.620078**.
- Random Forest: espacio de 864 combinaciones, **50** evaluadas × 5 folds = **250 ajustes**; mejor `n_estimators=800`, `max_depth=None`, `max_features=log2`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight=balanced`; F1 CV ≈ **0.686682**.
- No se usó Optuna. Los resultados corresponden a esta partición y a las etiquetas del dataset; no demuestran causalidad ni que cada eliminación implicara una trampa activa.

## Archivos del repositorio y entrega completa

- `salidas/`: métricas, configuraciones, gráficos y diagnósticos **ligeros** reproducidos del experimento. Se omiten los CSV voluminosos de predicciones por ventana.
- `datos/procesado/resumen_particiones.csv`: resumen de particiones; los demás CSV procesados se encuentran en la entrega completa.
- `modelos/*.joblib`: **no incluidos en GitHub** por tamaño. La entrega completa conserva los modelos serializados, incluido Random Forest (≈610 MB).
- `requirements.txt`: dependencias fijadas del entorno original; se convirtió de UTF-16 a UTF-8 para facilitar su instalación desde GitHub.

## Ejecución local

En **Windows / PowerShell**, con una versión de Python compatible con las dependencias fijadas:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Los scripts 1–4 acceden a la fuente pública mediante Internet; ejecutarlos de nuevo puede requerir tiempo y recursos importantes. Los scripts 5–10 necesitan los CSV y/o modelos de la **entrega completa**, que no están alojados aquí. La búsqueda de Random Forest originalmente consumió varias horas. **No se recomienda ejecutar de nuevo el script 9 sobre TEST**, que ya fue evaluado y dispone de un archivo marcador `TEST_EVALUADO.txt`.

**Alcance:** estudio comparativo sobre un dataset etiquetado. Las importancias de características no deben interpretarse como pruebas individuales de trampas.

## Extensión exploratoria posterior al jurado

**Estado:** XGBoost entrenado con F1 medio CV = **0.6854575364**. En VALIDACION, al seleccionar umbral 0.66, obtuvo precisión = **0.6702**, recall = **0.6104**, F1 = **0.6389**. El análisis de ablación en TRAIN mantuvo mejor F1 con los 72 predictores completos (media CV = **0.6855**). Todas estas comparaciones posteriores al TEST histórico son exploratorias. Se mantienen intactos los resultados originales de RL y RF.

| Orden | Script | Función |
|---|---|---|
| 11 | `11_diagnosticar_desbalance.py` | Resume distribución por clases y folds. Solo lee métricas históricas de TEST, sin predecir sobre TEST |
| 12 | `12_entrenar_xgboost.py` | Entrena XGBoost en TRAIN usando los 72 predictores originales y los mismos cinco folds agrupados por partida |
| 13 | `13_evaluar_xgboost_validacion.py` | Selecciona un umbral para XGBoost y reporta métricas solo en VALIDACION |
| 14 | `14_analizar_falsos_negativos_validacion.py` | Resume FN por partida y compara descriptivamente FN y TP entre ventanas positivas de VALIDACION |

### Ejecución (desde la raíz del repositorio)

Instalar el entorno original y la dependencia adicional:

```powershell
python -m pip install -r requirements-xgboost.txt
```

Ejecutar los scripts en orden:

```powershell
python 11_diagnosticar_desbalance.py
python 12_entrenar_xgboost.py
python 13_evaluar_xgboost_validacion.py
python 14_analizar_falsos_negativos_validacion.py
```

Los pasos 12–14 necesitan los archivos de la **entrega completa no incluidos en GitHub** (`datos/procesado/caracteristicas_train.csv` y `datos/procesado/caracteristicas_validacion.csv`). Los 72 predictores y el mapa de folds ya están incluidos en el repositorio.

El nuevo modelo se guarda en `modelos/xgboost_best_pipeline.joblib` y los resultados en `salidas/xgboost/`. Los CSV de predicciones por ventana se excluyen mediante `.gitignore`.

### Precauciones metodológicas

- **No repetir la evaluación final sobre TEST** como si fuera independiente de las decisiones tomadas después de conocer los resultados históricos. La validación del nuevo clasificador es exploratoria.
- `cheater=1` es la clase positiva. Cada fila representa una **ventana etiquetada**, no un jugador único.
- El peso `scale_pos_weight` se calcula con el subconjunto recibido por `fit` en cada fold y se contrasta con ausencia de ponderación.
- Se conserva F1 como criterio principal. Recall, precisión, ROC-AUC y PR-AUC son medidas complementarias.
- Las diferencias de medianas de FN frente a TP no establecen causalidad ni prueban trampas individuales.
- Antes de reclamar una mejora confirmatoria, acordar con el asesor una nueva prueba independiente o explicar la limitación del TEST histórico.

### Experimentos futuros (todavía no implementados)

Estudiar selección de características dentro de cada fold y aumento progresivo de datos con controles de origen y separación por partida. No mezclar esos cambios con la primera comparación de XGBoost.

### Compatibilidad del primer modelo guardado

El primer entrenamiento produjo `modelos/xgboost_best_pipeline.joblib` con el estimador serializado como `__main__.XGBPonderado`. La actualización de `13_evaluar_xgboost_validacion.py` registra esa clase al cargar, por lo que **no es necesario entrenar nuevamente**. Para futuras ejecuciones, `12_entrenar_xgboost.py` utiliza la clase estable del módulo `xgb_estimador.py`. Mantener ambas versiones de scikit-learn y XGBoost del entorno registrado en `mejor_configuracion_xgb.json`.

### Diagnóstico adicional por subgrupos (exploratorio)

`15_analizar_subgrupos_validacion.py` compara el recall, falsos negativos y falsos positivos por `kill_wallbang`, `weapon_semi_rifle_kill`, `kill_headshot` y `kill_through_smoke` y por la combinación wallbang/arma. Se ejecuta después de los pasos 13 y 14:

```powershell
python 15_analizar_subgrupos_validacion.py
```

Requiere `datos/procesado/caracteristicas_validacion.csv` y `salidas/xgboost/validacion/predicciones_validacion.csv`. Produce `salidas/xgboost/validacion/desempeno_por_subgrupos.csv`. Es un **análisis descriptivo posterior** y no debe usarse para modificar variables, seleccionar hiperparámetros o ajustar umbrales en VALIDACION. La comparación no prueba que ninguna característica cause el error. No utiliza TEST.

### Ablación exploratoria dentro de TRAIN

Después del diagnóstico de subgrupos de VALIDACION se propuso una hipótesis sobre `kill_wallbang` y las características de rifles semiautomáticos. **No es una hipótesis confirmatoria** porque se generó tras observar VALIDACION. Para probar su sensibilidad dentro de TRAIN sin volver a consultar VALIDACION ni TEST:

```powershell
python 16_ablacion_xgboost_train.py
```

Este script reutiliza los cinco folds por partida y los hiperparámetros ya seleccionados, con umbral fijo de 0.50. Compara 72 variables, sin `kill_wallbang`, sin las dos variables de rifle semiautomático y sin las tres. Produce `salidas/xgboost/ablacion_train/comparacion_ablacion.csv`, `metricas_por_fold.csv` y `metricas_oof_por_subgrupo.csv`. No reemplaza el modelo original ni produce una nueva evaluación final. La eliminación de variables se estudia solo como ablación, no como prueba de que esas variables sean incorrectas o causen un sesgo.

### Comparación de ponderación XGBoost (solo TRAIN)

Para aislar el efecto de `scale_pos_weight`, el archivo `17_comparar_ponderacion_xgboost_train.py` compara dos variantes: `balancear=False` (peso 1) y `balancear=True` (peso negativos / positivos de cada fold). Utiliza exactamente los mismos 72 predictores, los cinco folds agrupados por partida, los hiperparámetros fijos de XGBoost y umbral 0.50. **No vuelve a ejecutar la búsqueda de parámetros**, ni modifica el modelo existente, ni abre VALIDACION o TEST.

```powershell
python 17_comparar_ponderacion_xgboost_train.py
```

Escribe `salidas/xgboost/ponderacion_train/metricas_por_fold.csv`, `comparacion_ponderacion.csv` y `notas_metodologicas.json`. El contraste es **exploratorio**: los hiperparámetros fueron seleccionados anteriormente con una búsqueda que ya incluía la ponderación como opción. Mantenerlos constantes permite estudiar el efecto del peso con ese modelo, pero no demuestra que esos mismos hiperparámetros sean óptimos sin ponderar. Los resultados del TEST original continúan cerrados.
