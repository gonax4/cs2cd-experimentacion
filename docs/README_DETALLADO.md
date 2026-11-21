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
| 1 | `scripts/01_estudio_original/1_inspeccionar_Dataset.py` | Inspección de la estructura de una ventana original. |
| 2 | `scripts/01_estudio_original/2_validar_ventanas.py` | Verificación **preliminar de 100 ventanas**, no de todo el conjunto. |
| 3 | `scripts/01_estudio_original/3_crear_particiones.py` | Exclusión de ventanas aumentadas e índice/particiones agrupadas por partida. |
| 4 | `scripts/01_estudio_original/4_generar_caracteristicas.py` | Comprobación estructural por ventana y extracción de atributos tabulares. |
| 5 | `scripts/01_estudio_original/5_analizar_train.py` | Análisis de calidad y selección de variables exclusivamente con TRAIN. |
| 6 | `scripts/01_estudio_original/6_entrenar_regresion_logistica_final.py` | `Pipeline` con escalado y `GridSearchCV` por partidas. |
| 7 | `scripts/01_estudio_original/7_entrenar_random_forest.py` | `RandomizedSearchCV` con los mismos folds agrupados. |
| 8 | `scripts/01_estudio_original/8_seleccionar_umbrales_validacion.py` | Umbral de decisión seleccionado con VALIDACIÓN. |
| 9 | `scripts/01_estudio_original/9_evaluacion_final_test.py` | Evaluación final con TEST y modelos/umbrales fijados. |
| 10 | `scripts/01_estudio_original/10_interpretar_modelos_y_resultados.py` | Importancias y coeficientes de modelos guardados. |

`scripts/01_estudio_original/demostracion_lectura_cs2cd.py` sirve para inspeccionar algunos campos del CSV de TRAIN de forma **solo lectura** (requiere los CSV de la entrega completa).

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
| 11 | `scripts/02_extension_jurado/11_diagnosticar_desbalance.py` | Resume distribución por clases y folds. Solo lee métricas históricas de TEST, sin predecir sobre TEST |
| 12 | `scripts/02_extension_jurado/12_entrenar_xgboost.py` | Entrena XGBoost en TRAIN usando los 72 predictores originales y los mismos cinco folds agrupados por partida |
| 13 | `scripts/02_extension_jurado/13_evaluar_xgboost_validacion.py` | Selecciona un umbral para XGBoost y reporta métricas solo en VALIDACION |
| 14 | `scripts/02_extension_jurado/14_analizar_falsos_negativos_validacion.py` | Resume FN por partida y compara descriptivamente FN y TP entre ventanas positivas de VALIDACION |

### Ejecución (desde la raíz del repositorio)

Instalar el entorno original y la dependencia adicional:

```powershell
python -m pip install -r requirements-xgboost.txt
```

Ejecutar los scripts en orden:

```powershell
python scripts/02_extension_jurado/11_diagnosticar_desbalance.py
python scripts/02_extension_jurado/12_entrenar_xgboost.py
python scripts/02_extension_jurado/13_evaluar_xgboost_validacion.py
python scripts/02_extension_jurado/14_analizar_falsos_negativos_validacion.py
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

El primer entrenamiento produjo `modelos/xgboost_best_pipeline.joblib` con el estimador serializado como `__main__.XGBPonderado`. La actualización de `scripts/02_extension_jurado/13_evaluar_xgboost_validacion.py` registra esa clase al cargar, por lo que **no es necesario entrenar nuevamente**. Para futuras ejecuciones, `scripts/02_extension_jurado/12_entrenar_xgboost.py` utiliza la clase estable del módulo `scripts/02_extension_jurado/xgb_estimador.py`. Mantener ambas versiones de scikit-learn y XGBoost del entorno registrado en `mejor_configuracion_xgb.json`.

### Diagnóstico adicional por subgrupos (exploratorio)

`scripts/02_extension_jurado/15_analizar_subgrupos_validacion.py` compara el recall, falsos negativos y falsos positivos por `kill_wallbang`, `weapon_semi_rifle_kill`, `kill_headshot` y `kill_through_smoke` y por la combinación wallbang/arma. Se ejecuta después de los pasos 13 y 14:

```powershell
python scripts/02_extension_jurado/15_analizar_subgrupos_validacion.py
```

Requiere `datos/procesado/caracteristicas_validacion.csv` y `salidas/xgboost/validacion/predicciones_validacion.csv`. Produce `salidas/xgboost/validacion/desempeno_por_subgrupos.csv`. Es un **análisis descriptivo posterior** y no debe usarse para modificar variables, seleccionar hiperparámetros o ajustar umbrales en VALIDACION. La comparación no prueba que ninguna característica cause el error. No utiliza TEST.

### Ablación exploratoria dentro de TRAIN

Después del diagnóstico de subgrupos de VALIDACION se propuso una hipótesis sobre `kill_wallbang` y las características de rifles semiautomáticos. **No es una hipótesis confirmatoria** porque se generó tras observar VALIDACION. Para probar su sensibilidad dentro de TRAIN sin volver a consultar VALIDACION ni TEST:

```powershell
python scripts/02_extension_jurado/16_ablacion_xgboost_train.py
```

Este script reutiliza los cinco folds por partida y los hiperparámetros ya seleccionados, con umbral fijo de 0.50. Compara 72 variables, sin `kill_wallbang`, sin las dos variables de rifle semiautomático y sin las tres. Produce `salidas/xgboost/ablacion_train/comparacion_ablacion.csv`, `metricas_por_fold.csv` y `metricas_oof_por_subgrupo.csv`. No reemplaza el modelo original ni produce una nueva evaluación final. La eliminación de variables se estudia solo como ablación, no como prueba de que esas variables sean incorrectas o causen un sesgo.

### Comparación de ponderación XGBoost (solo TRAIN)

Para aislar el efecto de `scale_pos_weight`, el archivo `scripts/02_extension_jurado/17_comparar_ponderacion_xgboost_train.py` compara dos variantes: `balancear=False` (peso 1) y `balancear=True` (peso negativos / positivos de cada fold). Utiliza exactamente los mismos 72 predictores, los cinco folds agrupados por partida, los hiperparámetros fijos de XGBoost y umbral 0.50. **No vuelve a ejecutar la búsqueda de parámetros**, ni modifica el modelo existente, ni abre VALIDACION o TEST.

```powershell
python scripts/02_extension_jurado/17_comparar_ponderacion_xgboost_train.py
```

Escribe `salidas/xgboost/ponderacion_train/metricas_por_fold.csv`, `comparacion_ponderacion.csv` y `notas_metodologicas.json`. El contraste es **exploratorio**: los hiperparámetros fueron seleccionados anteriormente con una búsqueda que ya incluía la ponderación como opción. Mantenerlos constantes permite estudiar el efecto del peso con ese modelo, pero no demuestra que esos mismos hiperparámetros sean óptimos sin ponderar. Los resultados del TEST original continúan cerrados.

### Curvas precisión-recall del experimento de ponderación

El script 17 se ha ampliado para guardar probabilidades de predicción **fuera de fold** en `salidas/xgboost/ponderacion_train/predicciones_oof_train.csv`. Tras actualizar la rama, **es necesario ejecutar de nuevo el script 17 una sola vez** para generar ese archivo, porque los CSV antiguos contienen únicamente métricas agregadas y no permiten reconstruir curvas PR auténticas.

```powershell
python scripts/02_extension_jurado/17_comparar_ponderacion_xgboost_train.py
python scripts/02_extension_jurado/18_graficar_curvas_pr_train.py
```

El script 18 **no entrena**. Lee las probabilidades fuera de fold, dibuja `salidas/xgboost/ponderacion_train/curvas_precision_recall.png`, y guarda `resumen_curvas_pr.csv`. Presenta Average Precision (AP) calculado sobre todas las probabilidades fuera de fold, la prevalencia y los puntos del umbral fijo 0.50. La comparación con y sin ponderación sigue siendo exploratoria, ya que los hiperparámetros fijos se escogieron mediante una búsqueda previa sobre TRAIN. No se seleccionan nuevos umbrales ni se consultan VALIDACION o TEST.

### Comparación a recall equivalente: ponderación de XGBoost

Una vez generado `salidas/xgboost/ponderacion_train/predicciones_oof_train.csv` con el script 17, ejecutar:

```powershell
python scripts/02_extension_jurado/19_comparar_igual_recall_train.py
```

El script 19 **no realiza entrenamientos**. Con las mismas predicciones fuera de fold de TRAIN de ambos modelos, encuentra el **umbral más alto que alcance al menos** cada recall objetivo de 60 %, 70 % y 73 %. Compara precisión, F1 y falsos positivos a recalls prácticamente iguales, para separar mejor el efecto de `scale_pos_weight` de elegir el umbral común 0.50. Los recalls observados pueden diferir unas pocas ventanas por discretización o empates.

Archivos: `salidas/xgboost/ponderacion_train/comparacion_igual_recall.csv` (todos los resultados), `diferencias_igual_recall.csv` (diferencia entre modelos), `comparacion_igual_recall.png` (gráfico de falsos positivos) y `notas_igual_recall.json` (limitaciones). **Los umbrales se calculan post hoc sobre las mismas predicciones usadas para comparar métricas: no son umbrales validados independientemente ni deben trasladarse directamente a TEST.** Los hiperparámetros fueron elegidos previamente mediante CV en TRAIN; el análisis es exploratorio.

### Experimento 20: sobremuestreo positivo progresivo (solo TRAIN)

Como **línea base previa** al uso de las ventanas sintéticamente aumentadas del dataset, el archivo `scripts/02_extension_jurado/20_sobremuestreo_progresivo_xgboost_train.py` compara niveles de **duplicación aleatoria de filas positivas**: +0 %, +50 %, +100 %, +200 % y +300 % respecto a las positivas originales de cada subconjunto de entrenamiento de fold. Reutiliza los mismos cinco folds agrupados por partida, los 72 predictores y los hiperparámetros fijos, con **ponderación desactivada** y umbral fijo de 0.50 para aislar el efecto del remuestreo. El incremento es anidado y reproducible mediante semillas fijas. Solo se duplican filas del subconjunto de entrenamiento de cada fold; la evaluación interna siempre usa las filas originales de validación del fold.

```powershell
python scripts/02_extension_jurado/20_sobremuestreo_progresivo_xgboost_train.py
```

Resultados: `salidas/xgboost/sobremuestreo_train/comparacion_sobremuestreo.csv`, `metricas_por_fold.csv` y `notas_metodologicas.json`. El experimento ejecuta **25 ajustes**. **No es aumento sintético de telemetría**: duplicar ventanas no produce evidencia ni señales nuevas, y no equivale al aumento de los creadores. El dataset público `CS2CD/Context_window_256` incluye archivos con sufijos `_aug*` que nuestros scripts 3 y 4 originales omitieron deliberadamente. Antes de incorporar esas muestras preexistentes será necesario verificar su clave de origen, transformación y agrupación por partida, y añadir solo las correspondientes a TRAIN y al subconjunto de entrenamiento de cada fold. No abrir TEST. No reemplazar los modelos originales.

### Auditoría preliminar de ventanas aumentadas `_aug` (solo TRAIN)

**Hallazgo documental previo a cualquier entrenamiento adicional:** El artículo original [AntiCheatPT](https://arxiv.org/abs/2508.06348) informa que las variantes aumentadas se generan añadiendo ruido gaussiano a las coordenadas X/Y/Z de atacante y víctima, con el mismo ruido por eje y tick. El [código de aumentación publicado por los autores](https://github.com/Pinkvinus/CS2_cheat_detection/blob/main/DataAugmentation/dataaugmentation.py) también revela un detalle importante: `noise_std = 0.01`, recorte individual de cada coordenada al intervalo `[0, 1]` y conversión a `float32`. Por tanto, las distancias relativas **no están garantizadas exactamente**: el recorte puede afectar de manera diferente a los dos jugadores.

El script `scripts/01_estudio_original/4_generar_caracteristicas.py` utiliza **diferencias espaciales relativas, no coordenadas absolutas**, para sus 28 predictores espaciales (`delta_x/y/z` y `distancia_3d`, siete agregados por cada grupo). El resto de los 72 predictores seleccionados se basa en campos distintos de las coordenadas. Como consecuencia, si una versión `_aug` solo modifica posiciones absolutas sin cambiar diferencias, su representación tabular será prácticamente equivalente a la original. **No se debe afirmar que siempre es idéntica sin comprobar el efecto del recorte y la precisión numérica.**

Auditar una muestra de parejas original/`_aug` pertenecientes exclusivamente a TRAIN:

```powershell
python scripts/02_extension_jurado/21_auditar_aug_cs2cd.py
```

Requiere conexión a Internet y `datos/procesado/caracteristicas_train.csv`. Lee `CS2CD/Context_window_256` por streaming, descodifica solamente parejas de TRAIN, se detiene tras **20 parejas** o **50 000 registros** y **no entrena** ningún modelo. Es posible configurar `--pares 10 --max-registros 30000` para una primera ejecución más acotada. Los resultados quedan en `salidas/auditoria_aug/pares_auditados.csv`, `errores_auditoria.csv` y `resumen_auditoria.json`. La muestra es consecutiva, **no representativa** y puede incluir solo una clase según el orden del dataset. Solo se compara una variante aumentada por original. No se utilizan VALIDACION ni TEST para ajustar o evaluar.

**Decisión posterior:** Si el efecto de `_aug` en los predictores tabulares es ínfimo, documentar que la aumentación del modelo fuente fue diseñada para entrada secuencial/posicional y que para nuestras características tabulares aporta escasa variabilidad. Si se observan cambios relevantes, diseñar una evaluación adicional con separación estricta por partida y sin mezclar variantes originales/aumentadas entre los folds. No incorporar indiscriminadamente todas las ventanas aumentadas ni abrir TEST.

### Auditoría estratificada de aumentación (paso posterior al script 21)

**Primer resultado (20 parejas del script 21):** Las 20 parejas originales/`_aug0` pertenecían a la clase positiva, a la misma partida (`cdata_file_0`) y al mismo jugador. En 20/20 no cambiaron los canales no espaciales. Solo 2/20 parejas cambiaron alguna de las 28 características espaciales más de `1e-6`, y 1/20 más de `1e-3`. Este resultado es descriptivo **y no puede extrapolarse por sí solo** a otros jugadores, partidas ni a la clase negativa.

Para evitar depender únicamente del orden del streaming, el script `scripts/02_extension_jurado/22_auditar_aug_estratificado_train.py` selecciona **20 partidas por clase**, elige aleatoriamente una ventana original por partida en TRAIN (semilla 42) y busca su versión `_aug0` en CS2CD. Reutiliza la lógica comparativa del script 21. Se requerirá recorrer una parte amplia del dataset remoto para encontrar las claves seleccionadas. No se entrenan modelos ni se leen los Parquet de evaluación externa.

```powershell
git pull --ff-only origin mejoras-jurado-xgboost
python scripts/02_extension_jurado/22_auditar_aug_estratificado_train.py
```

Salidas: `salidas/auditoria_aug_estratificada/resumen_por_clase.csv`, `pares_auditados_estratificados.csv`, `claves_seleccionadas.csv`, `parejas_no_encontradas.csv`, `errores_auditoria.csv` y `notas_metodologicas.json`. Si el dataset no contiene `_aug0` para alguna ventana seleccionada, se registrará como pareja faltante y no se imputarán valores. La muestra garantiza diversidad por partida dentro de cada clase, pero **no representa aleatoriamente la distribución de todas las ventanas**. La auditoría solo establece el impacto observado de las variantes sobre las variables tabulares, no que un modelo temporal se beneficie o deje de beneficiarse del mismo mecanismo.

### Resultado de la auditoría estratificada `_aug0` (experimento 22, solo TRAIN)

La auditoría se ejecutó sobre **40 parejas originales/`_aug0`** elegidas en TRAIN: 20 positivas y 20 negativas, una por partida *dentro de cada clase*. Como dos partidas se repiten entre clases, se inspeccionaron **38 partidas distintas en total**. Se encontraron las 40 parejas sin errores tras recorrer 214 425 registros del streaming.

| Clase | Parejas válidas | Cambios > 1e-6 en alguna de las 28 características espaciales | Cambios > 1e-3 | Máximo cambio en característica espacial |
|---|---:|---:|---:|---:|
| Negativa (0) | 20 | 3 | 3 | 0.024926 |
| Positiva (1) | 20 | 5 | 5 | 0.016397 |
| **Total** | **40** | **8** | **8** | **0.024926** |

En las **40/40 parejas**, los canales no espaciales no cambiaron (`max_cambio_no_espacial = 0`). **32/40** tuvieron una variación espacial máxima no mayor que `1e-6`. **8/40** superaron `1e-3` y **4/40** superaron `0.01`. El máximo cambio relativo de coordenadas entre atacante y víctima fue `0.027148`.

**Interpretación acotada:** Las muestras aumentadas preservan exactamente los canales no espaciales y con frecuencia dejan prácticamente inalteradas las 28 características espaciales que calcula el proyecto. Sin embargo, existen diferencias espaciales no triviales en una minoría de casos, compatibles con el efecto del recorte por separado a `[0, 1]`. La magnitud absoluta por sí sola **no demuestra** si esos cambios afectarán las predicciones de XGBoost. La muestra se seleccionó para diversificar partidas y clases, pero es pequeña, no representa la distribución general de ventanas y solo considera `_aug0`.

**Decisión metodológica provisional:** No incorporar masivamente las variantes `_aug` ni atribuirles mejoras sin evaluar sus efectos. Considerar satisfecha la auditoría de viabilidad de la aumentación original y presentar la comparación final de modelos como limitada a la experimentación ya realizada, distinguiendo TEST histórico de evaluaciones exploratorias sobre TRAIN y VALIDACION. Si se emprende un estudio adicional con estas muestras, habrá que conservar el vínculo con sus partidas originales y limitar la aumentación a entrenamiento dentro de cada fold, con evaluación independiente.
