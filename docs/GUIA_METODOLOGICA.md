# Metodología integrada de clasificación de telemetría CS2CD

## Objetivo y alcance

Comparar **Regresión Logística (RL), Random Forest (RF) y XGBoost** para clasificar patrones de telemetría de jugadores etiquetados como tramposos en *Counter-Strike 2*, con el conjunto CS2CD. El estudio utiliza la **ventana** como unidad de análisis y la **partida** como agrupación de validación para reducir fugas entre ventanas correlacionadas. La clase etiquetada como `cheater` no equivale a demostrar una trampa en un instante concreto.

Esta es la **presentación metodológica consolidada**, que integra una ampliación posterior. No implica que los tres modelos se hubieran desarrollado simultáneamente. El repositorio ahora ofrece una **evaluación retrospectiva adicional ya realizada** de XGBoost sobre TEST; no equivale a una prueba independiente ciega.

## FASE 1. Preparación de los datos

**Dataset:** `CS2CD/Context_window_256`; cada ventana de 256 ticks contiene 44 canales. Los scripts originales retuvieron **90 707 ventanas** y descartaron **127 553 variantes `_aug`** para la comparación inicial.

**Segmento temporal:** Para extraer características se emplean ticks 0–223, terminando en el evento objetivo situado en el tick 223.

**Partición:** Las partidas se mezclan con semilla 42 y se asignan aproximadamente 70 % TRAIN, 15 % VALIDACIÓN y 15 % TEST. Los porcentajes corresponden al número de **partidas**, no a cantidades exactas de ventanas:

| Conjunto | Partidas | Ventanas | Uso metodológico |
|---|---:|---:|---|
| TRAIN | 556 | 63 411 | Selección de predictores, búsqueda de hiperparámetros, validación cruzada |
| VALIDACIÓN | 119 | 14 075 | Comparación externa, umbrales y análisis descriptivos |
| TEST | 119 | 13 221 | RL/RF históricos y XGBoost retrospectivo, ya ejecutados |

**Ingeniería de características:** Se aplica el mismo procedimiento a las tres particiones. Se extraen características agregadas de movimiento, direcciones, distancias relativas, disparos, eventos, armas y salud. Inicialmente existen **74 predictoras candidatas** y 3 indicadores diagnósticos. El análisis de TRAIN identificó dos predictoras constantes, `victim_health_min` y `victim_health_kill`, y se fijaron **72 predictores** para los tres modelos.

Archivos: `scripts/datos/inspeccionar_dataset.py`, `validar_ventanas.py`, `crear_particiones.py`, `generar_caracteristicas.py` y `analizar_train.py`.

**Flechas del diagrama:** Las tres ramas TRAIN, VALIDACIÓN y TEST pasan por ingeniería de características. La **selección de 72 predictores parte de TRAIN** y la lista elegida se aplica a los otros conjuntos. Las flechas a las fases posteriores salen de los últimos recuadros pertinentes de cada rama, no de posiciones arbitrarias del marco exterior.

## FASE 2. Entrenamiento y ajuste de los clasificadores

Las tres alternativas reciben los mismos 72 predictores y emplean **cinco folds de entrenamiento/validación interna agrupados por partida**, conforme al mapa de folds ya guardado.

| Modelo | Archivo | Procedimiento |
|---|---|---|
| RL | `scripts/modelos/entrenar_regresion_logistica_final.py` | Pipeline con `LogisticRegression`, `GridSearchCV`, 10 combinaciones × 5 folds |
| RF | `scripts/modelos/entrenar_random_forest.py` | `RandomForestClassifier`, `RandomizedSearchCV`, 50 combinaciones × 5 folds |
| XGBoost | `scripts/modelos/entrenar_xgboost.py` | `XGBClassifier`, `RandomizedSearchCV`, 20 combinaciones × 5 folds |

**Preprocesamiento:** La RL escala las 61 variables continuas mediante `StandardScaler` dentro del pipeline, conservando 11 binarias sin ese escalado. RF y XGBoost trabajan con los predictores sin aplicar el mismo escalado.

**Desbalance:** La proporción positiva en TRAIN es aproximadamente 21,17 %. Para XGBoost se exploró ponderación de la clase positiva, calculada exclusivamente con la parte de entrenamiento de cada fold. El archivo `scripts/modelos/xgb_estimador.py` implementa esa opción. En RL/RF se conservan las opciones de ponderación seleccionadas en sus propios ajustes.

**Importante:** Los ajustes y comparaciones de reponderación/sobremuestreo no son una etapa idéntica para RL, RF y XGBoost. Los experimentos adicionales se realizaron específicamente sobre XGBoost y no justifican escribir que los tres modelos recibieron las mismas técnicas complementarias.

## FASE 3. Validación, comparación y umbrales de decisión

Aplicar cada modelo junto con su pipeline a VALIDACIÓN. Generar probabilidades y calcular precisión, recall, F1, ROC-AUC, Average Precision (AP) y errores de clasificación. Estudiar el efecto del umbral de decisión sobre las predicciones.

En la comparación descriptiva sobre el mismo conjunto VALIDACIÓN:

| Modelo | Umbral seleccionado con VALIDACIÓN | F1 VALIDACIÓN | Recall VALIDACIÓN |
|---|---:|---:|---:|
| RL | 0,67 | 0,5879 | 0,5743 |
| RF | 0,53 | 0,6332 | 0,5998 |
| XGBoost | 0,66 | 0,6389 | 0,6104 |

Los resultados de XGBoost suponen una mejora pequeña de F1 respecto a RF. No se puede afirmar superioridad estadística sin análisis adicional. Como el umbral se eligió con ese mismo conjunto, los resultados son **descriptivos/seleccionados sobre VALIDACIÓN**, no una nueva confirmación independiente.

Archivos: `scripts/evaluacion/seleccionar_umbrales_validacion.py` (RL/RF) y `scripts/modelos/evaluar_xgboost_validacion.py` (XGBoost).

## FASE 4. Evaluación final e interpretación (histórica y retrospectiva)

Los modelos **RL y RF** se evaluaron históricamente sobre TEST aplicando sus hiperparámetros y umbrales previamente fijados. El archivo de evaluación es `scripts/evaluacion/evaluacion_final_test.py`, ya ejecutado. Sus resultados no deben volver a emplearse para seleccionar técnicas nuevas.

| Modelo | F1 TEST histórico | Recall TEST histórico |
|---|---:|---:|
| RL | 0,605918 | 0,567756 |
| RF | 0,642485 | 0,588689 |

**XGBoost puede incluirse en esta fase como una evaluación adicional retrospectiva**, usando el modelo `modelos/xgboost_best_pipeline.joblib` ya entrenado y el umbral 0,66 seleccionado previamente en VALIDACIÓN. El modo `python scripts/evaluacion/evaluacion_final_test.py --xgboost` **ya fue ejecutado una sola vez**. El detalle de los resultados se encuentra en [RESULTADOS_TEST_TRES_MODELOS.md](RESULTADOS_TEST_TRES_MODELOS.md). Es independiente del código de RL/RF: No altera sus archivos históricos y escribe en `salidas/evaluacion_xgboost_test_retrospectiva/`. Dado que TEST ya era conocido en el desarrollo original, **no debe presentarse esa nueva comparación como confirmación independiente** ni utilizarse para ajustar XGBoost o seleccionar estrategias nuevas. XGBoost obtuvo F1 **0,656064**, recall **0,605949**, precisión **0,715215** y ROC-AUC **0,868644** en TEST, con 657 falsos positivos y 1 073 falsos negativos. Estas métricas son retrospectivas y descriptivas.

La interpretación de importancia de características procede de los modelos guardados, no de calcular importancias a partir de las métricas de TEST. Archivo: `scripts/evaluacion/interpretar_modelos_y_resultados.py`.

## Análisis metodológicos complementarios (transversales)

Estos procedimientos complementan la metodología, **no forman una fase secuencial posterior a TEST**. En el diagrama se muestran como un bloque lateral o inferior con flechas discontinuas desde sus fuentes.

| Pregunta | Datos y estrategia | Archivo |
|---|---|---|
| ¿Qué desbalance existe? | Proporciones y folds, sin entrenar nuevos modelos | `scripts/datos/diagnosticar_desbalance.py` |
| ¿Dónde falla XGBoost? | Falsos negativos de VALIDACIÓN | `scripts/evaluacion/analizar_falsos_negativos_validacion.py` |
| ¿Hay diferencias entre subgrupos? | VALIDACIÓN, análisis descriptivo post hoc | `scripts/evaluacion/analizar_subgrupos_validacion.py` |
| ¿Conviene excluir variables? | Ablación de 72, 71, 70 y 69 atributos en CV de TRAIN | `scripts/modelos/ablacion_xgboost_train.py` |
| ¿Qué efecto tiene la ponderación? | XGBoost ponderado/no ponderado, cinco folds TRAIN | `scripts/modelos/comparar_ponderacion_xgboost_train.py` |
| ¿Cómo cambian las curvas PR? | Probabilidades fuera de fold de TRAIN | `scripts/evaluacion/graficar_curvas_pr_train.py` |
| ¿Qué pasa a igual recall? | Umbrales descriptivos sobre OOF de TRAIN | `scripts/evaluacion/comparar_igual_recall_train.py` |
| ¿Qué aporta duplicar positivos? | Sobremuestreo +0, +50, +100, +200 y +300 % SOLO en entrenamiento de cada fold | `scripts/modelos/sobremuestreo_progresivo_xgboost_train.py` |
| ¿Cómo cambia una variante `_aug`? | Auditoría original/aumentada de TRAIN (no entrenamiento) | `scripts/datos/auditar_aug_cs2cd.py` y `auditar_aug_estratificado_train.py` |

**Resultados interpretables con cautela:**
- La ponderación y el sobremuestreo aumentaron recall al umbral 0,50, pero no mejoraron de forma clara el AP general.
- El mejor F1 agregado entre los niveles de duplicación ensayados se observó con +100 % de positivos, pero se trató de una exploración en TRAIN.
- La auditoría estratificada de 40 parejas `_aug0` encontró **32/40** parejas sin cambios apreciables en las 28 características espaciales (umbral 1e-6) y **8/40** con cambios superiores a 0,001; no se entrenó un modelo con las variantes aumentadas.
- Las comparaciones post hoc y selección de umbrales deben declararse **exploratorias**; no demuestran causalidad ni mejoras confirmadas sobre datos independientes.

## Cómo dibujar las conexiones correctamente

1. **CS2CD → Ticks 0–223 → Partición por partida** → Tres conjuntos.
2. De cada conjunto, flecha a su **ingeniería de características**.
3. Solo desde TRAIN se selecciona la lista de **72 predictores**, utilizada después para TRAIN, VALIDACIÓN y TEST.
4. **TRAIN procesado → Cinco folds por partida → RL, RF y XGBoost → Modelos ajustados**.
5. **VALIDACIÓN procesada + Modelos ajustados → Predicción y comparación de los tres → Estudio de umbrales**.
6. Desde el resultado de umbrales, **RL/RF** continúan a TEST histórico. **XGBoost** tiene una rama retrospectiva adicional ya ejecutada con su umbral de VALIDACIÓN fijado (0,66).
7. **TEST procesado + RL/RF fijados → Métricas históricas**. Una segunda rama, marcada como **retrospectiva**, une TEST con el modelo XGBoost ya congelado, sin modificar los resultados anteriores.
8. Del entrenamiento XGBoost y las predicciones de TRAIN/VALIDACIÓN salen **flechas discontinuas** a los análisis complementarios. No salen de TEST.
9. **Análisis de importancia → Modelos entrenados**, no «métricas TEST → importancia».

El esquema integrado es una presentación lógica y metodológica del trabajo terminado, no una afirmación de que todas las decisiones estaban preespecificadas desde el principio.

## Rutas y ejecución

Ejecutar siempre desde la raíz del repositorio:

```cmd
python scripts/evaluacion/graficar_curvas_pr_train.py
python scripts/evaluacion/comparar_igual_recall_train.py
python scripts/datos/auditar_aug_estratificado_train.py
```

Los scripts de XGBoost y `xgb_estimador.py` permanecen juntos dentro de `scripts/modelos`. Las rutinas de auditoría 21/22 están juntas dentro de `scripts/datos`. Los CSV de `datos/procesado/`, los modelos en `modelos/` y los resultados en `salidas/` **mantienen sus rutas anteriores**.

Para la trazabilidad, consultar `docs/REGISTRO_HISTORICO.md` y `docs/EQUIVALENCIAS_SCRIPTS.md`.

### Tabla comparativa de TEST para la fase 4

| Modelo | Procedencia | Umbral | F1 | Recall | Precisión |
|---|---|---:|---:|---:|---:|
| RL | Histórica | 0,67 | 0,605918 | 0,567756 | 0,649580 |
| RF | Histórica | 0,53 | 0,642485 | 0,588689 | 0,707102 |
| XGBoost | Retrospectiva | 0,66 | **0,656064** | **0,605949** | **0,715215** |

La comparación de los tres modelos sobre las mismas 13 221 ventanas de TEST ya se ejecutó. Sin embargo, el conocimiento previo del TEST histórico impide tratar el análisis agregado como una nueva confirmación independiente ciega. No utilizar estas métricas para modificar el diseño experimental. [Detalles y procedencia](RESULTADOS_TEST_TRES_MODELOS.md).
