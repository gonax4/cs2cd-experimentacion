# Guía metodológica — CS2CD, Seminario de Investigación I

## 1. Regla básica: método científico ≠ cronología de programación

La comparación del artículo debe explicar **tres clasificadores** sobre los mismos datos y criterios de partición: RL, RF y XGBoost. En el desarrollo real hubo dos fases distintas:

- **Estudio original:** Ingesta, ingeniería de características, entrenamiento y evaluación de RL y RF, incluyendo una evaluación histórica en TEST.
- **Extensión tras el jurado:** Diagnóstico de desbalance, tercer clasificador XGBoost y experimentos adicionales de recall y aumentación, **sin repetir TEST**.

Esta separación preserva la trazabilidad. No debe escribirse que el script 9 evaluó XGBoost, ni presentar la extensión como si hubiera antecedido a la evaluación original de RL y RF.

## 2. Orden lógico para redactar la sección Metodología

### 2.1 Datos y unidad de análisis

Fuente: `CS2CD/Context_window_256` de Hugging Face. Cada ventana original contiene 256 ticks y 44 canales. Se utilizan los ticks 0–223 para extraer estadísticas de comportamiento. Se indexaron **90 707 ventanas originales**, excluyendo **127 553 variantes aumentadas**.

Archivos:
- `scripts/01_estudio_original/1_inspeccionar_Dataset.py`: Inspección de un Parquet.
- `scripts/01_estudio_original/2_validar_ventanas.py`: Prueba estructural preliminar sobre 100 ventanas.
- `scripts/01_estudio_original/3_crear_particiones.py`: Índice y separación TRAIN/VALIDACION/TEST por partida.
- `scripts/01_estudio_original/4_generar_caracteristicas.py`: Transformación de las ventanas a filas tabulares.
- `scripts/01_estudio_original/5_analizar_train.py`: Calidad y diagnóstico de características exclusivamente en TRAIN.

Precisión importante: El split `train` de Hugging Face **no** equivale a la partición experimental TRAIN.

### 2.2 Partición y validación cruzada

La partición por partida evita que las ventanas de la misma partida se mezclen entre conjuntos. TRAIN tiene 63 411 ventanas de 556 partidas, VALIDACION 14 075 de 119 partidas y TEST 13 221 de 119 partidas. Los tres modelos reutilizan el mapa de **cinco folds internos agrupados por partida**, guardado en `salidas/regresion_logistica/mapa_partidas_folds_internos.csv`.

En el artículo deben explicarse por separado la búsqueda en TRAIN, la evaluación en VALIDACIÓN y el TEST histórico.

### 2.3 Características y preprocesamiento

Partiendo de 44 señales por tick se obtuvieron 74 predictores candidatos y 3 diagnósticos. Tras retirar dos atributos constantes detectados en TRAIN, se utilizaron **72 predictores finales**. La lista de referencia está en `salidas/regresion_logistica/predictores_finales_72.txt`.

La Regresión Logística utiliza escalado dentro de su pipeline. No se debe suponer que RF o XGBoost requieren el mismo escalado.

### 2.4 Modelos comparados

| Modelo | Código | Protocolo de búsqueda |
|---|---|---|
| Regresión Logística | `scripts/01_estudio_original/6_entrenar_regresion_logistica_final.py` | GridSearchCV, 10 configuraciones × 5 folds |
| Random Forest | `scripts/01_estudio_original/7_entrenar_random_forest.py` | RandomizedSearchCV, 50 configuraciones × 5 folds |
| XGBoost | `scripts/02_extension_jurado/12_entrenar_xgboost.py` y `xgb_estimador.py` en la misma carpeta | RandomizedSearchCV, 20 configuraciones × 5 folds |

XGBoost fue incorporado **después** de completar la comparación original. Sus pruebas se reportan como extensión exploratoria. Para XGBoost ponderado, `scale_pos_weight` se calcula con la proporción de clases **solo del entrenamiento de cada fold**.

### 2.5 Métricas, umbrales y evaluación

Explicar precisión, recall, F1, ROC-AUC, Average Precision (AP), matrices de confusión y falsos positivos/negativos. El umbral de clasificación cambia el equilibrio precisión-recall.

- `scripts/01_estudio_original/8_seleccionar_umbrales_validacion.py`: Selección histórica sobre VALIDACIÓN para RL y RF.
- `scripts/02_extension_jurado/13_evaluar_xgboost_validacion.py`: Estudio de umbrales de XGBoost en VALIDACIÓN.
- `scripts/01_estudio_original/9_evaluacion_final_test.py`: **TEST histórico de RL y RF, ya ejecutado. No reejecutar.**
- `scripts/01_estudio_original/10_interpretar_modelos_y_resultados.py`: Interpretación de los modelos originales.

En tablas, **no mezclar** métricas de TRAIN, VALIDACIÓN y TEST. No hay evaluación nueva independiente de XGBoost en TEST.

## 3. Diagnósticos y pruebas exploratorias: Resultados complementarios

Esta fase se describe después de la comparación principal, sin presentar cada script como un paso obligatorio de producción.

| Pregunta de investigación | Script |
|---|---|
| ¿Cuál es el desbalance por partición y fold? | `11_diagnosticar_desbalance.py` |
| ¿Dónde se producen los falsos negativos de XGBoost? | `14_analizar_falsos_negativos_validacion.py` |
| ¿Qué subgrupos muestran mayor dificultad? | `15_analizar_subgrupos_validacion.py` |
| ¿Qué ocurre al eliminar características? | `16_ablacion_xgboost_train.py` |
| ¿Qué cambia con y sin ponderación de clases? | `17_comparar_ponderacion_xgboost_train.py` |
| ¿Cómo son las curvas PR y AP? | `18_graficar_curvas_pr_train.py` |
| ¿Qué sucede al comparar a igual recall? | `19_comparar_igual_recall_train.py` |
| ¿Qué logra duplicar positivos progresivamente? | `20_sobremuestreo_progresivo_xgboost_train.py` |
| ¿Qué cambios induce la aumentación original `_aug0`? | `21_auditar_aug_cs2cd.py` y `22_auditar_aug_estratificado_train.py` |

Todos los archivos de esta tabla están en `scripts/02_extension_jurado/`. Su finalidad es **exploratoria y de discusión**, no una nueva evaluación final.

### Resultado y límite de la auditoría `_aug`

En la muestra estratificada se analizaron 40 parejas original/`_aug0`: 20 por clase y **38 partidas distintas en total**. Los canales no espaciales permanecieron idénticos en 40/40, mientras que 8/40 parejas presentaron al menos una variación espacial mayor que 0.001. En esta muestra la representación tabular suele conservarse, aunque **no siempre**. No se entrenó un modelo usando directamente `_aug`. La muestra pequeña no permite generalizar a todos los casos.

### Resultado y límite del sobremuestreo progresivo

La duplicación aleatoria de positivos se aplicó **solo al entrenamiento de cada fold**. Con umbral 0.50, el nivel +100 % obtuvo el mejor F1 agregado de los niveles probados, con un intercambio precisión-recall y más falsos positivos. **Duplicar ventanas no añade señales independientes ni equivale a la aumentación de telemetría de los autores**.

## 4. Correspondencia con el artículo

| Apartado del artículo | Material recomendado |
|---|---|
| Metodología | Dataset, partición por partidas, 72 predictores, cinco folds, tres algoritmos, métricas y umbrales |
| Resultados principales | Comparativa de RL/RF/XGBoost **en VALIDACIÓN**, identificando sus umbrales, y una tabla aparte para TEST histórico |
| Resultados complementarios | Desbalance, falsos negativos, ponderación, PR, ablación, sobremuestreo y auditoría `_aug` |
| Discusión | Trade-off precisión/recall, dependencia del umbral, subgrupos, limitaciones por ventana/partida, ausencia de TEST nuevo |
| Conclusiones | Rendimientos observados, alcance de los tres modelos, limitaciones y trabajo futuro |

## 5. Cómo ejecutar tras la reorganización

Situarse siempre en la raíz del repositorio:

```cmd
python scripts/01_estudio_original/5_analizar_train.py
python scripts/02_extension_jurado/18_graficar_curvas_pr_train.py
python scripts/02_extension_jurado/19_comparar_igual_recall_train.py
python scripts/02_extension_jurado/22_auditar_aug_estratificado_train.py
```

Los CSV y modelos voluminosos se generan/leen en rutas de raíz: `datos/procesado`, `salidas`, `modelos`. **No ejecutes 3, 4, 6, 7, 9 o 12 indiscriminadamente**, pues algunos vuelven a generar artefactos costosos o históricos. Para verificar la organización, basta ejecutar un script de lectura o consulta de resultados que no reentrene.

**Nota metodológica:** Esta guía reorganiza archivos existentes, no cambia cifras, hiperparámetros ni los conjuntos utilizados. La extensión XGBoost sigue siendo una investigación posterior a la evaluación original.
