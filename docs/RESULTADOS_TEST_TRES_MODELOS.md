# Resultados de TEST de tres clasificadores: Comparación retrospectiva

## Procedencia de los resultados

Se conservaron las métricas históricas de Regresión Logística (RL) y Random Forest (RF) publicadas en `salidas/evaluacion_final_test/metricas_finales_test.csv`. Posteriormente, **sin reentrenar**, se aplicó el XGBoost previamente guardado (`modelos/xgboost_best_pipeline.joblib`) al mismo TEST con los **72 predictores** y el **umbral fijo 0,66**, seleccionado antes mediante VALIDACIÓN. La ejecución se realizó una sola vez mediante:

```cmd
python scripts/evaluacion/evaluacion_final_test.py --xgboost
```

Los resultados de XGBoost se proporcionaron mediante los archivos reales `metricas_xgboost_test.csv`, `metricas_xgboost_test.json`, `predicciones_xgboost_test.csv` y la imagen de matriz de confusión, generados localmente. Esta tabla del repositorio consolida las métricas verificadas sin publicar las **13 221 predicciones individuales**.

## Resultados observados en las mismas 13 221 ventanas de TEST

| Métrica | RL (histórico) | RF (histórico) | XGBoost (retrospectivo) |
|---|---:|---:|---:|
| Umbral de VALIDACIÓN | 0,67 | 0,53 | 0,66 |
| Accuracy | 0,847894 | 0,865063 | **0,869148** |
| Precisión | 0,649580 | 0,707102 | **0,715215** |
| Recall | 0,567756 | 0,588689 | **0,605949** |
| F1-score | 0,605918 | 0,642485 | **0,656064** |
| Especificidad | 0,920556 | 0,936750 | **0,937417** |
| ROC-AUC | 0,843366 | 0,854247 | **0,868644** |
| Verdaderos negativos | 9 664 | 9 834 | **9 841** |
| Falsos positivos | 834 | 664 | **657** |
| Falsos negativos | 1 177 | 1 120 | **1 073** |
| Verdaderos positivos | 1 546 | 1 603 | **1 650** |

Average Precision de XGBoost en TEST: **0,689754**. No se incorporan AP de RL y RF porque los archivos históricos consultados no las contenían.

**Diferencias XGBoost frente a RF:** F1 +0,013579, recall +0,017260, precisión +0,008112, 47 falsos negativos menos y 7 falsos positivos menos. Son **diferencias descriptivas**, no evidencias de significancia estadística.

La matriz de confusión de XGBoost tiene `TN=9841`, `FP=657`, `FN=1073`, `TP=1650`, en la disposición de clases `[[TN, FP], [FN, TP]]`.

Archivo de datos resumidos: [comparacion_rl_rf_xgb_test_retrospectiva.csv](../salidas/evaluacion_xgboost_test_retrospectiva/comparacion_rl_rf_xgb_test_retrospectiva.csv).

## Interpretación académica y precauciones

1. XGBoost mostró el **mayor F1 observado** entre los tres modelos en las mismas ventanas de TEST. No puede afirmarse que sea superior en la población ni que la diferencia sea estadísticamente significativa sin análisis adicional.
2. **Este TEST no fue una prueba ciega e independiente de un protocolo de tres modelos preespecificado.** Los resultados de TEST correspondientes a RL y RF ya se habían conocido antes de incorporar y estudiar XGBoost. Por tanto, el análisis adicional de XGBoost es **retrospectivo y exploratorio**, aunque el umbral 0,66 se había fijado en VALIDACIÓN antes de ejecutar la predicción sobre TEST.
3. Los resultados de TEST no deben utilizarse para modificar hiperparámetros, umbrales, características, ponderación o sobremuestreo. La pequeña mejora descriptiva no justifica volver a optimizar el procedimiento utilizando TEST.
4. El desempeño se mide por **ventanas de telemetría etiquetadas según el jugador**, no por pruebas concluyentes de que una trampa estuviera activa en cada eliminación.
5. Para la discusión, distinguir siempre las comparaciones con umbrales seleccionados en VALIDACIÓN de esta tabla de TEST, y no confundirlas con los estudios fuera de fold realizados exclusivamente en TRAIN.

## Impacto en la figura de metodología

La **fase 4** puede representar ahora los tres modelos:

- **RL y RF:** Evaluación histórica sobre TEST con sus umbrales fijados.
- **XGBoost:** Evaluación retrospectiva adicional sobre el mismo TEST con umbral fijado en VALIDACIÓN.
- **Convergencia:** Métricas y comparación **descriptiva** de los tres modelos, manteniendo señaladas las dos procedencias.

No es correcto denominar esta tabla «confirmación final independiente» de los tres clasificadores. Es correcto denominarla «comparación retrospectiva en TEST» y aclarar que la evaluación de XGBoost se hizo posteriormente.

Registro revisado con base en los archivos de salida proporcionados el 21 de noviembre de 2026.
