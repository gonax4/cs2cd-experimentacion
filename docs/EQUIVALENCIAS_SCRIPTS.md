# Equivalencias de scripts: numeración anterior y organización integrada

La reorganización cambia **nombres y carpetas**, no sus experimentos ni los datos procesados. Los comandos deben ejecutarse desde la raíz del repositorio.

| N.º y nombre anterior | Ruta actual | Función metodológica |
|---|---|---|
| 1_inspeccionar_Dataset.py | `scripts/datos/inspeccionar_dataset.py` | Inspección del dataset |
| 2_validar_ventanas.py | `scripts/datos/validar_ventanas.py` | Comprobación preliminar |
| 3_crear_particiones.py | `scripts/datos/crear_particiones.py` | Separación por partida |
| 4_generar_caracteristicas.py | `scripts/datos/generar_caracteristicas.py` | Ingeniería de predictores |
| 5_analizar_train.py | `scripts/datos/analizar_train.py` | Diagnóstico y selección en TRAIN |
| 6_entrenar_regresion_logistica_final.py | `scripts/modelos/entrenar_regresion_logistica_final.py` | Ajuste RL |
| 7_entrenar_random_forest.py | `scripts/modelos/entrenar_random_forest.py` | Ajuste RF |
| 8_seleccionar_umbrales_validacion.py | `scripts/evaluacion/seleccionar_umbrales_validacion.py` | Umbrales RL/RF |
| 9_evaluacion_final_test.py | `scripts/evaluacion/evaluacion_final_test.py` | TEST histórico RL/RF |
| 10_interpretar_modelos_y_resultados.py | `scripts/evaluacion/interpretar_modelos_y_resultados.py` | Interpretación RL/RF |
| 11_diagnosticar_desbalance.py | `scripts/datos/diagnosticar_desbalance.py` | Diagnóstico de clases |
| 12_entrenar_xgboost.py | `scripts/modelos/entrenar_xgboost.py` | Ajuste XGBoost |
| 13_evaluar_xgboost_validacion.py | `scripts/modelos/evaluar_xgboost_validacion.py` | Validación externa XGBoost |
| 14_analizar_falsos_negativos_validacion.py | `scripts/evaluacion/analizar_falsos_negativos_validacion.py` | Errores XGBoost |
| 15_analizar_subgrupos_validacion.py | `scripts/evaluacion/analizar_subgrupos_validacion.py` | Análisis de subgrupos |
| 16_ablacion_xgboost_train.py | `scripts/modelos/ablacion_xgboost_train.py` | Ablación de predictores |
| 17_comparar_ponderacion_xgboost_train.py | `scripts/modelos/comparar_ponderacion_xgboost_train.py` | Ponderación de clases |
| 18_graficar_curvas_pr_train.py | `scripts/evaluacion/graficar_curvas_pr_train.py` | Curvas PR |
| 19_comparar_igual_recall_train.py | `scripts/evaluacion/comparar_igual_recall_train.py` | Comparación a igual recall |
| 20_sobremuestreo_progresivo_xgboost_train.py | `scripts/modelos/sobremuestreo_progresivo_xgboost_train.py` | Remuestreo positivo |
| 21_auditar_aug_cs2cd.py | `scripts/datos/auditar_aug_cs2cd.py` | Auditoría preliminar |
| 22_auditar_aug_estratificado_train.py | `scripts/datos/auditar_aug_estratificado_train.py` | Auditoría estratificada |

Adicionales:

- `xgb_estimador.py` → `scripts/modelos/xgb_estimador.py`: Clase de XGBoost ponderado, junto a todos los scripts que la importan.
- `demostracion_lectura_cs2cd.py` → `scripts/datos/demostracion_lectura_cs2cd.py`: Demostración de lectura.

## Uso de los artefactos ya generados

**No es necesario repetir los entrenamientos por cambiar los nombres de script**. Los archivos generados mantienen sus ubicaciones en `salidas/`, `datos/procesado/` y `modelos/`, que se resuelven desde el directorio de trabajo.

Ejemplos:

```cmd
python scripts/evaluacion/graficar_curvas_pr_train.py
python scripts/modelos/comparar_ponderacion_xgboost_train.py
python scripts/datos/auditar_aug_estratificado_train.py
```

**No ejecutar nuevamente el script de TEST**: La evaluación histórica ya utilizó TEST y no se debe convertir en un conjunto de ajuste para XGBoost ni para las nuevas estrategias.

Esta equivalencia permite verificar las modificaciones de archivo y reconstruir la trayectoria experimental, sin presentarla necesariamente como el orden de exposición del artículo.
