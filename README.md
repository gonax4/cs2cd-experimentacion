# CS2CD — Comparación de Regresión Logística, Random Forest y XGBoost

Repositorio de investigación sobre clasificación de ventanas de telemetría de *Counter-Strike 2* mediante tres clasificadores. La organización sigue las **funciones metodológicas** del estudio revisado, no la fecha en que se desarrolló cada experimento.

> **Rigor de evaluación:** El diseño original evaluó RL y RF sobre TEST. XGBoost se incorporó durante la revisión y se evaluó sobre TRAIN (validación cruzada agrupada) y VALIDACIÓN externa. Se preparó un **modo adicional** para evaluar XGBoost retrospectivamente sobre TEST una única vez, sin reejecutar RL/RF. **Ya fue ejecutado localmente (resultados registrados) y no constituye una prueba independiente ciega**, porque TEST ya se observó durante el estudio original.

## Estructura del proyecto

```text
cs2cd-experimentacion/
├── scripts/
│   ├── datos/               # Ingesta, particiones, características, desbalance, auditoría _aug
│   ├── modelos/             # RL, RF, XGBoost, ponderación, ablación, sobremuestreo
│   └── evaluacion/          # Umbrales, validación, TEST histórico, errores, curvas PR
├── docs/
│   ├── GUIA_METODOLOGICA.md # Metodología integrada y flechas del diagrama
│   ├── EQUIVALENCIAS_SCRIPTS.md  # Rutas anteriores y nuevas
│   └── REGISTRO_HISTORICO.md     # Procedencia de la extensión y advertencias
├── datos/                   # Datos originales documentados y derivados
├── salidas/                 # Métricas, configuraciones y figuras
├── fuentes/                 # Documentación de procedencia de CS2CD
├── requirements.txt
└── requirements-xgboost.txt
```

El código mantiene sus cálculos y sus rutas de entrada/salida relativas a la raíz del repositorio. **La reorganización no reentrenó los modelos.** Posteriormente, XGBoost se evaluó retrospectivamente una sola vez en TEST, conservando los resultados históricos de RL/RF.

## Fases del diagrama metodológico

1. **Preparación:** Ingesta de ventanas originales, segmento 0–223, particiones por partida, ingeniería de características y selección de 72 predictores con TRAIN.
2. **Entrenamiento y ajuste:** RL, RF y XGBoost, con los mismos 5 folds agrupados por partida.
3. **Validación y umbrales:** Modelos y pipelines aplicados a VALIDACIÓN, evaluación de precisión, recall y F1, y selección de umbrales.
4. **Evaluación final:** RL y RF sobre TEST (evaluación histórica), más un **modo retrospectivo adicional** para XGBoost con el umbral fijado en VALIDACIÓN. La evaluación retrospectiva de XGBoost **ya fue ejecutada** y sus resultados se registraron. No recupera la independencia estadística de TEST.

**Análisis complementarios transversales:** Desbalance, falsos negativos, subgrupos, ponderación, ablación, curvas PR, sobremuestreo y auditoría de aumentación. Se conectan a TRAIN o VALIDACIÓN según el experimento y **no** a TEST como un proceso secuencial posterior.

## Archivos principales

| Propósito | Archivo |
|---|---|
| Partición por partida | `scripts/datos/crear_particiones.py` |
| Ingeniería de características | `scripts/datos/generar_caracteristicas.py` |
| Selección y análisis de predictores | `scripts/datos/analizar_train.py` |
| Regresión Logística | `scripts/modelos/entrenar_regresion_logistica_final.py` |
| Random Forest | `scripts/modelos/entrenar_random_forest.py` |
| XGBoost | `scripts/modelos/entrenar_xgboost.py` |
| Validación RL/RF | `scripts/evaluacion/seleccionar_umbrales_validacion.py` |
| Validación XGBoost | `scripts/modelos/evaluar_xgboost_validacion.py` |
| Evaluación histórica TEST (RL/RF) | `scripts/evaluacion/evaluacion_final_test.py` sin argumentos |
| Evaluación retrospectiva XGBoost sobre TEST | `scripts/evaluacion/evaluacion_final_test.py --xgboost`, que utiliza `evaluar_xgboost_test_retrospectivo.py` |
| Experimentos complementarios | Archivos de `scripts/modelos/`, `scripts/evaluacion/` y `scripts/datos/` |

## Ejecución desde la raíz del repositorio

```cmd
python scripts/evaluacion/graficar_curvas_pr_train.py
python scripts/evaluacion/comparar_igual_recall_train.py
python scripts/datos/auditar_aug_estratificado_train.py
```

Los scripts que entrenan XGBoost y el módulo `xgb_estimador.py` permanecen juntos en `scripts/modelos/`. La auditoría estratificada y su módulo de auditoría están juntos en `scripts/datos/`. Los archivos locales grandes de entrenamiento, modelos serializados y predicciones no se suben automáticamente al repositorio.

**TEST ya fue utilizado para los tres modelos:** El análisis histórico de RL/RF permanece protegido por `salidas/evaluacion_final_test/TEST_EVALUADO.txt` y el retrospectivo de XGBoost tiene el marcador `salidas/evaluacion_xgboost_test_retrospectiva/TEST_XGBOOST_EVALUADO.txt` en el entorno local. **No volver a ejecutar TEST para optimizar o repetir comparaciones.**

Comando utilizado en la ejecución retrospectiva de XGBoost (registro histórico, **no volver a ejecutar**):

```cmd
python scripts/evaluacion/evaluacion_final_test.py --xgboost
```

**Advertencia:** El análisis sigue siendo **retrospectivo**: Los resultados de TEST del estudio inicial ya eran conocidos antes de desarrollar el tercer modelo.

## Documentación

- [Guía metodológica unificada](docs/GUIA_METODOLOGICA.md)
- [Resultados TEST de los tres modelos](docs/RESULTADOS_TEST_TRES_MODELOS.md)
- [Mapa de nombres y rutas anteriores](docs/EQUIVALENCIAS_SCRIPTS.md)
- [Registro histórico y limitaciones](docs/REGISTRO_HISTORICO.md)
- [Dataset fuente](https://huggingface.co/datasets/CS2CD/Context_window_256)

## Comparación observada sobre TEST

| Modelo | Umbral | F1 | Recall | Precisión | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| RL (histórico) | 0,67 | 0,605918 | 0,567756 | 0,649580 | 0,843366 |
| RF (histórico) | 0,53 | 0,642485 | 0,588689 | 0,707102 | 0,854247 |
| XGBoost (**retrospectivo**) | 0,66 | **0,656064** | **0,605949** | **0,715215** | **0,868644** |

XGBoost tuvo el mayor F1 observado, pero la comparación no constituye una prueba independiente ciega ni demuestra significancia estadística. [Ver matriz de confusión, procedencia y limitaciones](docs/RESULTADOS_TEST_TRES_MODELOS.md).
