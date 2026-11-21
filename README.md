# CS2CD — Comparativa de clasificadores para detectar patrones asociados a trampas en CS2

Proyecto de **Seminario de Investigación I**. Se investiga la clasificación de ventanas de telemetría del dataset [CS2CD/Context_window_256](https://huggingface.co/datasets/CS2CD/Context_window_256) mediante **Regresión Logística (RL), Random Forest (RF) y XGBoost**.

> **Importante:** RL y RF fueron entrenados y evaluados sobre TEST en el estudio original. XGBoost se incorporó después de las observaciones del jurado. Sus resultados posteriores son exploratorios sobre TRAIN y VALIDACIÓN. **No se debe presentar un resultado de XGBoost en TEST ni comparar métricas de conjuntos diferentes como si fuesen equivalentes.**

## ¿Dónde está cada parte?

```text
cs2cd-experimentacion/
├── scripts/
│   ├── 01_estudio_original/    # Scripts 1–10; RL, RF y TEST histórico
│   └── 02_extension_jurado/    # Scripts 11–22; XGBoost y experimentos
├── docs/
│   ├── GUIA_METODOLOGICA.md    # Etapas y relación con las secciones del artículo
│   └── README_DETALLADO.md     # Documentación técnica completa e histórica
├── datos/
│   ├── crudo/                  # Documentación de la fuente
│   └── procesado/              # Datos derivados (CSV grandes ignorados)
├── salidas/                    # Configuraciones, métricas y gráficas
├── fuentes/                    # Procedencia del dataset
├── requirements.txt
└── requirements-xgboost.txt
```

**Empieza por [GUIA_METODOLOGICA.md](docs/GUIA_METODOLOGICA.md)**. Allí se ordenan los métodos según su papel científico en el artículo, sin confundirlos con el orden cronológico de desarrollo.

## Etapas científicas y archivos principales

| Etapa en la metodología | Archivos |
|---|---|
| Obtención, validación y particiones por partida | `scripts/01_estudio_original/1_...` a `5_...` |
| Regresión Logística y Random Forest | `scripts/01_estudio_original/6_...` y `7_...` |
| XGBoost (tercer clasificador, extensión posterior) | `scripts/02_extension_jurado/12_entrenar_xgboost.py` |
| Evaluación y umbrales sobre VALIDACIÓN | Original: `8_seleccionar_umbrales_validacion.py`. Extensión: `13_evaluar_xgboost_validacion.py` |
| Evaluación histórica sobre TEST | `scripts/01_estudio_original/9_evaluacion_final_test.py` **solo RL y RF, ya ejecutado** |
| Interpretación, diagnóstico y análisis exploratorio | `10_...` y `11_...`, `14_...` a `22_...` |

## Resultados de referencia

| Clasificador | VALIDACIÓN F1 | VALIDACIÓN recall | Umbral en VALIDACIÓN | TEST histórico |
|---|---:|---:|---:|---|
| Regresión Logística | 0.5879 | 0.5743 | 0.67 | Disponible (estudio original) |
| Random Forest | 0.6332 | 0.5998 | 0.53 | Disponible (estudio original) |
| XGBoost | 0.6389 | 0.6104 | 0.66 | **No evaluado** |

Estos resultados no implican superioridad estadística. Los experimentos 16–22 sobre TRAIN estudian características, ponderación, umbrales, sobremuestreo y viabilidad de la aumentación; no constituyen una nueva evaluación final independiente.

## Ejecución desde la raíz del repositorio

**Las rutas de lectura/escritura de los scripts son relativas a la raíz del proyecto.** Ejecuta siempre los comandos estando en `cs2cd-experimentacion`.

```cmd
python scripts/02_extension_jurado/19_comparar_igual_recall_train.py
python scripts/02_extension_jurado/20_sobremuestreo_progresivo_xgboost_train.py
python scripts/02_extension_jurado/22_auditar_aug_estratificado_train.py
```

Los scripts que usan `xgb_estimador.py` lo encuentran en su misma carpeta, y el experimento 22 encuentra el 21 mediante `__file__`.

**No ejecutes de nuevo el script 9:** TEST ya se evaluó y existe `salidas/evaluacion_final_test/TEST_EVALUADO.txt`.

## Fuentes y detalles

- [Guía metodológica y mapa de scripts](docs/GUIA_METODOLOGICA.md)
- [Descripción histórica de resultados, particiones y experimentos](docs/README_DETALLADO.md)
- [Procedencia del dataset](fuentes/ORIGEN_DATASET_CS2CD.txt)

Los archivos CSV de entrenamiento/validación, los modelos serializados y las predicciones voluminosas no están incluidos en GitHub. Se conservan localmente o en los entregables originales. Al cambiar de rama, **no borres ni sobrescribas esos datos locales**.
