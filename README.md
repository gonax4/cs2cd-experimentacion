# CS2CD — Comparación de Regresión Logística, Random Forest y XGBoost

Repositorio de investigación sobre clasificación de ventanas de telemetría de *Counter-Strike 2* mediante tres clasificadores. La organización sigue las **funciones metodológicas** del estudio revisado, no la fecha en que se desarrolló cada experimento.

> **Rigor de evaluación:** El diseño original evaluó RL y RF sobre TEST. XGBoost se incorporó durante la revisión y se evaluó sobre TRAIN (validación cruzada agrupada) y VALIDACIÓN externa, **pero no sobre TEST**. La metodología se presenta integrada para facilitar su lectura, sin atribuirle retrospectivamente una evaluación final que no se realizó.

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

El código mantiene sus cálculos y sus rutas de entrada/salida relativas a la raíz del repositorio. **No se reentrenó ni reevaluó TEST durante esta reorganización.**

## Fases del diagrama metodológico

1. **Preparación:** Ingesta de ventanas originales, segmento 0–223, particiones por partida, ingeniería de características y selección de 72 predictores con TRAIN.
2. **Entrenamiento y ajuste:** RL, RF y XGBoost, con los mismos 5 folds agrupados por partida.
3. **Validación y umbrales:** Modelos y pipelines aplicados a VALIDACIÓN, evaluación de precisión, recall y F1, y selección de umbrales.
4. **Evaluación final histórica:** RL y RF sobre TEST, con los umbrales previamente fijados. **XGBoost no cuenta con evaluación independiente en TEST.**

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
| Evaluación histórica TEST (RL/RF) | `scripts/evaluacion/evaluacion_final_test.py` |
| Experimentos complementarios | Archivos de `scripts/modelos/`, `scripts/evaluacion/` y `scripts/datos/` |

## Ejecución desde la raíz del repositorio

```cmd
python scripts/evaluacion/graficar_curvas_pr_train.py
python scripts/evaluacion/comparar_igual_recall_train.py
python scripts/datos/auditar_aug_estratificado_train.py
```

Los scripts que entrenan XGBoost y el módulo `xgb_estimador.py` permanecen juntos en `scripts/modelos/`. La auditoría estratificada y su módulo de auditoría están juntos en `scripts/datos/`. Los archivos locales grandes de entrenamiento, modelos serializados y predicciones no se suben automáticamente al repositorio.

**No vuelvas a ejecutar `scripts/evaluacion/evaluacion_final_test.py`:** El conjunto TEST ya se utilizó en la fase histórica y está señalado por `salidas/evaluacion_final_test/TEST_EVALUADO.txt`.

## Documentación

- [Guía metodológica unificada](docs/GUIA_METODOLOGICA.md)
- [Mapa de nombres y rutas anteriores](docs/EQUIVALENCIAS_SCRIPTS.md)
- [Registro histórico y limitaciones](docs/REGISTRO_HISTORICO.md)
- [Dataset fuente](https://huggingface.co/datasets/CS2CD/Context_window_256)
