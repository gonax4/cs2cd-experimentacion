# CS2CD — Comparación de Regresión Logística, Random Forest y XGBoost

Repositorio de investigación sobre clasificación de ventanas de telemetría de *Counter-Strike 2* mediante tres clasificadores. La organización sigue las **funciones metodológicas** del estudio revisado, no la fecha en que se desarrolló cada experimento.

> **Rigor de evaluación:** El diseño original evaluó RL y RF sobre TEST. XGBoost se incorporó durante la revisión y se evaluó sobre TRAIN (validación cruzada agrupada) y VALIDACIÓN externa. Se preparó un **modo adicional** para evaluar XGBoost retrospectivamente sobre TEST una única vez, sin reejecutar RL/RF. **Todavía requiere ejecución local y no constituye una prueba independiente ciega**, porque TEST ya se observó durante el estudio original.

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
4. **Evaluación final:** RL y RF sobre TEST (evaluación histórica), más un **modo retrospectivo adicional** para XGBoost con el umbral fijado en VALIDACIÓN. La nueva evaluación de XGBoost no se ha ejecutado aquí ni recupera la independencia estadística de TEST.

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

**No vuelvas a ejecutar la evaluación histórica de RL/RF:** Está protegida por `salidas/evaluacion_final_test/TEST_EVALUADO.txt`. Si decides abrir TEST una vez para XGBoost, el nuevo modo `--xgboost` guardará exclusivamente resultados adicionales en `salidas/evaluacion_xgboost_test_retrospectiva/` y creará un marcador propio. No cambia el umbral 0,66 ni los resultados originales.

```cmd
python scripts/evaluacion/evaluacion_final_test.py --xgboost
```

**Advertencia:** No ejecutes este comando desde una copia sin los archivos locales de `datos/procesado/`, `modelos/` y `salidas/`. El análisis sigue siendo **retrospectivo**: los resultados de TEST del estudio inicial ya eran conocidos antes de desarrollar el tercer modelo.

## Documentación

- [Guía metodológica unificada](docs/GUIA_METODOLOGICA.md)
- [Mapa de nombres y rutas anteriores](docs/EQUIVALENCIAS_SCRIPTS.md)
- [Registro histórico y limitaciones](docs/REGISTRO_HISTORICO.md)
- [Dataset fuente](https://huggingface.co/datasets/CS2CD/Context_window_256)
