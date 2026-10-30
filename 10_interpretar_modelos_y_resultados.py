"""Interpretación posterior a la evaluación final de CS2CD.

No entrena modelos, no carga TEST/VALIDACION crudos, no cambia
predictores, hiperparámetros ni umbrales. Reutiliza solamente
modelos guardados y archivos finales ya producidos.
"""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path('.')
RUTA_PREDICTORES = ROOT / 'salidas/regresion_logistica/predictores_finales_72.txt'
RUTA_RL = ROOT / 'modelos/regresion_logistica_best_pipeline.joblib'
RUTA_RF = ROOT / 'modelos/random_forest_best_pipeline.joblib'
RUTA_RESULTADOS_TEST = ROOT / 'salidas/evaluacion_final_test/metricas_finales_test.csv'
RUTA_MARCADOR_TEST = ROOT / 'salidas/evaluacion_final_test/TEST_EVALUADO.txt'
SALIDA = ROOT / 'salidas/interpretacion_modelos'


def grupo_de(caracteristica):
    if caracteristica.startswith('attacker_vel_'):
        return 'Movimiento'
    if caracteristica.startswith((
        'attacker_pitch_delta_', 'attacker_yaw_delta_',
        'attacker_pitch_head_delta_', 'attacker_yaw_head_delta_'
    )):
        return 'Apuntado y orientación'
    if caracteristica.startswith(('delta_x_', 'delta_y_', 'delta_z_', 'distancia_3d_')):
        return 'Relación espacial'
    if caracteristica.startswith('weapon_'):
        return 'Armas'
    if caracteristica.startswith((
        'attacker_flashed_', 'victim_noise_', 'attacker_shot_',
        'kill_', 'victim_health_'
    )):
        return 'Combate y estados'
    return 'Otros'


def grafico_horiz_barras(tabla, columna_valor, columna_nombre, titulo, archivo, top=20):
    seleccion = tabla.head(top).iloc[::-1]
    alto = max(5, 0.34 * len(seleccion) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, alto))
    ax.barh(seleccion[columna_nombre], seleccion[columna_valor])
    ax.set_title(titulo)
    ax.set_xlabel(columna_valor.replace('_', ' '))
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    fig.savefig(SALIDA / archivo, dpi=200, bbox_inches='tight')
    plt.close(fig)


print('INTERPRETACIÓN DE MODELOS Y RESUMEN DE RESULTADOS')
print('=================================================\n')

requeridos = [RUTA_PREDICTORES, RUTA_RL, RUTA_RF,
              RUTA_RESULTADOS_TEST, RUTA_MARCADOR_TEST]
for ruta in requeridos:
    if not ruta.exists():
        raise FileNotFoundError(f'Falta archivo requerido: {ruta}')

SALIDA.mkdir(parents=True, exist_ok=True)

predictores = [linea.strip() for linea in RUTA_PREDICTORES.read_text(encoding='utf-8').splitlines()
               if linea.strip()]
if len(predictores) != 72 or len(set(predictores)) != 72:
    raise ValueError('Se esperaban 72 nombres de predictores únicos y congelados.')

rl = joblib.load(RUTA_RL)
rf = joblib.load(RUTA_RF)

# ------------------------------------------------------------
# Coeficientes de Regresión Logística. ColumnTransformer puede
# reordenar continuas y binarias, por eso se recupera el orden
# del preprocesamiento YA AJUSTADO y no se asume el orden de X.
# ------------------------------------------------------------

if 'preprocesamiento' not in rl.named_steps or 'modelo' not in rl.named_steps:
    raise RuntimeError('Pipeline RL no tiene las etapas esperadas.')

preprocesador = rl.named_steps['preprocesamiento']
modelo_rl = rl.named_steps['modelo']
nombres_rl = [str(x) for x in preprocesador.get_feature_names_out()]
coeficientes = np.asarray(modelo_rl.coef_[0], dtype=float)

if len(nombres_rl) != 72 or len(coeficientes) != 72:
    raise RuntimeError('El pipeline RL no contiene 72 coeficientes.')
if set(nombres_rl) != set(predictores):
    raise RuntimeError('Los nombres del pipeline RL no coinciden con los 72 predictores.')

rl_df = pd.DataFrame({
    'caracteristica': nombres_rl,
    'coeficiente': coeficientes,
})
rl_df['magnitud_absoluta'] = rl_df['coeficiente'].abs()
rl_df['sentido_asociacion'] = np.where(
    rl_df['coeficiente'] > 0, 'Positiva',
    np.where(rl_df['coeficiente'] < 0, 'Negativa', 'Nula')
)
rl_df['grupo'] = rl_df['caracteristica'].map(grupo_de)
rl_df = rl_df.sort_values('magnitud_absoluta', ascending=False)
rl_df.to_csv(SALIDA / 'coeficientes_regresion_logistica.csv', index=False)

# Esta gráfica muestra magnitud absoluta; el signo queda en el CSV.
grafico_horiz_barras(
    rl_df, 'magnitud_absoluta', 'caracteristica',
    'Regresión Logística: mayor magnitud absoluta de coeficientes',
    'top20_coeficientes_rl.png', top=20
)

# ------------------------------------------------------------
# Importancias por disminución media de impureza (MDI) del RF.
# No son importancias causales ni equivalen a permutación.
# ------------------------------------------------------------

if 'modelo' not in rf.named_steps:
    raise RuntimeError('Pipeline RF no tiene la etapa modelo esperada.')

modelo_rf = rf.named_steps['modelo']
importancias = np.asarray(modelo_rf.feature_importances_, dtype=float)
if len(importancias) != 72:
    raise RuntimeError('Random Forest no contiene 72 importancias.')

rf_df = pd.DataFrame({
    'caracteristica': predictores,
    'importancia_mdi': importancias,
})
rf_df['grupo'] = rf_df['caracteristica'].map(grupo_de)
rf_df = rf_df.sort_values('importancia_mdi', ascending=False)
rf_df.to_csv(SALIDA / 'importancias_random_forest.csv', index=False)

grafico_horiz_barras(
    rf_df, 'importancia_mdi', 'caracteristica',
    'Random Forest: 20 mayores importancias MDI',
    'top20_importancias_rf.png', top=20
)

# Agregado descriptivo de grupos: depende de cuántas variables
# haya en cada grupo, por eso se reportan cantidad y promedio.
grupos_rf = (rf_df.groupby('grupo', as_index=False)
             .agg(n_caracteristicas=('caracteristica', 'size'),
                  importancia_total=('importancia_mdi', 'sum'),
                  importancia_media=('importancia_mdi', 'mean'))
             .sort_values('importancia_total', ascending=False))
grupos_rf.to_csv(SALIDA / 'importancias_grupos_rf.csv', index=False)

# ------------------------------------------------------------
# Documentación de las métricas YA calculadas en TEST.
# Este script no vuelve a ejecutar predict ni a abrir TEST.
# ------------------------------------------------------------

metricas = pd.read_csv(RUTA_RESULTADOS_TEST)
esperadas = {'modelo', 'f1', 'accuracy', 'precision', 'recall',
             'especificidad', 'roc_auc', 'falsos_positivos',
             'falsos_negativos'}
if not esperadas.issubset(metricas.columns) or len(metricas) != 2:
    raise ValueError('El archivo de métricas finales no tiene el formato esperado.')

metricas.to_csv(SALIDA / 'comparacion_final_modelos.csv', index=False)

graf = metricas.set_index('modelo')[
    ['f1', 'precision', 'recall', 'roc_auc', 'accuracy', 'especificidad']
].T
ax = graf.plot(kind='bar', figsize=(10, 5), rot=0)
ax.set_ylim(0, 1)
ax.set_ylabel('Puntuación')
ax.set_title('Comparación de modelos sobre TEST (umbrales congelados)')
ax.legend(title='Modelo')
ax.figure.tight_layout()
ax.figure.savefig(SALIDA / 'comparacion_metricas_test.png', dpi=200, bbox_inches='tight')
plt.close(ax.figure)

registro = {
    'uso_de_datos': 'Interpretación de modelos ajustados y resumen de métricas ya guardadas',
    'reentrenamiento': False,
    'reevaluacion_test': False,
    'n_predictores': 72,
    'coeficientes_rl': 'Signo y magnitud condicionales al preprocesamiento del pipeline',
    'importancias_rf': 'MDI/impureza; descriptivas, sujetas a sesgos de importancia',
    'nota_grupos': 'La suma de MDI por grupo depende también del número de características',
    'limite': 'Las asociaciones entre telemetría y etiquetas no prueban trampas en una eliminación individual',
}
(SALIDA / 'notas_interpretacion.json').write_text(
    json.dumps(registro, ensure_ascii=False, indent=2), encoding='utf-8'
)

print('Top 10 importancias de Random Forest:')
print(rf_df[['caracteristica', 'importancia_mdi']].head(10).to_string(index=False))
print('\nTop 10 coeficientes de RL por magnitud absoluta:')
print(rl_df[['caracteristica', 'coeficiente']].head(10).to_string(index=False))
print('\nResumen por grupos (Random Forest):')
print(grupos_rf.to_string(index=False))
print('\nArchivos de interpretación creados en:', SALIDA)
print('No se reentrenó ningún modelo y TEST no volvió a evaluarse.')
