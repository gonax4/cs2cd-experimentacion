"""Demostración de solo lectura para una sustentación técnica de CS2CD.

No se conecta a TEST, no ajusta modelos y no modifica ningún CSV.
Ejecutar desde la raíz del proyecto: python demostracion_lectura_cs2cd.py
"""
from pathlib import Path
import pandas as pd

ruta = Path('datos/procesado/caracteristicas_train.csv')
if not ruta.is_file():
    raise FileNotFoundError(f'Abre terminal en la raíz del proyecto. Falta: {ruta}')

campos = [
    'clave', 'partida_id', 'particion', 'etiqueta',
    'attacker_vel_media', 'attacker_shot_conteo',
    'weapon_transiciones', 'diag_weapon_ticks_sin_categoria',
    'victim_health_min', 'victim_health_kill',
]
train = pd.read_csv(ruta, usecols=campos)
print('\n1) Origen local de resultados transformados:', ruta)
print('   Ventanas TRAIN:', len(train))
print('   Partidas TRAIN:', train['partida_id'].nunique())
print('\n2) SELECT equivalente: solo cinco columnas y cinco filas')
seleccion = ['partida_id', 'etiqueta', 'attacker_vel_media',
             'attacker_shot_conteo', 'weapon_transiciones']
print(train[seleccion].head(5).to_string(index=False))
print('\n3) WHERE equivalente: primeras cinco filas de clase positiva (etiqueta = 1)')
print(train.loc[train['etiqueta'].eq(1), seleccion].head(5).to_string(index=False))
print('\n4) Problema realmente observado: ticks sin categoria valida de arma')
incidencias = train.loc[train['diag_weapon_ticks_sin_categoria'].gt(0)]
print('   Ventanas afectadas:', len(incidencias))
print(incidencias[['partida_id','diag_weapon_ticks_sin_categoria',
                  'weapon_transiciones']].head(5).to_string(index=False))
print('\n5) Verificacion de dos columnas constantes y de valores faltantes')
for nombre in ['victim_health_min', 'victim_health_kill']:
    print(f'   {nombre}: nunique={train[nombre].nunique()},'
          f' valores={train[nombre].unique()[:5]}')
print('   NaN de las diez columnas cargadas:', int(train.isna().sum().sum()))
print('\nFIN. No se guardo, modifico ni entreno nada.')
