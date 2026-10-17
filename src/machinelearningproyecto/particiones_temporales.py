"""Ejecutar después de cargar el Excel en df y antes de las regresiones.

Requiere: pandas y numpy.
Conserva el test desde enero de 2025 y los cinco folds originales.
Configura features y target para que coincidan con tu regresión.
"""
import numpy as np
import pandas as pd

# TODO 1: organizarlo en funciones
# TODO 2: que cada una de las files que haga un modelo importe estas funciones

def leer_excel():
# Leer el archivo especificando la hoja correcta: 'Regresion'
    df = pd.read_excel('../../data/euribor_datos.xlsx', sheet_name='Regresion', header=5)
    return df
    #df.head(10)
    #print(df.head(5))

"""
No harua falta ya que esas varibales no se meten en la division de los datos
#copiamos el daframe en uno sobre el que trabajaremos
df_model = df.copy()
drop_columns = ["Estado bonos", "Estado IPC", "Estado IPP Eurostat", "Estado deuda BCE"]
df_model = df_model.drop(columns=drop_columns, errors="ignore")
"""


features = [
    'Inflación HICP interanual (%)',
    'IPP país (2021=100)',
    'Desempleo país (%)',
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'IPI variación interanual (%)',
]
target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
variables_pais = features + ['País ISO']

def separacion_variables_entrenamiento(features, target, variables_pais):
    df = leer_excel()
    # 1. Preparación: conservar juntos todos los países del mismo mes.
    datos = df[['Fecha'] + variables_pais + [target]].copy()
    datos['Fecha'] = pd.to_datetime(datos['Fecha'], dayfirst=True, errors='coerce')
    datos['País ISO'] = datos['País ISO'].astype('string').str.strip().replace('', pd.NA)
    for columna in features + [target]:
        datos[columna] = pd.to_numeric(datos[columna], errors='coerce')
    datos[features + [target]] = datos[features + [target]].replace([np.inf, -np.inf], np.nan)
    datos = datos.dropna(subset=['Fecha'] + variables_pais + [target])
    datos = datos.sort_values(['Fecha', 'País ISO'], kind='stable').reset_index(drop=True)
    datos['Mes'] = datos['Fecha'].dt.to_period('M')

    # 2. Test original: desde enero de 2025, sin solapamiento.
    train_val = datos.loc[datos['Fecha'] < '2025-01-01'].copy().reset_index(drop=True)
    test = datos.loc[datos['Fecha'] >= '2025-01-01'].copy()

    # 3. División train / validación temporal original.
    # El objetivo es aproximadamente el 20 % de FILAS DE TODA LA MUESTRA,
    # como en la última celda del notebook, respetando meses completos.
    conteo = train_val.groupby('Mes', sort=True).size()
    if len(conteo) < 5:
        raise ValueError('Se necesitan al menos cinco meses anteriores al test.')
    objetivo_val = round(len(datos) * 0.20)
    acumulado_inverso = conteo.iloc[::-1].cumsum().iloc[::-1]
    candidatos_val = acumulado_inverso[acumulado_inverso < len(train_val)]
    mes_corte = (candidatos_val - objetivo_val).abs().idxmin()

    train = train_val.loc[train_val['Mes'] < mes_corte].copy()
    val = train_val.loc[train_val['Mes'] >= mes_corte].copy()
    assert train['Fecha'].max() < val['Fecha'].min()
    if not test.empty:
        assert val['Fecha'].max() < test['Fecha'].min()

    # Para las regresiones numéricas; País ISO no se introduce sin codificar.
    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]
    # Si tu modelo codifica País ISO, usa train/val/test[variables_pais].

    # 4. Cinco bloques originales: meses consecutivos y filas similares.
    # Son una evaluación alternativa a la división anterior sobre train_val.
    # ATENCIÓN: entrenar con los otros cuatro bloques incluye datos futuros en
    # los folds 1–4. Se conserva el KFold original, no una ventana expansiva.
    n_splits = 5
    acumulado = conteo.cumsum().to_numpy()
    cortes = []
    anterior = 0
    for k in range(1, n_splits):
        candidatos = np.arange(anterior + 1, len(conteo) - (n_splits - k) + 1)
        errores = np.abs(acumulado[candidatos - 1] - k * len(train_val) / n_splits)
        corte = int(candidatos[np.argmin(errores)])
        cortes.append(corte)
        anterior = corte

    bloques = np.split(conteo.index.to_numpy(), cortes)
    mapa_folds = {
        mes: numero
        for numero, meses in enumerate(bloques, start=1)
        for mes in meses
    }
    train_val['Fold'] = train_val['Mes'].map(mapa_folds)

    # Índices POSICIONALES para usar con iloc o cv=folds en scikit-learn.
    X_cv, y_cv = train_val[features], train_val[target]
    folds = []
    for numero in range(1, n_splits + 1):
        mascara = train_val['Fold'].eq(numero).to_numpy()
        folds.append((np.flatnonzero(~mascara), np.flatnonzero(mascara)))

    # Uso junto a tu regresión (sin entrenar nada aquí):
    # for numero, (idx_train, idx_val) in enumerate(folds, start=1):
    #     X_train_fold, y_train_fold = X_cv.iloc[idx_train], y_cv.iloc[idx_train]
    #     X_val_fold, y_val_fold = X_cv.iloc[idx_val], y_cv.iloc[idx_val]
    #     ... ajustar un modelo nuevo y evaluar este fold ...
    # Los índices de folds corresponden a X_cv/y_cv, no a X_train/y_train.
    # La validación final comparte datos con los folds: no es un holdout
    # independiente si ya se usaron los cinco folds para seleccionar el modelo.

    # 5. Mostrar claramente las divisiones, sin guardar archivos ni ajustar modelos.
    resumen_division = pd.DataFrame([
        {'Conjunto': nombre, 'Filas': len(d), 'Inicio': d['Fecha'].min(),
        'Fin': d['Fecha'].max()}
        for nombre, d in [('Train', train), ('Validación', val), ('Test', test)]
    ])
    resumen_folds = train_val.groupby('Fold').agg(
        Inicio=('Fecha', 'min'), Fin=('Fecha', 'max'),
        Filas=('Fecha', 'size'), Meses=('Mes', 'nunique'),
    )
    print('\nDIVISIÓN TEMPORAL TRAIN / VALIDACIÓN / TEST')
    print(resumen_division.to_string(index=False))
    print('\nCINCO BLOQUES DE VALIDACIÓN ORIGINALES')
    print(resumen_folds.to_string())


    return X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv



"""
ESTO ESLO QUE DEBERIA DAR AL EEJECUTAR LA FILE: 

DIVISIÓN TEMPORAL TRAIN / VALIDACIÓN / TEST
  Conjunto  Filas     Inicio        Fin
     Train   1296 2010-06-30 2023-07-31
Validación    434 2023-08-31 2024-12-31
      Test    463 2025-01-31 2026-07-31

CINCO BLOQUES DE VALIDACIÓN ORIGINALES
         Inicio        Fin  Filas  Meses
Fold                                    
1    2010-06-30 2020-04-30    334     94
2    2020-05-31 2021-07-31    362     15
3    2021-08-31 2022-09-30    346     14
4    2022-10-31 2023-10-31    332     13
5    2023-11-30 2024-12-31    356     14

"""


