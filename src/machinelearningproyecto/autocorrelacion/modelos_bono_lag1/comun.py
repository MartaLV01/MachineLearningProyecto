"""OLS agrupado con rendimiento del mismo país en el mes anterior.

Sólo escribe resultados junto a este archivo. No ejecuta los modelos originales.
"""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson

BASE = Path(__file__).resolve().parent
ORIGINALES = BASE.parent.parent
PROYECTO = BASE.parents[3]
TARGET = 'Rendimiento BCE'
LAG = 'Rendimiento_BCE_mes_anterior'
MODELOS = ('2Var', '4Var_Inflacion', '4Var_Produccion', '6Var')


def configuracion(modelo):
    archivo = ORIGINALES / f'Model_LinearRegression_{modelo}.py'
    valores = {}
    for nodo in ast.parse(archivo.read_text()).body:
        if isinstance(nodo, ast.Assign) and isinstance(nodo.targets[0], ast.Name):
            if nodo.targets[0].id in ('features', 'target'):
                valores[nodo.targets[0].id] = ast.literal_eval(nodo.value)
    if valores['target'] != TARGET:
        raise ValueError('La variable objetivo original ha cambiado.')
    return valores['features']


def preparar(modelo, features):
    datos = pd.read_excel(PROYECTO / 'data/euribor_datos.xlsx', sheet_name='Regresion', header=5)
    datos = datos[['Fecha', 'País ISO', TARGET, *features]].copy()
    datos['Fecha'] = pd.to_datetime(datos.Fecha, dayfirst=True, errors='coerce')
    datos['País ISO'] = datos['País ISO'].astype('string').str.strip().replace('', pd.NA)
    for col in [TARGET, *features]:
        datos[col] = pd.to_numeric(datos[col], errors='coerce')
    datos = datos.replace([np.inf, -np.inf], np.nan).dropna(subset=['Fecha', 'País ISO'])
    datos['mes'] = datos.Fecha.dt.year * 12 + datos.Fecha.dt.month
    if datos.duplicated(['País ISO', 'mes']).any():
        raise ValueError('Más de una observación por país y mes en el Excel.')
    datos = datos.sort_values(['País ISO', 'mes'])
    grupos = datos.groupby('País ISO', sort=False)
    datos[LAG] = grupos[TARGET].shift(1)
    datos['fecha_bono_anterior'] = grupos.Fecha.shift(1)
    datos['mes_anterior'] = grupos.mes.shift(1)
    datos.loc[datos.mes - datos.mes_anterior != 1, [LAG, 'fecha_bono_anterior']] = [np.nan, pd.NaT]
    # Recuperar las asignaciones existentes: no recalcular cortes tras perder filas.
    archivos = sorted(BASE.parent.glob(f'*_{modelo}.csv'))
    if not archivos:
        raise ValueError('Ejecutar primero analizar.py para disponer de las particiones originales.')
    original = pd.concat([pd.read_csv(p) for p in archivos], ignore_index=True)
    original['Fecha'] = pd.to_datetime(original.Fecha)
    original = original.rename(columns={'prediccion': 'prediccion_original'})
    unidos = original[['Fecha', 'País ISO', 'conjunto', 'prediccion_original']].merge(
        datos, on=['Fecha', 'País ISO'], how='left', validate='one_to_one')
    if unidos[[TARGET, *features]].isna().any().any():
        raise ValueError('Los datos actuales no coinciden con la muestra original.')
    disponibles = unidos.dropna(subset=[LAG]).copy()
    assert (disponibles.mes - disponibles.mes_anterior == 1).all()
    assert (disponibles.fecha_bono_anterior < disponibles.Fecha).all()
    return unidos, disponibles.sort_values(['Fecha', 'País ISO'])


def evaluar(filas, columna):
    y = filas[TARGET].to_numpy()
    e = y - filas[columna].to_numpy()
    sst = np.sum((y-y.mean())**2)
    return dict(n=len(filas), R2=float(1-np.sum(e**2)/sst) if sst > 0 else np.nan,
                RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))))


def ejecutar(modelo):
    features = configuracion(modelo)
    originales, datos = preparar(modelo, features)
    variables = [*features, LAG]
    train = datos.loc[datos.conjunto == 'train']
    if train.empty:
        raise ValueError('No hay entrenamiento disponible.')
    ajuste = OLS(train[TARGET], add_constant(train[variables], has_constant='add'), missing='raise').fit()
    datos['prediccion_lag1'] = ajuste.predict(add_constant(datos[variables], has_constant='add'))
    # Referencia de persistencia: predecir directamente el bono del mes anterior.
    datos['prediccion_persistencia'] = datos[LAG]
    datos['error_lag1'] = datos[TARGET] - datos.prediccion_lag1
    ic = ajuste.conf_int()
    coef = pd.DataFrame({'b': ajuste.params, 'error_estandar': ajuste.bse,
        't': ajuste.tvalues, 'p_valor': ajuste.pvalues,
        'IC95_inferior': ic.iloc[:, 0], 'IC95_superior': ic.iloc[:, 1],
        'significativo_5pct': ajuste.pvalues < .05})
    coef.index.name = 'variable'
    coef.to_csv(BASE / f'{modelo}_coeficientes.csv')
    entrenamiento = pd.Series({'n': int(ajuste.nobs), 'R2': ajuste.rsquared,
        'R2_ajustado': ajuste.rsquared_adj, 'F': ajuste.fvalue,
        'p_valor_F': ajuste.f_pvalue, 'RMSE': np.sqrt(np.mean(ajuste.resid**2)),
        'MAE': np.mean(np.abs(ajuste.resid)), 'AIC': ajuste.aic, 'BIC': ajuste.bic})
    entrenamiento.rename_axis('metrica').to_csv(BASE / f'{modelo}_ajuste_train.csv', header=['valor'])
    metricas, particiones, paises, dw = [], [], [], []
    for conjunto in ('train', 'validacion', 'test'):
        bloque = datos.loc[datos.conjunto == conjunto]
        previos = originales.loc[originales.conjunto == conjunto]
        particiones.append(dict(conjunto=conjunto, n_original=len(previos), n_con_retardo=len(bloque),
            excluidas_sin_retardo=len(previos)-len(bloque), inicio_original=previos.Fecha.min(),
            fin_original=previos.Fecha.max(), inicio_utilizado=bloque.Fecha.min(), fin_utilizado=bloque.Fecha.max()))
        for variante in ('original', 'lag1', 'persistencia'):
            metricas.append(dict(modelo=modelo, conjunto=conjunto, variante=variante,
                **evaluar(bloque, f'prediccion_{variante}')))
        for pais, grupo in bloque.groupby('País ISO'):
            for variante in ('original', 'lag1', 'persistencia'):
                paises.append(dict(modelo=modelo, pais=pais, conjunto=conjunto, variante=variante,
                    **evaluar(grupo, f'prediccion_{variante}')))
            grupo = grupo.sort_values('mes')
            for tramo, serie in grupo.groupby(grupo.mes.diff().ne(1).cumsum()):
                if len(serie) >= 2:
                    dw.append(dict(pais=pais, conjunto=conjunto, tramo=tramo, n=len(serie),
                        inicio=serie.Fecha.min(), fin=serie.Fecha.max(),
                        DW_original=durbin_watson(serie[TARGET]-serie.prediccion_original),
                        DW_lag1=durbin_watson(serie.error_lag1)))
    pd.DataFrame(particiones).to_csv(BASE / f'{modelo}_particiones.csv', index=False)
    pd.DataFrame(metricas).to_csv(BASE / f'{modelo}_metricas.csv', index=False)
    pd.DataFrame(paises).to_csv(BASE / f'{modelo}_metricas_por_pais.csv', index=False)
    pd.DataFrame(dw).to_csv(BASE / f'{modelo}_durbin_watson.csv', index=False)
    datos.to_csv(BASE / f'{modelo}_predicciones.csv', index=False)
    print(f'\n{modelo}: {len(variables)} predictores más constante')
    print(pd.DataFrame(metricas).round(4).to_string(index=False))
    return pd.DataFrame(metricas), coef.loc[LAG, 'b']


def ejecutar_todos():
    tablas, coeficientes = [], {}
    for modelo in MODELOS:
        tabla, coeficientes[modelo] = ejecutar(modelo)
        tablas.append(tabla)
    resumen = pd.concat(tablas, ignore_index=True)
    resumen.to_csv(BASE / 'comparacion_general.csv', index=False)
    texto = ['# Modelos con el bono del mismo país un mes antes', '',
        'Cuatro OLS agrupados: 2, 4 de inflación, 4 de producción y 6 variables originales, más Rendimiento BCE del mismo país en el mes anterior (3, 5, 5 y 7 predictores). Cada modelo comparte coeficientes entre países. Todos los coeficientes se ajustan sólo con entrenamiento.', '',
        'El retardo se crea por país sobre el Excel antes de eliminar observaciones por otras variables. Sólo se admite el mes inmediatamente anterior. Se conservan las asignaciones originales a entrenamiento, validación y test; las filas sin bono anterior se excluyen sin recalcular cortes. Los CSV de particiones muestran las pérdidas.', '',
        'Se evalúan predicciones sucesivas a un mes, con el bono real anterior ya observado. No se utilizan resultados del mes actual o posterior para construir el retardo. Esto no representa predecir todo el período de una vez; además las variables macroeconómicas contemporáneas de los modelos originales tendrían que estar disponibles en el momento de predecir.', '',
        '## Comparación sobre exactamente las mismas filas', '',
        'Original: modelo previo sin retardo, ya entrenado sobre su muestra original. Lag1: OLS nuevo con el bono anterior, entrenado en las filas de entrenamiento con retardo disponible. La diferencia combina el predictor añadido y la pérdida de filas de entrenamiento. Persistencia: usar directamente el bono anterior como predicción, sin ajustar parámetros.', '',
        '| Conjunto | Modelo | N | RMSE original | RMSE lag1 | RMSE persistencia | R² lag1 | MAE lag1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for conjunto in ('train', 'validacion', 'test'):
        for modelo in MODELOS:
            filas = resumen.loc[(resumen.modelo == modelo)&(resumen.conjunto == conjunto)].set_index('variante')
            a, b, c = (filas.loc[v] for v in ('original', 'lag1', 'persistencia'))
            texto.append(f'| {conjunto} | {modelo} | {int(b.n)} | {a.RMSE:.4f} | {b.RMSE:.4f} | {c.RMSE:.4f} | {b.R2:.4f} | {b.MAE:.4f} |')
    texto += ['', '## Interpretación de los resultados', '',
        'Con los resultados actuales, los cuatro modelos con retardo reducen claramente RMSE y MAE frente a los originales en validación y test. El bono del mes anterior aporta mucha información predictiva.', '',
        'Sin embargo, la referencia de persistencia consigue menor RMSE y MAE que los cuatro modelos con retardo tanto en validación como en test. Por tanto, estos resultados no demuestran que las variables macroeconómicas añadan capacidad predictiva al bono anterior; no elegiría un modelo más complejo sólo por su mejora frente al original.', '',
        'Entre los nuevos modelos, el de 4 variables de inflación obtiene el menor RMSE de validación. Su elección sería provisional y basada en validación; el test se presenta como evaluación retrospectiva.', '',
        '## Coeficiente del bono anterior', '']
    texto += [f'- {m}: {v:.4f}.' for m,v in coeficientes.items()]
    texto += ['', '## Ejecutar y consultar resultados', '',
        'Desde la raíz del proyecto:', '', '```bash',
        './src/machinelearningproyecto/autocorrelacion/.venv/bin/python src/machinelearningproyecto/autocorrelacion/modelos_bono_lag1/ejecutar_todos.py', '```', '',
        'También puedes ejecutar cualquiera de los cuatro archivos modelo_*.py con el mismo Python. No se necesitan paquetes nuevos.', '',
        '- comparacion_general.csv: R², RMSE y MAE por modelo, conjunto y variante.',
        '- *_coeficientes.csv: coeficientes, errores estándar, t, p-valores, IC al 95 % y significación.',
        '- *_ajuste_train.csv: N, R² ajustado, F, p-valor F, AIC y BIC del nuevo modelo OLS.',
        '- *_metricas_por_pais.csv y *_durbin_watson.csv: resultados por país; DW por tramo continuo.',
        '- *_predicciones.csv: observaciones, fecha del bono anterior, retardo, predicciones y errores.',
        '- *_particiones.csv: tamaños y fechas antes y después de exigir el retardo.', '',
        'Los errores estándar e inferencias son los OLS convencionales y pueden verse afectados por dependencia temporal. Un buen rendimiento frente al modelo original no basta: debe compararse también con persistencia. El test ya ha sido consultado durante este trabajo; la confirmación independiente requiere datos posteriores.', '']
    (BASE / 'README.md').write_text('\n'.join(texto), encoding='utf-8')
