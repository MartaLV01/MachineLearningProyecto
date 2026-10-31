"""Ejecutar con Python que disponga de pandas, openpyxl y statsmodels."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson

BASE = Path(__file__).resolve().parent
PROYECTO = BASE.parents[2]

def main():
    fuente = pd.read_excel(PROYECTO / 'data/euribor_datos.xlsx', sheet_name='Regresion', header=5)
    resumen = []
    for archivo in sorted(BASE.parent.glob('Model_LinearRegression_*.py')):
        # Leer la configuración sin ejecutar los scripts originales (tienen efectos al importar).
        config = {n.targets[0].id: ast.literal_eval(n.value)
                  for n in ast.parse(archivo.read_text()).body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id in ('features', 'target')}
        features, target = config['features'], config['target']
        datos = fuente[['Fecha', 'País ISO', *features, target]].copy()
        datos['Fecha'] = pd.to_datetime(datos['Fecha'], dayfirst=True, errors='coerce')
        datos['País ISO'] = datos['País ISO'].astype('string').str.strip().replace('', pd.NA)
        for col in [*features, target]:
            datos[col] = pd.to_numeric(datos[col], errors='coerce')
        datos = datos.replace([np.inf, -np.inf], np.nan).dropna()
        datos = datos.sort_values(['Fecha', 'País ISO'], kind='stable').reset_index(drop=True)
        datos['Mes'] = datos['Fecha'].dt.to_period('M')
        anteriores = datos.loc[datos.Fecha < '2025-01-01']
        conteo = anteriores.groupby('Mes').size()
        acumulado = conteo.iloc[::-1].cumsum().iloc[::-1]
        corte = (acumulado[acumulado < len(anteriores)] - round(len(datos) * .20)).abs().idxmin()
        datos['conjunto'] = np.where(datos.Fecha >= '2025-01-01', 'test', np.where(datos.Mes < corte, 'train', 'validacion'))
        train = datos.loc[datos.conjunto == 'train']
        ajuste = sm.OLS(train[target], sm.add_constant(train[features], has_constant='add'), missing='raise').fit()
        datos['prediccion'] = ajuste.predict(sm.add_constant(datos[features], has_constant='add'))
        datos['error'] = datos[target] - datos.prediccion
        nombre = archivo.stem.removeprefix('Model_LinearRegression_')
        for pais, grupo in datos.groupby('País ISO'):
            grupo = grupo.sort_values('Fecha')
            grupo[['Fecha', 'País ISO', 'conjunto', target, 'prediccion', 'error']].to_csv(BASE / f'{pais}_{nombre}.csv', index=False)
            for conjunto, bloque in grupo.groupby('conjunto'):
                # Los retardos sólo representan meses si la secuencia es continua.
                meses = bloque.Mes.astype('int64').to_numpy()
                continuo = len(set(meses)) == len(meses) and np.all(np.diff(meses) == 1)
                errores = bloque.error.to_numpy()
                lag = min(12, len(errores) // 5)
                fila = dict(pais=pais, modelo=nombre, conjunto=conjunto, n=len(bloque), inicio=str(bloque.Fecha.min().date()), fin=str(bloque.Fecha.max().date()), meses_continuos=bool(continuo), retardo=lag, durbin_watson=float(durbin_watson(errores)), p_ljung_box=np.nan)
                if continuo and lag >= 1 and np.std(errores) > 0:
                    fila['p_ljung_box'] = float(acorr_ljungbox(errores, lags=[lag], model_df=0).lb_pvalue.iloc[0])
                resumen.append(fila)
    tabla = pd.DataFrame(resumen)
    validos = tabla.p_ljung_box.notna()
    tabla['p_ajustado_holm'] = np.nan
    tabla.loc[validos, 'p_ajustado_holm'] = multipletests(tabla.loc[validos, 'p_ljung_box'], method='holm')[1]
    tabla['diagnostico'] = np.where(~validos, 'No evaluable: serie corta, constante o meses discontinuos', np.where(tabla.p_ajustado_holm < .05, 'Evidencia de autocorrelacion', 'Sin evidencia suficiente'))
    tabla.to_csv(BASE / 'resumen_autocorrelacion.csv', index=False)
    for (pais, modelo), filas in tabla.groupby(['pais', 'modelo']):
        (BASE / f'{pais}_{modelo}.md').write_text(f'# {pais} — {modelo}\n\n' + filas.to_string(index=False) + '\n', encoding='utf-8')
    print(tabla.diagnostico.value_counts().to_string())

if __name__ == '__main__':
    main()
