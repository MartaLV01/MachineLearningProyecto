"""Corrección AR(1) agrupada de los cuatro modelos existentes.

Ejecutar después de analizar.py. Sólo lee y escribe dentro de esta carpeta.
Evaluación a un mes: al predecir t se supone observado el resultado de t-1.
El OLS base y phi quedan congelados tras entrenamiento.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson

BASE = Path(__file__).resolve().parent
MODELOS = ['2Var', '4Var_Inflacion', '4Var_Produccion', '6Var']


def metricas(filas, prediccion):
    errores = filas['Rendimiento BCE'].to_numpy() - np.asarray(prediccion)
    return float(np.sqrt(np.mean(errores ** 2))), float(np.mean(np.abs(errores)))


def r2_prediccion(filas, prediccion):
    y = filas['Rendimiento BCE'].to_numpy()
    denominador = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - np.asarray(prediccion)) ** 2) / denominador) if denominador > 0 else np.nan


def ejecutar():
    resultados, parametros, predicciones, por_pais = [], [], [], []
    coeficientes, ajustes, metricas_prediccion, dw_paises = [], [], [], []
    for modelo in MODELOS:
        archivos = sorted(BASE.glob(f'*_{modelo}.csv'))
        if not archivos:
            raise ValueError(f'Faltan errores originales para {modelo}; ejecutar analizar.py.')
        datos = pd.concat([pd.read_csv(p) for p in archivos], ignore_index=True)
        datos['Fecha'] = pd.to_datetime(datos['Fecha'])
        datos['mes'] = datos.Fecha.dt.year * 12 + datos.Fecha.dt.month
        if datos.duplicated(['País ISO', 'mes']).any():
            raise ValueError('Hay más de una observación por país y mes.')
        datos = datos.sort_values(['País ISO', 'Fecha']).reset_index(drop=True)
        grupos = datos.groupby('País ISO', sort=False)
        datos['error_lag1'] = grupos.error.shift(1)
        datos['conjunto_anterior'] = grupos.conjunto.shift(1)
        datos['mes_anterior'] = grupos.mes.shift(1)
        datos.loc[datos.mes - datos.mes_anterior != 1, 'error_lag1'] = np.nan
        train = datos.loc[(datos.conjunto == 'train') & (datos.conjunto_anterior == 'train')].dropna(subset=['error_lag1'])
        if train.empty:
            raise ValueError('No hay pares consecutivos de entrenamiento.')
        # OLS sin constante: error_t = phi * error_(t-1).
        x, y = train.error_lag1.to_numpy(), train.error.to_numpy()
        if np.dot(x, x) == 0:
            raise ValueError('Los errores de entrenamiento no permiten estimar phi.')
        ajuste_ar = OLS(train.error, train[['error_lag1']], missing='raise').fit()
        phi = float(ajuste_ar.params['error_lag1'])
        ic = ajuste_ar.conf_int()
        coeficientes.append(dict(modelo=modelo, variable='error_lag1', b=phi,
            error_estandar=ajuste_ar.bse.iloc[0], t=ajuste_ar.tvalues.iloc[0],
            p_valor=ajuste_ar.pvalues.iloc[0], IC95_inferior=ic.iloc[0, 0],
            IC95_superior=ic.iloc[0, 1], significativo_5pct=ajuste_ar.pvalues.iloc[0] < .05))
        ajustes.append(dict(modelo=modelo, n=int(ajuste_ar.nobs),
            R2_no_centrado=ajuste_ar.rsquared, R2_ajustado_no_centrado=ajuste_ar.rsquared_adj,
            F=ajuste_ar.fvalue, p_valor_F=ajuste_ar.f_pvalue,
            RMSE_error_train=float(np.sqrt(np.mean(ajuste_ar.resid ** 2))),
            MAE_error_train=float(np.mean(np.abs(ajuste_ar.resid))),
            AIC=ajuste_ar.aic, BIC=ajuste_ar.bic))
        parametros.append(dict(modelo=modelo, phi=phi, pares_train=len(train), paises_train=train['País ISO'].nunique()))
        datos['modelo'] = modelo
        datos['phi'] = phi
        datos['correccion_disponible'] = datos.error_lag1.notna()
        # Sin observación del mes anterior se conserva la predicción base.
        datos['prediccion_AR1'] = datos.prediccion + phi * datos.error_lag1.fillna(0)
        datos['error_AR1'] = datos['Rendimiento BCE'] - datos.prediccion_AR1
        predicciones.append(datos)
        for conjunto in ['train', 'validacion', 'test']:
            bloque = datos.loc[datos.conjunto == conjunto]
            for muestra in ['pares_consecutivos', 'todas_las_filas']:
                evaluacion = bloque.loc[bloque.correccion_disponible] if muestra == 'pares_consecutivos' else bloque
                if evaluacion.empty:
                    continue
                rmse, mae = metricas(evaluacion, evaluacion.prediccion)
                rmse_ar, mae_ar = metricas(evaluacion, evaluacion.prediccion_AR1)
                for variante, columna in [('original', 'prediccion'), ('AR1', 'prediccion_AR1')]:
                    rmse_v, mae_v = metricas(evaluacion, evaluacion[columna])
                    metricas_prediccion.append(dict(modelo=modelo, conjunto=conjunto,
                        muestra=muestra, variante=variante, n=len(evaluacion),
                        R2=r2_prediccion(evaluacion, evaluacion[columna]), RMSE=rmse_v, MAE=mae_v))
                resultados.append(dict(modelo=modelo, conjunto=conjunto, muestra=muestra, n=len(evaluacion), rmse_base=rmse, rmse_AR1=rmse_ar, mejora_rmse_pct=100*(1-rmse_ar/rmse), mae_base=mae, mae_AR1=mae_ar, mejora_mae_pct=100*(1-mae_ar/mae)))
            for pais, evaluacion in bloque.loc[bloque.correccion_disponible].groupby('País ISO'):
                rmse, mae = metricas(evaluacion, evaluacion.prediccion)
                rmse_ar, mae_ar = metricas(evaluacion, evaluacion.prediccion_AR1)
                por_pais.append(dict(modelo=modelo, pais=pais, conjunto=conjunto, n=len(evaluacion), R2_base=r2_prediccion(evaluacion, evaluacion.prediccion), R2_AR1=r2_prediccion(evaluacion, evaluacion.prediccion_AR1), rmse_base=rmse, rmse_AR1=rmse_ar, mae_base=mae, mae_AR1=mae_ar))
            for pais, serie in bloque.groupby('País ISO'):
                serie = serie.sort_values('mes')
                tramos = serie.mes.diff().ne(1).cumsum()
                for tramo, filas in serie.groupby(tramos):
                    # Mismas observaciones para ambas variantes, sin unir huecos ni países.
                    filas = filas.loc[filas.correccion_disponible]
                    if len(filas) < 2:
                        continue
                    dw_paises.append(dict(modelo=modelo, pais=pais, conjunto=conjunto,
                        tramo=int(tramo), inicio=filas.Fecha.min(), fin=filas.Fecha.max(), n=len(filas),
                        DW_base=float(durbin_watson(filas.error)),
                        DW_AR1=float(durbin_watson(filas.error_AR1))))
    pd.DataFrame(coeficientes).to_csv(BASE / 'AR1_coeficientes.csv', index=False)
    pd.DataFrame(ajustes).to_csv(BASE / 'AR1_metricas_ajuste.csv', index=False)
    pd.DataFrame(metricas_prediccion).to_csv(BASE / 'AR1_metricas_prediccion.csv', index=False)
    pd.DataFrame(dw_paises).to_csv(BASE / 'AR1_durbin_watson_por_pais.csv', index=False)
    resumen = pd.DataFrame(resultados)
    pd.DataFrame(parametros).to_csv(BASE / 'AR1_parametros.csv', index=False)
    resumen.to_csv(BASE / 'AR1_comparacion.csv', index=False)
    pd.DataFrame(por_pais).to_csv(BASE / 'AR1_por_pais.csv', index=False)
    pd.concat(predicciones, ignore_index=True).to_csv(BASE / 'AR1_predicciones.csv', index=False)
    lineas = ['# Resultados de la corrección AR(1)', '',
        'Se estima un phi común a todos los países por cada modelo base, exclusivamente con pares de errores consecutivos de entrenamiento. El retardo pertenece siempre al mismo país y al mes inmediatamente anterior. No se reestiman los modelos base ni se usan validación o test para ajustar phi.', '',
        '**Escenario: predicción a un mes con el resultado del mes anterior ya observado.** En validación y test se actualiza el error anterior observado, pero no los parámetros. Esto no equivale a pronosticar todo el período de una sola vez. Los errores de entrenamiento son residuos del ajuste, por lo que sus métricas son descriptivas y optimistas.', '',
        '## Coeficientes', '', '| Modelo | Phi | Pares de entrenamiento |', '|---|---:|---:|']
    for r in parametros:
        lineas.append(f"| {r['modelo']} | {r['phi']:.4f} | {r['pares_train']} |")
    lineas += ['', '## Comparación en las mismas filas con mes anterior disponible', '', '| Conjunto | Modelo | N | RMSE original | RMSE AR1 | Mejora RMSE | MAE original | MAE AR1 | Mejora MAE |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in resultados:
        if r['muestra'] == 'pares_consecutivos':
            lineas.append(f"| {r['conjunto']} | {r['modelo']} | {r['n']} | {r['rmse_base']:.4f} | {r['rmse_AR1']:.4f} | {r['mejora_rmse_pct']:.1f}% | {r['mae_base']:.4f} | {r['mae_AR1']:.4f} | {r['mejora_mae_pct']:.1f}% |")
    lineas += ['', 'Un porcentaje positivo indica reducción del error; uno negativo, empeoramiento. Los promedios ponderan observaciones, no países. Las muestras pueden diferir entre modelos por valores ausentes.', '',
        'AR1_comparacion.csv incluye también todas las filas: si falta el mes anterior se utiliza el modelo original sin corrección. AR1_por_pais.csv permite comprobar excepciones por país. AR1_predicciones.csv conserva cada predicción, el retardo y su disponibilidad.', '',
        'El test ya se había inspeccionado al recomendar el retardo: estos resultados son retrospectivos y no una confirmación independiente de la elección. Para confirmar la mejora conviene una evaluación temporal nueva. No se ha comparado AR(2) ni seleccionado otro orden.', '']
    lineas += ['## Métricas equivalentes a los modelos originales', '',
        'AR1_coeficientes.csv: b, error estándar, t, p-valor, intervalo al 95 % y significación del coeficiente phi. Son inferencias OLS convencionales, condicionales a los residuos estimados del modelo base; no incorporan la incertidumbre de esa primera etapa.', '',
        'AR1_metricas_ajuste.csv: N, R² y R² ajustado no centrados (regresión sin constante), F, p-valor F, RMSE, MAE, AIC y BIC del ajuste error_t = phi × error_(t−1) en entrenamiento. No son el R², AIC o BIC del modelo completo sobre Rendimiento BCE ni se deben comparar con los AIC/BIC de los OLS originales.', '',
        'AR1_metricas_prediccion.csv: R² centrado, RMSE y MAE sobre Rendimiento BCE, para original y AR1 en entrenamiento, validación y test; tanto en pares consecutivos como en todas las filas. Estas son las métricas para comparar predicciones finales.', '',
        'AR1_durbin_watson_por_pais.csv: Durbin–Watson de los errores originales y corregidos por país, conjunto y tramo mensual continuo; no se concatenan países para calcularlo.', '']
    (BASE / 'AR1_resultados.md').write_text('\n'.join(lineas), encoding='utf-8')
    for nombre, tabla in [('Coeficientes AR1', coeficientes), ('Ajuste de los errores en entrenamiento', ajustes),
                          ('Métricas de predicción', metricas_prediccion), ('Durbin–Watson por país y tramo', dw_paises)]:
        print('\n' + nombre)
        salida = pd.DataFrame(tabla)
        numericas = salida.select_dtypes(include='number').columns
        salida[numericas] = salida[numericas].round(4)
        print(salida.to_string(index=False))
    print(pd.DataFrame(parametros).to_string(index=False))
    print(resumen.loc[(resumen.conjunto != 'train') & (resumen.muestra == 'pares_consecutivos')].to_string(index=False))


if __name__ == '__main__':
    ejecutar()
