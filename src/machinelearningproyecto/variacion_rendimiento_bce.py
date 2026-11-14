"""Añade la variación porcentual mensual del rendimiento BCE por país."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

COLUMNA = 'Variación mensual Rendimiento BCE (%)'
FUENTE = Path(__file__).resolve().parents[2] / 'data' / 'euribor_datos.xlsx'


def añadir_variacion(df):
    """Usa el mes anterior natural; sin dato anterior o con denominador cero, NaN."""
    resultado = df.copy()
    pais = resultado['País ISO'].astype('string').str.strip().replace('', pd.NA)
    mes = pd.to_datetime(resultado['Fecha'], dayfirst=True, errors='coerce').dt.to_period('M')
    valor = pd.to_numeric(resultado['Rendimiento BCE'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    claves = pd.DataFrame({'pais': pais, 'mes': mes, 'valor': valor})
    validas = claves.dropna(subset=['pais', 'mes'])
    if validas.duplicated(['pais', 'mes']).any():
        raise ValueError('Hay más de una observación para un mismo país y mes.')
    anterior = validas[['pais', 'mes', 'valor']].copy()
    anterior['mes'] = anterior['mes'] + 1
    anterior = anterior.set_index(['pais', 'mes'])['valor']
    previo = pd.Series(anterior.reindex(pd.MultiIndex.from_arrays([pais, mes])).to_numpy(), index=df.index)
    resultado[COLUMNA] = ((valor - previo) / previo.replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=FUENTE.with_name('euribor_datos_variacion.csv'))
    args = parser.parse_args()
    datos = pd.read_excel(FUENTE, sheet_name='Regresion', header=5)
    añadir_variacion(datos).to_csv(args.salida, index=False)
    print(f'Datos guardados en {args.salida}')


if __name__ == '__main__':
    main()