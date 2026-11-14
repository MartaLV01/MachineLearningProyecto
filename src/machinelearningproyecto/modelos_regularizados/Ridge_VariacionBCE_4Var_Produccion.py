"""Regresión Ridge para la tasa de variación mensual del rendimiento BCE."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from machinelearningproyecto import particiones_temporales

features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Desempleo país (%)',
    'IPI variación interanual (%)',
]
target = 'Variación mensual Rendimiento BCE (%)'
variables_pais = features + ['País ISO']


def main():
    X_train, y_train, X_val, y_val, _, _, _, _ = (
        particiones_temporales.separacion_variables_entrenamiento(features, target, variables_pais)
    )
    modelo = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    modelo.fit(X_train, y_train)
    for nombre, X, y in [('Entrenamiento', X_train, y_train), ('Validación', X_val, y_val)]:
        pred = modelo.predict(X)
        print(f'{nombre}: R²={r2_score(y, pred):.4f}, '
              f'RMSE={np.sqrt(mean_squared_error(y, pred)):.4f}, '
              f'MAE={mean_absolute_error(y, pred):.4f}')


if __name__ == '__main__':
    main()
