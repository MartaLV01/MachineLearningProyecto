import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

from sklearn.linear_model import LinearRegression
from machinelearningproyecto import particiones_temporales
from statsmodels.stats.stattools import durbin_watson

from machinelearningproyecto import modelos


features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Desempleo país (%)',
    'IPI variación interanual (%)',
    'Rendimiento cupon 0 mes anterior'
]
# ANTES: target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
target = 'Rendimiento cupon 0'
variables_pais = features + ['País ISO']

archivo = 'euribor_datos_variacion_1.csv'

X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv, folds = particiones_temporales.separacion_variables_entrenamiento(
    archivo, features, target, variables_pais, devolver_folds=True)


"""
Usando OLS para usar metricas y calcular coef ya que desde linear regression no se puede
HECHO PARA EL TRAIN
"""
modelos.model_OLS(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv)

"""

DIVISIÓN TEMPORAL TRAIN / VALIDACIÓN / TEST
  Conjunto  Filas     Inicio        Fin
     Train   1255 2010-07-31 2023-07-31
Validación    433 2023-08-31 2024-12-31
      Test    461 2025-01-31 2026-07-31

CINCO BLOQUES DE VALIDACIÓN ORIGINALES
         Inicio        Fin  Filas  Meses
Fold                                    
1    2010-07-31 2020-05-31    328     92
2    2020-06-30 2021-08-31    358     15
3    2021-09-30 2022-09-30    317     13
4    2022-10-31 2023-11-30    356     14
5    2023-12-31 2024-12-31    329     13
Tabla de coeficientes:
                                        b  Error estándar         t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                           
const                              0.0156          0.0257    0.6087   0.5429          -0.0347           0.0660                False
Títulos públicos (% PIB, mensual)  0.0005          0.0004    1.3144   0.1889          -0.0002           0.0012                False
Euríbor 12M (% anual)              0.0499          0.0058    8.5435   0.0000           0.0385           0.0614                 True
Desempleo país (%)                -0.0042          0.0032   -1.3083   0.1910          -0.0105           0.0021                False
IPI variación interanual (%)       0.0001          0.0007    0.1502   0.8806          -0.0013           0.0016                False
Rendimiento cupon 0 mes anterior   0.9920          0.0056  178.0944   0.0000           0.9811           1.0030                 True

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1255.0000
R² entrenamiento         0.9688
R² ajustado              0.9687
F                     7767.3710
p-valor de F             0.0000
Durbin-Watson            1.9963
RMSE entrenamiento       0.2994
MAE entrenamiento        0.1040
AIC                    546.2244
BIC                    577.0338

Tabla de validacion
                    Valor
R² validación    0.942697
RMSE validación  0.501513
MAE validación   0.242142

"""