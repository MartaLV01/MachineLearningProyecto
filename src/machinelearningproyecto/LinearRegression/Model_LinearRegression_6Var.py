
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

from machinelearningproyecto import particiones_temporales
from statsmodels.stats.stattools import durbin_watson

from machinelearningproyecto import modelos


features = [
    'Inflación HICP interanual (%)',
    'IPP país (2021=100)',
    'Desempleo país (%)',
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
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
const                             -0.0458          0.0705   -0.6500   0.5158          -0.1840           0.0924                False
Inflación HICP interanual (%)      0.0074          0.0036    2.0504   0.0405           0.0003           0.0144                 True
IPP país (2021=100)                0.0002          0.0008    0.2149   0.8299          -0.0013           0.0016                False
Desempleo país (%)                -0.0014          0.0034   -0.4068   0.6842          -0.0080           0.0052                False
Títulos públicos (% PIB, mensual)  0.0005          0.0004    1.4314   0.1526          -0.0002           0.0012                False
Euríbor 12M (% anual)              0.0358          0.0078    4.5642   0.0000           0.0204           0.0512                 True
IPI variación interanual (%)      -0.0002          0.0007   -0.2475   0.8045          -0.0017           0.0013                False
Rendimiento cupon 0 mes anterior   0.9911          0.0056  175.5231   0.0000           0.9800           1.0022                 True

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1255.0000
R² entrenamiento         0.9691
R² ajustado              0.9690
F                     5591.6952
p-valor de F             0.0000
Durbin-Watson            2.0091
RMSE entrenamiento       0.2980
MAE entrenamiento        0.1067
AIC                    538.7622
BIC                    579.8414

Tabla de validacion
                    Valor
R² validación    0.945608
RMSE validación  0.488608
MAE validación   0.206365

"""