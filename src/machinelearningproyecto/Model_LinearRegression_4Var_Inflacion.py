import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from machinelearningproyecto import particiones_temporales
from statsmodels.stats.stattools import durbin_watson

from machinelearningproyecto import modelos


features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Inflación HICP interanual (%)',
    'IPP país (2021=100)',
]
target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
variables_pais = features + ['País ISO']


X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv = particiones_temporales.separacion_variables_entrenamiento(
    features, target, variables_pais)


"""
Usando OLS para usar metricas y calcular coef ya que desde linear regression no se puede
HECHO PARA EL TRAIN
"""
modelos.model_OLS(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv)

"""
Tabla de coeficientes:
                                        b  Error estándar        t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                          
const                              2.4108          0.3430   7.0287   0.0000           1.7379           3.0837                 True
Títulos públicos (% PIB, mensual)  0.0023          0.0017   1.3592   0.1743          -0.0010           0.0056                False
Euríbor 12M (% anual)              0.4718          0.0372  12.6898   0.0000           0.3989           0.5447                 True
Inflación HICP interanual (%)      0.1334          0.0167   7.9813   0.0000           0.1006           0.1662                 True
IPP país (2021=100)               -0.0254          0.0037  -6.9057   0.0000          -0.0326          -0.0182                 True

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1296.0000
R² entrenamiento         0.2102
R² ajustado              0.2077
F                       85.8751
p-valor de F             0.0000
Durbin-Watson            1.8643
RMSE entrenamiento       1.5186
MAE entrenamiento        0.9639
AIC                   4770.7836
BIC                   4796.6188

                                           b  Error estándar          t  ...  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                ...                                                       
const                              2.410804        0.342995   7.028692  ...         1.737916         3.083693                 True
Títulos públicos (% PIB, mensual)  0.002280        0.001677   1.359181  ...        -0.001011         0.005571                False
Euríbor 12M (% anual)              0.471804        0.037180  12.689790  ...         0.398864         0.544743                 True
Inflación HICP interanual (%)      0.133385        0.016712   7.981296  ...         0.100599         0.166171                 True
IPP país (2021=100)               -0.025425        0.003682  -6.905710  ...        -0.032648        -0.018202                 True

[5 rows x 7 columns]
                             Valor
Métrica                           
N.º de observaciones  1.296000e+03
R² entrenamiento      2.101562e-01
R² ajustado           2.077089e-01
F                     8.587509e+01
p-valor de F          9.965271e-65
Durbin-Watson         1.864316e+00
RMSE entrenamiento    1.518592e+00
MAE entrenamiento     9.639442e-01
AIC                   4.770784e+03
BIC                   4.796619e+03
                    Valor
R² validación   -0.729721
RMSE validación  2.752204
MAE validación   2.214354

"""