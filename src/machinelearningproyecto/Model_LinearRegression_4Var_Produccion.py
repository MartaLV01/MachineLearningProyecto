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
    'Desempleo país (%)',
    'IPI variación interanual (%)',
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
const                              0.6271          0.1289   4.8634   0.0000           0.3741           0.8801                 True
Títulos públicos (% PIB, mensual)  0.0024          0.0018   1.3251   0.1854          -0.0012           0.0060                False
Euríbor 12M (% anual)              0.4363          0.0279  15.6456   0.0000           0.3816           0.4910                 True
Desempleo país (%)                -0.0495          0.0163  -3.0338   0.0025          -0.0814          -0.0175                 True
IPI variación interanual (%)      -0.0019          0.0038  -0.5083   0.6113          -0.0095           0.0056                False

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1296.0000
R² entrenamiento         0.1758
R² ajustado              0.1733
F                       68.8536
p-valor de F             0.0000
Durbin-Watson            1.8255
RMSE entrenamiento       1.5512
MAE entrenamiento        0.9977
AIC                   4825.9258
BIC                   4851.7610

                                           b  Error estándar          t  ...  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                ...                                                       
const                              0.627098        0.128943   4.863357  ...         0.374136         0.880059                 True
Títulos públicos (% PIB, mensual)  0.002400        0.001811   1.325105  ...        -0.001153         0.005953                False
Euríbor 12M (% anual)              0.436321        0.027888  15.645620  ...         0.381610         0.491031                 True
Desempleo país (%)                -0.049455        0.016301  -3.033843  ...        -0.081435        -0.017476                 True
IPI variación interanual (%)      -0.001949        0.003834  -0.508304  ...        -0.009470         0.005572                False

[5 rows x 7 columns]
                             Valor
Métrica                           
N.º de observaciones  1.296000e+03
R² entrenamiento      1.758247e-01
R² ajustado           1.732711e-01
F                     6.885360e+01
p-valor de F          7.070043e-53
Durbin-Watson         1.825459e+00
RMSE entrenamiento    1.551245e+00
MAE entrenamiento     9.976602e-01
AIC                   4.825926e+03
BIC                   4.851761e+03
                    Valor
R² validación   -0.204155
RMSE validación  2.296327
MAE validación   1.913687

"""