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
const                              0.3995          0.1052   3.7983   0.0002           0.1932           0.6059                 True
Títulos públicos (% PIB, mensual)  0.0005          0.0017   0.2992   0.7648          -0.0028           0.0038                False
Euríbor 12M (% anual)              0.4471          0.0276  16.2214   0.0000           0.3930           0.5011                 True

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1296.0000
R² entrenamiento         0.1699
R² ajustado              0.1686
F                      132.2815
p-valor de F             0.0000
Durbin-Watson            1.8601
RMSE entrenamiento       1.5569
MAE entrenamiento        1.0130
AIC                   4831.2762
BIC                   4846.7773

                                           b  Error estándar          t  ...  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                ...                                                       
const                              0.399512        0.105182   3.798291  ...         0.193166         0.605859                 True
Títulos públicos (% PIB, mensual)  0.000508        0.001697   0.299205  ...        -0.002821         0.003837                False
Euríbor 12M (% anual)              0.447080        0.027561  16.221370  ...         0.393010         0.501149                 True

[3 rows x 7 columns]
                             Valor
Métrica                           
N.º de observaciones  1.296000e+03
R² entrenamiento      1.698570e-01
R² ajustado           1.685729e-01
F                     1.322815e+02
p-valor de F          5.399552e-53
Durbin-Watson         1.860108e+00
RMSE entrenamiento    1.556851e+00
MAE entrenamiento     1.013037e+00
AIC                   4.831276e+03
BIC                   4.846777e+03
                    Valor
R² validación   -0.215670
RMSE validación  2.307281
MAE validación   1.920788

"""