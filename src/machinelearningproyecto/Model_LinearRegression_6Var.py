
import pandas as pd
import numpy as np
import statsmodels.api as sm

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
const                              2.4651          0.3455   7.1343   0.0000           1.7872           3.1430                 True
Inflación HICP interanual (%)      0.1306          0.0176   7.4153   0.0000           0.0960           0.1651                 True
IPP país (2021=100)               -0.0250          0.0037  -6.6995   0.0000          -0.0323          -0.0177                 True
Desempleo país (%)                -0.0148          0.0168  -0.8809   0.3785          -0.0478           0.0182                False
Títulos públicos (% PIB, mensual)  0.0027          0.0018   1.5257   0.1273          -0.0008           0.0062                False
Euríbor 12M (% anual)              0.4658          0.0376  12.3898   0.0000           0.3921           0.5396                 True
IPI variación interanual (%)      -0.0042          0.0038  -1.1002   0.2714          -0.0116           0.0033                False

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1296.0000
R² entrenamiento         0.2114
R² ajustado              0.2077
F                       57.5784
p-valor de F             0.0000
Durbin-Watson            1.8519
RMSE entrenamiento       1.5174
MAE entrenamiento        0.9620
AIC                   4772.7979
BIC                   4808.9672

                                           b  Error estándar          t  ...  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                ...                                                       
const                              2.465110        0.345531   7.134257  ...         1.787245         3.142976                 True
Inflación HICP interanual (%)      0.130596        0.017612   7.415344  ...         0.096045         0.165146                 True
IPP país (2021=100)               -0.025001        0.003732  -6.699481  ...        -0.032322        -0.017680                 True
Desempleo país (%)                -0.014809        0.016812  -0.880877  ...        -0.047790         0.018172                False
Títulos públicos (% PIB, mensual)  0.002707        0.001774   1.525663  ...        -0.000774         0.006187                False
Euríbor 12M (% anual)              0.465838        0.037599  12.389797  ...         0.392077         0.539599                 True
IPI variación interanual (%)      -0.004158        0.003779  -1.100201  ...        -0.011572         0.003256                False

[7 rows x 7 columns]
                             Valor
Métrica                           
N.º de observaciones  1.296000e+03
R² entrenamiento      2.113654e-01
R² ajustado           2.076945e-01
F                     5.757843e+01
p-valor de F          3.243272e-63
Durbin-Watson         1.851925e+00
RMSE entrenamiento    1.517429e+00
MAE entrenamiento     9.620262e-01
AIC                   4.772798e+03
BIC                   4.808967e+03
                    Valor
R² validación   -0.710000
RMSE validación  2.736469
MAE validación   2.204025

"""