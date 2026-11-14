
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
target = 'Variación mensual Rendimiento BCE (%)'
variables_pais = features + ['País ISO']


X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv = particiones_temporales.separacion_variables_entrenamiento(
    features, target, variables_pais)



"""
Usando OLS para usar metricas y calcular coef ya que desde linear regression no se puede
HECHO PARA EL TRAIN
"""
modelos.model_OLS(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv)

