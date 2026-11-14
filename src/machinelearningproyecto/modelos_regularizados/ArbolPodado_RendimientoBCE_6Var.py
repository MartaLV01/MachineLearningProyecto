"""Árbol con poda de complejidad; ccp_alpha se fija inicialmente en 0,01."""
from machinelearningproyecto import particiones_temporales
from machinelearningproyecto import modelos


features = [
    'Inflación HICP interanual (%)',
    'IPP país (2021=100)',
    'Desempleo país (%)',
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'IPI variación interanual (%)',
]
target = 'Rendimiento BCE'
variables_pais = features + ['País ISO']


X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv = particiones_temporales.separacion_variables_entrenamiento(
    features, target, variables_pais)

modelos.model_DecisionTree(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv, ccp_alpha=0.01)
