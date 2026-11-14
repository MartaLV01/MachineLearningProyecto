from machinelearningproyecto import particiones_temporales
from machinelearningproyecto import modelos


features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Desempleo país (%)',
    'IPI variación interanual (%)',
]
target = 'Variación mensual Rendimiento BCE (%)'
variables_pais = features + ['País ISO']


X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv = particiones_temporales.separacion_variables_entrenamiento(
    features, target, variables_pais)

modelos.model_DecisionTree(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv)
