# Este acrhivo es para usar los arboles con 2 varibales:    'Títulos públicos (% PIB, mensual)', 'Euríbor 12M (% anual)'

import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from machinelearningproyecto import particiones_temporales
from machinelearningproyecto import modelos

# Ajustado por pais

features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Desempleo país (%)',
    'IPI variación interanual (%)',
]
target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
variables_pais = features + ['País ISO']


X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv, folds = particiones_temporales.separacion_variables_entrenamiento(
    features, target, variables_pais)

# DecisionTreeRegressor()

tree_param_grid = {
    "max_depth": [None, 3, 5, 10, 15, 20, 50],
    "min_samples_leaf": [1],
    # "preprocessing__high_card__min_frequency": [20],
    "min_samples_split": [2,10,15,20,30,50,100]
}

tree_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=tree_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=folds, # son los folds que hemos calculado anteriorimente teneindo en caunta el data leakage por los valores temporales
    n_jobs=1,
    return_train_score=True,
)
 
#tree_search.fit(X_train, y_train)
tree_search.fit(X_cv, y_cv)
print("\nMODELOS\n", modelos.search_results_table(tree_search))

print("\nMEJORES RESULTADOS")
print("Mejores hiperparámetros:", tree_search.best_params_)
print("Mejor RMSE:", -tree_search.best_score_)
print("Mejor estimador:", tree_search.best_estimator_)

# UNA VEZ VISTOS LOS MEJORES HIPERPARAMTROS ENTRENMAOS NUESTRO MODELO
model_tree = DecisionTreeRegressor(max_depth= 5, min_samples_leaf= 1, min_samples_split= 10, random_state=42)
model_tree.fit(X_train, y_train)

# Predicciones sobre entrenamiento y validación
y_train_pred = model_tree.predict(X_train)
y_val_pred = model_tree.predict(X_val)

# Métricas de entrenamiento
metricas_train = {
    "R2": r2_score(y_train, y_train_pred),
    "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
    "MAE": mean_absolute_error(y_train, y_train_pred)
}

# Métricas de validación
metricas_val = {
    "R2": r2_score(y_val, y_val_pred),
    "RMSE": np.sqrt(mean_squared_error(y_val, y_val_pred)),
    "MAE": mean_absolute_error(y_val, y_val_pred)
}

print("\nMÉTRICAS DE ENTRENAMIENTO")
print(pd.Series(metricas_train))

print("\nMÉTRICAS DE VALIDACIÓN")
print(pd.Series(metricas_val))


"""
DIVISIÓN TEMPORAL TRAIN / VALIDACIÓN / TEST
  Conjunto  Filas     Inicio        Fin
     Train   1296 2010-06-30 2023-07-31
Validación    434 2023-08-31 2024-12-31
      Test    463 2025-01-31 2026-07-31

CINCO BLOQUES DE VALIDACIÓN ORIGINALES
         Inicio        Fin  Filas  Meses
Fold                                    
1    2010-06-30 2020-04-30    334     94
2    2020-05-31 2021-07-31    362     15
3    2021-08-31 2022-09-30    346     14
4    2022-10-31 2023-10-31    332     13
5    2023-11-30 2024-12-31    356     14

MODELOS
     rank_test_score   CV_RMSE    CV_STD  TRAIN_RMSE                                             params
15                1  1.927401  0.597392    1.117896  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
16                1  1.927401  0.597392    1.117896  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
14                3  1.928375  0.598536    1.110013  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
18                4  1.944730  0.607760    1.126887  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
17                5  1.945972  0.608308    1.123353  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
25                6  1.957318  0.533236    0.822480  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
13                7  1.959657  0.490413    1.504864  {'max_depth': 3, 'min_samples_leaf': 1, 'min_s...
12                8  1.960387  0.486517    1.489217  {'max_depth': 3, 'min_samples_leaf': 1, 'min_s...
8                 8  1.960387  0.486517    1.489217  {'max_depth': 3, 'min_samples_leaf': 1, 'min_s...
9                 8  1.960387  0.486517    1.489217  {'max_depth': 3, 'min_samples_leaf': 1, 'min_s...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 10}
Mejor RMSE: 1.9274009057400658
Mejor estimador: DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)

MÉTRICAS DE ENTRENAMIENTO
R2      0.541057
RMSE    1.157576
MAE     0.724332
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2     -0.304398
RMSE    2.389998
MAE     2.088503
dtype: float64

"""