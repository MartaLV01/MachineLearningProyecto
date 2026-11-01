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
    'Inflación HICP interanual (%)',
    'IPP país (2021=100)',
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
model_tree = DecisionTreeRegressor(max_depth= 15, min_samples_leaf= 1, min_samples_split= 15, random_state=42)
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
30                1  1.772562  0.622742    0.541078  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
31                2  1.774691  0.627134    0.665172  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
23                3  1.776318  0.618761    0.625689  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
2                 4  1.777994  0.626095    0.539842  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
44                4  1.777994  0.626095    0.539842  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
37                4  1.777994  0.626095    0.539842  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
3                 7  1.787056  0.640541    0.664482  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
45                7  1.787056  0.640541    0.664482  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
38                7  1.787056  0.640541    0.664482  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
32               10  1.793446  0.633998    0.774030  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 15}
Mejor RMSE: 1.7725620714168613
Mejor estimador: DecisionTreeRegressor(max_depth=15, min_samples_split=15, random_state=42)


ESTAS METRICAS SON CON LOS VALORES HIPERPARAMETROS QUE MEJOR DAN
MÉTRICAS DE ENTRENAMIENTO
R2      0.873910
RMSE    0.606752
MAE     0.276552
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2     -0.405782
RMSE    2.481142
MAE     1.991638
dtype: float64

"""