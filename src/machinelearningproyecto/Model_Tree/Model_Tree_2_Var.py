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
model_tree = DecisionTreeRegressor(max_depth= 15, min_samples_leaf= 1, min_samples_split= 30, random_state=42)
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
32                1  1.885238  0.555614    1.004210  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
4                 2  1.885238  0.555614    1.004024  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
39                2  1.885238  0.555614    1.004024  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
46                2  1.885238  0.555614    1.004024  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
33                5  1.889930  0.579754    1.152857  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
5                 5  1.889930  0.579754    1.152857  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
26                5  1.889930  0.579754    1.155084  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
40                5  1.889930  0.579754    1.152857  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
47                5  1.889930  0.579754    1.152857  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
8                10  1.898187  0.579605    1.561580  {'max_depth': 3, 'min_samples_leaf': 1, 'min_s...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 30}
Mejor RMSE: 1.8852383150065168
Mejor estimador: DecisionTreeRegressor(max_depth=15, min_samples_split=30, random_state=42)

"""

"""
CON LOS MEJORES DATOS DE ENTRENAMIENTO

MÉTRICAS DE ENTRENAMIENTO
R2      0.707532
RMSE    0.924080
MAE     0.497636
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2     -0.211372
RMSE    2.303198
MAE     1.915244
dtype: float64

"""


"""
INICIO HECHO "POR FUERZA BRUTA"

resultado_2 = modelos.model_DecisionTree(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv,
                            max_depth=5, min_samples_leaf=5, min_samples_split=2)
print("\nUSANDO: max_depth=5, min_samples_leaf=5, min_samples_split=2")
print("\nMODELO:\n ", resultado_2['modelo'])
print("\nTABLA DE MÉTRICAS:\n", resultado_2['metricas'])
print("\nPREDICCIONES:\n ", resultado_2['predicciones'])
print("\nRESIDUOS:\n ", resultado_2['residuos'])
print("\nIMPORTANCIAS:\n ", resultado_2['importancias'])

resultado_10 = modelos.model_DecisionTree(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv,
                            max_depth=5, min_samples_leaf=5, min_samples_split=10)
print("\nUSANDO: max_depth=5, min_samples_leaf=5, min_samples_split=10")
print("\nMODELO:\n ", resultado_10['modelo'])
print("\nTABLA DE MÉTRICAS:\n", resultado_10['metricas'])
print("\nPREDICCIONES:\n ", resultado_10['predicciones'])
print("\nRESIDUOS:\n ", resultado_10['residuos'])
print("\nIMPORTANCIAS:\n ", resultado_10['importancias'])

resultado_15 = modelos.model_DecisionTree(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv,
                            max_depth=5, min_samples_leaf=5, min_samples_split=15)
print("\nUSANDO: max_depth=5, min_samples_leaf=5, min_samples_split=15")
print("\nMODELO:\n ", resultado_15['modelo'])
print("\nTABLA DE MÉTRICAS:\n", resultado_15['metricas'])
print("\nPREDICCIONES:\n ", resultado_15['predicciones'])
print("\nRESIDUOS:\n ", resultado_15['residuos'])
print("\nIMPORTANCIAS:\n ", resultado_15['importancias'])

resultado_20 = modelos.model_DecisionTree(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv,
                            max_depth=5, min_samples_leaf=5, min_samples_split=20)
print("\nUSANDO: max_depth=5, min_samples_leaf=5, min_samples_split=20")
print("\nMODELO:\n ", resultado_20['modelo'])
print("\nTABLA DE MÉTRICAS:\n", resultado_20['metricas'])
print("\nPREDICCIONES:\n ", resultado_20['predicciones'])
print("\nRESIDUOS:\n ", resultado_20['residuos'])
print("\nIMPORTANCIAS:\n ", resultado_20['importancias'])

resultado_30 = modelos.model_DecisionTree(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv,
                            max_depth=5, min_samples_leaf=5, min_samples_split=30)
print("\nUSANDO: max_depth=5, min_samples_leaf=5, min_samples_split=30")
print("\nMODELO:\n ", resultado_30['modelo'])
print("\nTABLA DE MÉTRICAS:\n", resultado_30['metricas'])
print("\nPREDICCIONES:\n ", resultado_30['predicciones'])
print("\nRESIDUOS:\n ", resultado_30['residuos'])
print("\nIMPORTANCIAS:\n ", resultado_30['importancias'])

# USANDO:
#    max_depth=5, min_samples_leaf=5, min_samples_split=2

"""
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

Métricas Decision Tree:
            N.º de observaciones      R²    RMSE     MAE
Conjunto                                                
Train                       1296  0.5213  1.1823  0.7164
Validación                   434 -0.1321  2.2265  1.8629

Importancias por reducción de impureza (no son coeficientes ni p-valores):
Variable
Títulos públicos (% PIB, mensual)    0.561
Euríbor 12M (% anual)                0.439
Name: Importancia, dtype: float64

USANDO: max_depth=5, min_samples_leaf=5, min_samples_split=2

MODELO:
  DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=42)

TABLA DE MÉTRICAS:
             N.º de observaciones        R²      RMSE       MAE
Conjunto                                                      
Train                       1296  0.521284  1.182251  0.716377
Validación                   434 -0.132073  2.226536  1.862869

PREDICCIONES:
  {'Train': 0       4.200000
1       0.399113
2       1.585653
3       4.200000
4       0.399113
          ...   
1291    1.585653
1292    6.026420
1293    0.703048
1294    5.282137
1295    1.645895
Name: Predicción, Length: 1296, dtype: float64, 'Validación': 1296    1.585653
1297    1.585653
1298    0.703048
1299    1.645895
1300    3.857178
          ...   
1725    6.026420
1726    5.282137
1727    0.399113
1728    0.703048
1729    5.282137
Name: Predicción, Length: 434, dtype: float64}

ERRORES:
  {'Train': 0       0.000000
1       3.701387
2      -1.540353
3       0.000000
4       3.701387
          ...   
1291    0.933747
1292   -0.385920
1293    2.856352
1294   -2.362237
1295   -1.647995
Name: Residuo, Length: 1296, dtype: float64, 'Validación': 1296    1.821547
1297    1.472147
1298   -0.717948
1299   -1.551095
1300    1.596922
          ...   
1725   -1.426120
1726   -2.161237
1727    5.299587
1728    2.647152
1729   -2.342637
Name: Residuo, Length: 434, dtype: float64}

IMPORTANCIAS:
  Variable
Títulos públicos (% PIB, mensual)    0.560964
Euríbor 12M (% anual)                0.439036
Name: Importancia, dtype: float64
"""


"""
print("\nUSANDO: max_depth=5, min_samples_leaf=5, min_samples_split=10")


 uv run python -m Model_Tree_2_Var

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

Métricas Decision Tree:
            N.º de observaciones      R²    RMSE     MAE
Conjunto                                                
Train                       1296  0.5213  1.1823  0.7164
Validación                   434 -0.1321  2.2265  1.8629

Importancias por reducción de impureza (no son coeficientes ni p-valores):
Variable
Títulos públicos (% PIB, mensual)    0.561
Euríbor 12M (% anual)                0.439
Name: Importancia, dtype: float64

USANDO: max_depth=5, min_samples_leaf=5, min_samples_split=2

MODELO:
  DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, min_samples_split=10,
                      random_state=42)

TABLA DE MÉTRICAS:
             N.º de observaciones        R²      RMSE       MAE
Conjunto                                                      
Train                       1296  0.521284  1.182251  0.716377
Validación                   434 -0.132073  2.226536  1.862869

PREDICCIONES:
  {'Train': 0       4.200000
1       0.399113
2       1.585653
3       4.200000
4       0.399113
          ...   
1291    1.585653
1292    6.026420
1293    0.703048
1294    5.282137
1295    1.645895
Name: Predicción, Length: 1296, dtype: float64, 'Validación': 1296    1.585653
1297    1.585653
1298    0.703048
1299    1.645895
1300    3.857178
          ...   
1725    6.026420
1726    5.282137
1727    0.399113
1728    0.703048
1729    5.282137
Name: Predicción, Length: 434, dtype: float64}

RESIDUOS:
  {'Train': 0       0.000000
1       3.701387
2      -1.540353
3       0.000000
4       3.701387
          ...   
1291    0.933747
1292   -0.385920
1293    2.856352
1294   -2.362237
1295   -1.647995
Name: Residuo, Length: 1296, dtype: float64, 'Validación': 1296    1.821547
1297    1.472147
1298   -0.717948
1299   -1.551095
1300    1.596922
          ...   
1725   -1.426120
1726   -2.161237
1727    5.299587
1728    2.647152
1729   -2.342637
Name: Residuo, Length: 434, dtype: float64}

IMPORTANCIAS:
  Variable
Títulos públicos (% PIB, mensual)    0.560964
Euríbor 12M (% anual)                0.439036
Name: Importancia, dtype: float64

"""