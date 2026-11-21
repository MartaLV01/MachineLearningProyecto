import numpy as np
import pandas as pd

# from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from machinelearningproyecto import particiones_temporales
from machinelearningproyecto import modelos

features = [
    'Inflación HICP interanual (%)',
    'IPP país (2021=100)',
    'Desempleo país (%)',
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'IPI variación interanual (%)',
    'Rendimiento cupon 0 mes anterior'
]
# ANTES: target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
target = 'Rendimiento cupon 0'
variables_pais = features + ['País ISO']

archivo =  'euribor_datos_variacion_1.csv'

X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv, folds = particiones_temporales.separacion_variables_entrenamiento(
    archivo, features, target, variables_pais, devolver_folds=True)

randomforest_param_grid = {
    "n_estimators": [200, 400], # Número de árboles.
    "max_depth": [None, 3, 5, 10, 15, 20, 50], # Límite de profundidad.
    "min_samples_leaf": [1, 5, 10],
    # "preprocessing__high_card__min_frequency": [20],
    "min_samples_split": [2,10,15,20,30,50,100]
}

# Crear el modelo con una configuración inicial reproducible.
modelo_rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
)

ramdomforest_search = GridSearchCV(
    estimator=modelo_rf,
    param_grid=randomforest_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=folds, # son los folds que hemos calculado anteriorimente teneindo en caunta el data leakage por los valores temporales
    n_jobs=1,
    return_train_score=True,
)

# HACEMOS ESTO PARA SACAR LOS MEJORES HIPERPARAMETROS DEL MODELO RANDOM FOREST
ramdomforest_search.fit(X_cv, y_cv)
print("\nMODELOS\n", modelos.search_results_table(ramdomforest_search))

print("\nMEJORES RESULTADOS")
print("Mejores hiperparámetros:", ramdomforest_search.best_params_)
print("Mejor RMSE:", -ramdomforest_search.best_score_)
print("Mejor estimador:", ramdomforest_search.best_estimator_)

# UNA VEZ VISTOS LOS MEJORES HIPERPARAMTROS ENTRENMAOS NUESTRO MODELO
# ESTAMOS DICIENDO QUE NUESTRO MODELOS SEA QUE HEMOS HAYADO QUE ES MEJOR
# LO ENTRENAMOS CON LOS DATOS DE ENTRENAMIENTO
model_random_tree = ramdomforest_search.best_estimator_
model_random_tree.fit(X_train, y_train)

# Generar predicciones.
train_pred_rf = model_random_tree.predict(X_train)
val_pred_rf = model_random_tree.predict(X_val)

# Métricas de entrenamiento
metricas_train = {
    "R2": r2_score(y_train, train_pred_rf),
    "RMSE": np.sqrt(mean_squared_error(y_train, train_pred_rf)),
    "MAE": mean_absolute_error(y_train, train_pred_rf)
}

# Métricas de validación
metricas_val = {
    "R2": r2_score(y_val, val_pred_rf),
    "RMSE": np.sqrt(mean_squared_error(y_val, val_pred_rf)),
    "MAE": mean_absolute_error(y_val, val_pred_rf)
}

print("\nMÉTRICAS DE ENTRENAMIENTO")
print(pd.Series(metricas_train))

print("\nMÉTRICAS DE VALIDACIÓN")
print(pd.Series(metricas_val))


# 4. Tabla de métricas para cada conjunto.
filas = []

for conjunto, y_real, y_pred in [
    ("Entrenamiento", y_train, train_pred_rf),
    ("Validación", y_val, val_pred_rf),
]:
    filas.append({
        "Conjunto": conjunto,
        "N.º de observaciones": len(y_real),
        "R²": r2_score(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "MAE": mean_absolute_error(y_real, y_pred),
    })

tabla_metricas_rf = pd.DataFrame(filas).set_index("Conjunto")
print("\ntabla_metricas_rf")
print(tabla_metricas_rf)

# 5. Importancia de las variables basada en reducción de impureza.
tabla_importancias_rf = pd.DataFrame({
    "Variable": X_train.columns,
    "Importancia": model_random_tree.feature_importances_,
}).sort_values(
    "Importancia", ascending=False
).reset_index(drop=True)

print("\ntabla_importancias_rf")
print(tabla_importancias_rf)

"""
$ uv run RandomTree_6Var.py

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
1                  1  1.687344  0.686873    0.185447  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
253                1  1.687344  0.686873    0.185447  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
252                3  1.687388  0.685299    0.183832  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
0                  3  1.687388  0.685299    0.183832  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
169                5  1.689515  0.691449    0.187022  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
126                6  1.690250  0.686876    0.229165  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
168                7  1.691554  0.693989    0.185065  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
211                8  1.691902  0.693073    0.185646  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
127                9  1.694917  0.694706    0.230436  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
210               10  1.696368  0.697013    0.183485  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
Mejor RMSE: 1.6873436291567627
Mejor estimador: RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)

MÉTRICAS DE ENTRENAMIENTO
R2      0.979083
RMSE    0.247125
MAE     0.123716
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2      0.031404
RMSE    2.059510
MAE     1.734186
dtype: float64

tabla_metricas_rf
               N.º de observaciones        R²      RMSE       MAE
Conjunto                                                         
Entrenamiento                  1296  0.979083  0.247125  0.123716
Validación                      434  0.031404  2.059510  1.734186

tabla_importancias_rf
                            Variable  Importancia
0                IPP país (2021=100)     0.251826
1      Inflación HICP interanual (%)     0.241544
2              Euríbor 12M (% anual)     0.213996
3                 Desempleo país (%)     0.137395
4  Títulos públicos (% PIB, mensual)     0.122197
5       IPI variación interanual (%)     0.033042


"""