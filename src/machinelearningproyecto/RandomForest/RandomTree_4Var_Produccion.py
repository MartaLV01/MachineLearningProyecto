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
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Desempleo país (%)',
    'IPI variación interanual (%)',
]
target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
variables_pais = features + ['País ISO']

X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv, folds = particiones_temporales.separacion_variables_entrenamiento(
    features, target, variables_pais, devolver_folds=True)

randomforest_param_grid = {
    "n_estimators": [300, 500], # Número de árboles.
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
210                1  1.775553  0.620568    0.197228  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
0                  2  1.775625  0.620451    0.197391  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
252                2  1.775625  0.620451    0.197391  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
168                4  1.776198  0.619725    0.197672  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
169                5  1.778710  0.619159    0.199629  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
211                6  1.779001  0.619973    0.199004  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
253                7  1.779699  0.621255    0.198912  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
1                  7  1.779699  0.621255    0.198912  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
126                9  1.784598  0.624809    0.242444  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
127               10  1.789288  0.624745    0.245581  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
Mejor RMSE: 1.7755527776118665
Mejor estimador: RandomForestRegressor(max_depth=20, n_estimators=300, n_jobs=-1,
                      random_state=42)

MÉTRICAS DE ENTRENAMIENTO
R2      0.978967
RMSE    0.247809
MAE     0.121711
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2      0.124918
RMSE    1.957568
MAE     1.589701
dtype: float64

tabla_metricas_rf
               N.º de observaciones        R²      RMSE       MAE
Conjunto                                                         
Entrenamiento                  1296  0.978967  0.247809  0.121711
Validación                      434  0.124918  1.957568  1.589701

tabla_importancias_rf
                            Variable  Importancia
0                 Desempleo país (%)     0.347838
1              Euríbor 12M (% anual)     0.299367
2  Títulos públicos (% PIB, mensual)     0.261476
3       IPI variación interanual (%)     0.091320

"""