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
# model_random_tree = RandomForestRegressor(ramdomforest_search.best_params_) 
"""
Esto es usado para reentrenar el modelo, pero con best_estimator_ ya tenemos el mejor modelo calculado,
 por lo que no haria falta reentrenarlo con los mejore shiperparametros
En caso de querer reentrenarlo, no solo habria que pasarle best_params_ si no temabien los otros 3 hieperparamentros
 ya que no se le han pasado al grid y por lo que no estan incluidos en best_params_
model_random_tree = RandomForestRegressor(
    **ramdomforest_search.best_params_, n_estimators=300, random_state=42, n_jobs=-1
)
"""

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
$ uv run RandomTree_2Var.py

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
99                 1  1.813211  0.601223    0.746588  {'max_depth': 5, 'min_samples_leaf': 5, 'min_s...
101                2  1.813211  0.601223    0.746588  {'max_depth': 5, 'min_samples_leaf': 5, 'min_s...
103                3  1.814641  0.603609    0.751273  {'max_depth': 5, 'min_samples_leaf': 5, 'min_s...
98                 4  1.815019  0.603493    0.743601  {'max_depth': 5, 'min_samples_leaf': 5, 'min_s...
100                5  1.815019  0.603493    0.743601  {'max_depth': 5, 'min_samples_leaf': 5, 'min_s...
89                 6  1.815952  0.596260    0.737577  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
91                 7  1.816101  0.601378    0.761567  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
105                8  1.816139  0.607826    0.772920  {'max_depth': 5, 'min_samples_leaf': 5, 'min_s...
102                9  1.816422  0.605584    0.748172  {'max_depth': 5, 'min_samples_leaf': 5, 'min_s...
85                10  1.817132  0.585836    0.724492  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 500}
Mejor RMSE: 1.8132114773169155
Mejor estimador: RandomForestRegressor(max_depth=5, min_samples_leaf=5, n_estimators=500,
                      n_jobs=-1, random_state=42)

MÉTRICAS DE ENTRENAMIENTO
R2      0.619384
RMSE    1.054178
MAE     0.663525
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2      0.048711
RMSE    2.041027
MAE     1.764077
dtype: float64

tabla_metricas_rf
               N.º de observaciones        R²      RMSE       MAE
Conjunto                                                         
Entrenamiento                  1296  0.619384  1.054178  0.663525
Validación                      434  0.048711  2.041027  1.764077

tabla_importancias_rf
                            Variable  Importancia
0  Títulos públicos (% PIB, mensual)     0.515228
1              Euríbor 12M (% anual)     0.484772

"""


"""
Al hacerlo con (es lo actual de la file):
 target = 'Rendimiento cupon 0'
 archivo =  'euribor_datos_variacion_1.csv'
y añadir a features: 'Rendimiento cupon 0 mes anterior'

$ uv run RandomTree_2Var.py 
\MachineLearningProyecto\src\machinelearningproyecto\particiones_temporales.py:66: UserWarning: Parsing dates in %Y-%m-%d format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.
  datos['Fecha'] = pd.to_datetime(datos['Fecha'], dayfirst=True, errors='coerce')

DIVISIÓN TEMPORAL TRAIN / VALIDACIÓN / TEST
  Conjunto  Filas     Inicio        Fin
     Train   1255 2010-07-31 2023-07-31
Validación    433 2023-08-31 2024-12-31
      Test    461 2025-01-31 2026-07-31

CINCO BLOQUES DE VALIDACIÓN ORIGINALES
         Inicio        Fin  Filas  Meses
Fold                                    
1    2010-07-31 2020-05-31    328     92
2    2020-06-30 2021-08-31    358     15
3    2021-09-30 2022-09-30    317     13
4    2022-10-31 2023-11-30    356     14
5    2023-12-31 2024-12-31    329     13

MODELOS
      rank_test_score   CV_RMSE    CV_STD  TRAIN_RMSE                                             params
92                 1  0.608506  0.276305    0.211603  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
176                2  0.608853  0.276906    0.198637  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
218                3  0.608853  0.276906    0.198636  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
260                4  0.608853  0.276906    0.198636  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
8                  4  0.608853  0.276906    0.198636  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
93                 6  0.609194  0.277875    0.211744  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
134                7  0.609575  0.278530    0.198555  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
135                8  0.609616  0.279351    0.198833  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
261                9  0.610059  0.279940    0.198893  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
219               10  0.610059  0.279940    0.198893  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 30, 'n_estimators': 200}
Mejor RMSE: 0.6085060570760648
Mejor estimador: RandomForestRegressor(max_depth=5, min_samples_split=30, n_estimators=200,
                      n_jobs=-1, random_state=42)

MÉTRICAS DE ENTRENAMIENTO
R2      0.975388
RMSE    0.266065
MAE     0.108050
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2      0.922380
RMSE    0.583687
MAE     0.236343
dtype: float64

tabla_metricas_rf
               N.º de observaciones        R²      RMSE       MAE
Conjunto                                                         
Entrenamiento                  1255  0.975388  0.266065  0.108050
Validación                      433  0.922380  0.583687  0.236343

tabla_importancias_rf
                            Variable  Importancia
0   Rendimiento cupon 0 mes anterior     0.993067
1              Euríbor 12M (% anual)     0.006668
2  Títulos públicos (% PIB, mensual)     0.000266

"""