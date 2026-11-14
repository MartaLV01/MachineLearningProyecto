# Random Forest
En esta carpeta estamos viendo que resultados obtenermos si hacemos un Random Forest

No estan regularizados

Se ejecutan con el `uv run nombrefile.py` Estando en el directorio correspondiente


## `Model_RandomTree_2Var.py`
**Variables usadas:**

```python
features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
]
target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
variables_pais = features + ['País ISO']
```

**Resultados**

```text

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
```

## `Model_RandomTree_4Var_Inflacion.py`
**Variables usadas:**

```python
features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Inflación HICP interanual (%)',
    'IPP país (2021=100)',
]
target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
variables_pais = features + ['País ISO']
```

**Resultados**

```text

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
169                1  1.669915  0.743156    0.203383  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
168                2  1.670192  0.741398    0.202254  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
210                3  1.671815  0.743180    0.200964  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
0                  4  1.671864  0.743368    0.201209  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
252                4  1.671864  0.743368    0.201209  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
253                6  1.672177  0.745543    0.202225  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
1                  7  1.672177  0.745543    0.202225  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
127                8  1.672929  0.746494    0.244895  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
211                9  1.672931  0.746192    0.202014  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
126               10  1.673030  0.744673    0.244171  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
Mejor RMSE: 1.669914652857087
Mejor estimador: RandomForestRegressor(max_depth=15, n_estimators=500, n_jobs=-1,
                      random_state=42)

MÉTRICAS DE ENTRENAMIENTO
R2      0.975404
RMSE    0.267982
MAE     0.144845
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2     -0.158661
RMSE    2.252531
MAE     1.879214
dtype: float64

tabla_metricas_rf
               N.º de observaciones        R²      RMSE       MAE
Conjunto                                                         
Entrenamiento                  1296  0.975404  0.267982  0.144845
Validación                      434 -0.158661  2.252531  1.879214

tabla_importancias_rf
                            Variable  Importancia
0      Inflación HICP interanual (%)     0.294267
1                IPP país (2021=100)     0.285016
2              Euríbor 12M (% anual)     0.226722
3  Títulos públicos (% PIB, mensual)     0.193995

"""
```

## `Model_RandomTree_4Var_Produccion.py`
**Variables usadas:**

```python
features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Desempleo país (%)',
    'IPI variación interanual (%)',
]
target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
variables_pais = features + ['País ISO']
```

**Resultados**

```text

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
```

## `Model_RandomTree_6Var.py`
**Variables usadas:**

```python
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
```

**Resultados**

```text

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
```