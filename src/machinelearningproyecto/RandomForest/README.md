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
    'Rendimiento cupon 0 mes anterior'
]
# ANTES: target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
target = 'Rendimiento cupon 0'
variables_pais = features + ['País ISO']

archivo =  'euribor_datos_variacion_1.csv'
```

**Resultados**

```text

$ uv run RandomTree_2Var.py 

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

```

## `Model_RandomTree_4Var_Inflacion.py`
**Variables usadas:**

```python
features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Inflación HICP interanual (%)',
    'IPP país (2021=100)',
    'Rendimiento cupon 0 mes anterior'
]
# ANTES: target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
target = 'Rendimiento cupon 0'
variables_pais = features + ['País ISO']

archivo =  'euribor_datos_variacion_1.csv'
```

**Resultados**

```text

$ uv run RandomTree_4Var_Inflacion.py 

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
85                 1  0.602929  0.299808    0.150683  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
84                 2  0.603385  0.299770    0.148218  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
126                3  0.605256  0.303231    0.109720  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
127                4  0.605355  0.302595    0.112337  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
168                5  0.606632  0.305319    0.107794  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
169                6  0.607724  0.305701    0.110402  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
1                  7  0.609002  0.307260    0.110069  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
253                7  0.609002  0.307260    0.110069  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
211                9  0.609189  0.307160    0.110053  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
210               10  0.609684  0.308661    0.107693  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
Mejor RMSE: 0.6029291223071269
Mejor estimador: RandomForestRegressor(max_depth=5, n_estimators=400, n_jobs=-1, random_state=42)

MÉTRICAS DE ENTRENAMIENTO
R2      0.987837
RMSE    0.187037
MAE     0.093247
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2      0.929907
RMSE    0.554664
MAE     0.274903
dtype: float64

tabla_metricas_rf
               N.º de observaciones        R²      RMSE       MAE
Conjunto                                                         
Entrenamiento                  1255  0.987837  0.187037  0.093247
Validación                      433  0.929907  0.554664  0.274903

tabla_importancias_rf
                            Variable  Importancia
0   Rendimiento cupon 0 mes anterior     0.909240
1                IPP país (2021=100)     0.071611
2              Euríbor 12M (% anual)     0.010854
3      Inflación HICP interanual (%)     0.007232
4  Títulos públicos (% PIB, mensual)     0.001064
```

## `Model_RandomTree_4Var_Produccion.py`
**Variables usadas:**

```python
features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
    'Desempleo país (%)',
    'IPI variación interanual (%)',
    'Rendimiento cupon 0 mes anterior'
]
# ANTES: target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
target = 'Rendimiento cupon 0'
variables_pais = features + ['País ISO']

archivo =  'euribor_datos_variacion_1.csv'
```

**Resultados**

```text

$ uv run RandomTree_4Var_Produccion.py

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
134                1  0.612496  0.287898    0.197597  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
218                2  0.613051  0.287896    0.197289  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
260                3  0.613051  0.287896    0.197289  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
8                  3  0.613051  0.287896    0.197289  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
176                5  0.613057  0.287897    0.197290  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
135                6  0.614548  0.290441    0.197548  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
177                7  0.614781  0.290588    0.197359  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
261                8  0.614810  0.290592    0.197357  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
219                9  0.614810  0.290592    0.197357  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
9                 10  0.614810  0.290592    0.197357  {'max_depth': None, 'min_samples_leaf': 1, 'mi...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 30, 'n_estimators': 200}
Mejor RMSE: 0.6124955274933629
Mejor estimador: RandomForestRegressor(max_depth=10, min_samples_split=30, n_estimators=200,
                      n_jobs=-1, random_state=42)

MÉTRICAS DE ENTRENAMIENTO
R2      0.979258
RMSE    0.244254
MAE     0.088855
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2      0.923471
RMSE    0.579569
MAE     0.232263
dtype: float64

tabla_metricas_rf
               N.º de observaciones        R²      RMSE       MAE
Conjunto                                                         
Entrenamiento                  1255  0.979258  0.244254  0.088855
Validación                      433  0.923471  0.579569  0.232263

tabla_importancias_rf
                            Variable  Importancia
0   Rendimiento cupon 0 mes anterior     0.988444
1              Euríbor 12M (% anual)     0.006787
2  Títulos públicos (% PIB, mensual)     0.002030
3                 Desempleo país (%)     0.001545
4       IPI variación interanual (%)     0.001195
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
    'Rendimiento cupon 0 mes anterior'
]
# ANTES: target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
target = 'Rendimiento cupon 0'
variables_pais = features + ['País ISO']

archivo =  'euribor_datos_variacion_1.csv'
```

**Resultados**

```text

$ uv run RandomTree_6Var.py 
C:\Users\marta\MachineLearning\MachineLearningProyecto\src\machinelearningproyecto\particiones_temporales.py:66: UserWarning: Parsing dates in %Y-%m-%d format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.
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
168                1  0.608379  0.303272    0.108112  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
84                 2  0.610918  0.302864    0.147756  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
126                3  0.612121  0.306212    0.109905  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
169                4  0.612296  0.303839    0.111393  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
210                5  0.612906  0.301628    0.107754  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
85                 6  0.614014  0.303866    0.150069  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...
252                7  0.614024  0.303503    0.107912  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
0                  7  0.614024  0.303503    0.107912  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
253                9  0.616874  0.304877    0.111081  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
1                 10  0.616874  0.304877    0.111081  {'max_depth': None, 'min_samples_leaf': 1, 'mi...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Mejor RMSE: 0.6083792457101116
Mejor estimador: RandomForestRegressor(max_depth=15, n_estimators=200, n_jobs=-1,
                      random_state=42)

MÉTRICAS DE ENTRENAMIENTO
R2      0.996057
RMSE    0.106492
MAE     0.037691
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2      0.931302
RMSE    0.549118
MAE     0.228750
dtype: float64

tabla_metricas_rf
               N.º de observaciones        R²      RMSE       MAE
Conjunto                                                         
Entrenamiento                  1255  0.996057  0.106492  0.037691
Validación                      433  0.931302  0.549118  0.228750

tabla_importancias_rf
                            Variable  Importancia
0   Rendimiento cupon 0 mes anterior     0.902566
1                IPP país (2021=100)     0.071300
2              Euríbor 12M (% anual)     0.010255
3      Inflación HICP interanual (%)     0.006451
4       IPI variación interanual (%)     0.004089
5                 Desempleo país (%)     0.003036
6  Títulos públicos (% PIB, mensual)     0.002303

```