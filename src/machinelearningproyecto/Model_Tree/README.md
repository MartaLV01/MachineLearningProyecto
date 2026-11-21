# Decision Tree
En esta carpeta estamos viendo que resultados obtenermos si arboles

No estan regularizados

Se ejecutan con el `uv run nombrefile.py` Estando en el directorio correspondiente

**Modelo antiguo**
**No esta actualizado con la variable 'Rendimiento cupon 0 mes anterior'. Ni corregido funcion separacion variables**

## `Model_Decision_Tree_2_Var.py`
**Variables usadas:**

```python
features = [
    'Títulos públicos (% PIB, mensual)',
    'Euríbor 12M (% anual)',
]
target = 'Rendimiento BCE'  # O: 'Bono soberano 10 años (% anual)'
variables_pais = features + ['País ISO']
````

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
```

## `Model_Decision_Tree_4_Var_Inflacion.py`
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
```

## `Model_Decision_Tree_4_Var_Produccion.py`
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
```

## `Model_Decision_Tree_6_Var.py`
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
     rank_test_score   CV_RMSE    CV_STD    TRAIN_RMSE                                             params
0                 1  1.704936  0.559028  8.190034e-17  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
42                1  1.704936  0.559028  8.190034e-17  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
37                3  1.731298  0.505575  4.556643e-01  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...
2                 3  1.731298  0.505575  4.556643e-01  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
44                3  1.731298  0.505575  4.556643e-01  {'max_depth': 50, 'min_samples_leaf': 1, 'min_...
31                6  1.733051  0.569061  5.742366e-01  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
25                7  1.734926  0.568335  7.220650e-01  {'max_depth': 10, 'min_samples_leaf': 1, 'min_...
6                 8  1.741517  0.446396  1.098166e+00  {'max_depth': None, 'min_samples_leaf': 1, 'mi...
34                8  1.741517  0.446396  1.098166e+00  {'max_depth': 15, 'min_samples_leaf': 1, 'min_...
41                8  1.741517  0.446396  1.098166e+00  {'max_depth': 20, 'min_samples_leaf': 1, 'min_...

MEJORES RESULTADOS
Mejores hiperparámetros: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Mejor RMSE: 1.704936348844668
Mejor estimador: DecisionTreeRegressor(random_state=42)


¡¡¡¡¡¡¡ESTOS RESULTADOS DAN OVERFITTING!!!!!!
El r^2 esta diciendo que predice perfecto (paso por todos los puentos del entrenamiento)
  cuando en validacion da negativo, es decir que es peor que si coguiero ael valor de la medio (valor medio)

MÉTRICAS DE ENTRENAMIENTO
R2      1.000000e+00
RMSE    9.568193e-17
MAE     1.356190e-17
dtype: float64

MÉTRICAS DE VALIDACIÓN
R2     -0.259363
RMSE    2.348378
MAE     1.859018
dtype: float64

"""
```