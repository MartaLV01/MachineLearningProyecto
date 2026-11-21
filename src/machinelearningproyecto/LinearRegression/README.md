# Modelo OLS
En esta carpeta estamos viendo que resultados obtenermos si hacemos un modelo Lineal (OLS)

No estan regularizados

Cada file se runea con `uv run nombrefile.py` (estando en el directorio correspondiente)

## `Model_LinearRegression_2Var.py`
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
````

**Resultados**

```text
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
Tabla de coeficientes:
                                        b  Error estándar         t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                           
const                             -0.0027          0.0208   -0.1283   0.8979          -0.0434           0.0381                False
Títulos públicos (% PIB, mensual)  0.0003          0.0003    0.9030   0.3667          -0.0004           0.0010                False
Euríbor 12M (% anual)              0.0504          0.0058    8.7149   0.0000           0.0390           0.0617                 True
Rendimiento cupon 0 mes anterior   0.9925          0.0056  178.6322   0.0000           0.9816           1.0034                 True

Métricas del modelo:
                           Valor
Métrica                         
N.º de observaciones   1255.0000
R² entrenamiento          0.9688
R² ajustado               0.9687
F                     12947.5602
p-valor de F              0.0000
Durbin-Watson             1.9963
RMSE entrenamiento        0.2996
MAE entrenamiento         0.1029
AIC                     543.9874
BIC                     564.5270

Tabla de validacion
                    Valor
R² validación    0.942707
RMSE validación  0.501470
MAE validación   0.242386
```


## `Model_LinearRegression_4Var_Inflacion.py`
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

archivo = 'euribor_datos_variacion_1.csv'
```

**Resultados**

```text

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
Tabla de coeficientes:
                                        b  Error estándar         t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                           
const                             -0.0501          0.0698   -0.7179   0.4730          -0.1871           0.0868                False
Títulos públicos (% PIB, mensual)  0.0005          0.0003    1.3878   0.1655          -0.0002           0.0011                False
Euríbor 12M (% anual)              0.0361          0.0078    4.6406   0.0000           0.0208           0.0513                 True
Inflación HICP interanual (%)      0.0077          0.0034    2.2500   0.0246           0.0010           0.0144                 True
IPP país (2021=100)                0.0001          0.0007    0.1634   0.8702          -0.0013           0.0016                False
Rendimiento cupon 0 mes anterior   0.9912          0.0056  175.7650   0.0000           0.9801           1.0022                 True

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1255.0000
R² entrenamiento         0.9691
R² ajustado              0.9690
F                     7839.4664
p-valor de F             0.0000
Durbin-Watson            2.0101
RMSE entrenamiento       0.2980
MAE entrenamiento        0.1069
AIC                    534.9891
BIC                    565.7985

Tabla de validacion
                    Valor
R² validación    0.945743
RMSE validación  0.487999
MAE validación   0.204826
```

## `Model_LinearRegression_4Var_Produccion.py`
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

archivo = 'euribor_datos_variacion_1.csv'
```

**Resultados**

```text

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
Tabla de coeficientes:
                                        b  Error estándar         t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                           
const                              0.0156          0.0257    0.6087   0.5429          -0.0347           0.0660                False
Títulos públicos (% PIB, mensual)  0.0005          0.0004    1.3144   0.1889          -0.0002           0.0012                False
Euríbor 12M (% anual)              0.0499          0.0058    8.5435   0.0000           0.0385           0.0614                 True
Desempleo país (%)                -0.0042          0.0032   -1.3083   0.1910          -0.0105           0.0021                False
IPI variación interanual (%)       0.0001          0.0007    0.1502   0.8806          -0.0013           0.0016                False
Rendimiento cupon 0 mes anterior   0.9920          0.0056  178.0944   0.0000           0.9811           1.0030                 True

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1255.0000
R² entrenamiento         0.9688
R² ajustado              0.9687
F                     7767.3710
p-valor de F             0.0000
Durbin-Watson            1.9963
RMSE entrenamiento       0.2994
MAE entrenamiento        0.1040
AIC                    546.2244
BIC                    577.0338

Tabla de validacion
                    Valor
R² validación    0.942697
RMSE validación  0.501513
MAE validación   0.242142
```

## `Model_LinearRegression_6Var.py`
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

archivo = 'euribor_datos_variacion_1.csv'
```

**Resultados**

```text
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
Tabla de coeficientes:
                                        b  Error estándar         t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                           
const                             -0.0458          0.0705   -0.6500   0.5158          -0.1840           0.0924                False
Inflación HICP interanual (%)      0.0074          0.0036    2.0504   0.0405           0.0003           0.0144                 True
IPP país (2021=100)                0.0002          0.0008    0.2149   0.8299          -0.0013           0.0016                False
Desempleo país (%)                -0.0014          0.0034   -0.4068   0.6842          -0.0080           0.0052                False
Títulos públicos (% PIB, mensual)  0.0005          0.0004    1.4314   0.1526          -0.0002           0.0012                False
Euríbor 12M (% anual)              0.0358          0.0078    4.5642   0.0000           0.0204           0.0512                 True
IPI variación interanual (%)      -0.0002          0.0007   -0.2475   0.8045          -0.0017           0.0013                False
Rendimiento cupon 0 mes anterior   0.9911          0.0056  175.5231   0.0000           0.9800           1.0022                 True

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1255.0000
R² entrenamiento         0.9691
R² ajustado              0.9690
F                     5591.6952
p-valor de F             0.0000
Durbin-Watson            2.0091
RMSE entrenamiento       0.2980
MAE entrenamiento        0.1067
AIC                    538.7622
BIC                    579.8414

Tabla de validacion
                    Valor
R² validación    0.945608
RMSE validación  0.488608
MAE validación   0.206365
```

