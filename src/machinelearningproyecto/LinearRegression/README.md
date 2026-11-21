# Modelo OLS
En esta carpeta estamos viendo que resultados obtenermos si hacemos un modelo Lineal (OLS)

No estan regularizados

Cada file se runea con `uv run nombrefile.py` (estando en el directorio correspondiente)

**ESTAOS RESULTADOS NO ESTAN ACTUALIZADOS!!!**

**PARA VER LOS RESULTADOS IR A CADA UNA DE LAS FILES PARA VER SUS PARAMETROS**

**LOS RESULTADOS NO ESTAN ACTUALIZADOS TAMPOCO EN LAS FILES**

## `Model_LinearRegression_2Var.py`
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
Tabla de coeficientes:

                            b  Error estándar        t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                          
const                  0.3995          0.1052   3.7983   0.0002           0.1932           0.6059                 True
Títulos públicos (% PIB, mensual)  0.0005          0.0017   0.2992   0.7648          -0.0028           0.0038                False
Euríbor 12M (% anual)              0.4471          0.0276  16.2214   0.0000           0.3930           0.5011                 True

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1296.0000
R² entrenamiento         0.1699
R² ajustado              0.1686
F                      132.2815
p-valor de F             0.0000
Durbin-Watson            1.8601
RMSE entrenamiento       1.5569
MAE entrenamiento        1.0130
AIC                   4831.2762
BIC                   4846.7773

                                           b  Error estándar          t  ...  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                ...                                                       
const                              0.399512        0.105182   3.798291  ...         0.193166         0.605859                 True
Títulos públicos (% PIB, mensual)  0.000508        0.001697   0.299205  ...        -0.002821         0.003837                False
Euríbor 12M (% anual)              0.447080        0.027561  16.221370  ...         0.393010         0.501149                 True

[3 rows x 7 columns]
                             Valor
Métrica                           
N.º de observaciones  1.296000e+03
R² entrenamiento      1.698570e-01
R² ajustado           1.685729e-01
F                     1.322815e+02
p-valor de F          5.399552e-53
Durbin-Watson         1.860108e+00
RMSE entrenamiento    1.556851e+00
MAE entrenamiento     1.013037e+00
AIC                   4.831276e+03
BIC                   4.846777e+03
                    Valor
R² validación   -0.215670
RMSE validación  2.307281
MAE validación   1.920788

"""
```


## `Model_LinearRegression_4Var_Inflacion.py`
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
Tabla de coeficientes:
                                        b  Error estándar        t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                          
const                              2.4108          0.3430   7.0287   0.0000           1.7379           3.0837                 True
Títulos públicos (% PIB, mensual)  0.0023          0.0017   1.3592   0.1743          -0.0010           0.0056                False
Euríbor 12M (% anual)              0.4718          0.0372  12.6898   0.0000           0.3989           0.5447                 True
Inflación HICP interanual (%)      0.1334          0.0167   7.9813   0.0000           0.1006           0.1662                 True
IPP país (2021=100)               -0.0254          0.0037  -6.9057   0.0000          -0.0326          -0.0182                 True

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1296.0000
R² entrenamiento         0.2102
R² ajustado              0.2077
F                       85.8751
p-valor de F             0.0000
Durbin-Watson            1.8643
RMSE entrenamiento       1.5186
MAE entrenamiento        0.9639
AIC                   4770.7836
BIC                   4796.6188

                                           b  Error estándar          t  ...  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                ...                                                       
const                              2.410804        0.342995   7.028692  ...         1.737916         3.083693                 True
Títulos públicos (% PIB, mensual)  0.002280        0.001677   1.359181  ...        -0.001011         0.005571                False
Euríbor 12M (% anual)              0.471804        0.037180  12.689790  ...         0.398864         0.544743                 True
Inflación HICP interanual (%)      0.133385        0.016712   7.981296  ...         0.100599         0.166171                 True
IPP país (2021=100)               -0.025425        0.003682  -6.905710  ...        -0.032648        -0.018202                 True

[5 rows x 7 columns]
                             Valor
Métrica                           
N.º de observaciones  1.296000e+03
R² entrenamiento      2.101562e-01
R² ajustado           2.077089e-01
F                     8.587509e+01
p-valor de F          9.965271e-65
Durbin-Watson         1.864316e+00
RMSE entrenamiento    1.518592e+00
MAE entrenamiento     9.639442e-01
AIC                   4.770784e+03
BIC                   4.796619e+03
                    Valor
R² validación   -0.729721
RMSE validación  2.752204
MAE validación   2.214354

"""
```

## `Model_LinearRegression_4Var_Produccion.py`
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
Tabla de coeficientes:
                                        b  Error estándar        t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                          
const                              0.6271          0.1289   4.8634   0.0000           0.3741           0.8801                 True
Títulos públicos (% PIB, mensual)  0.0024          0.0018   1.3251   0.1854          -0.0012           0.0060                False
Euríbor 12M (% anual)              0.4363          0.0279  15.6456   0.0000           0.3816           0.4910                 True
Desempleo país (%)                -0.0495          0.0163  -3.0338   0.0025          -0.0814          -0.0175                 True
IPI variación interanual (%)      -0.0019          0.0038  -0.5083   0.6113          -0.0095           0.0056                False

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1296.0000
R² entrenamiento         0.1758
R² ajustado              0.1733
F                       68.8536
p-valor de F             0.0000
Durbin-Watson            1.8255
RMSE entrenamiento       1.5512
MAE entrenamiento        0.9977
AIC                   4825.9258
BIC                   4851.7610

                                           b  Error estándar          t  ...  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                ...                                                       
const                              0.627098        0.128943   4.863357  ...         0.374136         0.880059                 True
Títulos públicos (% PIB, mensual)  0.002400        0.001811   1.325105  ...        -0.001153         0.005953                False
Euríbor 12M (% anual)              0.436321        0.027888  15.645620  ...         0.381610         0.491031                 True
Desempleo país (%)                -0.049455        0.016301  -3.033843  ...        -0.081435        -0.017476                 True
IPI variación interanual (%)      -0.001949        0.003834  -0.508304  ...        -0.009470         0.005572                False

[5 rows x 7 columns]
                             Valor
Métrica                           
N.º de observaciones  1.296000e+03
R² entrenamiento      1.758247e-01
R² ajustado           1.732711e-01
F                     6.885360e+01
p-valor de F          7.070043e-53
Durbin-Watson         1.825459e+00
RMSE entrenamiento    1.551245e+00
MAE entrenamiento     9.976602e-01
AIC                   4.825926e+03
BIC                   4.851761e+03
                    Valor
R² validación   -0.204155
RMSE validación  2.296327
MAE validación   1.913687

"""
```

## `Model_LinearRegression_4Var_Produccion.py`
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
Tabla de coeficientes:
                                        b  Error estándar        t  p-valor  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                                                                          
const                              2.4651          0.3455   7.1343   0.0000           1.7872           3.1430                 True
Inflación HICP interanual (%)      0.1306          0.0176   7.4153   0.0000           0.0960           0.1651                 True
IPP país (2021=100)               -0.0250          0.0037  -6.6995   0.0000          -0.0323          -0.0177                 True
Desempleo país (%)                -0.0148          0.0168  -0.8809   0.3785          -0.0478           0.0182                False
Títulos públicos (% PIB, mensual)  0.0027          0.0018   1.5257   0.1273          -0.0008           0.0062                False
Euríbor 12M (% anual)              0.4658          0.0376  12.3898   0.0000           0.3921           0.5396                 True
IPI variación interanual (%)      -0.0042          0.0038  -1.1002   0.2714          -0.0116           0.0033                False

Métricas del modelo:
                          Valor
Métrica                        
N.º de observaciones  1296.0000
R² entrenamiento         0.2114
R² ajustado              0.2077
F                       57.5784
p-valor de F             0.0000
Durbin-Watson            1.8519
RMSE entrenamiento       1.5174
MAE entrenamiento        0.9620
AIC                   4772.7979
BIC                   4808.9672

                                           b  Error estándar          t  ...  IC 95% inferior  IC 95% superior  Significativo al 5%
Variable                                                                ...                                                       
const                              2.465110        0.345531   7.134257  ...         1.787245         3.142976                 True
Inflación HICP interanual (%)      0.130596        0.017612   7.415344  ...         0.096045         0.165146                 True
IPP país (2021=100)               -0.025001        0.003732  -6.699481  ...        -0.032322        -0.017680                 True
Desempleo país (%)                -0.014809        0.016812  -0.880877  ...        -0.047790         0.018172                False
Títulos públicos (% PIB, mensual)  0.002707        0.001774   1.525663  ...        -0.000774         0.006187                False
Euríbor 12M (% anual)              0.465838        0.037599  12.389797  ...         0.392077         0.539599                 True
IPI variación interanual (%)      -0.004158        0.003779  -1.100201  ...        -0.011572         0.003256                False

[7 rows x 7 columns]
                             Valor
Métrica                           
N.º de observaciones  1.296000e+03
R² entrenamiento      2.113654e-01
R² ajustado           2.076945e-01
F                     5.757843e+01
p-valor de F          3.243272e-63
Durbin-Watson         1.851925e+00
RMSE entrenamiento    1.517429e+00
MAE entrenamiento     9.620262e-01
AIC                   4.772798e+03
BIC                   4.808967e+03
                    Valor
R² validación   -0.710000
RMSE validación  2.736469
MAE validación   2.204025

"""
```

