# Resultados de la corrección AR(1)

Se estima un phi común a todos los países por cada modelo base, exclusivamente con pares de errores consecutivos de entrenamiento. El retardo pertenece siempre al mismo país y al mes inmediatamente anterior. No se reestiman los modelos base ni se usan validación o test para ajustar phi.

**Escenario: predicción a un mes con el resultado del mes anterior ya observado.** En validación y test se actualiza el error anterior observado, pero no los parámetros. Esto no equivale a pronosticar todo el período de una sola vez. Los errores de entrenamiento son residuos del ajuste, por lo que sus métricas son descriptivas y optimistas.

## Coeficientes

| Modelo | Phi | Pares de entrenamiento |
|---|---:|---:|
| 2Var | 0.9978 | 1255 |
| 4Var_Inflacion | 0.9948 | 1255 |
| 4Var_Produccion | 0.9974 | 1255 |
| 6Var | 0.9949 | 1255 |

## Comparación en las mismas filas con mes anterior disponible

| Conjunto | Modelo | N | RMSE original | RMSE AR1 | Mejora RMSE | MAE original | MAE AR1 | Mejora MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2Var | 1255 | 1.5426 | 0.3091 | 80.0% | 1.0042 | 0.1149 | 88.6% |
| validacion | 2Var | 433 | 2.3090 | 0.4840 | 79.0% | 1.9221 | 0.1768 | 90.8% |
| test | 2Var | 461 | 1.6988 | 0.2503 | 85.3% | 1.3345 | 0.1334 | 90.0% |
| train | 4Var_Inflacion | 1255 | 1.5125 | 0.3429 | 77.3% | 0.9585 | 0.1610 | 83.2% |
| validacion | 4Var_Inflacion | 433 | 2.7535 | 0.5024 | 81.8% | 2.2145 | 0.2153 | 90.3% |
| test | 4Var_Inflacion | 461 | 2.2813 | 0.2723 | 88.1% | 1.7741 | 0.1564 | 91.2% |
| train | 4Var_Produccion | 1255 | 1.5383 | 0.3095 | 79.9% | 0.9889 | 0.1190 | 88.0% |
| validacion | 4Var_Produccion | 433 | 2.2978 | 0.4844 | 78.9% | 1.9145 | 0.1773 | 90.7% |
| test | 4Var_Produccion | 461 | 1.6768 | 0.2527 | 84.9% | 1.3157 | 0.1348 | 89.8% |
| train | 6Var | 1255 | 1.5115 | 0.3412 | 77.4% | 0.9565 | 0.1619 | 83.1% |
| validacion | 6Var | 433 | 2.7376 | 0.5030 | 81.6% | 2.2041 | 0.2148 | 90.3% |
| test | 6Var | 461 | 2.2607 | 0.2725 | 87.9% | 1.7605 | 0.1568 | 91.1% |

Un porcentaje positivo indica reducción del error; uno negativo, empeoramiento. Los promedios ponderan observaciones, no países. Las muestras pueden diferir entre modelos por valores ausentes.

AR1_comparacion.csv incluye también todas las filas: si falta el mes anterior se utiliza el modelo original sin corrección. AR1_por_pais.csv permite comprobar excepciones por país. AR1_predicciones.csv conserva cada predicción, el retardo y su disponibilidad.

El test ya se había inspeccionado al recomendar el retardo: estos resultados son retrospectivos y no una confirmación independiente de la elección. Para confirmar la mejora conviene una evaluación temporal nueva. No se ha comparado AR(2) ni seleccionado otro orden.

## Métricas equivalentes a los modelos originales

AR1_coeficientes.csv: b, error estándar, t, p-valor, intervalo al 95 % y significación del coeficiente phi. Son inferencias OLS convencionales, condicionales a los residuos estimados del modelo base; no incorporan la incertidumbre de esa primera etapa.

AR1_metricas_ajuste.csv: N, R² y R² ajustado no centrados (regresión sin constante), F, p-valor F, RMSE, MAE, AIC y BIC del ajuste error_t = phi × error_(t−1) en entrenamiento. No son el R², AIC o BIC del modelo completo sobre Rendimiento BCE ni se deben comparar con los AIC/BIC de los OLS originales.

AR1_metricas_prediccion.csv: R² centrado, RMSE y MAE sobre Rendimiento BCE, para original y AR1 en entrenamiento, validación y test; tanto en pares consecutivos como en todas las filas. Estas son las métricas para comparar predicciones finales.

AR1_durbin_watson_por_pais.csv: Durbin–Watson de los errores originales y corregidos por país, conjunto y tramo mensual continuo; no se concatenan países para calcularlo.
