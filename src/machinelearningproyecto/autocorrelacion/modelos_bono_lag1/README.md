# Modelos con el bono del mismo país un mes antes

Cuatro OLS agrupados: 2, 4 de inflación, 4 de producción y 6 variables originales, más Rendimiento BCE del mismo país en el mes anterior (3, 5, 5 y 7 predictores). Cada modelo comparte coeficientes entre países. Todos los coeficientes se ajustan sólo con entrenamiento.

El retardo se crea por país sobre el Excel antes de eliminar observaciones por otras variables. Sólo se admite el mes inmediatamente anterior. Se conservan las asignaciones originales a entrenamiento, validación y test; las filas sin bono anterior se excluyen sin recalcular cortes. Los CSV de particiones muestran las pérdidas.

Se evalúan predicciones sucesivas a un mes, con el bono real anterior ya observado. No se utilizan resultados del mes actual o posterior para construir el retardo. Esto no representa predecir todo el período de una vez; además las variables macroeconómicas contemporáneas de los modelos originales tendrían que estar disponibles en el momento de predecir.

## Comparación sobre exactamente las mismas filas

Original: modelo previo sin retardo, ya entrenado sobre su muestra original. Lag1: OLS nuevo con el bono anterior, entrenado en las filas de entrenamiento con retardo disponible. La diferencia combina el predictor añadido y la pérdida de filas de entrenamiento. Persistencia: usar directamente el bono anterior como predicción, sin ajustar parámetros.

| Conjunto | Modelo | N | RMSE original | RMSE lag1 | RMSE persistencia | R² lag1 | MAE lag1 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 2Var | 1255 | 1.5426 | 0.2996 | 0.3120 | 0.9688 | 0.1029 |
| train | 4Var_Inflacion | 1255 | 1.5125 | 0.2980 | 0.3120 | 0.9691 | 0.1069 |
| train | 4Var_Produccion | 1255 | 1.5383 | 0.2994 | 0.3120 | 0.9688 | 0.1040 |
| train | 6Var | 1255 | 1.5115 | 0.2980 | 0.3120 | 0.9691 | 0.1067 |
| validacion | 2Var | 433 | 2.3090 | 0.5015 | 0.4792 | 0.9427 | 0.2424 |
| validacion | 4Var_Inflacion | 433 | 2.7535 | 0.4880 | 0.4792 | 0.9457 | 0.2048 |
| validacion | 4Var_Produccion | 433 | 2.2978 | 0.5015 | 0.4792 | 0.9427 | 0.2421 |
| validacion | 6Var | 433 | 2.7376 | 0.4886 | 0.4792 | 0.9456 | 0.2064 |
| test | 2Var | 461 | 1.6988 | 0.2803 | 0.2464 | 0.9659 | 0.1895 |
| test | 4Var_Inflacion | 461 | 2.2813 | 0.2637 | 0.2464 | 0.9698 | 0.1619 |
| test | 4Var_Produccion | 461 | 1.6768 | 0.2804 | 0.2464 | 0.9659 | 0.1903 |
| test | 6Var | 461 | 2.2607 | 0.2645 | 0.2464 | 0.9697 | 0.1634 |

## Interpretación de los resultados

Con los resultados actuales, los cuatro modelos con retardo reducen claramente RMSE y MAE frente a los originales en validación y test. El bono del mes anterior aporta mucha información predictiva.

Sin embargo, la referencia de persistencia consigue menor RMSE y MAE que los cuatro modelos con retardo tanto en validación como en test. Por tanto, estos resultados no demuestran que las variables macroeconómicas añadan capacidad predictiva al bono anterior; no elegiría un modelo más complejo sólo por su mejora frente al original.

Entre los nuevos modelos, el de 4 variables de inflación obtiene el menor RMSE de validación. Su elección sería provisional y basada en validación; el test se presenta como evaluación retrospectiva.

## Coeficiente del bono anterior

- 2Var: 0.9925.
- 4Var_Inflacion: 0.9912.
- 4Var_Produccion: 0.9920.
- 6Var: 0.9911.

## Ejecutar y consultar resultados

Desde la raíz del proyecto:

```bash
./src/machinelearningproyecto/autocorrelacion/.venv/bin/python src/machinelearningproyecto/autocorrelacion/modelos_bono_lag1/ejecutar_todos.py
```

También puedes ejecutar cualquiera de los cuatro archivos modelo_*.py con el mismo Python. No se necesitan paquetes nuevos.

- comparacion_general.csv: R², RMSE y MAE por modelo, conjunto y variante.
- *_coeficientes.csv: coeficientes, errores estándar, t, p-valores, IC al 95 % y significación.
- *_ajuste_train.csv: N, R² ajustado, F, p-valor F, AIC y BIC del nuevo modelo OLS.
- *_metricas_por_pais.csv y *_durbin_watson.csv: resultados por país; DW por tramo continuo.
- *_predicciones.csv: observaciones, fecha del bono anterior, retardo, predicciones y errores.
- *_particiones.csv: tamaños y fechas antes y después de exigir el retardo.

Los errores estándar e inferencias son los OLS convencionales y pueden verse afectados por dependencia temporal. Un buen rendimiento frente al modelo original no basta: debe compararse también con persistencia. El test ya ha sido consultado durante este trabajo; la confirmación independiente requiere datos posteriores.
