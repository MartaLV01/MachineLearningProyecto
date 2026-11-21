# Modelos con el bono del mismo país un mes antes

Cuatro OLS agrupados: 2, 4 de inflación, 4 de producción y 6 variables originales, más Bono soberano 10 años (% anual) del mismo país en el mes anterior (3, 5, 5 y 7 predictores). Cada modelo comparte coeficientes entre países. Todos los coeficientes se ajustan sólo con entrenamiento.

El retardo se crea por país sobre el Excel antes de eliminar observaciones por otras variables. Sólo se admite el mes inmediatamente anterior. Se conservan las fechas de corte originales de validación y test, pero la muestra se prepara con el bono a 10 años; las filas sin bono anterior se excluyen sin recalcular cortes. Los CSV de particiones muestran las pérdidas.

Se evalúan predicciones sucesivas a un mes, con el bono real anterior ya observado. No se utilizan resultados del mes actual o posterior para construir el retardo. Esto no representa predecir todo el período de una vez; además las variables macroeconómicas contemporáneas de los modelos originales tendrían que estar disponibles en el momento de predecir.

## Comparación sobre exactamente las mismas filas

Original: OLS sin retardo reajustado para el bono a 10 años con las variables originales y toda su muestra de entrenamiento disponible. Lag1: OLS nuevo con el bono anterior, entrenado en las filas de entrenamiento con retardo disponible. La diferencia combina el predictor añadido y la pérdida de filas de entrenamiento. Persistencia: usar directamente el bono anterior como predicción, sin ajustar parámetros.

| Conjunto | Modelo | N | RMSE original | RMSE lag1 | RMSE persistencia | R² lag1 | MAE lag1 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 2Var | 1252 | 1.3653 | 0.2770 | 0.2835 | 0.9773 | 0.1846 |
| train | 4Var_Inflacion | 1252 | 1.2304 | 0.2656 | 0.2835 | 0.9791 | 0.1744 |
| train | 4Var_Produccion | 1252 | 1.3514 | 0.2762 | 0.2835 | 0.9775 | 0.1836 |
| train | 6Var | 1252 | 1.2299 | 0.2656 | 0.2835 | 0.9792 | 0.1743 |
| validacion | 2Var | 433 | 1.1690 | 0.2190 | 0.1997 | 0.9623 | 0.1479 |
| validacion | 4Var_Inflacion | 433 | 1.3866 | 0.1996 | 0.1997 | 0.9687 | 0.1462 |
| validacion | 4Var_Produccion | 433 | 1.1387 | 0.2189 | 0.1997 | 0.9624 | 0.1478 |
| validacion | 6Var | 433 | 1.3739 | 0.1995 | 0.1997 | 0.9687 | 0.1455 |
| test | 2Var | 461 | 1.3044 | 0.1581 | 0.1578 | 0.9799 | 0.1181 |
| test | 4Var_Inflacion | 461 | 1.9115 | 0.1655 | 0.1578 | 0.9779 | 0.1178 |
| test | 4Var_Produccion | 461 | 1.2683 | 0.1584 | 0.1578 | 0.9798 | 0.1181 |
| test | 6Var | 461 | 1.8964 | 0.1644 | 0.1578 | 0.9782 | 0.1169 |

## Interpretación

El objetivo es el bono soberano a 10 años, tanto en el valor actual como en el retardo. Las métricas se calculan sobre las mismas filas para las tres variantes. La cobertura puede diferir de Rendimiento BCE, aunque se mantengan las fechas de corte.

- validacion: menor RMSE con retardo: 6Var (0.1995).
  El modelo con retardo supera a persistencia en RMSE en 2 de 4 configuraciones.
- test: menor RMSE con retardo: 2Var (0.1581).
  El modelo con retardo supera a persistencia en RMSE en 0 de 4 configuraciones.

## Coeficiente del bono anterior

- 2Var: 0.9935.
- 4Var_Inflacion: 0.9711.
- 4Var_Produccion: 0.9916.
- 6Var: 0.9712.

## Ejecutar y consultar resultados

Desde la raíz del proyecto:

```bash
./src/machinelearningproyecto/autocorrelacion/.venv/bin/python src/machinelearningproyecto/autocorrelacion/modelos_bono_10_anios_lag1/ejecutar_todos.py
```

También puedes ejecutar cualquiera de los cuatro archivos modelo_*.py con el mismo Python. No se necesitan paquetes nuevos.

- comparacion_general.csv: R², RMSE y MAE por modelo, conjunto y variante.
- *_coeficientes.csv: coeficientes, errores estándar, t, p-valores, IC al 95 % y significación.
- *_ajuste_train.csv: N, R² ajustado, F, p-valor F, AIC y BIC del nuevo modelo OLS.
- *_metricas_por_pais.csv y *_durbin_watson.csv: resultados por país; DW por tramo continuo.
- *_predicciones.csv: observaciones, fecha del bono anterior, retardo, predicciones y errores.
- *_particiones.csv: tamaños y fechas antes y después de exigir el retardo.

Los errores estándar e inferencias son los OLS convencionales y pueden verse afectados por dependencia temporal. Un buen rendimiento frente al modelo original no basta: debe compararse también con persistencia. El test ya ha sido consultado durante este trabajo; la confirmación independiente requiere datos posteriores.
