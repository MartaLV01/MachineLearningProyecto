import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from machinelearningproyecto import particiones_temporales
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import r2_score



"""
Usando OLS para usar metricas y calcular coef ya que desde linear regression no se puede
HECHO PARA EL TRAIN
"""
def model_OLS(X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv):
    X_train_sm = sm.add_constant(X_train, has_constant="add")

    # Ajustar la regresión lineal por mínimos cuadrados ordinarios
    modelo_ols = sm.OLS(
        y_train,
        X_train_sm,
        missing="raise"
    ).fit()

    # Predicciones de entrenamiento y validación
    train_pred_simple = modelo_ols.predict(X_train_sm)

    X_val_sm = sm.add_constant(X_val, has_constant="add")
    X_val_sm = X_val_sm.reindex(columns=X_train_sm.columns)

    val_pred_simple = modelo_ols.predict(X_val_sm)

    ic = modelo_ols.conf_int()

    tabla_coeficientes = pd.DataFrame({
        "b": modelo_ols.params,
        "Error estándar": modelo_ols.bse,
        "t": modelo_ols.tvalues,
        "p-valor": modelo_ols.pvalues,
        "IC 95% inferior": ic.iloc[:, 0],
        "IC 95% superior": ic.iloc[:, 1],
        "Significativo al 5%": modelo_ols.pvalues < 0.05,
    })
    tabla_coeficientes.index.name = "Variable"

    print("Tabla de coeficientes:")
    print(tabla_coeficientes.round(4))

    # Métricas del ajuste en entrenamiento
    residuos = modelo_ols.resid

    # Métricas del ajuste.
    tabla_metricas = pd.Series({
        "N.º de observaciones": int(modelo_ols.nobs),
        "R² entrenamiento": modelo_ols.rsquared,
        "R² ajustado": modelo_ols.rsquared_adj,
        "F": modelo_ols.fvalue,
        "p-valor de F": modelo_ols.f_pvalue,
        "Durbin-Watson": durbin_watson(residuos),
        "RMSE entrenamiento": np.sqrt(np.mean(residuos ** 2)),
        "MAE entrenamiento": np.mean(np.abs(residuos)),
        "AIC": modelo_ols.aic,
        "BIC": modelo_ols.bic,
    }, name="Valor").to_frame()

    tabla_metricas.index.name = "Métrica"

    print("\nMétricas del modelo:")
    print(tabla_metricas.round(4))



    errores_val = y_val.to_numpy() - val_pred_simple

    tabla_validacion = pd.Series({
        "R² validación": r2_score(y_val, val_pred_simple),
        "RMSE validación": np.sqrt(np.mean(errores_val ** 2)),
        "MAE validación": np.mean(np.abs(errores_val)),
    }, name="Valor").to_frame()


    print("\n", tabla_coeficientes)
    print(tabla_metricas)
    print(tabla_validacion)