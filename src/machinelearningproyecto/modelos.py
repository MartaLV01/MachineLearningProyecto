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


"""
Usando Tree para usar metricas y calcular coef
HECHO PARA EL TRAIN
"""
def model_DecisionTree(
    X_train, y_train, X_val, y_val, X_test, y_test, X_cv, y_cv,
    *, max_depth, min_samples_leaf,
    min_samples_split, ccp_alpha=0.0, random_state=42,
):

    # el * indica que las varibales que vienen despues deben pasarse por nombre
    # ccp_alpha controla la poda del arbol, en 0 indica que no hay poda, es el predetermidado de scikit-learn.
    # Solo se usa loa valores de max profundidas y min de hojas, sin ningun otro limite
    """Ajustar un árbol de regresión con la misma entrada que model_OLS.
 
    Sólo se entrena con train; se respetan las particiones recibidas.
    X_cv e y_cv se aceptan por compatibilidad, sin realizar validación cruzada.
    Las variables deben ser numéricas; País ISO requiere codificación previa.
 
    Devuelve el estimador, las métricas, predicciones, errores e importancias.
    """
    from sklearn.tree import DecisionTreeRegressor
 
    modelo = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )
    modelo.fit(X_train, np.asarray(y_train).ravel())
 
    conjuntos = [('Train', X_train, y_train), ('Validación', X_val, y_val)]
    
    filas, predicciones, residuos = [], {}, {}
    for nombre, X, y in conjuntos:
        if len(X) == 0:
            continue
        pred = modelo.predict(X)
        observado = np.asarray(y).ravel()
        residuo = observado - pred
        indice = y.index if isinstance(y, (pd.Series, pd.DataFrame)) else None
        predicciones[nombre] = pd.Series(pred, index=indice, name='Predicción')
        residuos[nombre] = pd.Series(residuo, index=indice, name='Residuo')
        filas.append({
            'Conjunto': nombre,
            'N.º de observaciones': len(observado),
            'R²': r2_score(observado, pred) if len(observado) >= 2 else np.nan,
            'RMSE': np.sqrt(np.mean(residuo ** 2)),
            'MAE': np.mean(np.abs(residuo)),
        })
 
    tabla_metricas = pd.DataFrame(filas).set_index('Conjunto')
    nombres = (list(X_train.columns) if isinstance(X_train, pd.DataFrame)
               else [f'Variable {i + 1}' for i in range(modelo.n_features_in_)])
    importancias = pd.Series(
        modelo.feature_importances_, index=nombres, name='Importancia',
    ).sort_values(ascending=False)
    importancias.index.name = 'Variable'
 
    print('\nMétricas Decision Tree:')
    print(tabla_metricas.round(4))
    print('\nImportancias por reducción de impureza (no son coeficientes ni p-valores):')
    print(importancias.round(4))
 
    return {
        'modelo': modelo,
        'metricas': tabla_metricas,
        'predicciones': predicciones,
        'residuos': residuos,
        'importancias': importancias,
    }


def search_results_table(search, top_n=10):
    results = pd.DataFrame(search.cv_results_).copy()
    results["CV_RMSE"] = -results["mean_test_score"]
    results["TRAIN_RMSE"] = -results["mean_train_score"]
    results["CV_STD"] = results["std_test_score"]
    columns = [
        "rank_test_score", "CV_RMSE", "CV_STD", "TRAIN_RMSE", "params"
    ]
    return results.sort_values("rank_test_score")[columns].head(top_n)