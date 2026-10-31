"""Actualizar el README a partir de los errores y diagnósticos existentes."""
import csv
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
NOMBRES = dict(AT='Austria', BE='Bélgica', BG='Bulgaria', HR='Croacia', CY='Chipre', CZ='Chequia', DE='Alemania', DK='Dinamarca', EE='Estonia', ES='España', FI='Finlandia', FR='Francia', GR='Grecia', EL='Grecia', HU='Hungría', IE='Irlanda', IT='Italia', LT='Lituania', LU='Luxemburgo', LV='Letonia', MT='Malta', NL='Países Bajos', PL='Polonia', PT='Portugal', RO='Rumanía', SE='Suecia', SI='Eslovenia', SK='Eslovaquia')

def main():
    with (BASE / 'resumen_autocorrelacion.csv').open() as f:
        resumen = list(csv.DictReader(f))
    positivos = [r for r in resumen if r['diagnostico'] == 'Evidencia de autocorrelacion']
    lineas = ['# Resumen de autocorrelación por país y modelo', '',
        f'Hay evidencia de autocorrelación en **{len(positivos)} de {len(resumen)} casos** (país × modelo × conjunto). Se usa Ljung–Box con p ajustado por Holm < 0,05.', '',
        '**Cómo leer las tablas:** cada celda muestra las autocorrelaciones de los errores en los retardos **1 / 2 / 12 meses**, respectivamente. Por ejemplo, `0,80 / 0,60 / −0,10` significa ρ(1)=0,80, ρ(2)=0,60 y ρ(12)=−0,10. El signo indica correlación positiva o negativa. «—» significa que el contraste global no detectó evidencia suficiente o no pudo evaluarse; consulta el CSV para distinguirlos.', '',
        '**AR y retardos:** la notación habitual es AR(1), AR(2) o AR(12); los errores anteriores son e(t−1), e(t−2), e(t−12). Las cifras de estas tablas son ACF (correlaciones), no coeficientes de una regresión AR ni una selección del orden AR. El contraste global tampoco demuestra que cada retardo individual sea significativo. Para determinar un orden AR habría que estudiar PACF y comparar ajustes AR.', '',
        'Los retardos de 12 meses en validación y test son descriptivos y poco precisos por el tamaño reducido de esas muestras. Se usa la ACF muestral centrada, con denominador común Σ(e−media)².', '']
    modelos = ['2Var', '4Var_Inflacion', '4Var_Produccion', '6Var']
    cache = {}
    for r in positivos:
        clave = (r['pais'], r['modelo'])
        if clave not in cache:
            with (BASE / f'{clave[0]}_{clave[1]}.csv').open() as f:
                cache[clave] = list(csv.DictReader(f))
        x = np.array([float(f['error']) for f in cache[clave] if f['conjunto'] == r['conjunto']])
        x -= x.mean()
        valores = [np.dot(x[k:], x[:-k]) / np.dot(x, x) if len(x) > k else np.nan for k in (1, 2, 12)]
        r['acf'] = ' / '.join(f'{v:.2f}'.replace('.', ',') if np.isfinite(v) else 'NA' for v in valores)
    for conjunto, titulo in [('train', 'Entrenamiento'), ('validacion', 'Validación'), ('test', 'Test')]:
        filas = [r for r in positivos if r['conjunto'] == conjunto]
        lineas += [f'## {titulo}', '', '| País | 2 variables | 4: inflación | 4: producción | 6 variables |', '|---|---|---|---|---|']
        for pais in sorted({r['pais'] for r in filas}):
            celdas = [next((r['acf'] for r in filas if r['pais'] == pais and r['modelo'] == modelo), '—') for modelo in modelos]
            lineas.append('| ' + ' | '.join([f'{NOMBRES.get(pais, pais)} ({pais})', *celdas]) + ' |')
        lineas.append('')
    lineas += ['## Archivos y reproducción', '',
        '- `resumen_autocorrelacion.csv`: diagnóstico y p-valores por país, modelo y conjunto.',
        '- `<PAIS>_<MODELO>.csv`: errores, observaciones y predicciones ordenados por fecha.',
        '- `<PAIS>_<MODELO>.md`: diagnóstico individual.',
        '- Ejecutar `.venv/bin/python analizar.py` y después `.venv/bin/python resumir_readme.py` desde esta carpeta.', '',
        'Se reproducen los modelos OLS originales agrupando todos los países y ajustando únicamente con entrenamiento. Los errores se separan por país y conjunto para analizarlos. Las configuraciones y el Excel externos se leen sin modificarlos. Las series con huecos mensuales, duplicados, pocos datos o errores constantes quedan sin diagnóstico inferencial. No detectar autocorrelación no demuestra independencia.', '']
    (BASE / 'README.md').write_text('\n'.join(lineas), encoding='utf-8')

if __name__ == '__main__':
    main()
