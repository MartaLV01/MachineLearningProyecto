# Resumen de autocorrelación por país y modelo

Hay evidencia de autocorrelación en **165 de 308 casos** (país × modelo × conjunto). Se usa Ljung–Box con p ajustado por Holm < 0,05.

**Cómo leer las tablas:** cada celda muestra las autocorrelaciones de los errores en los retardos **1 / 2 / 12 meses**, respectivamente. Por ejemplo, `0,80 / 0,60 / −0,10` significa ρ(1)=0,80, ρ(2)=0,60 y ρ(12)=−0,10. El signo indica correlación positiva o negativa. «—» significa que el contraste global no detectó evidencia suficiente o no pudo evaluarse; consulta el CSV para distinguirlos.

**AR y retardos:** la notación habitual es AR(1), AR(2) o AR(12); los errores anteriores son e(t−1), e(t−2), e(t−12). Las cifras de estas tablas son ACF (correlaciones), no coeficientes de una regresión AR ni una selección del orden AR. El contraste global tampoco demuestra que cada retardo individual sea significativo. Para determinar un orden AR habría que estudiar PACF y comparar ajustes AR.

Los retardos de 12 meses en validación y test son descriptivos y poco precisos por el tamaño reducido de esas muestras. Se usa la ACF muestral centrada, con denominador común Σ(e−media)².

## Entrenamiento

| País | 2 variables | 4: inflación | 4: producción | 6 variables |
|---|---|---|---|---|
| Austria (AT) | 0,90 / 0,86 / 0,42 | 0,89 / 0,85 / 0,37 | 0,90 / 0,86 / 0,42 | 0,89 / 0,84 / 0,36 |
| Bulgaria (BG) | 0,93 / 0,85 / -0,22 | 0,92 / 0,83 / -0,16 | 0,93 / 0,85 / -0,21 | 0,92 / 0,83 / -0,17 |
| Dinamarca (DK) | 0,85 / 0,66 / -0,22 | 0,84 / 0,64 / -0,07 | 0,84 / 0,66 / -0,22 | 0,84 / 0,64 / -0,06 |
| España (ES) | 0,85 / 0,67 / -0,18 | 0,80 / 0,58 / -0,20 | 0,88 / 0,73 / -0,18 | 0,80 / 0,58 / -0,22 |
| Finlandia (FI) | 0,85 / 0,66 / -0,22 | 0,85 / 0,65 / -0,26 | 0,85 / 0,67 / -0,23 | 0,85 / 0,65 / -0,27 |
| Francia (FR) | 0,96 / 0,92 / 0,53 | 0,96 / 0,93 / 0,53 | 0,96 / 0,93 / 0,54 | 0,96 / 0,93 / 0,53 |
| Grecia (GR) | 0,83 / 0,66 / -0,25 | 0,84 / 0,69 / -0,31 | 0,86 / 0,71 / -0,24 | 0,84 / 0,70 / -0,31 |
| Croacia (HR) | 0,91 / 0,75 / -0,13 | 0,93 / 0,81 / -0,13 | 0,91 / 0,75 / -0,12 | 0,93 / 0,81 / -0,14 |
| Hungría (HU) | 0,93 / 0,85 / 0,09 | 0,92 / 0,84 / 0,12 | 0,93 / 0,85 / 0,09 | 0,92 / 0,84 / 0,13 |
| Irlanda (IE) | 0,74 / 0,49 / 0,08 | 0,72 / 0,45 / 0,04 | 0,74 / 0,50 / 0,08 | 0,73 / 0,47 / 0,05 |
| Luxemburgo (LU) | 0,94 / 0,87 / -0,02 | — | 0,94 / 0,88 / 0,01 | 0,56 / 0,36 / -0,01 |
| Malta (MT) | 0,87 / 0,72 / -0,17 | 0,89 / 0,75 / -0,38 | 0,87 / 0,73 / -0,18 | 0,89 / 0,75 / -0,38 |
| Polonia (PL) | 0,95 / 0,89 / 0,06 | 0,95 / 0,88 / 0,04 | 0,95 / 0,88 / 0,06 | 0,95 / 0,88 / 0,03 |
| Portugal (PT) | 0,79 / 0,67 / -0,31 | 0,85 / 0,74 / -0,42 | 0,80 / 0,67 / -0,30 | 0,85 / 0,74 / -0,41 |
| Rumanía (RO) | 0,86 / 0,67 / -0,20 | 0,87 / 0,71 / -0,12 | 0,87 / 0,68 / -0,21 | 0,87 / 0,71 / -0,13 |
| Suecia (SE) | 0,85 / 0,66 / -0,12 | 0,85 / 0,64 / -0,20 | 0,85 / 0,66 / -0,15 | 0,85 / 0,64 / -0,21 |
| Eslovenia (SI) | 0,84 / 0,66 / -0,31 | 0,87 / 0,71 / -0,31 | 0,85 / 0,66 / -0,31 | 0,87 / 0,71 / -0,32 |
| Eslovaquia (SK) | 0,94 / 0,88 / 0,15 | 0,95 / 0,89 / 0,13 | 0,94 / 0,88 / 0,15 | 0,95 / 0,88 / 0,11 |

## Validación

| País | 2 variables | 4: inflación | 4: producción | 6 variables |
|---|---|---|---|---|
| Bulgaria (BG) | 0,81 / 0,56 / -0,35 | 0,77 / 0,56 / -0,38 | 0,81 / 0,55 / -0,36 | 0,77 / 0,56 / -0,38 |
| Chipre (CY) | 0,82 / 0,60 / -0,35 | 0,82 / 0,54 / -0,39 | 0,82 / 0,60 / -0,34 | 0,82 / 0,54 / -0,39 |
| Alemania (DE) | 0,79 / 0,56 / -0,36 | 0,72 / 0,47 / -0,35 | 0,80 / 0,57 / -0,36 | 0,73 / 0,48 / -0,35 |
| España (ES) | 0,77 / 0,50 / -0,36 | 0,81 / 0,55 / -0,42 | 0,76 / 0,48 / -0,34 | 0,80 / 0,54 / -0,42 |
| Finlandia (FI) | 0,74 / 0,47 / -0,36 | — | 0,76 / 0,49 / -0,36 | — |
| Francia (FR) | 0,71 / 0,62 / -0,36 | — | 0,70 / 0,62 / -0,36 | — |
| Croacia (HR) | 0,79 / 0,49 / -0,40 | 0,77 / 0,47 / -0,40 | 0,78 / 0,48 / -0,40 | — |
| Hungría (HU) | 0,86 / 0,69 / -0,41 | 0,88 / 0,73 / -0,39 | 0,86 / 0,69 / -0,41 | 0,88 / 0,73 / -0,39 |
| Irlanda (IE) | 0,81 / 0,56 / -0,35 | 0,81 / 0,55 / -0,35 | 0,80 / 0,56 / -0,34 | 0,81 / 0,57 / -0,35 |
| Lituania (LT) | 0,83 / 0,62 / -0,37 | 0,74 / 0,52 / -0,39 | 0,83 / 0,61 / -0,38 | 0,74 / 0,52 / -0,39 |
| Luxemburgo (LU) | 0,81 / 0,56 / -0,35 | 0,82 / 0,50 / -0,20 | 0,81 / 0,56 / -0,36 | 0,81 / 0,51 / -0,21 |
| Polonia (PL) | 0,81 / 0,61 / -0,39 | 0,79 / 0,59 / -0,38 | 0,81 / 0,61 / -0,39 | 0,79 / 0,59 / -0,38 |
| Eslovenia (SI) | — | 0,81 / 0,51 / -0,41 | — | 0,80 / 0,52 / -0,41 |

## Test

| País | 2 variables | 4: inflación | 4: producción | 6 variables |
|---|---|---|---|---|
| Bélgica (BE) | 0,83 / 0,71 / -0,41 | 0,79 / 0,65 / -0,34 | 0,83 / 0,70 / -0,41 | 0,79 / 0,64 / -0,33 |
| Chipre (CY) | 0,78 / 0,59 / -0,28 | 0,71 / 0,52 / -0,28 | 0,78 / 0,59 / -0,27 | 0,71 / 0,52 / -0,28 |
| Chequia (CZ) | 0,68 / 0,44 / -0,25 | — | — | — |
| Alemania (DE) | 0,74 / 0,54 / -0,29 | 0,74 / 0,54 / -0,32 | 0,74 / 0,53 / -0,29 | 0,74 / 0,54 / -0,32 |
| España (ES) | 0,85 / 0,72 / -0,41 | 0,86 / 0,70 / -0,39 | 0,86 / 0,72 / -0,41 | 0,86 / 0,70 / -0,39 |
| Finlandia (FI) | 0,86 / 0,72 / -0,42 | 0,81 / 0,66 / -0,38 | 0,86 / 0,72 / -0,41 | 0,82 / 0,67 / -0,38 |
| Grecia (GR) | 0,80 / 0,66 / -0,36 | 0,79 / 0,58 / -0,34 | 0,79 / 0,65 / -0,32 | 0,79 / 0,57 / -0,31 |
| Italia (IT) | 0,85 / 0,71 / -0,40 | 0,81 / 0,61 / -0,31 | 0,85 / 0,71 / -0,40 | 0,81 / 0,62 / -0,32 |
| Luxemburgo (LU) | 0,73 / 0,55 / -0,34 | 0,83 / 0,60 / -0,34 | 0,73 / 0,54 / -0,31 | 0,84 / 0,62 / -0,34 |
| Malta (MT) | 0,80 / 0,60 / -0,28 | 0,78 / 0,56 / -0,25 | 0,79 / 0,58 / -0,25 | 0,77 / 0,55 / -0,23 |
| Polonia (PL) | 0,88 / 0,74 / -0,46 | 0,87 / 0,73 / -0,45 | 0,88 / 0,74 / -0,46 | 0,87 / 0,73 / -0,45 |
| Rumanía (RO) | — | 0,73 / 0,46 / -0,40 | — | 0,73 / 0,47 / -0,40 |
| Suecia (SE) | 0,74 / 0,63 / -0,37 | — | 0,73 / 0,63 / -0,37 | — |
| Eslovenia (SI) | 0,83 / 0,71 / -0,41 | 0,85 / 0,69 / -0,40 | 0,82 / 0,70 / -0,41 | 0,85 / 0,69 / -0,39 |

## Archivos y reproducción

- `resumen_autocorrelacion.csv`: diagnóstico y p-valores por país, modelo y conjunto.
- `<PAIS>_<MODELO>.csv`: errores, observaciones y predicciones ordenados por fecha.
- `<PAIS>_<MODELO>.md`: diagnóstico individual.
- Ejecutar `.venv/bin/python analizar.py` y después `.venv/bin/python resumir_readme.py` desde esta carpeta.

Se reproducen los modelos OLS originales agrupando todos los países y ajustando únicamente con entrenamiento. Los errores se separan por país y conjunto para analizarlos. Las configuraciones y el Excel externos se leen sin modificarlos. Las series con huecos mensuales, duplicados, pocos datos o errores constantes quedan sin diagnóstico inferencial. No detectar autocorrelación no demuestra independencia.
