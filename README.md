# Inteligencia Computacional - Prácticas 1 y 2

Scripts de MATLAB para el análisis, preprocesado, entrenamiento y comparación de modelos de clasificación sobre los datasets **Iris** y **QSAR**, siguiendo la metodología experimental de la asignatura.

---

## Orden de ejecución

Los scripts deben ejecutarse **en orden**, ya que cada uno depende de los resultados del anterior:

```
paso1_analisis → paso2_preprocesado → paso3_aprendizaje → paso4_comparacion
```

---

## Paso 1 — Análisis exploratorio (EDA)

**¿Qué hace?**
Analiza el dataset en crudo sin modificar nada. Su objetivo es entender la estructura de los datos antes de tomar ninguna decisión.

**Al ejecutarlo imprime:**
- Dimensiones del dataset (nº muestras × nº variables)
- Nº de valores perdidos (NaN) por variable
- Tabla de estadísticas descriptivas: media, mediana, desviación típica, mínimo y máximo (primeras 10 variables)
- Nº de muestras con outliers según IQR, y en qué porcentaje del dataset
- Balance de clases: cuántas muestras hay por clase y su porcentaje

**Figuras generadas:**
| Figura | Para qué sirve |
|---|---|
| `fig_*_clases.png` | Ver si las clases están balanceadas. Importante para elegir métricas en el paso 3 |
| `fig_*_boxplot.png` | Ver la escala de cada variable y detectar outliers visualmente |
| `fig_*_dispersion.png` | Ver relaciones entre variables y si las clases se separan bien (anticipa el rendimiento de LDA/QDA) |
| `fig_*_correlacion.png` | Ver si hay variables redundantes (alta correlación puede desestabilizar LDA/QDA) |

**Instrucciones:**
1. Ejecutar en MATLAB.
2. Seleccionar dataset:
   - **Iris** → pulsar `1` o **Enter** (por defecto)
   - **QSAR** → pulsar `2`

> Este script no genera ningún archivo `.mat`. Es solo de consulta y análisis visual.

---

## Paso 2 — Preprocesado

**¿Qué hace?**
Aplica las transformaciones necesarias sobre los datos crudos y justifica cada decisión. Genera los datos limpios que usarán todos los modelos.

**Al ejecutarlo imprime:**
- Nº de NaN detectados y acción tomada (imputación por media si los hay)
- Nº de muestras con outliers (IQR) y decisión justificada (se mantienen)
- Pares de variables con correlación |r| ≥ 0.90 y decisión justificada (se mantienen, el enunciado prohíbe reducción de dimensión)
- Ratio de dispersión entre variables (max std / min std) y decisión de normalizar
- Confirmación de normalización Z-score aplicada

**Figuras generadas:**
| Figura | Para qué sirve |
|---|---|
| `fig_*_boxplot_outliers.png` | Respaldo visual del análisis IQR |
| `fig_*_normalizacion.png` | Comparación de escalas antes y después del Z-score. Justifica visualmente la normalización |

**Archivos generados:**
- `Datos_Iris_Preprocesados.mat` → variables `X` (inputs normalizados) e `Y` (outputs)
- `Datos_QSAR_Preprocesados.mat` → variables `X` (inputs normalizados) e `Y` (outputs)

**Instrucciones:**
1. Ejecutar en MATLAB.
2. Seleccionar dataset:
   - **Iris** → pulsar `1` o **Enter** (por defecto)
   - **QSAR** → pulsar `2`

> ⚠️ Ejecutar para **ambos datasets** antes de pasar al Paso 3.

---

## Paso 3 — Aprendizaje

**¿Qué hace?**
Entrena un modelo sobre los datos preprocesados usando validación cruzada con 10 folds y guarda las métricas de rendimiento. Debe ejecutarse varias veces para cubrir todos los modelos necesarios.

**⚠️ Requisito previo:** Haber ejecutado el Paso 2 para el dataset que se quiere analizar.

**Al ejecutarlo imprime:**
- Tabla de métricas por clase (Recall, Specificity, Precision, F1) con media ± std sobre los 10 folds
- Métricas globales (Accuracy, F1, Recall, Specificity, Precision) con media ± std
- Comparación Train vs Test en F1 para detectar sobreajuste

**Instrucciones:**
1. Ejecutar en MATLAB.
2. Seleccionar dataset:
   - **Iris** → pulsar `1` o **Enter** (por defecto)
   - **QSAR** → pulsar `2`
3. Seleccionar modelo:
   - `1` → LDA (Discriminante Lineal)
   - `2` → QDA (Discriminante Cuadrático)
   - `3` → Árbol de Decisión
4. Si se elige Árbol, seleccionar parámetro a modificar:
   - `1` → `MaxNumSplits` (profundidad máxima)
   - `2` → `MinLeafSize` (mínimo de ejemplos en una hoja)
   - `3` → `MinParentSize` (mínimo de ejemplos para dividir un nodo)
5. Si se elige Árbol, introducir el valor numérico del parámetro. El script muestra valores sugeridos justificados según el dataset.

**Archivos generados:**
- `Resultados_<Dataset>_linear.mat` → resultados LDA
- `Resultados_<Dataset>_pseudoquadratic.mat` → resultados QDA
- `Resultados_<Dataset>_tree_<Parametro>_<Valor>.mat` → resultados de cada versión de árbol

**Figuras generadas:**
| Figura | Para qué sirve |
|---|---|
| `fig_*_tree_*.png` | Visualización del árbol entrenado en el Fold 1. Permite ver cómo cambia la estructura al modificar parámetros |

> ⚠️ Antes de pasar al Paso 4 hay que haber ejecutado este script para: **LDA**, **QDA** y **mínimo 3 versiones de árbol** (con distintos valores del mismo parámetro), para cada dataset.

---

## Paso 4 — Comparación de modelos

**¿Qué hace?**
Realiza la comparación estadística en dos fases: primero selecciona el mejor árbol entre todas las versiones entrenadas, y luego enfrenta ese árbol con LDA y QDA para determinar el modelo final.

**⚠️ Requisito previo:** Haber ejecutado el Paso 3 para LDA, QDA y al menos 3 versiones de árbol del dataset a comparar.

**Fase 1 — Comparación de árboles:**
- Carga automáticamente todos los archivos `Resultados_<Dataset>_tree_*.mat`
- Imprime tabla resumen con F1 y Accuracy (media ± std) de cada versión
- Aplica test estadístico (Lilliefors → ANOVA o Kruskal-Wallis)
- Si hay diferencias significativas: selecciona el árbol con mayor F1
- Si no hay diferencias: selecciona el árbol más sencillo (más restringido)
- Genera figura comparativa de boxplots entre versiones de árbol

**Fase 2 — Comparación final (mejor árbol vs LDA vs QDA):**
- Carga el árbol ganador de la Fase 1 junto con LDA y QDA
- Imprime tabla resumen final con F1 y Accuracy (media ± std)
- Aplica test estadístico final
- Concluye con el modelo ganador: si no hay diferencias significativas, se elige LDA por ser el más sencillo
- Genera figura comparativa de boxplots final

**Figuras generadas:**
| Figura | Para qué sirve |
|---|---|
| `fig_*_comparativa_trees.png` | Boxplots F1 de todas las versiones de árbol. Base para justificar la elección del árbol ganador |
| `fig_*_comparativa_final.png` | Boxplots F1 de LDA vs QDA vs mejor árbol. Figura principal del informe para la conclusión final |

**Instrucciones:**
1. Ejecutar en MATLAB.
2. Seleccionar dataset:
   - **Iris** → pulsar `1` o **Enter** (por defecto)
   - **QSAR** → pulsar `2`

---

## Archivos de datos

| Archivo | Contenido |
|---|---|
| `iris.mat` | Variables `INPUTS` (150×4) y `OUTPUTS` (150×1) |
| `qsar_data.mat` | Variables `INPUTS_qsar` (1055×41) y `OUTPUTS_qsar` (1055×1) |
| `Datos_Iris_Preprocesados.mat` | Generado por Paso 2. Variables `X`, `Y` |
| `Datos_QSAR_Preprocesados.mat` | Generado por Paso 2. Variables `X`, `Y` |
