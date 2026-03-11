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
Entrena los modelos de clasificación sobre los datos preprocesados usando validación cruzada y guarda las métricas de rendimiento.

**⚠️ Requisito previo:** Haber ejecutado el Paso 2 para el dataset que se quiere analizar.

**Instrucciones:**
1. Ejecutar en MATLAB.
2. Seleccionar dataset (Iris / QSAR).
3. Seleccionar modelo a entrenar.

> Al finalizar genera un archivo `.mat` con los resultados necesarios para el Paso 4.

---

## Paso 4 — Comparación de modelos

**¿Qué hace?**
Carga los resultados de todos los modelos entrenados y determina cuál es el mejor para cada dataset mediante comparación estadística.

**⚠️ Requisito previo:** Haber ejecutado el Paso 3 con **todos los modelos** del dataset a comparar.

**Instrucciones:**
1. Ejecutar en MATLAB.
2. Seleccionar dataset (Iris / QSAR).

---

## Archivos de datos

| Archivo | Contenido |
|---|---|
| `iris.mat` | Variables `INPUTS` (150×4) y `OUTPUTS` (150×1) |
| `qsar_data.mat` | Variables `INPUTS_qsar` (1055×41) y `OUTPUTS_qsar` (1055×1) |
| `Datos_Iris_Preprocesados.mat` | Generado por Paso 2. Variables `X`, `Y` |
| `Datos_QSAR_Preprocesados.mat` | Generado por Paso 2. Variables `X`, `Y` |
