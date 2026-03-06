
# Inteligencia Computacional - Evaluación de Modelos

Se añaden scripts interactivos para calcular y comparar los resultados únicamente del dataset y modelo escogidos, facilitando el análisis guiado.

---

## Script del Paso 2

Este script entrena un modelo específico sobre el dataset escogido utilizando validación cruzada y guarda las métricas de rendimiento.

**Instrucciones de ejecución:**

1. Ejecutar el script en MATLAB.
2. Escoger el dataset que se utilizará para realizar el test. Para escoger:
   * **IRIS** -> Escribir `1` o simplemente pulsar **Enter** (opción por defecto).
   * **QSAR** -> Escribir `2`.
3. Seleccionar el modelo que se usará:
   * **Linear** -> Escribir `1` o simplemente pulsar **Enter** (opción por defecto).
   * **Quadratic** -> Escribir `2`.

> **Nota:** Al finalizar, el script generará automáticamente un archivo `.mat` con los resultados, el cual es necesario para el siguiente paso.

---

## Script del Paso 3

Este script toma los resultados previos y ejecuta la comparación estadística para determinar qué modelo funciona mejor.

**⚠️ Requisito previo:**
Antes de ejecutarlo, hay que asegurarse de que tenemos los resultados del paso anterior con **todos los modelos** del dataset que queremos analizar (es decir, haber ejecutado el Paso 2 tanto para Linear como para Quadratic en el mismo dataset).

**Instrucciones de ejecución:**

1. Ejecutar el script en MATLAB.
2. Al ejecutarlo, nos pide especificar qué dataset vamos a comparar:
   * **IRIS** -> Escribir `1` o simplemente pulsar **Enter** (opción por defecto).
   * **QSAR** -> Escribir `2`.
