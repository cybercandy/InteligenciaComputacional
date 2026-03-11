%% --- PASO 2: PREPROCESADO DE DATOS ---
clearvars; clc; close all;

% ==========================================
% 2.1. SELECCIÓN DE DATASET
% ==========================================
fprintf('Selecciona el dataset a preprocesar:\n');
fprintf('  1. Iris\n');
fprintf('  2. QSAR\n');
opcion_dataset = input('Elige una opción [Por defecto: Iris]: ');

if isempty(opcion_dataset)
    opcion_dataset = 1;
end

% Cargar los datos correspondientes
if opcion_dataset == 1
    load('iris.mat');
    INPUTS_raw = INPUTS;
    OUTPUTS_raw = OUTPUTS;
    nombre_dataset = 'Iris';
elseif opcion_dataset == 2
    load('qsar_data.mat');
    INPUTS_raw = INPUTS_qsar;
    OUTPUTS_raw = categorical(OUTPUTS_qsar); % Convertir a categórica para clasificación
    nombre_dataset = 'QSAR';
else
    error('Opción no válida. Ejecuta de nuevo el script.');
end

fprintf('\n=== PREPROCESADO DEL DATASET: %s ===\n\n', nombre_dataset);
[n_muestras, n_vars] = size(INPUTS_raw);
fprintf('Dimensiones: %d muestras x %d variables\n\n', n_muestras, n_vars);

% ==========================================
% 2.2. TRATAMIENTO DE VALORES PERDIDOS (NaN)
% ==========================================
fprintf('--- 1. VALORES PERDIDOS ---\n');
num_nan = sum(isnan(INPUTS_raw), 'all');
fprintf('  NaN detectados: %d\n', num_nan);

if(num_nan > 0)
    % Rellenar NaN con la media de cada variable
    INPUTS_clean = fillmissing(INPUTS_raw, 'constant', mean(INPUTS_raw, 'omitnan'));
    fprintf('  → Imputación por media de cada variable aplicada.\n');
else 
    INPUTS_clean = INPUTS_raw;
    fprintf('  → No se requiere imputación.\n');
end

% ==========================================
% 2.3. OUTLIERS (IQR)
% ==========================================

fprintf('\n--- 2. OUTLIERS ---\n');

% Usamos IQR (Interquartile Range) como método para detectar valores
% atípicos
outliers_iqr = false(n_muestras, n_vars);
for v = 1:n_vars
    x = INPUTS_clean(:, v);
    Q1 = quantile(x, 0.25);
    Q3 = quantile(x, 0.75);
    IQR_val = Q3 - Q1;
    outliers_iqr(:, v) = x < (Q1 - 1.5*IQR_val) | x > (Q3 + 1.5*IQR_val);
end

filas_outlier = any(outliers_iqr, 2);
n_outliers = sum(filas_outlier);
fprintf('  → Muestras con outlier en ≥1 variable: %d (%.1f%%)\n', ...
    n_outliers, 100*n_outliers/n_muestras);
fprintf('  → Se mantienen los outliers.\n');

% Boxplot como respaldo visual al IQR 
% Permite ver la distribución de muestras por variables, donde los bigotes
% son los valores extremos
figure('Name', sprintf('%s - Boxplot outliers', nombre_dataset));
boxplot(INPUTS_clean);
title(sprintf('%s: Distribución por variable (outliers visibles)', nombre_dataset));
xlabel('Variable'); ylabel('Valor'); grid on;
saveas(gcf, sprintf('fig_%s_boxplot.png', lower(nombre_dataset)));

% ==========================================
% 2.4. CORRELACIÓN ENTRE VARIABLES
% ==========================================
fprintf('\n--- 3. CORRELACIÓN ENTRE VARIABLES ---\n');
R         = corrcoef(INPUTS_clean, 'Rows', 'pairwise');
umbral    = 0.90;
n_pares   = 0;

for i = 1:n_vars
    for j = i+1:n_vars
        if abs(R(i,j)) >= umbral
            fprintf('  Variables %d y %d: r = %.4f\n', i, j, R(i,j));
            n_pares = n_pares + 1;
        end
    end
end

if n_pares == 0
    fprintf('  No se encontraron pares con |r| >= %.2f.\n', umbral);
end
fprintf('  → DECISIÓN: Se mantienen todas las variables.\n');
fprintf('     Justificación: El enunciado prohíbe reducción de dimensión.\n');
fprintf('     Alta correlación puede afectar la estabilidad de LDA/QDA.\n');

% ==========================================
% 2.5. DISPERSIÓN Y NORMALIZACIÓN
% ==========================================
fprintf('\n--- 4. DISPERSIÓN DE VARIABLES ---\n');
stds      = std(INPUTS_clean);
ratio_std = max(stds) / min(stds);
fprintf('  Std mínima: %.4f | Std máxima: %.4f\n', min(stds), max(stds));
fprintf('  Ratio max/min desviaciones: %.2f\n', ratio_std);

if ratio_std > 2
    fprintf('  → Alta diferencia de escalas. NORMALIZACIÓN NECESARIA.\n');
else
    fprintf('  → Diferencia de escalas moderada. Normalización recomendada.\n');
end

fprintf('\n--- 5. NORMALIZACIÓN Z-SCORE ---\n');
X = normalize(INPUTS_clean, 'zscore');
Y = OUTPUTS_raw;

% ==========================================
% 2.5.1. EVIDENCIA VISUAL: IMPORTANCIA DE LA NORMALIZACIÓN
% ==========================================

mostrar_grafica = false;%true % Cambiar a false si no se quiere ver la gráfica
if mostrar_grafica
    figure('Name', sprintf('Evidencia de normalización: %s', nombre_dataset));
    
    % Gráfica superior: Datos sin normalizar
    subplot(2,1,1);
    boxplot(INPUTS_raw);
    title(sprintf('%s: Escalas Originales', nombre_dataset));
    ylabel('Magnitud Real');
    grid on;
    
    % Gráfica inferior: Datos normalizados
    subplot(2,1,2);
    boxplot(X);
    title(sprintf('%s: Tras Normalización Z-Score (Media 0, Varianza 1)', nombre_dataset));
    ylabel('Valor Tipificado');
    grid on;
end

% ==========================================
% 2.6. GUARDAR DATOS PREPROCESADOS
% ==========================================
nombre_archivo = sprintf('Datos_%s_Preprocesados.mat', nombre_dataset);
save(nombre_archivo, 'X', 'Y');
fprintf('\nPreprocesado completado. Datos (X, Y) guardados en: %s\n', nombre_archivo);