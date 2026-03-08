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

% ==========================================
% 2.2. TRATAMIENTO DE VALORES PERDIDOS (NaN)
% ==========================================
num_nan = sum(isnan(INPUTS_raw), 'all');
fprintf('1. Valores perdidos detectados: %d\n', num_nan);

if(num_nan > 0)
    % Rellenar NaN con la media de cada variable
    INPUTS_clean = fillmissing(INPUTS_raw, 'constant', mean(INPUTS_raw, 'omitnan'));
else 
    INPUTS_clean = INPUTS_raw;
end

% ==========================================
% 2.3. DETECCIÓN Y MANEJO DE OUTLIERS
% ==========================================

%%%% COMPROBAR ESTO
% Recortar outliers usando la desviación absoluta de la mediana (MAD)
INPUTS_no_outliers = filloutliers(INPUTS_clean, 'clip', 'median');
fprintf('2. Outliers limitados (método clip con MAD).\n');

% ==========================================
% 2.4. NORMALIZACIÓN (Z-SCORE)
% ==========================================
X = normalize(INPUTS_no_outliers, 'zscore');
Y = OUTPUTS_raw;
fprintf('3. Datos normalizados mediante Z-score.\n');

% ==========================================
% 2.5. EVIDENCIA VISUAL: IMPORTANCIA DE LA NORMALIZACIÓN
% ==========================================
mostrar_grafica = true;%false % Cambiar a false si no se quiere ver la gráfica
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