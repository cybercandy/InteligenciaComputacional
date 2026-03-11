%% --- PASO 1: ANÁLISIS DEL CONJUNTO DE DATOS (EDA) ---
clearvars; clc; close all;

% ==========================================
% 1.1. SELECCIÓN DE DATASET
% ==========================================
fprintf('Selecciona el dataset a analizar:\n');
fprintf('  1. Iris\n');
fprintf('  2. QSAR\n');
opcion_dataset = input('Elige una opción [Por defecto: Iris]: ');

if isempty(opcion_dataset)
    opcion_dataset = 1;
end

% Cargar los datos y asignarlos a variables genéricas
if opcion_dataset == 1
    load('iris.mat');
    datos_actuales = INPUTS;
    nombre_dataset = 'IRIS';
elseif opcion_dataset == 2
    load('qsar_data.mat');
    datos_actuales = INPUTS_qsar;
    nombre_dataset = 'QSAR';
else
    error('Opción no válida. Ejecuta de nuevo el script.');
end

fprintf('\n====== INICIANDO ANÁLISIS DEL DATASET: %s ======\n\n', nombre_dataset);

% ==========================================
% 1.2. ANÁLISIS Y ESTADÍSTICAS
% ==========================================

% --- Valores perdidos (NaN) ---
num_nan = sum(isnan(datos_actuales), 'all');
fprintf('Valores perdidos detectados: %d\n\n', num_nan);

% --- Estadísticas descriptivas ---
% Limitamos la salida en consola a 10 variables máximo para no saturar
num_vars_mostrar = min(10, size(datos_actuales, 2));

fprintf('--- Estadísticas descriptivas (Primeras %d variables) ---\n', num_vars_mostrar);
fprintf('Media: '); disp(mean(datos_actuales(:, 1:num_vars_mostrar), 'omitnan'));
fprintf('Mediana: '); disp(median(datos_actuales(:, 1:num_vars_mostrar), 'omitnan'));
fprintf('Desviación estándar: '); disp(std(datos_actuales(:, 1:num_vars_mostrar), 'omitnan'));

if size(datos_actuales, 2) > 10
    fprintf('*(Nota: Hay %d variables en total, se muestran solo las primeras 10 en consola)*\n\n', size(datos_actuales, 2));
end

% ==========================================
% 1.3. VISUALIZACIONES
% ==========================================
% Para las gráficas de dispersión y cajas, limitamos a 5-15 variables si es muy grande
num_vars_plot_cajas = min(15, size(datos_actuales, 2));
num_vars_plot_dispersion = min(5, size(datos_actuales, 2));

% --- Detección de outliers (Boxplot) ---

figure('Name', sprintf('Detección de outliers - %s', nombre_dataset));
boxplot(datos_actuales(:, 1:num_vars_plot_cajas));
title(sprintf('%s: Distribución (Mostrando %d vars)', nombre_dataset, num_vars_plot_cajas));
xlabel('Variables'); ylabel('Valores');
grid on;

% --- Matriz de dispersión (Relaciones) ---
figure('Name', sprintf('Relaciones entre variables - %s', nombre_dataset));
plotmatrix(datos_actuales(:, 1:num_vars_plot_dispersion));
title(sprintf('%s: Matriz de dispersión (Mostrando %d vars)', nombre_dataset, num_vars_plot_dispersion));

% --- Matriz de correlación ---
figure('Name', sprintf('Matriz de correlación - %s', nombre_dataset));
imagesc(corrcoef(datos_actuales, 'Rows', 'pairwise')); 
colorbar;
title(sprintf('Correlación - %s', nombre_dataset));