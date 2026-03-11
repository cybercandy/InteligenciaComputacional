%% --- PASO 4: Análisis de resultados y comparación ---
clearvars; clc; close all;

% ==========================================
% 4.1. CONFIGURACIÓN INTERACTIVA
% ==========================================
fprintf('--- CONFIGURACIÓN DE LA COMPARACIÓN ESTADÍSTICA ---\n');
fprintf('Selecciona el dataset del que quieres comparar LDA vs QDA:\n');
fprintf('  1. Iris\n');
fprintf('  2. QSAR\n');
opcion_dataset = input('Elige una opción [Por defecto: 1]: ');
if isempty(opcion_dataset), opcion_dataset = 1; end

lista_datasets = {'Iris', 'QSAR'}; 
dataset_comparar = lista_datasets{opcion_dataset}; 

fprintf('\n>>> Comparando modelos para el dataset "%s" <<<\n\n', dataset_comparar);

% ==========================================
% 4.2. CARGA DE RESULTADOS
% ==========================================
try
    res_LDA = load(sprintf('Resultados_%s_linear.mat', dataset_comparar));
    res_QDA = load(sprintf('Resultados_%s_pseudoquadratic.mat', dataset_comparar));
catch
    error('Faltan archivos .mat. Asegúrate de ejecutar el Script 2 para AMBOS modelos de %s primero.', dataset_comparar);
end

% ==========================================
% 4.3. PREPARACIÓN Y TEST ESTADÍSTICO
% ==========================================
% Construimos la matriz exacta que espera tu función (10 filas x 2 columnas)
Resultados_F1_Folds = [res_LDA.F1_Test_Folds, res_QDA.F1_Test_Folds];
modelos_comparados = {res_LDA.modelo_usar, res_QDA.modelo_usar};

fprintf('======================================================\n');
fprintf('ANÁLISIS ESTADÍSTICO PARA DATASET: %s\n', dataset_comparar);
fprintf('Comparando LDA (%s) vs QDA (%s)\n', modelos_comparados{1}, modelos_comparados{2});
fprintf('======================================================\n');

[p] = testEstadistico(Resultados_F1_Folds, modelos_comparados);