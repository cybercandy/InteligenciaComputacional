%% --- PASO 4: Comparación de Modelos ---
clearvars; clc; close all;

% ==========================================
% 4.1. SELECCIÓN DE DATASET
% ==========================================
fprintf('Selecciona el dataset a comparar:\n');
fprintf('  1. Iris\n');
fprintf('  2. QSAR\n');
opcion_dataset = input('Elige una opción [Por defecto: Iris]: ');
if isempty(opcion_dataset), opcion_dataset = 1; end

lista_datasets = {'Iris', 'QSAR'};
dataset_comparar = lista_datasets{opcion_dataset};
fprintf('\n>>> Comparando modelos para: %s <<<\n\n', dataset_comparar);

% ==========================================
% 4.2. FASE 1: COMPARACIÓN DE ÁRBOLES
% ==========================================
fprintf('========================================================\n');
fprintf('FASE 1: COMPARACIÓN DE VERSIONES DE ÁRBOL\n');
fprintf('========================================================\n');

% Buscar todos los archivos de árboles del dataset
patron_trees = sprintf('Resultados_%s_tree_*.mat', dataset_comparar);
archivos_trees = dir(patron_trees);

if length(archivos_trees) < 2
    error(['Se necesitan al menos 2 versiones de árbol para comparar.\n' ...
           'Ejecuta paso3_aprendizaje con distintas configuraciones de árbol.']);
end

n_trees = length(archivos_trees);
F1_trees = zeros(10, n_trees);
etiquetas_trees = cell(n_trees, 1);
resumen_trees = struct();

fprintf('Versiones de árbol encontradas:\n');
for k = 1:n_trees
    res = load(archivos_trees(k).name);
    F1_trees(:, k) = res.F1_ts_folds;
    etiquetas_trees{k} = sprintf('Tree_%s', res.etiqueta_version);
    resumen_trees(k).nombre   = etiquetas_trees{k};
    resumen_trees(k).mean_F1  = res.mean_F1_ts;
    resumen_trees(k).std_F1   = res.std_F1_ts;
    resumen_trees(k).mean_ACC = res.mean_ACC_ts;
    resumen_trees(k).std_ACC  = res.std_ACC_ts;
    fprintf('  [%d] %s → F1 = %.4f ± %.4f\n', k, etiquetas_trees{k}, ...
        resumen_trees(k).mean_F1, resumen_trees(k).std_F1);
end

% --- Tabla resumen árboles ---
fprintf('\n--- Tabla resumen árboles ---\n');
fprintf('%-35s %-22s %-22s\n', 'Modelo', 'F1-Score (media±std)', 'Accuracy (media±std)');
fprintf('%s\n', repmat('-', 1, 80));
for k = 1:n_trees
    fprintf('%-35s %.4f ± %.4f        %.4f ± %.4f\n', ...
        resumen_trees(k).nombre, ...
        resumen_trees(k).mean_F1,  resumen_trees(k).std_F1, ...
        resumen_trees(k).mean_ACC, resumen_trees(k).std_ACC);
end
fprintf('%s\n\n', repmat('-', 1, 80));

% --- Test estadístico entre árboles ---
fprintf('--- Test estadístico entre versiones de árbol ---\n');
max_len = max(cellfun(@length, etiquetas_trees));
etiquetas_trees_fmt = char(cellfun(@(e) sprintf('%-*s', max_len, e), ...
    etiquetas_trees, 'UniformOutput', false));

[P_trees] = testEstadistico(F1_trees, etiquetas_trees_fmt);

% --- Selección del mejor árbol ---
fprintf('\n--- Selección del mejor árbol ---\n');
fprintf('p-valor obtenido: %.4f\n', P_trees);

F1_medias_trees = [resumen_trees.mean_F1];

% REVISAR CRITERIO PARA ESCOGER EL MÁS SENCILLO     
% Para IRIS las diferencias no son significativas aunque lo modifiquemos
% Nos quedaríamos con los valores por defecto de los parámetros
if P_trees >= 0.10
    fprintf('→ No hay diferencias significativas entre versiones.\n');
    fprintf('→ Se selecciona el árbol más sencillo (mayor restricción).\n');

    % Cargar los valores del parámetro de cada árbol
    valores_param = zeros(n_trees, 1);
    for k = 1:n_trees
        res_tmp = load(archivos_trees(k).name);
        valores_param(k) = res_tmp.valor_param(1);
    end

    % El más sencillo es el de mayor valor de MinParentSize/MinLeafSize
    % o el de menor MaxNumSplits según el parámetro usado
    res_tmp = load(archivos_trees(1).name);
    if res_tmp.opcion_param == 1
        % MaxNumSplits: más sencillo = valor más pequeño
        [~, idx_mejor_tree] = min(valores_param);
    else
        % MinLeafSize o MinParentSize: más sencillo = valor más grande
        [~, idx_mejor_tree] = max(valores_param);
    end

    fprintf('→ Árbol seleccionado: %s\n', resumen_trees(idx_mejor_tree).nombre);
else
    fprintf('→ Hay diferencias significativas. Se selecciona el de mayor F1.\n');
    [~, idx_mejor_tree] = max(F1_medias_trees);
    fprintf('→ Mejor árbol: %s (F1 = %.4f)\n', ...
        resumen_trees(idx_mejor_tree).nombre, F1_medias_trees(idx_mejor_tree));
end

mejor_tree_nombre  = resumen_trees(idx_mejor_tree).nombre;
mejor_tree_F1      = F1_trees(:, idx_mejor_tree);
mejor_tree_archivo = archivos_trees(idx_mejor_tree).name;

% Figura comparativa árboles
figure('Name', sprintf('%s - Comparativa árboles', dataset_comparar), ...
    'Position', [100 100 900 500]);
boxplot(F1_trees, 'Labels', etiquetas_trees);
ylabel('F1-Score (Test)');
title(sprintf('%s: Comparativa versiones de árbol (%d folds)', dataset_comparar, 10));
grid on; xtickangle(30);
saveas(gcf, sprintf('fig_%s_comparativa_trees.png', lower(dataset_comparar)));

% ==========================================
% 4.3. FASE 2: COMPARACIÓN FINAL (MEJOR ÁRBOL vs LDA vs QDA)
% ==========================================
fprintf('\n========================================================\n');
fprintf('FASE 2: COMPARACIÓN FINAL — MEJOR ÁRBOL vs LDA vs QDA\n');
fprintf('========================================================\n');

% Cargar LDA y QDA
try
    res_LDA = load(sprintf('Resultados_%s_linear.mat',          dataset_comparar));
    res_QDA = load(sprintf('Resultados_%s_pseudoquadratic.mat', dataset_comparar));
catch
    error(['Faltan resultados de LDA o QDA para %s.\n' ...
           'Ejecuta paso3_aprendizaje para los modelos LDA y QDA.'], dataset_comparar);
end

% Construir matriz de comparación final
F1_final     = [res_LDA.F1_ts_folds, res_QDA.F1_ts_folds, mejor_tree_F1];
etiquetas_final = {'linear', 'pseudoquadratic', mejor_tree_nombre};

% Tabla resumen final
resumen_final(1).nombre   = 'LDA';
resumen_final(1).mean_F1  = res_LDA.mean_F1_ts;
resumen_final(1).std_F1   = res_LDA.std_F1_ts;
resumen_final(1).mean_ACC = res_LDA.mean_ACC_ts;
resumen_final(1).std_ACC  = res_LDA.std_ACC_ts;

resumen_final(2).nombre   = 'QDA';
resumen_final(2).mean_F1  = res_QDA.mean_F1_ts;
resumen_final(2).std_F1   = res_QDA.std_F1_ts;
resumen_final(2).mean_ACC = res_QDA.mean_ACC_ts;
resumen_final(2).std_ACC  = res_QDA.std_ACC_ts;

res_tree = load(mejor_tree_archivo);
resumen_final(3).nombre   = mejor_tree_nombre;
resumen_final(3).mean_F1  = res_tree.mean_F1_ts;
resumen_final(3).std_F1   = res_tree.std_F1_ts;
resumen_final(3).mean_ACC = res_tree.mean_ACC_ts;
resumen_final(3).std_ACC  = res_tree.std_ACC_ts;

fprintf('\n--- Tabla resumen final ---\n');
fprintf('%-35s %-22s %-22s\n', 'Modelo', 'F1-Score (media±std)', 'Accuracy (media±std)');
fprintf('%s\n', repmat('-', 1, 80));
for k = 1:3
    fprintf('%-35s %.4f ± %.4f        %.4f ± %.4f\n', ...
        resumen_final(k).nombre, ...
        resumen_final(k).mean_F1,  resumen_final(k).std_F1, ...
        resumen_final(k).mean_ACC, resumen_final(k).std_ACC);
end
fprintf('%s\n\n', repmat('-', 1, 80));

% Justificación de métrica
fprintf('Métrica de comparación: F1-Score\n');
if strcmp(dataset_comparar, 'QSAR')
    fprintf('Justificación: QSAR puede presentar desbalance → F1 más robusto que Accuracy.\n\n');
else
    fprintf('Justificación: Iris está balanceado → F1 y Accuracy equivalentes.\n\n');
end

% --- Test estadístico final ---
fprintf('--- Test estadístico final ---\n');
max_len = max(cellfun(@length, etiquetas_final));
etiquetas_final_fmt = char(cellfun(@(e) sprintf('%-*s', max_len, e), ...
    etiquetas_final, 'UniformOutput', false));

[P_final] = testEstadistico(F1_final, etiquetas_final_fmt);

% --- Conclusión final ---
fprintf('\n========================================================\n');
fprintf('CONCLUSIÓN FINAL\n');
fprintf('========================================================\n');
fprintf('p-valor obtenido: %.4f\n', P_final);

F1_medias_final = [resumen_final.mean_F1];
[F1_ord, idx_ord] = sort(F1_medias_final, 'descend');

fprintf('\nRanking por F1-Score:\n');
for k = 1:3
    fprintf('  %d. %-35s F1 = %.4f ± %.4f\n', k, ...
        resumen_final(idx_ord(k)).nombre, F1_ord(k), ...
        resumen_final(idx_ord(k)).std_F1);
end

if P_final >= 0.10
    fprintf('\n→ No hay diferencias significativas entre los tres modelos.\n');
    fprintf('→ Se selecciona el MÁS SENCILLO: LDA\n');
    fprintf('   Justificación: LDA es el modelo más simple (lineal, sin hiperparámetros).\n');
else
    fprintf('\n→ Existen diferencias significativas (p < 0.10).\n');
    fprintf('→ Ver gráfico multcompare para identificar el mejor modelo.\n');
    fprintf('→ Entre los no significativamente distintos, elegir el más sencillo.\n');
end

% Figura comparativa final
figure('Name', sprintf('%s - Comparativa final', dataset_comparar), ...
    'Position', [100 100 900 500]);
boxplot(F1_final, 'Labels', {'LDA', 'QDA', mejor_tree_nombre});
ylabel('F1-Score (Test)');
title(sprintf('%s: Comparativa final LDA vs QDA vs Árbol (%d folds)', dataset_comparar, 10));
grid on; xtickangle(30);
saveas(gcf, sprintf('fig_%s_comparativa_final.png', lower(dataset_comparar)));