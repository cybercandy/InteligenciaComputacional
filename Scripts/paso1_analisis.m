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
    INPUTS_raw = INPUTS;
    OUTPUTS_raw = OUTPUTS;
    nombre_dataset = 'IRIS';
elseif opcion_dataset == 2
    load('qsar_data.mat');
    INPUTS_raw = INPUTS_qsar;
    OUTPUTS_raw = categorical(OUTPUTS_qsar);
    nombre_dataset = 'QSAR';
else
    error('Opción no válida. Ejecuta de nuevo el script.');
end

fprintf('\n====== INICIANDO ANÁLISIS DEL DATASET: %s ======\n\n', nombre_dataset);

[n_muestras, n_vars] = size(INPUTS_raw);
fprintf('Dimensiones: %d muestras x %d variables\n\n', n_muestras, n_vars);

% ==========================================
% 1.2. ANÁLISIS Y ESTADÍSTICAS
% ==========================================

% --- Valores perdidos (NaN) ---
fprintf('--- 1. VALORES PERDIDOS ---\n');
num_nan = sum(isnan(INPUTS_raw), 'all');
fprintf('  NaN totales: %d\n', num_nan);

% ==========================================
% 1.3. ESTADÍSTICAS DESCRIPTIVAS
% ==========================================
fprintf('\n--- 2. ESTADÍSTICAS DESCRIPTIVAS ---\n');
num_vars_consola = min(10, n_vars);
fprintf('(Mostrando primeras %d de %d variables)\n\n', num_vars_consola, n_vars);

fprintf('  %-10s %-12s %-12s %-12s %-12s %-12s\n', ...
    'Variable', 'Media', 'Mediana', 'Std', 'Min', 'Max');
for v = 1:num_vars_consola
    x = INPUTS_raw(:, v);
    fprintf('  Var %-6d %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
        v, mean(x,'omitnan'), median(x,'omitnan'), ...
        std(x,'omitnan'), min(x), max(x));
end
if n_vars > 10
    fprintf('  ... (%d variables restantes no mostradas)\n', n_vars - 10);
end

% ==========================================
% 1.4. OUTLIERS
% ==========================================
% Los analizaremos en el script paso2_preprocesado

% ==========================================
% 1.5. BALANCE DE CLASES
% ==========================================
fprintf('\n--- 4. BALANCE DE CLASES ---\n');
clases = OUTPUTS_raw;
categorias  = unique(clases);
n_clases    = length(categorias);
conteo      = zeros(n_clases, 1);
etiquetas   = strings(n_clases, 1);

for i = 1:n_clases
    conteo(i)    = sum(clases == categorias(i));
    etiquetas(i) = string(categorias(i));
    fprintf('  Clase %s: %d muestras (%.1f%%)\n', ...
        etiquetas(i), conteo(i), 100*conteo(i)/n_muestras);
end

% Desequilibrio: ratio entre clase mayor y menor
ratio_clases = max(conteo) / min(conteo);
fprintf('  Ratio max/min clases: %.2f', ratio_clases);
if ratio_clases > 1.5
    fprintf(' → Desbalance moderado/alto\n');
else
    fprintf(' → Clases balanceadas\n');
end


% ==========================================
% 1.6. FIGURAS
% ==========================================

% -- Figura 1: Balance de clases --
figure('Name', sprintf('%s - Balance de clases', nombre_dataset), ...
    'Position', [100 100 500 400]);
bar(1:n_clases, conteo, 'FaceColor', [0.2 0.6 0.8]);
xticks(1:n_clases); xticklabels(etiquetas);
xlabel('Clase'); ylabel('Nº muestras');
title(sprintf('%s: Balance de clases', nombre_dataset));
text(1:n_clases, conteo, num2str(conteo), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
grid on;
saveas(gcf, sprintf('fig_%s_clases.png', lower(nombre_dataset)));

% -- Figura 2: Boxplot por variable --
num_vars_box = min(15, n_vars);
figure('Name', sprintf('%s - Boxplot variables', nombre_dataset), ...
    'Position', [100 100 900 400]);
boxplot(INPUTS_raw(:, 1:num_vars_box));
title(sprintf('%s: Distribución por variable (primeras %d)', ...
    nombre_dataset, num_vars_box));
xlabel('Variable'); ylabel('Valor'); grid on;
saveas(gcf, sprintf('fig_%s_boxplot.png', lower(nombre_dataset)));

% -- Figura 3: Matriz de dispersión (primeras 5 variables) --
num_vars_scatter = min(5, n_vars);
figure('Name', sprintf('%s - Matriz de dispersión', nombre_dataset), ...
    'Position', [100 100 700 600]);
plotmatrix(INPUTS_raw(:, 1:num_vars_scatter));
title(sprintf('%s: Matriz de dispersión (primeras %d variables)', ...
    nombre_dataset, num_vars_scatter));
saveas(gcf, sprintf('fig_%s_dispersion.png', lower(nombre_dataset)));

% -- Figura 4: Matriz de correlación --
figure('Name', sprintf('%s - Correlación', nombre_dataset), ...
    'Position', [100 100 700 600]);
R = corrcoef(INPUTS_raw, 'Rows', 'pairwise');
imagesc(R); colorbar; colormap('jet'); clim([-1 1]);
title(sprintf('%s: Matriz de correlación', nombre_dataset));
xlabel('Variable'); ylabel('Variable');
xticks(1:n_vars); yticks(1:n_vars);
axis square;
saveas(gcf, sprintf('fig_%s_correlacion.png', lower(nombre_dataset)));

fprintf('\n====== EDA completado: %s ======\n', nombre_dataset);