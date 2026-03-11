%% --- PASO 3: Aprendizaje y Evaluación ---
clearvars; clc; close all;
 rng(42); % para que sea reproducible

% ==========================================
% 3.1. CONFIGURACIÓN APRENDIZAJE
% ==========================================

% --- 1. SELECCIÓN DE DATASET ---
fprintf('Selecciona el dataset a analizar:\n');
fprintf('  1. Iris\n');
fprintf('  2. QSAR\n');
opcion_dataset = input('Elige una opción [Por defecto: Iris]: ');
if isempty(opcion_dataset)
    opcion_dataset = 1;
end

% Lista con los nombres (el orden debe coincidir con los números de arriba)
lista_datasets = {'Iris', 'QSAR'};
dataset_usar = lista_datasets{opcion_dataset}; % Saca el nombre usando el número


% --- 2. SELECCIÓN DE MODELO ---
fprintf('\nSelecciona el modelo a entrenar:\n');
fprintf('  1. LDA  - Discriminante Lineal\n');
fprintf('  2. QDA  - Discriminante Cuadrático\n');
fprintf('  3. Tree - Árbol de Decisión\n');
opcion_modelo = input('Elige una opción [Por defecto: Linear]: ');
if isempty(opcion_modelo)
    opcion_modelo = 1;
end

lista_modelos = {'linear', 'pseudoquadratic', 'tree'};
modelo_usar = lista_modelos{opcion_modelo};

% --- 3. CONFIGURACIÓN ÁRBOL (solo si es tree) ---
opcion_param = 0;
valor_param  = 0;
etiqueta_version = '';
usar_defecto = false;
nombres_param = {'MaxNumSplits', 'MinLeafSize', 'MinParentSize'};

if strcmp(modelo_usar, 'tree')

    fprintf('\nParámetro a modificar:\n');
    fprintf('  1. MaxNumSplits  - Profundidad máxima del árbol\n');
    fprintf('  2. MinLeafSize   - Mínimo de ejemplos en una hoja\n');
    fprintf('  3. MinParentSize - Mínimo de ejemplos para dividir un nodo\n');
    opcion_param = input('Elige parámetro [1-3]: ');
    if isempty(opcion_param), opcion_param = 1; end

    % Valores sugeridos según dataset
    fprintf('\n--- Valores sugeridos para %s ---\n', dataset_usar);
    if strcmp(dataset_usar, 'Iris')
        % Iris: 150 muestras, 3 clases, ~50 por clase
        switch opcion_param
            case 1
                fprintf('  MaxNumSplits: 2 (árbol mínimo), 5 (moderado), 10 (extenso)\n');
                fprintf('  Justificación: Con 150 muestras y clases bien separadas,\n');
                fprintf('  árboles profundos tienden a sobreajustar.\n');
            case 2
                fprintf('  MinLeafSize: 20 (restrictivo), 10 (moderado), 5 (permisivo)\n');
                fprintf('  Justificación: Con 50 muestras por clase, hojas muy pequeñas\n');
                fprintf('  pueden memorizar el conjunto de entrenamiento.\n');
            case 3
                fprintf('  MinParentSize: 20 (restrictivo), 10 (moderado), 5 (permisivo)\n');
                fprintf('  Justificación: Similar a MinLeafSize pero controla la división\n');
                fprintf('  de nodos internos.\n');
        end
    else
        % QSAR: 1055 muestras, 2 clases
        switch opcion_param
            case 1
                fprintf('  MaxNumSplits: 5 (árbol simple), 20 (moderado), 50 (extenso)\n');
                fprintf('  Justificación: Con 1055 muestras y 41 variables, el árbol por\n');
                fprintf('  defecto puede ser muy profundo. Limitar evita sobreajuste.\n');
            case 2
                fprintf('  MinLeafSize: 50 (restrictivo), 20 (moderado), 5 (permisivo)\n');
                fprintf('  Justificación: Con más de 1000 muestras, hojas con al menos\n');
                fprintf('  50 ejemplos siguen siendo representativas.\n');
            case 3
                fprintf('  MinParentSize: 50 (restrictivo), 20 (moderado), 10 (permisivo)\n');
                fprintf('  Justificación: Controla cuántos ejemplos mínimos necesita un\n');
                fprintf('  nodo para poder dividirse.\n');
        end
    end

    valor_param = input('\nIntroduce el valor del parámetro: ');

    if isempty(valor_param)
        % Árbol por defecto - usar un valor centinela
        usar_defecto = true;
        valor_param  = 0;  % 0 indica "por defecto"
        etiqueta_version = sprintf('%s_default', nombres_param{opcion_param});
        fprintf('→ Se usará el valor por defecto de %s\n', nombres_param{opcion_param});
    else
        usar_defecto = false;
        etiqueta_version = sprintf('%s_%d', nombres_param{opcion_param}, valor_param);
        fprintf('→ %s = %d\n', nombres_param{opcion_param}, valor_param);
    end
end

K = 10;
fprintf('\n>>> Dataset: %s | Modelo: %s', dataset_usar, modelo_usar);
if strcmp(modelo_usar, 'tree')
    fprintf(' (%s)', etiqueta_version);
end
fprintf(' | K=%d folds <<<\n\n', K);

% ==========================================
% 3.2. CARGA DE DATOS Y PREPARACIÓN
% ==========================================

% Cargamos el .mat correspondiente generado en el Paso 2 (paso2_preprocesado)
nombre_archivo_datos = sprintf('Datos_%s_Preprocesados.mat', dataset_usar);
try
    load(nombre_archivo_datos);
catch ME
    fprintf('No se pudo cargar el archivo. Error: %s\n', ME.message);
end

% X e Y -> Son las variables normalizadas
DATA_X = X;
DATA_Y = Y;

[n_muestras, n_vars] = size(DATA_X);
unique_classes = unique(DATA_Y);
NumClass = length(unique_classes);
CV = cvpartition(DATA_Y, 'Kfold', K);

fprintf('Muestras: %d | Variables: %d | Clases: %d\n\n', n_muestras, n_vars, NumClass);

% ==========================================
% 3.3. ENTRENAMIENTO Y EVALUACIÓN (K-Folds)
% ==========================================

% Métricas por fold y clase [K x NumClass]
Recall_tr    = zeros(K, NumClass); Recall_ts    = zeros(K, NumClass);
Spec_tr      = zeros(K, NumClass); Spec_ts      = zeros(K, NumClass);
Precision_tr = zeros(K, NumClass); Precision_ts = zeros(K, NumClass);
ACC_tr       = zeros(K, NumClass); ACC_ts       = zeros(K, NumClass);
F1_tr        = zeros(K, NumClass); F1_ts        = zeros(K, NumClass);

nombres_param = {'MaxNumSplits', 'MinLeafSize', 'MinParentSize'};

for i = 1:K
    trIdx = CV.training(i);
    tsIdx = CV.test(i);
    X_tr = DATA_X(trIdx,:); Y_tr = DATA_Y(trIdx);
    X_ts = DATA_X(tsIdx,:); Y_ts = DATA_Y(tsIdx);

    % --- Entrenar modelo ---
    if strcmp(modelo_usar, 'tree')
        if usar_defecto
            % Sin restricciones - árbol completo
            Mdl = fitctree(X_tr, Y_tr);
        else
            Mdl = fitctree(X_tr, Y_tr, param_name, valor_param);
        end

        % Visualizar árbol solo en el primer fold
        if i == 1
            fprintf('--- Árbol entrenado (Fold 1) ---\n');
            view(Mdl, 'Mode', 'graph');
            saveas(gcf, sprintf('fig_%s_tree_%s.png', lower(dataset_usar), etiqueta_version));
        end
    % Discriminantes
    elseif strcmp (modelo_usar, 'pseudoquadratic')
        Mdl = fitcdiscr(X_tr, Y_tr, 'DiscrimType', 'pseudoquadratic');
    else
        Mdl = fitcdiscr(X_tr, Y_tr, 'DiscrimType', 'linear');
    end

    % --- Predicciones ---
    Predict_ts = predict(Mdl, X_ts);
    Predict_tr = predict(Mdl, X_tr);

    % --- Matrices de confusión ---
    CM_ts = confusionmat(Y_ts, Predict_ts);
    CM_tr = confusionmat(Y_tr, Predict_tr);

    % --- Métricas por clase ---
    for c = 1:NumClass
        [Recall_ts(i,c), Spec_ts(i,c), Precision_ts(i,c), ~, ACC_ts(i,c), F1_ts(i,c)] = ...
            performanceIndexes(CM_ts, c);
        [Recall_tr(i,c), Spec_tr(i,c), Precision_tr(i,c), ~, ACC_tr(i,c), F1_tr(i,c)] = ...
            performanceIndexes(CM_tr, c);
    end
end

% ==========================================
% 3.4. CÁLCULO DE ESTADÍSTICOS FINALES
% ==========================================

% Media entre clases por fold → vector [K x 1]
F1_ts_folds  = mean(F1_ts,  2);
F1_tr_folds  = mean(F1_tr,  2);
ACC_ts_folds = mean(ACC_ts, 2);

% Estadísticos globales (media y std sobre los K folds)
% TEST
mean_F1_ts        = mean(F1_ts_folds);
std_F1_ts         = std(F1_ts_folds);
mean_ACC_ts       = mean(ACC_ts_folds);
std_ACC_ts        = std(ACC_ts_folds);
mean_Recall_ts    = mean(mean(Recall_ts,    2));
std_Recall_ts     = std( mean(Recall_ts,    2));
mean_Spec_ts      = mean(mean(Spec_ts,      2));
std_Spec_ts       = std( mean(Spec_ts,      2));
mean_Precision_ts = mean(mean(Precision_ts, 2));
std_Precision_ts  = std( mean(Precision_ts, 2));

% TRAIN (para detectar sobreajuste)
mean_F1_tr  = mean(F1_tr_folds);
std_F1_tr   = std(F1_tr_folds);

% ==========================================
% 3.5. RESULTADOS EN CONSOLA
% ==========================================
fprintf('\n========== RESULTADOS: %s | %s', dataset_usar, modelo_usar);
if strcmp(modelo_usar, 'tree'), fprintf(' (%s)', etiqueta_version); end
fprintf(' ==========\n\n');

% --- Por clase ---
fprintf('--- Métricas por clase (media ± std sobre %d folds) ---\n', K);
fprintf('%-12s %-18s %-18s %-18s %-18s\n', ...
    'Clase', 'Recall', 'Specificity', 'Precision', 'F1-Score');
for c = 1:NumClass
    fprintf('%-12s %.4f ± %.4f   %.4f ± %.4f   %.4f ± %.4f   %.4f ± %.4f\n', ...
        string(unique_classes(c)), ...
        mean(Recall_ts(:,c)),    std(Recall_ts(:,c)), ...
        mean(Spec_ts(:,c)),      std(Spec_ts(:,c)), ...
        mean(Precision_ts(:,c)), std(Precision_ts(:,c)), ...
        mean(F1_ts(:,c)),        std(F1_ts(:,c)));
end

% --- Globales ---
fprintf('\n--- Métricas globales (media ± std) ---\n');
fprintf('  Accuracy  : %.4f ± %.4f\n', mean_ACC_ts,       std_ACC_ts);
fprintf('  F1-Score  : %.4f ± %.4f\n', mean_F1_ts,        std_F1_ts);
fprintf('  Recall    : %.4f ± %.4f\n', mean_Recall_ts,    std_Recall_ts);
fprintf('  Specificity: %.4f ± %.4f\n', mean_Spec_ts,     std_Spec_ts);
fprintf('  Precision : %.4f ± %.4f\n', mean_Precision_ts, std_Precision_ts);

% --- Train vs Test (sobreajuste) ---
fprintf('\n--- Comparación Train vs Test (F1-Score) ---\n');
fprintf('  F1 Train : %.4f ± %.4f\n', mean_F1_tr, std_F1_tr);
fprintf('  F1 Test  : %.4f ± %.4f\n', mean_F1_ts, std_F1_ts);
gap = mean_F1_tr - mean_F1_ts;
if gap > 0.05
    fprintf('  x Gap Train-Test = %.4f → posible sobreajuste\n', gap);
else
    fprintf('  v Gap Train-Test = %.4f → buena generalización\n', gap);
end

% ==========================================
% 3.6. GUARDAR RESULTADOS
% ==========================================
if strcmp(modelo_usar, 'tree')
    nombre_resultados = sprintf('Resultados_%s_%s_%s.mat', ...
        dataset_usar, modelo_usar, etiqueta_version);
else
    nombre_resultados = sprintf('Resultados_%s_%s.mat', dataset_usar, modelo_usar);
end

save(nombre_resultados, ...
    'F1_ts_folds', 'F1_tr_folds', 'ACC_ts_folds', ...
    'Recall_ts', 'Spec_ts', 'Precision_ts', 'ACC_ts', 'F1_ts', ...
    'mean_F1_ts', 'std_F1_ts', 'mean_ACC_ts', 'std_ACC_ts', ...
    'modelo_usar', 'dataset_usar', 'etiqueta_version', 'opcion_param', 'valor_param');

fprintf('\nResultados guardados en: %s\n', nombre_resultados);