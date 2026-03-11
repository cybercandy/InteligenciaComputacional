%% --- PASO 3: Aprendizaje y Evaluación ---
clearvars; clc; close all;
 rng('shuffle');

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
fprintf('  1. Linear (LDA)\n');
fprintf('  2. Pseudoquadratic (QDA)\n');
fprintf('  3. Decision Tree (Árboles)\n');

opcion_modelo = input('Elige una opción [Por defecto: Linear]: ');

if isempty(opcion_modelo)
    opcion_modelo = 1;
end

lista_modelos = {'linear', 'pseudoquadratic', 'tree'};
modelo_usar = lista_modelos{opcion_modelo};

% --- 3. SELECCIÓN DE VERSIÓN (Solo si es Árbol) ---
opcion_v = 1; % Valor por defecto para LDA/QDA
if strcmp(modelo_usar, 'tree')
    fprintf('\nConfiguración del Árbol de Decisión:\n');
    fprintf('  1. Por defecto (Árbol extenso/completo)\n');
    fprintf('  2. Limitar profundidad (MaxNumSplits = 15)\n');
    fprintf('  3. Limitar tamaño de hojas (MinLeafSize = 30)\n');
    opcion_v = input('Elige una versión de árbol [1-3]: ');
    if isempty(opcion_v), opcion_v = 1; end
end

% Configuración de etiquetas para Iris. Modelo árbol
resName = ''; predNames = {};
if strcmp(dataset_usar, 'Iris')
    resName = 'Iris type';
    predNames = {'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'};
end

K = 10; % Número de Folds

fprintf('\n>>> Ejecutando dataset %s | Modelo: %s (v%d) <<<\n\n', dataset_usar, modelo_usar, opcion_v);

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

% Asignación según el dataset
%if strcmp(dataset_usar, 'Iris')
%    DATA_X = X_iris; DATA_Y = Y_iris;
%else
%    DATA_X = X_qsar; DATA_Y = Y_qsar;
%end

DATA_X = X;
DATA_Y = Y;


unique_classes = unique(DATA_Y);
NumClass = length(unique_classes);
CV = cvpartition(DATA_Y, 'Kfold', K);


% ==========================================
% 3.3. ENTRENAMIENTO Y EVALUACIÓN (K-Folds)
% ==========================================

% Inicialización de métricas
Recall_tr = zeros(K, NumClass);
Spec_tr = zeros(K, NumClass);
Precision_tr = zeros(K, NumClass);
ACC_tr = zeros(K, NumClass);
F1_tr = zeros(K, NumClass);

Recall_ts = zeros(K, NumClass);
Spec_ts = zeros(K, NumClass);
Precision_ts = zeros(K, NumClass);
ACC_ts = zeros(K, NumClass);
F1_ts = zeros(K, NumClass);



for i = 1:K
    trIdx = CV.training(i);
    tsIdx = CV.test(i);
    X_tr = DATA_X(trIdx,:); Y_tr = DATA_Y(trIdx);
    X_ts = DATA_X(tsIdx,:); Y_ts = DATA_Y(tsIdx);

    % --- Entrenar modelo ---
    if strcmp(modelo_usar, 'tree')
        % Configuración según la versión elegida
        switch opcion_v
            case 1 % Por defecto
                Mdl = fitctree(X_tr, Y_tr, 'PredictorNames', predNames, 'ResponseName', resName);
            case 2 % Profundidad
                Mdl = fitctree(X_tr, Y_tr, 'MaxNumSplits', 15, 'PredictorNames', predNames);
            case 3 % Hojas
                Mdl = fitctree(X_tr, Y_tr, 'MinLeafSize', 30, 'PredictorNames', predNames);
        end
        
        % Visualización solo en el primer Fold
        if i == 1
            fprintf('\n--- Visualización del Árbol (Fold 1) ---\n');
            view(Mdl); % Modo texto
            view(Mdl, 'Mode', 'graph'); % Gráfico
        end
        % Discriminantes
    elseif strcmp (modelo_usar, 'pseudoquadratic')
        Mdl = fitcdiscr(X_tr, Y_tr, 'DiscrimType', 'pseudoquadratic');
    else
        Mdl = fitcdiscr(X_tr, Y_tr, 'DiscrimType', 'linear');

    end

    % --- Evaluación en TEST ---
    Predict_ts = predict(Mdl, X_ts);
    CM_ts = confusionmat(Y_ts, Predict_ts);

    % --- Evaluación en TRAIN ---
    Predict_tr = predict(Mdl, X_tr);
    CM_tr = confusionmat(Y_tr, Predict_tr);


    for c = 1:NumClass
        [Recall_ts(i, c), Spec_ts(i,c), Precision_ts(i,c), ~, ACC_ts(i,c), F1_ts(i,c)] = performanceIndexes(CM_ts, c);
        [Recall_tr(i, c), Spec_tr(i,c), Precision_tr(i,c), ~, ACC_tr(i,c), F1_tr(i,c)] = performanceIndexes(CM_tr, c);
    end
end

% Media de F1 entre todas las clases del fold actual
F1_Test_Folds = mean(F1_ts, 2);
F1_Train_Folds = mean(F1_tr, 2);

% Cálculo de medias finales
meanACC = mean(mean(ACC_ts, 1));
meanF1 = mean(mean(F1_ts, 1));
meanRecall = mean(mean(Recall_ts, 1));
meanSpec = mean(mean(Spec_ts, 1));
meanPrecision = mean(mean(Precision_ts, 1));


% ==========================================
% 2.4. RESULTADOS Y GUARDADO
% ==========================================
fprintf('--- RESULTADOS FINALES ---\n');
fprintf('F1-Score Medio TRAIN : %.4f\n', mean(F1_Train_Folds));
fprintf('F1-Score Medio TEST  : %.4f\n', mean(F1_Test_Folds));

% Guardamos solo lo necesario para el paso 3
nombre_archivo_resultados = sprintf('Resultados_%s_%s_v%d.mat', dataset_usar, modelo_usar, opcion_v);
save(nombre_archivo_resultados, 'F1_Test_Folds', 'F1_Train_Folds', 'modelo_usar', 'dataset_usar','opcion_v');

fprintf('--- Resultados guardados en: %s\n', nombre_archivo_resultados);