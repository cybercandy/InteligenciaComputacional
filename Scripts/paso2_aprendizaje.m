%% --- PASO 2: Aprendizaje y Evaluación ---
clearvars; clc; close all;
 rng('shuffle');

% ==========================================
% 2.1. CONFIGURACIÓN APRENDIZAJE
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

opcion_modelo = input('Elige una opción [Por defecto: Linear]: ');

if isempty(opcion_modelo)
    opcion_modelo = 1;
end

lista_modelos = {'linear', 'pseudoquadratic'};
modelo_usar = lista_modelos{opcion_modelo};

K = 10; % Número de Folds


fprintf('\n>>> Iniciando análisis para el dataset "%s" usando el modelo "%s" <<<\n\n', dataset_usar, modelo_usar);


% ==========================================
% 2.2. CARGA DE DATOS Y PREPARACIÓN
% ==========================================

% Cargamos el .mat correspondiente generado en el Paso 1
% (paso1_preparacion)
nombre_archivo_datos = sprintf('Datos_%s_Preprocesados.mat', dataset_usar);
load(nombre_archivo_datos);

% Asignación según el dataset
if strcmp(dataset_usar, 'Iris')
    DATA_X = X_iris; DATA_Y = Y_iris;
else
    DATA_X = X_qsar; DATA_Y = Y_qsar;
end


unique_classes = unique(DATA_Y);
NumClass = length(unique_classes);
CV = cvpartition(DATA_Y, 'Kfold', K);


% ==========================================
% 2.3. ENTRENAMIENTO Y EVALUACIÓN (K-Folds)
% ==========================================

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

    % Entrenar modelo
    Mdl = fitcdiscr(DATA_X(trIdx,:), DATA_Y(trIdx), 'DiscrimType', modelo_usar);

    % --- Evaluación en TEST ---
    PREDICT_TEST = predict(Mdl, DATA_X(tsIdx,:));
    CM_ts = confusionmat(DATA_Y(tsIdx), PREDICT_TEST);

    % --- Evaluación en TRAIN ---
    PREDICT_TRAIN = predict(Mdl, DATA_X(trIdx,:));
    CM_tr = confusionmat(DATA_Y(trIdx), PREDICT_TRAIN);


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
nombre_archivo_resultados = sprintf('Resultados_%s_%s.mat', dataset_usar, modelo_usar);
save(nombre_archivo_resultados, 'F1_Test_Folds', 'F1_Train_Folds', 'modelo_usar', 'dataset_usar');
fprintf('Resultados guardados en: %s\n', nombre_archivo_resultados);