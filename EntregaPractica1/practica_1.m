%% Práctica 1. Inteligencia Computacional
% Lucía
% Nerea
% Andrea

clear all; clc; rng('shuffle');

%% --- PASO 1: Preprocesado de datos ---
load('iris.mat');       % Carga INPUTS y OUTPUTS
load('qsar_data.mat');  % Carga INPUTS_qsar y OUTPUTS_qsar

% 1.1. Comprobar NaN
missing_values_iris = sum(isnan(INPUTS), 'all');
fprintf('Valores perdidos en iris: %d\n', missing_values_iris);
missing_values_qsar = sum(isnan(INPUTS_qsar), 'all');
fprintf('Valores perdidos en qsar: %d\n', missing_values_qsar);

% 1.2. Normalización Z-score (Limpieza)
X_iris = normalize(INPUTS, 'zscore');
Y_iris = OUTPUTS;
X_qsar = normalize(INPUTS_qsar, 'zscore');
Y_qsar = categorical(OUTPUTS_qsar);

% ==========================================================
% EVIDENCIA VISUAL: IMPORTANCIA DE LA NORMALIZACIÓN (QSAR)
% ==========================================================
figure('Name', 'Evidencia de Normalización: QSAR');

% Gráfica superior: Datos sin normalizar
subplot(2,1,1); 
boxplot(INPUTS_qsar); 
title('QSAR: Escalas Originales (Sin Normalizar)');
ylabel('Magnitud Real');
grid on;

% Gráfica inferior: Datos normalizados
subplot(2,1,2); 
boxplot(X_qsar); 
title('QSAR: Tras Normalización Z-Score (Media 0, Varianza 1)');
ylabel('Valor Tipificado');
grid on;
% ==========================================================


%% --- PASO 2. Aprendizaje  ---
modelos = {'linear', 'pseudoquadratic'}; % LDA y QDA
K = 10;
datasets = {'Iris', 'QSAR'};

% Se calcula para cada dataset
for d = 1:length(datasets)
    if d == 1
        DATA_X = X_iris; DATA_Y = Y_iris;
    else
        DATA_X = X_qsar; DATA_Y = Y_qsar;
    end

    unique_classes = unique(DATA_Y);
    NumClass = length(unique_classes);
    CV = cvpartition(DATA_Y, 'Kfold', K); % Mismos particionamientos para ambos modelos

    % Matrices para guardar la métrica global de los 10 folds para los 2 modelos
    Resultados_F1_Folds = zeros(K, length(modelos));

    % Bucle para recorrer cada modelo (LDA y QDA)
    for m = 1:length(modelos)
        Recall_ts = zeros(K, NumClass);
        Spec_ts = zeros(K, NumClass);
        Precision_ts = zeros(K, NumClass);
        ACC_ts = zeros(K, NumClass);
        F1_ts = zeros(K, NumClass);

        for i = 1:K
            trIdx = CV.training(i);
            tsIdx = CV.test(i);

            % Entrenamiento y Predicción
            Mdl = fitcdiscr(DATA_X(trIdx,:), DATA_Y(trIdx), 'DiscrimType', modelos{m});
            PREDICT_TEST = predict(Mdl, DATA_X(tsIdx,:));
            [CM_ts, ORDERCM] = confusionmat(DATA_Y(tsIdx), PREDICT_TEST);

            % Métricas por clase
            for c = 1:NumClass
                [Recall_ts(i,c), Spec_ts(i,c), Precision_ts(i,c), ~, ACC_ts(i,c), F1_ts(i,c)] = ...
                    performanceIndexes(CM_ts, c);
            end
        end

        % Promedio en la dimensión 2 (entre las clases) para tener el F1 general de cada fold
        Resultados_F1_Folds(:, m) = mean(F1_ts, 2);

        % Cálculo de medias finales
        meanACC = mean(mean(ACC_ts, 1));
        meanF1 = mean(mean(F1_ts, 1));
        meanRecall = mean(mean(Recall_ts, 1));
        meanSpec = mean(mean(Spec_ts, 1));
        meanPrecision = mean(mean(Precision_ts, 1));

        fprintf('\n--- DATASET: %s | MODELO: %s ---\n', datasets{d}, modelos{m});
        fprintf('Métrica media global -> F1: %.2f | Accuracy: %.2f | Recall: %.2f | Spec: %.2f | Precision: %.2f\n', ...
            meanF1, meanACC, meanRecall, meanSpec, meanPrecision);

        % %% --- GUARDADO DE RESULTADOS DEL MODELO ACTUAL --- %%
        nombre_archivo_modelo = sprintf('Resultados_%s_%s.mat', datasets{d}, modelos{m});
        save(nombre_archivo_modelo, 'meanACC', 'meanF1', 'meanRecall', 'meanSpec', 'meanPrecision', ...
            'ACC_ts', 'F1_ts', 'Recall_ts', 'Spec_ts', 'Precision_ts', 'ORDERCM');
    end

    %% PASO 3. Análisis de resultados y comparación de modelos

    fprintf('\n======================================================\n');
    fprintf('ANÁLISIS ESTADÍSTICO PARA DATASET: %s\n', datasets{d});
    fprintf('Comparando LDA (%s) vs QDA (%s)\n', modelos{1}, modelos{2});

    % Uso función testEstadístico
    [p] = testEstadistico(Resultados_F1_Folds, modelos);

    % Obtenemos los rendimientos pareados (10 valores por modelo)
    metric_LDA = Resultados_F1_Folds(:, 1);
    metric_QDA = Resultados_F1_Folds(:, 2);

    % Calculamos la diferencia
    diferencias = metric_LDA - metric_QDA;

    % Evitamos error si los modelos tienen exactamente el mismo rendimiento
    if std(diferencias) == 0
        fprintf('-> Las diferencias entre modelos son EXACTAMENTE cero (Varianza = 0).\n');
        fprintf('-> RESULTADO FINAL: Ambos modelos tienen exactamente el mismo rendimiento en todos los folds.\n');
    else
        % Test de Lilliefors para normalidad (h=0 significa que es Normal)

        warning('off', 'stats:lillietest:OutOfRangePLow');
        warning('off', 'stats:lillietest:OutOfRangePHigh');
        [h_norm, p_norm] = lillietest(diferencias);
        warning('on', 'stats:lillietest:OutOfRangePLow');
        warning('on', 'stats:lillietest:OutOfRangePHigh');


        if h_norm == 0
            fprintf('-> Las diferencias SIGUEN una distribución Normal (p-valor Lilliefors = %.4f >= 0.05).\n', p_norm);
            fprintf('-> Se procede a aplicar el TEST-T PAREADO (Paramétrico).\n');
            [h_test, p_test] = ttest(metric_LDA, metric_QDA);
            nombre_test = 'Test-t pareado';
        else
            fprintf('-> Las diferencias NO SIGUEN una distribución Normal (p-valor Lilliefors = %.4f < 0.05).\n', p_norm);
            fprintf('-> Se procede a aplicar el TEST DE WILCOXON (No paramétrico).\n');
            [p_test, h_test] = signrank(metric_LDA, metric_QDA);
            nombre_test = 'Test de Wilcoxon';
        end

        % Interpretación de la significancia
        if h_test == 1
            fprintf('-> RESULTADO: Hay diferencias SIGNIFICATIVAS (p = %.4f) entre los modelos según %s.\n', p_test, nombre_test);
        else
            fprintf('-> RESULTADO: NO hay diferencias significativas (p = %.4f) entre los modelos según %s.\n', p_test, nombre_test);
        end
    end
    fprintf('======================================================\n');
end