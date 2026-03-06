%% --- PASO 1: Preprocesado de datos ---
clearvars; clc; close all;

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

save('Datos_Iris_Preprocesados.mat', 'X_iris', 'Y_iris');
save('Datos_QSAR_Preprocesados.mat', 'X_qsar', 'Y_qsar');

% ==========================================================
% EVIDENCIA VISUAL: IMPORTANCIA DE LA NORMALIZACIÓN (QSAR)
% ==========================================================

% Poner a true para ver la gráfica
mostrar_grafica = false; %true

if mostrar_grafica

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
end

% ==========================================================

fprintf(['Preprocesado completado. Datos guardados en: ' ...
    'Datos_Iris_Preprocesados.mat y Datos_QSAR_Preprocesados.mat\n']);