
% ============================================================
% ASSIGNMENT 2 - ARTIFICIAL NEURAL NETWORKS
% Baseline de regressao linear - Concrete
%
% Usa a mesma divisao treino/validacao/teste das redes.
% Ajusta os coeficientes apenas com dados de treino.
% ============================================================

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);

addpath(genpath(project_dir));

output_dir = fullfile( ...
    project_dir, 'results', 'concrete', 'evaluation');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Carregar os dados ja divididos e normalizados.
data = concrete_preprocess();

% Ajustar a regressao linear com o conjunto de treino.
% A previsao produzida ainda esta normalizada.
tic;

[y_pred_normalizado, model] = linear_regression_baseline( ...
    data.Xtr, ...
    data.ytr, ...
    data.Xte);

tempo = toc;

% Converter previsoes e valores reais para MPa.
y_pred = y_pred_normalizado(:) * ...
    data.desvio_y + data.media_y;

y_real = data.yte_original(:);

% Calcular os erros.
erros = y_real - y_pred;

mse = mean(erros .^ 2);
rmse = sqrt(mse);
mae = mean(abs(erros));

ss_res = sum(erros .^ 2);
ss_tot = sum((y_real - mean(y_real)) .^ 2);

r2 = 1 - ss_res / ss_tot;

% Verificar resultados.
if any(~isfinite([mse, rmse, mae, r2]))
    error('O baseline produziu metricas invalidas.');
end

% Apresentar os resultados.
fprintf('\n');
fprintf('====================================\n');
fprintf('BASELINE REGRESSAO LINEAR - CONCRETE\n');
fprintf('====================================\n');

fprintf('Observacoes de teste: %d\n', numel(y_real));
fprintf('MSE:  %.4f MPa^2\n', mse);
fprintf('RMSE: %.4f MPa\n', rmse);
fprintf('MAE:  %.4f MPa\n', mae);
fprintf('R2:   %.4f\n', r2);
fprintf('Tempo de ajuste e previsao: %.4f s\n', tempo);

% Guardar dados e resultados.
save( ...
    fullfile(output_dir, 'baseline_linear.mat'), ...
    'model', ...
    'y_real', ...
    'y_pred', ...
    'mse', ...
    'rmse', ...
    'mae', ...
    'r2', ...
    'tempo' ...
);

% Guardar CSV para o relatorio.
csv_file = fullfile( ...
    output_dir, 'baseline_linear.csv');

fid = fopen(csv_file, 'w');

if fid < 0
    error('Nao foi possivel criar o ficheiro CSV.');
end

fprintf(fid, 'Modelo,MSE,RMSE,MAE,R2,Tempo\n');

fprintf(fid, ...
    'Regressao_Linear,%.8f,%.8f,%.8f,%.8f,%.6f\n', ...
    mse, rmse, mae, r2, tempo);

fclose(fid);

fprintf('\nResultados guardados em:\n%s\n', output_dir);
fprintf('====================================\n');
