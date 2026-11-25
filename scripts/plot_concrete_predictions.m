
% ============================================================
% ASSIGNMENT 2 - ARTIFICIAL NEURAL NETWORKS
% Previsoes e residuos - Dataset Concrete
%
% Modelo: Wide + BFGS
% Repeticao selecionada pela mediana do MSE de validacao.
% Nao realiza novos treinos.
% ============================================================

clear;
clc;
close all;

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);

addpath(genpath(project_dir));

final_dir = fullfile( ...
    project_dir, 'results', 'concrete', 'final');

output_dir = fullfile( ...
    project_dir, 'results', 'concrete', 'plots');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% ============================================================
% SELECIONAR UMA REPETICAO REPRESENTATIVA
% ============================================================

n_repeticoes = 30;
mse_validacao = zeros(n_repeticoes, 1);

for rep = 1:n_repeticoes

    ficheiro = fullfile( ...
        final_dir, ...
        sprintf('Wide_BFGS_rep%02d.mat', rep));

    if ~exist(ficheiro, 'file')
        error('Ficheiro nao encontrado: %s', ficheiro);
    end

    modelo = load(ficheiro);

    mse_validacao(rep) = modelo.mse_validacao;

end

% Para 30 repeticoes, selecionar o valor mais proximo
% da mediana. Em caso de empate, utilizar a menor repeticao.
mediana_validacao = median(mse_validacao);

[~, repeticao_escolhida] = min( ...
    abs(mse_validacao - mediana_validacao));

fprintf('\n');
fprintf('====================================\n');
fprintf('PREVISOES E RESIDUOS - CONCRETE\n');
fprintf('====================================\n');

fprintf('Modelo: Wide + BFGS\n');
fprintf('Repeticao selecionada: %d\n', ...
    repeticao_escolhida);

fprintf('MSE de validacao: %.6f\n', ...
    mse_validacao(repeticao_escolhida));

% ============================================================
% CARREGAR MODELO E CONJUNTO DE TESTE
% ============================================================

ficheiro = fullfile( ...
    final_dir, ...
    sprintf('Wide_BFGS_rep%02d.mat', ...
        repeticao_escolhida));

modelo = load(ficheiro);

data = concrete_preprocess();

y_real = data.yte_original(:);

% Previsoes em escala normalizada.
y_normalizado = mlp_predict( ...
    modelo.theta_best, ...
    data.Xte, ...
    modelo.layers(2));

% Converter previsoes para MPa.
y_pred = ...
    y_normalizado(:) * data.desvio_y + ...
    data.media_y;

residuos = y_real - y_pred;

% Metricas da repeticao selecionada.
mse = mean(residuos .^ 2);
rmse = sqrt(mse);
mae = mean(abs(residuos));

ss_res = sum(residuos .^ 2);
ss_tot = sum((y_real - mean(y_real)) .^ 2);

r2 = 1 - ss_res / ss_tot;

fprintf('\nRESULTADOS NO TESTE\n');
fprintf('MSE: %.4f MPa^2\n', mse);
fprintf('RMSE: %.4f MPa\n', rmse);
fprintf('MAE: %.4f MPa\n', mae);
fprintf('R2: %.4f\n', r2);

% ============================================================
% GRAFICO 1 - VALORES REAIS VS PREVISTOS
% ============================================================

fig = figure( ...
    'visible', 'off', ...
    'position', [100, 100, 900, 700]);

scatter( ...
    y_real, ...
    y_pred, ...
    35, ...
    [0.20, 0.45, 0.80], ...
    'filled');

hold on;
grid on;
box on;

limite_min = min([y_real; y_pred]);
limite_max = max([y_real; y_pred]);

plot( ...
    [limite_min, limite_max], ...
    [limite_min, limite_max], ...
    'k--', ...
    'linewidth', 1.7);

xlabel('Resistencia real (MPa)');
ylabel('Resistencia prevista (MPa)');

title(sprintf( ...
    'Concrete - Wide + BFGS (repeticao %d)', ...
    repeticao_escolhida));

legend( ...
    {'Previsoes', 'Previsao perfeita'}, ...
    'location', 'northwest');

axis equal;

xlim([limite_min, limite_max]);
ylim([limite_min, limite_max]);

set(gca, 'fontsize', 12);

caminho = fullfile( ...
    output_dir, 'previsoes_vs_reais.png');

print(fig, caminho, '-dpng', '-r160');

close(fig);

fprintf('\nGuardado: previsoes_vs_reais.png\n');

% ============================================================
% GRAFICO 2 - ANALISE DOS RESIDUOS
% ============================================================

fig = figure( ...
    'visible', 'off', ...
    'position', [100, 100, 900, 700]);

scatter( ...
    y_pred, ...
    residuos, ...
    35, ...
    [0.90, 0.45, 0.25], ...
    'filled');

hold on;
grid on;
box on;

plot( ...
    [min(y_pred), max(y_pred)], ...
    [0, 0], ...
    'k--', ...
    'linewidth', 1.7);

xlabel('Resistencia prevista (MPa)');
ylabel('Residuo: real - previsto (MPa)');

title(sprintf( ...
    'Residuos - Wide + BFGS (repeticao %d)', ...
    repeticao_escolhida));

set(gca, 'fontsize', 12);

caminho = fullfile( ...
    output_dir, 'analise_residuos.png');

print(fig, caminho, '-dpng', '-r160');

close(fig);

fprintf('Guardado: analise_residuos.png\n');

% ============================================================
% GUARDAR DADOS DOS GRAFICOS
% ============================================================

save( ...
    fullfile(output_dir, 'dados_previsoes.mat'), ...
    'y_real', ...
    'y_pred', ...
    'residuos', ...
    'repeticao_escolhida', ...
    'mse', ...
    'rmse', ...
    'mae', ...
    'r2' ...
);

fprintf('\n====================================\n');
fprintf('GRAFICOS GERADOS COM SUCESSO\n');
fprintf('====================================\n');
