% ============================================================
% ARTIFICIAL NEURAL NETWORKS - ASSIGNMENT 2 (2026/27)
% GNU Octave / MATLAB - script principal + biblioteca ann_lib.m
%
% USO:
%   1) Colocar este ficheiro e ann_lib.m na raiz do repositorio.
%   2) Manter os dois CSVs nas pastas data/ e src/concrete/.
%   3) Executar: main_assignment2
%
% MODOS:
%   'reuse' (legado): avalia 540 modelos ja treinados em
%       results/concrete/final e results/melbourne/final.
%       Nao volta a treinar; exporta tabelas, baselines, figuras.
%       Os checkpoints historicos de Melbourne usaram 100 epocas
%       para Adam/GD por uma incompatibilidade de configuracao.
%       O BFGS antigo nao foi limitado a 200 avaliaces completas.
%       Por isso, 'reuse' NAO demonstra equivalencia de orcamento.
%
%   'full': executa novo tuning e treinos finais para ambos os
%       datasets com max_epochs=200, batch=32 e BFGS limitado
%       a 200 avaliaces full-batch (incluindo pesquisa linear).
%       Guarda separadamente em results_compliant/; usa checkpoints
%       para retomar execucoes. Este e o modo reproduzivel.
% ============================================================

clear;
clc;
close all;

cfg = struct();
cfg.mode = 'full';            % entrega reproduzivel; 'reuse' so para checkpoints antigos
cfg.budget = 200;            % full-data epoch equivalents
cfg.batch_size = 32;
cfg.n_repeticoes_tuning = 5;
cfg.n_repeticoes_final = 30;
cfg.learning_rates = logspace(-4,-1,6);

root = fileparts(mfilename('fullpath'));
if isempty(root), root = pwd; end
addpath(root);
L = ann_lib();

fprintf('\n==============================================\n');
fprintf('ASSIGNMENT 2 - REDES NEURONAIS ARTIFICIAIS\n');
fprintf('Modo: %s | Orcamento: %d epoch-equivalents\n', ...
    cfg.mode,cfg.budget);
fprintf('==============================================\n');

if strcmp(cfg.mode,'full')
    base_results = fullfile(root,'results_compliant');
elseif strcmp(cfg.mode,'reuse')
    base_results = fullfile(root,'results');
    fprintf(['AVISO: checkpoints historicos de Melbourne usam 100 ' ...
        'epocas de Adam/GD; BFGS nao respeitou um limite comum.\n']);
else
    error('cfg.mode deve ser reuse ou full.');
end

names = {'concrete','melbourne'};
archs = {'Reference','Wide','Deep'};
methods = {'Adam','Gradient_Descent','BFGS'};

for ds=1:numel(names)
    name = names{ds};
    fprintf('\n========== DATASET: %s ==========\n',upper(name));
    D = L.prepare_dataset(L,name,root);
    workdir = fullfile(base_results,name);
    if ~exist(workdir,'dir'),mkdir(workdir);end
    plotdir = fullfile(workdir,'plots');
    if ~exist(plotdir,'dir'),mkdir(plotdir);end

    L.run_eda(D,plotdir);

    if strcmp(cfg.mode,'full')
        % A selecao do hiperparametro so utiliza treino/validacao.
        best_rates = L.full_tuning(L,D,cfg,workdir,archs);
        L.full_final(L,D,cfg,workdir,archs,methods,best_rates);
    end

    modelsdir = fullfile(workdir,'final');
    evaldir = fullfile(workdir,'evaluation');
    if ~exist(modelsdir,'dir')
        error('Pasta dos modelos inexistente: %s',modelsdir);
    end
    if strcmp(cfg.mode,'reuse') || cfg.n_repeticoes_final==30
        L.evaluate_final(L,D,modelsdir,evaldir);
        L.plot_convergence(D,modelsdir,plotdir);
    else
        fprintf('Teste reduzido: avaliacao de 30 seeds ignorada.\n');
    end
end

fprintf('\n==============================================\n');
fprintf('ESTUDO CONCLUIDO: tabelas, baselines e graficos.\n');
fprintf('Resultados: %s\n',base_results);
fprintf('==============================================\n');


