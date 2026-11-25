
% Assignment 2 - Redes Neuronais Artificiais
% Comparacao inicial: Adam, Gradient Descent e BFGS
% Dataset Concrete - Arquitetura A (10 neuronios)

clear;
clc;
close all;

% Localizar a pasta principal do projeto.
pasta_script = fileparts(mfilename('fullpath'));
pasta_projeto = fileparts(pasta_script);

addpath(genpath(pasta_projeto));

% Carregar dados normalizados.
ficheiro_csv = fullfile( ...
    pasta_projeto, 'src', 'concrete', 'Concrete_Data.csv');

data = concrete_preprocess(ficheiro_csv);

Xtr = data.Xtr;
ytr = data.ytr;

Xva = data.Xva;
yva = data.yva;

% Configurar arquitetura A.
rng(42);

n_inputs = size(Xtr, 2);
n_hidden = 10;

% Inicializacao Xavier.
limite1 = sqrt(6 / (n_inputs + n_hidden));
limite2 = sqrt(6 / (n_hidden + 1));

W1 = (2 * rand(n_hidden, n_inputs) - 1) * limite1;
b1 = zeros(n_hidden, 1);

W2 = (2 * rand(1, n_hidden) - 1) * limite2;
b2 = 0;

initial_blocks = {W1, b1, W2, b2};

% Criar funcoes objetivo.
[objective_train, theta0] = make_network_objective( ...
    initial_blocks, ...
    @(blocks) mlp_loss_gradient(blocks, Xtr, ytr));

objective_val = make_network_objective( ...
    initial_blocks, ...
    @(blocks) mlp_loss_gradient(blocks, Xva, yva));

objective_batch = @(theta, indices) batch_objective( ...
    theta, indices, initial_blocks, Xtr, ytr);

% Configuracoes dos algoritmos.
opts = struct();
opts.batch_size = 32;
opts.max_epochs = 200;
opts.seed = 42;

% Treinar com Adam.
fprintf('\nA treinar com Adam...\n');

opts.learning_rate = 0.001;

tempo_inicio = tic;

[theta_adam, hist_adam] = train_adam( ...
    theta0, objective_batch, objective_train, ...
    objective_val, size(Xtr, 1), opts);

tempo_adam = toc(tempo_inicio);

% Treinar com Mini-batch Gradient Descent.
fprintf('A treinar com Gradient Descent...\n');

opts.learning_rate = 0.01;

tempo_inicio = tic;

[theta_gd, hist_gd] = train_gradient_descent( ...
    theta0, objective_batch, objective_train, ...
    objective_val, size(Xtr, 1), opts);

tempo_gd = toc(tempo_inicio);

% Treinar com BFGS.
fprintf('A treinar com BFGS...\n');

opts_bfgs = struct();

opts_bfgs.max_iter = 200;
opts_bfgs.grad_tol = 1e-6;
opts_bfgs.step_tol = 1e-10;
opts_bfgs.max_params = 2000;

tempo_inicio = tic;

[theta_bfgs, info_bfgs] = bfgs_optimize( ...
    objective_train, theta0, opts_bfgs);

tempo_bfgs = toc(tempo_inicio);

% Avaliar os tres modelos na validacao.
mse_adam = objective_val(theta_adam);
mse_gd = objective_val(theta_gd);
mse_bfgs = objective_val(theta_bfgs);

% Apresentar resultados.
fprintf('\n====================================\n');
fprintf('RESULTADOS - CONCRETE - REFERENCE\n');
fprintf('====================================\n');

fprintf('\nAdam:\n');
fprintf('MSE validacao: %.6f\n', mse_adam);
fprintf('Tempo: %.3f segundos\n', tempo_adam);

fprintf('\nGradient Descent:\n');
fprintf('MSE validacao: %.6f\n', mse_gd);
fprintf('Tempo: %.3f segundos\n', tempo_gd);

fprintf('\nBFGS:\n');
fprintf('MSE validacao: %.6f\n', mse_bfgs);
fprintf('Tempo: %.3f segundos\n', tempo_bfgs);
fprintf('Iteracoes: %d\n', info_bfgs.iterations);
fprintf('Avaliacoes: %d\n', info_bfgs.n_evals);
fprintf('Estado: %s\n', info_bfgs.status);

% Guardar resultados numericos.
pasta_resultados = fullfile( ...
    pasta_projeto, 'results', 'concrete');

if ~exist(pasta_resultados, 'dir')
    mkdir(pasta_resultados);
end

resultados = [
    mse_adam, tempo_adam;
    mse_gd, tempo_gd;
    mse_bfgs, tempo_bfgs
];

csvwrite(fullfile( ...
    pasta_resultados, 'comparacao_inicial.csv'), resultados);

fprintf('\nComparacao inicial concluida.\n');
