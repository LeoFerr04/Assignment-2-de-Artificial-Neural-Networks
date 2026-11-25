
% Assignment 2 - Redes Neuronais Artificiais
% Treino da arquitetura C (Deep) com o dataset Concrete
% 10 camadas escondidas com 10 neuronios cada

clear;
clc;
close all;

% Localizar a pasta principal do projeto.
pasta_script = fileparts(mfilename('fullpath'));
pasta_projeto = fileparts(pasta_script);

addpath(genpath(pasta_projeto));

% Carregar os dados normalizados.
ficheiro_csv = fullfile( ...
    pasta_projeto, 'src', 'concrete', 'Concrete_Data.csv');

data = concrete_preprocess(ficheiro_csv);

Xtr = data.Xtr;
ytr = data.ytr;

Xva = data.Xva;
yva = data.yva;

% Arquitetura C: 8 entradas, 10 camadas de 10 neuronios,
% e uma saida linear.
layers = [size(Xtr, 2), repmat(10, 1, 10), 1];

% Inicializacao Xavier, com semente fixa.
rng(42);

n_layers = numel(layers) - 1;
initial_blocks = cell(1, 2 * n_layers);

for i = 1:n_layers

    n_in = layers(i);
    n_out = layers(i + 1);

    limite = sqrt(6 / (n_in + n_out));

    W = (2 * rand(n_out, n_in) - 1) * limite;
    b = zeros(n_out, 1);

    initial_blocks{2 * i - 1} = W;
    initial_blocks{2 * i} = b;
end

% Criar as funcoes de perda e gradiente.
[objective_train, theta0] = make_network_objective( ...
    initial_blocks, ...
    @(blocks) mlp_deep_loss_gradient(blocks, Xtr, ytr));

objective_val = make_network_objective( ...
    initial_blocks, ...
    @(blocks) mlp_deep_loss_gradient(blocks, Xva, yva));

objective_batch = @(theta, indices) batch_objective_deep( ...
    theta, indices, initial_blocks, Xtr, ytr);

% Opcoes de treino.
opts = struct();

opts.batch_size = 32;
opts.max_epochs = 200;
opts.seed = 42;

% Treinar com Adam.
fprintf('\nA iniciar treino Deep com Adam...\n');

opts.learning_rate = 0.001;

[theta_adam, hist_adam] = train_adam( ...
    theta0, objective_batch, objective_train, ...
    objective_val, size(Xtr, 1), opts);

% Treinar com Mini-batch Gradient Descent.
fprintf('A iniciar treino Deep com Gradient Descent...\n');

opts.learning_rate = 0.01;

[theta_gd, hist_gd] = train_gradient_descent( ...
    theta0, objective_batch, objective_train, ...
    objective_val, size(Xtr, 1), opts);

% Mostrar os resultados.
fprintf('\nRESULTADOS - CONCRETE DEEP\n');

fprintf('Adam - perda de validacao: %.6f\n', ...
    hist_adam.best_val_loss);

fprintf('Gradient Descent - perda de validacao: %.6f\n', ...
    hist_gd.best_val_loss);

% Criar grafico comparativo.
figure('Name', 'Concrete - Arquitetura Deep');

plot(hist_adam.train_loss, 'LineWidth', 1.5);
hold on;

plot(hist_gd.train_loss, 'LineWidth', 1.5);

xlabel('Epocas');
ylabel('Perda de treino (MSE)');
title('Concrete - Deep: Adam vs Gradient Descent');
legend('Adam', 'Gradient Descent');
grid on;

% Guardar o grafico.
pasta_resultados = fullfile( ...
    pasta_projeto, 'results', 'concrete');

if ~exist(pasta_resultados, 'dir')
    mkdir(pasta_resultados);
end

print(fullfile(pasta_resultados, 'treino_deep.png'), ...
    '-dpng', '-r150');

fprintf('\nTreino Deep concluido.\n');

% Funcao auxiliar para avaliar mini-batches da rede Deep.
function [loss, gradient] = batch_objective_deep( ...
    theta, indices, initial_blocks, Xtr, ytr)

    objective = make_network_objective( ...
        initial_blocks, ...
        @(blocks) mlp_deep_loss_gradient( ...
            blocks, Xtr(indices, :), ytr(indices)));

    [loss, gradient] = objective(theta);
end
