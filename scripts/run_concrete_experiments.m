
% Assignment 2 - Redes Neuronais Artificiais
% Experiencia inicial com o dataset Concrete
% Arquitetura A: 1 camada escondida com 10 neuronios

clear;
clc;
close all;

% Adicionar as funcoes do projeto
addpath(genpath(pwd));

% Carregar os dados preparados
data = concrete_preprocess();

Xtr = data.Xtr;
ytr = data.ytr;

Xva = data.Xva;
yva = data.yva;

% Configuracao da arquitetura A
rng(42);

n_inputs = size(Xtr, 2);
n_hidden = 100;

% Inicializacao Xavier
limite1 = sqrt(6 / (n_inputs + n_hidden));
limite2 = sqrt(6 / (n_hidden + 1));

W1 = (2 * rand(n_hidden, n_inputs) - 1) * limite1;
b1 = zeros(n_hidden, 1);

W2 = (2 * rand(1, n_hidden) - 1) * limite2;
b2 = 0;

initial_blocks = {W1, b1, W2, b2};

% Converter os pesos num vetor
[objective_train, theta0] = make_network_objective( ...
    initial_blocks, ...
    @(blocks) mlp_loss_gradient(blocks, Xtr, ytr));

objective_val = make_network_objective( ...
    initial_blocks, ...
    @(blocks) mlp_loss_gradient(blocks, Xva, yva));

% Funcao para avaliar mini-batches
objective_batch = @(theta, indices) batch_objective( ...
    theta, indices, initial_blocks, Xtr, ytr);

% Configuracoes comuns
opts = struct();

opts.batch_size = 32;
opts.max_epochs = 200;
opts.seed = 42;

% Treino com Adam
fprintf('\nA iniciar treino com Adam...\n');

opts.learning_rate = 0.001;

[theta_adam, hist_adam] = train_adam( ...
    theta0, objective_batch, objective_train, ...
    objective_val, size(Xtr, 1), opts);

% Treino com Gradient Descent
fprintf('A iniciar treino com Gradient Descent...\n');

opts.learning_rate = 0.01;

[theta_gd, hist_gd] = train_gradient_descent( ...
    theta0, objective_batch, objective_train, ...
    objective_val, size(Xtr, 1), opts);

% Resultados
fprintf('\nRESULTADOS - CONCRETE\n');

fprintf('Adam - perda validacao: %.6f\n', ...
    hist_adam.best_val_loss);

fprintf('Gradient Descent - perda validacao: %.6f\n', ...
    hist_gd.best_val_loss);

% Graficos
figure('Name', 'Treino - Concrete');

plot(hist_adam.train_loss, 'LineWidth', 1.5);
hold on;

plot(hist_gd.train_loss, 'LineWidth', 1.5);

xlabel('Epocas');
ylabel('Perda de treino');
title('Concrete - Adam vs Gradient Descent');
legend('Adam', 'Gradient Descent');
grid on;

% Guardar resultados
% Localizar a pasta principal do projeto.
pasta_script = fileparts(mfilename('fullpath'));
pasta_projeto = fileparts(pasta_script);

% Criar a pasta de resultados.
pasta_resultados = fullfile(pasta_projeto, 'results', 'concrete');

if ~exist(pasta_resultados, 'dir')
    mkdir(pasta_resultados);
end

% Guardar o grafico.
nome_ficheiro = sprintf('treino_%d_neuronios.png', n_hidden);

print(fullfile(pasta_resultados, nome_ficheiro), ...
    '-dpng', '-r150');

fprintf('\nExperiencia inicial concluida.\n');
