
% Teste de integração dos algoritmos Adam e Gradient Descent.
% Compatível com GNU Octave e MATLAB.

clear;
clc;

addpath(genpath(pwd));

% Fixar a semente para garantir resultados reproduzíveis.
rng(42);

% Criar um conjunto de dados artificial.
n_samples = 100;
X = linspace(-1, 1, n_samples)';
y = 2 * X + 1;

% Dividir os dados em treino e validação.
Xtr = X(1:80, :);
ytr = y(1:80);

Xva = X(81:100, :);
yva = y(81:100);

% Criar uma rede com uma camada escondida de 10 neurónios.
n_hidden = 10;

W1 = randn(n_hidden, 1) * 0.1;
b1 = zeros(n_hidden, 1);
W2 = randn(1, n_hidden) * 0.1;
b2 = 0;

initial_blocks = {W1, b1, W2, b2};

% Converter os parâmetros da rede num vetor.
[~, theta0] = make_network_objective( ...
    initial_blocks, ...
    @(blocks) mlp_loss_gradient(blocks, Xtr, ytr));

% Criar funções de perda e gradiente.
objective_train = make_network_objective( ...
    initial_blocks, ...
    @(blocks) mlp_loss_gradient(blocks, Xtr, ytr));

objective_val = make_network_objective( ...
    initial_blocks, ...
    @(blocks) mlp_loss_gradient(blocks, Xva, yva));

% Função que recebe os índices do mini-batch.
objective_batch = @(theta, indices) batch_objective( ...
    theta, indices, initial_blocks, Xtr, ytr);

% Configuração dos algoritmos.
opts = struct( ...
    'learning_rate', 0.01, ...
    'batch_size', 32, ...
    'max_epochs', 200, ...
    'seed', 42);

% Treinar com Adam.
[theta_adam, hist_adam] = train_adam( ...
    theta0, objective_batch, objective_train, ...
    objective_val, size(Xtr, 1), opts);

% Treinar com Gradient Descent.
[theta_gd, hist_gd] = train_gradient_descent( ...
    theta0, objective_batch, objective_train, ...
    objective_val, size(Xtr, 1), opts);

% Apresentar os resultados.
fprintf('\nResultados finais:\n');
fprintf('Adam - perda de validação: %.6f\n', ...
    hist_adam.best_val_loss);

fprintf('Gradient Descent - perda de validação: %.6f\n', ...
    hist_gd.best_val_loss);

% Comparar a evolução do treino.
figure;

plot(hist_adam.train_loss, 'LineWidth', 1.5);
hold on;

plot(hist_gd.train_loss, 'LineWidth', 1.5);

xlabel('Épocas');
ylabel('Perda de treino');
title('Comparação entre Adam e Gradient Descent');
legend('Adam', 'Gradient Descent');
grid on;

% Função auxiliar para calcular o gradiente do mini-batch.
function [loss, gradient] = batch_objective( ...
    theta, indices, initial_blocks, Xtr, ytr)

    objective = make_network_objective( ...
        initial_blocks, ...
        @(blocks) mlp_loss_gradient( ...
            blocks, Xtr(indices, :), ytr(indices)));

    [loss, gradient] = objective(theta);
end
