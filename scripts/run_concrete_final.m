
% Assignment 2 - Experiencias finais Concrete
% Adam e Mini-batch Gradient Descent
% 3 arquiteturas x 2 algoritmos x 30 repeticoes

clear;
clc;

% Localizar a pasta principal do projeto.
pasta_script = fileparts(mfilename('fullpath'));
pasta_projeto = fileparts(pasta_script);

addpath(genpath(pasta_projeto));

% Carregar os dados.
ficheiro_csv = fullfile( ...
    pasta_projeto, 'src', 'concrete', 'Concrete_Data.csv');

data = concrete_preprocess(ficheiro_csv);

Xtr = data.Xtr;
ytr = data.ytr;
Xva = data.Xva;
yva = data.yva;

% Configuracao fixa das experiencias.
arquiteturas = {'Reference', 'Wide', 'Deep'};
algoritmos = {'Adam', 'Gradient_Descent'};

% Taxas selecionadas na afinacao anterior.
learning_rates = [
    0.0251189, 0.1;
    0.00630957, 0.1;
    0.00158489, 0.1
];

n_repeticoes = 30;
max_epochs = 200;
batch_size = 32;

% Uma repeticao para verificar o funcionamento.
modo_teste = false;

% Pasta de resultados.
pasta_resultados = fullfile( ...
    pasta_projeto, 'results', 'concrete', 'final');

if ~exist(pasta_resultados, 'dir')
    mkdir(pasta_resultados);
end

fprintf('\nEXPERIENCIAS FINAIS - CONCRETE\n');

for a = 1:numel(arquiteturas)

    if a == 1
        layers = [size(Xtr, 2), 10, 1];
    elseif a == 2
        layers = [size(Xtr, 2), 100, 1];
    else
        layers = [size(Xtr, 2), repmat(10, 1, 10), 1];
    end

    for rep = 1:n_repeticoes

        % Sementes 1 a 30 para inicializacao.
        rng(rep);

        n_layers = numel(layers) - 1;
        blocks = cell(1, 2 * n_layers);

        % Inicializacao Xavier.
        for i = 1:n_layers

            n_in = layers(i);
            n_out = layers(i + 1);

            limite = sqrt(6 / (n_in + n_out));

            W = (2 * rand(n_out, n_in) - 1) * limite;
            b = zeros(n_out, 1);

            blocks{2 * i - 1} = W;
            blocks{2 * i} = b;
        end

        % Mesmos pesos iniciais para os algoritmos.
        if a == 3
            loss_fn = @(b, X, y) ...
                mlp_deep_loss_gradient(b, X, y);
        else
            loss_fn = @(b, X, y) ...
                mlp_loss_gradient(b, X, y);
        end

        [objective_train, theta0] = ...
            make_network_objective( ...
                blocks, @(b) loss_fn(b, Xtr, ytr));

        objective_val = make_network_objective( ...
            blocks, @(b) loss_fn(b, Xva, yva));

        if a == 3
            objective_batch = @(theta, indices) ...
                batch_objective_deep( ...
                    theta, indices, blocks, Xtr, ytr);
        else
            objective_batch = @(theta, indices) ...
                batch_objective( ...
                    theta, indices, blocks, Xtr, ytr);
        end

        for alg = 1:numel(algoritmos)

            nome_ficheiro = sprintf( ...
                '%s_%s_rep%02d.mat', ...
                arquiteturas{a}, algoritmos{alg}, rep);

            caminho = fullfile(pasta_resultados, nome_ficheiro);

            % Permite retomar uma execucao interrompida.
            if exist(caminho, 'file')
                fprintf('Ja concluido: %s\n', nome_ficheiro);
                continue;
            end

            opts = struct();
            opts.learning_rate = learning_rates(a, alg);
            opts.batch_size = batch_size;
            opts.max_epochs = max_epochs;

            % Ordem dos mini-batches reproduzivel.
            opts.seed = 42;

            fprintf('\n%s | %s | Repeticao %d/%d\n', ...
                arquiteturas{a}, algoritmos{alg}, ...
                rep, n_repeticoes);

            inicio = tic;

            if alg == 1
                [theta_best, history] = train_adam( ...
                    theta0, objective_batch, ...
                    objective_train, objective_val, ...
                    size(Xtr, 1), opts);
            else
                [theta_best, history] = ...
                    train_gradient_descent( ...
                        theta0, objective_batch, ...
                        objective_train, objective_val, ...
                        size(Xtr, 1), opts);
            end

            tempo = toc(inicio);

            mse_validacao = objective_val(theta_best);

            % Guardar resultados de treino e validacao.
            % O conjunto de teste nao e utilizado nesta fase.
            save(caminho, ...
                'theta_best', 'theta0', 'history', ...
                'tempo', 'mse_validacao', 'opts', 'layers');

            fprintf('MSE validacao: %.6f\n', mse_validacao);
            fprintf('Tempo: %.2f segundos\n', tempo);

            if modo_teste
                fprintf('\nTeste inicial concluido.\n');
                return;
            end
        end
    end
end

fprintf('\nExperiencias concluidas.\n');
