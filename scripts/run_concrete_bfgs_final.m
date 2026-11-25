
% Assignment 2 - Redes Neuronais Artificiais
% Experiencias BFGS com o dataset Concrete
% 3 arquiteturas x 30 repeticoes = 90 treinos

clear;
clc;

% Localizar a pasta principal do projeto.
pasta_script = fileparts(mfilename('fullpath'));
pasta_projeto = fileparts(pasta_script);

addpath(genpath(pasta_projeto));

% Carregar o dataset normalizado.
ficheiro_csv = fullfile( ...
    pasta_projeto, 'src', 'concrete', 'Concrete_Data.csv');

data = concrete_preprocess(ficheiro_csv);

Xtr = data.Xtr;
ytr = data.ytr;

Xva = data.Xva;
yva = data.yva;

% Configuracoes das experiencias.
arquiteturas = {'Reference', 'Wide', 'Deep'};

n_repeticoes = 30;

% Comecar com apenas uma experiencia.
modo_teste = false;

% Configuracoes BFGS.
opts_bfgs = struct();

opts_bfgs.max_iter = 200;
opts_bfgs.grad_tol = 1e-6;
opts_bfgs.step_tol = 1e-10;
opts_bfgs.max_params = 2000;
opts_bfgs.verbose = false;

% Pasta para guardar os resultados.
pasta_resultados = fullfile( ...
    pasta_projeto, 'results', 'concrete', 'final');

if ~exist(pasta_resultados, 'dir')
    mkdir(pasta_resultados);
end

fprintf('\n====================================\n');
fprintf('EXPERIENCIAS BFGS - CONCRETE\n');
fprintf('====================================\n');

% Percorrer as tres arquiteturas.
for a = 1:numel(arquiteturas)

    if a == 1
        layers = [size(Xtr, 2), 10, 1];

    elseif a == 2
        layers = [size(Xtr, 2), 100, 1];

    else
        layers = [ ...
            size(Xtr, 2), ...
            repmat(10, 1, 10), ...
            1];
    end

    fprintf('\nArquitetura: %s\n', arquiteturas{a});

    % Executar 30 repeticoes.
    for rep = 1:n_repeticoes

        % Nome do ficheiro de cada repeticao.
        nome_ficheiro = sprintf( ...
            '%s_BFGS_rep%02d.mat', ...
            arquiteturas{a}, rep);

        caminho = fullfile( ...
            pasta_resultados, nome_ficheiro);

        % Evitar repetir experiencias terminadas.
        if exist(caminho, 'file')

            fprintf('Ja concluido: %s\n', nome_ficheiro);

            continue;
        end

        % Utilizar as mesmas sementes 1 a 30
        % que foram usadas nos outros algoritmos.
        rng(rep);

        % Inicializacao Xavier.
        n_layers = numel(layers) - 1;

        blocks = cell(1, 2 * n_layers);

        for i = 1:n_layers

            n_in = layers(i);
            n_out = layers(i + 1);

            limite = sqrt(6 / (n_in + n_out));

            W = (2 * rand(n_out, n_in) - 1) * limite;
            b = zeros(n_out, 1);

            blocks{2 * i - 1} = W;
            blocks{2 * i} = b;
        end

        % Selecionar a funcao de perda.
        if a == 3

            loss_fn = @(b, X, y) ...
                mlp_deep_loss_gradient(b, X, y);

        else

            loss_fn = @(b, X, y) ...
                mlp_loss_gradient(b, X, y);
        end

        % Criar a funcao objetivo de treino.
        [objective_train, theta0] = ...
            make_network_objective( ...
                blocks, ...
                @(b) loss_fn(b, Xtr, ytr));

        % Criar a funcao objetivo de validacao.
        objective_val = make_network_objective( ...
            blocks, ...
            @(b) loss_fn(b, Xva, yva));

        fprintf( ...
            '\n%s | BFGS | Repeticao %d/%d\n', ...
            arquiteturas{a}, rep, n_repeticoes);

        fprintf('Numero de parametros: %d\n', ...
            numel(theta0));

        % Iniciar a contagem do tempo.
        inicio = tic;

        % Treinar com BFGS e selecao por validacao.
        [theta_best, info] = bfgs_optimize( ...
            objective_train, ...
            theta0, ...
            opts_bfgs, ...
            objective_val);

        tempo = toc(inicio);

        % Avaliar o melhor modelo na validacao.
        mse_validacao = objective_val(theta_best);

        % Guardar o historico.
        history = struct();

        history.train_loss = info.loss_history;
        history.val_loss = info.val_loss_history;
        history.n_evals = info.n_evals;
        history.iterations = info.iterations;

        % Guardar resultados individualmente.
        save(caminho, ...
            'theta_best', ...
            'theta0', ...
            'history', ...
            'info', ...
            'tempo', ...
            'mse_validacao', ...
            'opts_bfgs', ...
            'layers');

        % Apresentar resultados.
        fprintf('MSE validacao: %.6f\n', ...
            mse_validacao);

        fprintf('Tempo: %.2f segundos\n', tempo);

        fprintf('Iteracoes: %d\n', ...
            info.iterations);

        fprintf('Avaliacoes: %d\n', ...
            info.n_evals);

        fprintf('Melhor iteracao: %d\n', ...
            info.best_iteration);

        fprintf('Estado: %s\n', ...
            info.status);

        % Terminar depois da primeira experiencia.
        if modo_teste

            fprintf('\nTeste inicial concluido.\n');

            return;
        end
    end
end

fprintf('\n====================================\n');
fprintf('EXPERIENCIAS BFGS CONCLUIDAS\n');
fprintf('====================================\n');
