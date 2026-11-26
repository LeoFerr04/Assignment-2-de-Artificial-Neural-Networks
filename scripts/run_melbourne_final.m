
function run_melbourne_final()
    % ========================================================
    % ASSIGNMENT 2 - ARTIFICIAL NEURAL NETWORKS
    % Experiencias finais - Melbourne
    %
    % Reference, Wide, Deep
    % Adam, Gradient Descent, BFGS
    % 30 repeticoes por combinacao
    %
    % Total: 270 experiencias
    % ========================================================

    clc;

    script_dir = fileparts(mfilename('fullpath'));
    project_dir = fileparts(script_dir);

    addpath(project_dir);

    lib = ann_lib();

    output_dir = fullfile( ...
        project_dir, 'results', 'melbourne', 'final');

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % ========================================================
    % MODO DE TESTE
    % ========================================================

    modo_teste = false;

    % ========================================================
    % CARREGAR E PREPARAR MELBOURNE
    % ========================================================

    csv_path = fullfile( ...
        project_dir, 'data', ...
        'daily-minimum-temperatures-in-me.csv');

    [dates, temperatures] = ...
        lib.load_melbourne(csv_path);

    p = 7;

    [X, y, target_dates] = ...
        lib.create_melbourne_windows( ...
            dates, temperatures, p);

    [X_seasonal, ~] = ...
        lib.add_melbourne_seasonality( ...
            X, target_dates);

    [idx_train, idx_val, idx_test] = ...
        lib.split_melbourne_chronological( ...
            target_dates, 0.70, 0.15);

    [Xtr, Xva, ~, ytr, yva, ~, ~] = ...
        lib.normalize_melbourne_data( ...
            X_seasonal(idx_train, :), ...
            X_seasonal(idx_val, :), ...
            X_seasonal(idx_test, :), ...
            y(idx_train), ...
            y(idx_val), ...
            y(idx_test));

    fprintf('\n');
    fprintf('====================================\n');
    fprintf('EXPERIENCIAS FINAIS - MELBOURNE\n');
    fprintf('====================================\n');

    fprintf('Treino: %d observacoes\n', size(Xtr, 1));
    fprintf('Validacao: %d observacoes\n', size(Xva, 1));
    fprintf('Teste reservado: %d observacoes\n', numel(idx_test));

    % ========================================================
    % CONFIGURACOES
    % ========================================================

    arquiteturas = { ...
        'Reference', 'Wide', 'Deep' ...
    };

    algoritmos = { ...
        'Adam', 'Gradient_Descent', 'BFGS' ...
    };

    n_inputs = size(Xtr, 2);

    layers_all = {
        [n_inputs, 10, 1], ...
        [n_inputs, 100, 1], ...
        [n_inputs, repmat(10, 1, 10), 1]
    };

    % Taxas selecionadas na afinacao de Melbourne.
    % Colunas: Adam, Gradient Descent.
    % Linhas: Reference, Wide, Deep.

    learning_rates = [ ...
        0.00630957, 0.10000000; ...
        0.00630957, 0.10000000; ...
        0.00039811, 0.02511886 ...
    ];

    n_repeticoes = 30;

    epochs = 200;
    batch_size = 32;

    fprintf('Experiencias previstas: %d\n', ...
        numel(arquiteturas) * ...
        numel(algoritmos) * n_repeticoes);

    % ========================================================
    % EXECUTAR EXPERIENCIAS
    % ========================================================

    for a = 1:numel(arquiteturas)

        arquitetura = arquiteturas{a};
        layers = layers_all{a};

        fprintf('\n====================================\n');
        fprintf('Arquitetura: %s\n', arquitetura);
        fprintf('====================================\n');

        for alg = 1:numel(algoritmos)

            algoritmo = algoritmos{alg};

            for rep = 1:n_repeticoes

                nome_ficheiro = sprintf( ...
                    '%s_%s_rep%02d.mat', ...
                    arquitetura, algoritmo, rep);

                caminho = fullfile( ...
                    output_dir, nome_ficheiro);

                % Nao repetir experiencias concluidas.
                if exist(caminho, 'file')
                    fprintf('Ja concluido: %s\n', nome_ficheiro);
                    continue;
                end

                fprintf('\n%s | %s | Repeticao %d/%d\n', ...
                    arquitetura, algoritmo, ...
                    rep, n_repeticoes);

                % Mesma inicializacao para os tres
                % algoritmos na mesma arquitetura e semente.
                rng(rep);

                n_layers = numel(layers) - 1;

                blocks = cell(1, 2 * n_layers);

                for k = 1:n_layers

                    n_in = layers(k);
                    n_out = layers(k + 1);

                    limite = sqrt(6 / (n_in + n_out));

                    W = (2 * rand(n_out, n_in) - 1) ...
                        * limite;

                    b = zeros(n_out, 1);

                    blocks{2*k - 1} = W;
                    blocks{2*k} = b;

                end

                % Preparar funcao objetivo sobre o treino.
                loss_fn = @(blocos) ...
                    lib.mlp_deep_loss_gradient( ...
                        blocos, Xtr, ytr);

                [objective_bfgs, theta0] = ...
                    lib.make_network_objective( ...
                        blocks, loss_fn);

                % Objetivo dos minibatches.
                objective_batch = @(theta, indices) ...
                    melbourne_final_loss_grad( ...
                        theta, layers, ...
                        Xtr(indices, :), ...
                        ytr(indices), lib);

                % Perda completa do treino.
                objective_train = @(theta) ...
                    melbourne_final_loss_only( ...
                        theta, layers, Xtr, ytr, lib);

                % Perda de validacao.
                objective_val = @(theta) ...
                    melbourne_final_loss_only( ...
                        theta, layers, Xva, yva, lib);

                fprintf('Numero de parametros: %d\n', ...
                    numel(theta0));

                % =================================================
                % TREINO
                % =================================================

                inicio = tic;

                info = struct();

                if strcmp(algoritmo, 'Adam') || ...
                        strcmp(algoritmo, 'Gradient_Descent')

                    learning_rate = ...
                        learning_rates(a, alg);

                    opts = struct( ...
                        'learning_rate', learning_rate, ...
                        'epochs', epochs, ...
                        'batch_size', batch_size, ...
                        'seed', 42);

                    if strcmp(algoritmo, 'Adam')

                        [theta_best, history] = ...
                            lib.train_adam( ...
                                theta0, ...
                                objective_batch, ...
                                objective_train, ...
                                objective_val, ...
                                size(Xtr, 1), ...
                                opts);

                    else

                        [theta_best, history] = ...
                            lib.train_gradient_descent( ...
                                theta0, ...
                                objective_batch, ...
                                objective_train, ...
                                objective_val, ...
                                size(Xtr, 1), ...
                                opts);

                    end

                else

                    opts = struct( ...
                        'max_iter', 200, ...
                        'grad_tol', 1e-6, ...
                        'step_tol', 1e-10, ...
                        'max_params', 2000, ...
                        'verbose', false);

                    % O quarto argumento permite selecionar
                    % os melhores pesos pela validacao.
                    [theta_best, info] = ...
                        lib.bfgs_optimize( ...
                            objective_bfgs, ...
                            theta0, ...
                            opts, ...
                            objective_val);

                    history = struct();

                    history.train_loss = ...
                        info.loss_history;

                    history.val_loss = ...
                        info.val_loss_history;

                    history.n_evals = info.n_evals;

                    history.iterations = info.iterations;

                end

                tempo = toc(inicio);

                mse_validacao = objective_val(theta_best);

                if ~isfinite(mse_validacao)
                    error( ...
                        'MSE de validacao invalido: %s', ...
                        nome_ficheiro);
                end

                % =================================================
                % GUARDAR RESULTADOS
                % =================================================

                save( ...
                    caminho, ...
                    'theta_best', ...
                    'theta0', ...
                    'history', ...
                    'info', ...
                    'tempo', ...
                    'mse_validacao', ...
                    'opts', ...
                    'layers', ...
                    'rep', ...
                    'arquitetura', ...
                    'algoritmo');

                fprintf('MSE validacao: %.6f\n', ...
                    mse_validacao);

                fprintf('Tempo: %.2f segundos\n', tempo);

                if strcmp(algoritmo, 'BFGS')

                    fprintf('Iteracoes: %d\n', ...
                        info.iterations);

                    fprintf('Avaliacoes: %d\n', ...
                        info.n_evals);

                    fprintf('Estado: %s\n', ...
                        info.status);

                end

                % Parar apos a primeira experiencia nova.
                if modo_teste
                    fprintf('\nTeste inicial concluido.\n');
                    return;
                end

            end

        end

    end

    fprintf('\n====================================\n');
    fprintf('270 EXPERIENCIAS MELBOURNE CONCLUIDAS\n');
    fprintf('====================================\n');

end


% ============================================================
% FUNCAO AUXILIAR - PERDA E GRADIENTE
% ============================================================

function [loss, gradient] = melbourne_final_loss_grad( ...
        theta, layers, X, y, lib)

    n_layers = numel(layers) - 1;

    blocks = cell(1, 2 * n_layers);

    pos = 1;

    for k = 1:n_layers

        n_in = layers(k);
        n_out = layers(k + 1);

        n_weights = n_in * n_out;

        W = reshape( ...
            theta(pos:pos + n_weights - 1), ...
            n_out, n_in);

        pos = pos + n_weights;

        b = theta(pos:pos + n_out - 1);

        pos = pos + n_out;

        blocks{2*k - 1} = W;
        blocks{2*k} = b;

    end

    [loss, grad_blocks] = ...
        lib.mlp_deep_loss_gradient( ...
            blocks, X, y);

    gradient = zeros(numel(theta), 1);

    pos = 1;

    for k = 1:numel(grad_blocks)

        g = grad_blocks{k}(:);

        gradient(pos:pos + numel(g) - 1) = g;

        pos = pos + numel(g);

    end

end


% ============================================================
% FUNCAO AUXILIAR - PERDA MSE
% ============================================================

function loss = melbourne_final_loss_only( ...
        theta, layers, X, y, lib)

    y_pred = lib.mlp_deep_predict( ...
        theta, X, layers);

    loss = mean((y_pred(:) - y(:)).^2);

end
