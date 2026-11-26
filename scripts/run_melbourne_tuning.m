
function run_melbourne_tuning()
    % =========================================================
    % ASSIGNMENT 2 - ARTIFICIAL NEURAL NETWORKS
    % Hyperparameter Tuning - Melbourne
    %
    % 3 arquiteturas
    % 2 algoritmos: Adam e Gradient Descent
    % 6 learning rates
    % 5 repeticoes por configuracao
    %
    % Total: 180 experiencias
    % =========================================================

    clc;

    script_dir = fileparts(mfilename('fullpath'));
    project_dir = fileparts(script_dir);

    addpath(project_dir);

    lib = ann_lib();

    output_dir = fullfile( ...
        project_dir, 'results', 'melbourne', 'tuning');

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    checkpoint_file = fullfile( ...
        output_dir, 'resultados_tuning.mat');

    % =========================================================
    % CARREGAR MELBOURNE
    % =========================================================

    csv_path = fullfile( ...
        project_dir, 'data', ...
        'daily-minimum-temperatures-in-me.csv');

    [dates, temperatures] = ...
        lib.load_melbourne(csv_path);

    % Janela de 7 dias.
    p = 7;

    [X, y, target_dates] = ...
        lib.create_melbourne_windows( ...
            dates, temperatures, p);

    % Caracteristicas sazonais.
    [X_seasonal, ~] = ...
        lib.add_melbourne_seasonality( ...
            X, target_dates);

    % Divisao cronologica fixa.
    [idx_train, idx_val, idx_test] = ...
        lib.split_melbourne_chronological( ...
            target_dates, 0.70, 0.15);

    % Normalizacao baseada exclusivamente no treino.
    [Xtr, Xva, ~, ytr, yva, ~, ~] = ...
        lib.normalize_melbourne_data( ...
            X_seasonal(idx_train, :), ...
            X_seasonal(idx_val, :), ...
            X_seasonal(idx_test, :), ...
            y(idx_train), ...
            y(idx_val), ...
            y(idx_test));

    fprintf('\n');
    fprintf('========================================\n');
    fprintf('TUNING - MELBOURNE\n');
    fprintf('========================================\n');

    fprintf('Treino: %d observacoes\n', size(Xtr, 1));
    fprintf('Validacao: %d observacoes\n', size(Xva, 1));
    fprintf('Entradas: %d\n', size(Xtr, 2));

    % =========================================================
    % CONFIGURACOES
    % =========================================================

    arquiteturas = {'Reference', 'Wide', 'Deep'};
    algoritmos = {'Adam', 'Gradient_Descent'};

    n_inputs = size(Xtr, 2);

    layers_all = {
        [n_inputs, 10, 1], ...
        [n_inputs, 100, 1], ...
        [n_inputs, repmat(10, 1, 10), 1]
    };

    learning_rates = logspace(-4, -1, 6);

    n_repeticoes = 5;
    modo_teste = false;

    epochs = 200;
    batch_size = 32;

    % Cada entrada: arquitetura, algoritmo, taxa, repeticao.
    % O NaN representa uma experiencia por executar.
    resultados = NaN(3, 2, 6, n_repeticoes);

    tempos = NaN(3, 2, 6, n_repeticoes);

    % Retomar experiencias previamente guardadas.
    if exist(checkpoint_file, 'file')

        anterior = load(checkpoint_file);

        if isfield(anterior, 'resultados') && ...
                isequal(size(anterior.resultados), ...
                    size(resultados))

            resultados = anterior.resultados;

            if isfield(anterior, 'tempos')
                tempos = anterior.tempos;
            end

            fprintf('Checkpoint anterior carregado.\n');

        end
    end

    fprintf('Total previsto: %d experiencias\n', ...
        numel(resultados));

    % =========================================================
    % EXECUTAR TUNING
    % =========================================================

    for a = 1:numel(arquiteturas)

        layers = layers_all{a};

        fprintf('\n====================================\n');
        fprintf('Arquitetura: %s\n', arquiteturas{a});
        fprintf('====================================\n');

        for alg = 1:numel(algoritmos)

            algoritmo = algoritmos{alg};

            for lr_idx = 1:numel(learning_rates)

                learning_rate = learning_rates(lr_idx);

                for rep = 1:n_repeticoes

                    % Ignorar experiencias ja concluidas.
                    if isfinite(resultados(a,alg,lr_idx,rep))
                        continue;
                    end

                    fprintf('\n%s | %s | LR %.7f | Rep %d/%d\n', ...
                        arquiteturas{a}, ...
                        algoritmo, ...
                        learning_rate, ...
                        rep, n_repeticoes);

                    % Mesmos pesos iniciais para os
                    % algoritmos numa dada repeticao.
                    rng(rep);

                    blocks = cell(1, 2 * (numel(layers) - 1));

                    for k = 1:(numel(layers) - 1)

                        n_in = layers(k);
                        n_out = layers(k + 1);

                        limite = sqrt(6 / (n_in + n_out));

                        W = (2 * rand(n_out, n_in) - 1) ...
                            * limite;

                        b = zeros(n_out, 1);

                        blocks{2*k - 1} = W;
                        blocks{2*k} = b;

                    end

                    % Converter os parametros iniciais.
                    loss_fn = @(blocos) ...
                        lib.mlp_deep_loss_gradient( ...
                            blocos, Xtr, ytr);

                    [~, theta0] = ...
                        lib.make_network_objective( ...
                            blocks, loss_fn);

                    % Funcoes objetivo.
                    objective_batch = @(theta, indices) ...
                        melbourne_tuning_loss_grad( ...
                            theta, layers, ...
                            Xtr(indices, :), ...
                            ytr(indices), lib);

                    objective_train = @(theta) ...
                        melbourne_tuning_loss_only( ...
                            theta, layers, Xtr, ytr, lib);

                    objective_val = @(theta) ...
                        melbourne_tuning_loss_only( ...
                            theta, layers, Xva, yva, lib);

                    opts = struct( ...
                        'learning_rate', learning_rate, ...
                        'epochs', epochs, ...
                        'batch_size', batch_size, ...
                        'seed', 42);

                    inicio = tic;

                    if strcmp(algoritmo, 'Adam')

                        [theta_best, ~] = lib.train_adam( ...
                            theta0, ...
                            objective_batch, ...
                            objective_train, ...
                            objective_val, ...
                            size(Xtr, 1), ...
                            opts);

                    else

                        [theta_best, ~] = ...
                            lib.train_gradient_descent( ...
                                theta0, ...
                                objective_batch, ...
                                objective_train, ...
                                objective_val, ...
                                size(Xtr, 1), ...
                                opts);

                    end

                    tempo = toc(inicio);

                    mse_val = objective_val(theta_best);

                    resultados(a,alg,lr_idx,rep) = mse_val;
                    tempos(a,alg,lr_idx,rep) = tempo;

                    % Guardar apos cada experiencia.
                    save( ...
                        checkpoint_file, ...
                        'resultados', ...
                        'tempos', ...
                        'arquiteturas', ...
                        'algoritmos', ...
                        'learning_rates', ...
                        'n_repeticoes', ...
                        'epochs', ...
                        'batch_size');

                    fprintf('MSE validacao: %.6f\n', mse_val);
                    fprintf('Tempo: %.2f segundos\n', tempo);

                    if modo_teste
                        fprintf('\nTeste inicial concluido.\n');
                        return;
                    end

                end

            end

        end

    end

    % =========================================================
    % SELECIONAR MELHORES TAXAS
    % =========================================================

    melhores_rates = zeros(3, 2);

    fprintf('\n');
    fprintf('========================================\n');
    fprintf('MELHORES TAXAS - MELBOURNE\n');
    fprintf('========================================\n');

    for a = 1:3

        for alg = 1:2

            medias = zeros(1, numel(learning_rates));

            for lr_idx = 1:numel(learning_rates)

                valores = reshape( ...
                    resultados(a,alg,lr_idx,:), ...
                    [], 1);

                medias(lr_idx) = mean(valores);

            end

            [melhor_mse, idx] = min(medias);

            melhores_rates(a,alg) = learning_rates(idx);

            fprintf('%s | %s\n', ...
                arquiteturas{a}, algoritmos{alg});

            fprintf('Melhor LR: %.8f\n', ...
                melhores_rates(a,alg));

            fprintf('MSE medio validacao: %.6f\n\n', ...
                melhor_mse);

        end

    end

    save( ...
        checkpoint_file, ...
        'resultados', ...
        'tempos', ...
        'melhores_rates', ...
        'arquiteturas', ...
        'algoritmos', ...
        'learning_rates', ...
        'n_repeticoes', ...
        'epochs', ...
        'batch_size');

    fprintf('========================================\n');
    fprintf('TUNING MELBOURNE CONCLUIDO\n');
    fprintf('========================================\n');

end


% =============================================================
% FUNCOES AUXILIARES
% =============================================================

function [loss, gradient] = melbourne_tuning_loss_grad( ...
        theta, layers, X, y, lib)

    n_layers = numel(layers) - 1;

    blocks = cell(1, 2 * n_layers);

    pos = 1;

    for k = 1:n_layers

        n_in = layers(k);
        n_out = layers(k+1);

        n_weights = n_in * n_out;

        W = reshape( ...
            theta(pos:pos+n_weights-1), ...
            n_out, n_in);

        pos = pos + n_weights;

        b = theta(pos:pos+n_out-1);

        pos = pos + n_out;

        blocks{2*k-1} = W;
        blocks{2*k} = b;

    end

    [loss, grad_blocks] = ...
        lib.mlp_deep_loss_gradient(blocks, X, y);

    gradient = [];

    for k = 1:numel(grad_blocks)
        gradient = [gradient; grad_blocks{k}(:)];
    end

end


function loss = melbourne_tuning_loss_only( ...
        theta, layers, X, y, lib)

    y_pred = lib.mlp_deep_predict(theta, X, layers);

    loss = mean((y_pred(:) - y(:)).^2);

end
