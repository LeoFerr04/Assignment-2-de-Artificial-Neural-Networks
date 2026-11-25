
% Assignment 2 - Redes Neuronais Artificiais
% Afinacao automatica dos learning rates - Concrete
% Compativel com MATLAB e GNU Octave

clear;
clc;

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

% Configuracoes das experiencias.
arquiteturas = {'Reference', 'Wide', 'Deep'};
algoritmos = {'Adam', 'Gradient Descent'};

% Seis valores numa escala logaritmica.
learning_rates = logspace(-4, -1, 6);

n_repeticoes = 5;
max_epochs = 200;
batch_size = 32;

% Para testar inicialmente apenas uma experiencia,
% manter true. Alterar para false na execucao completa.
modo_teste = false;

% Preparar a pasta dos resultados.
pasta_resultados = fullfile( ...
    pasta_projeto, 'results', 'concrete', 'tuning');

if ~exist(pasta_resultados, 'dir')
    mkdir(pasta_resultados);
end

ficheiro_resultados = fullfile( ...
    pasta_resultados, 'resultados_tuning.mat');

% Cada linha representa uma experiencia:
% arquitetura, algoritmo, taxa, repeticao,
% perda de validacao e tempo.
if exist(ficheiro_resultados, 'file')
    carregado = load(ficheiro_resultados);
    resultados = carregado.resultados;
else
    resultados = zeros(0, 6);
end

fprintf('\nINICIO DA AFINACAO - CONCRETE\n');

for a = 1:numel(arquiteturas)

    % Configurar a arquitetura.
    if a == 1
        layers = [size(Xtr, 2), 10, 1];
    elseif a == 2
        layers = [size(Xtr, 2), 100, 1];
    else
        layers = [size(Xtr, 2), repmat(10, 1, 10), 1];
    end

    fprintf('\nArquitetura: %s\n', arquiteturas{a});

    for lr_idx = 1:numel(learning_rates)

        learning_rate = learning_rates(lr_idx);

        for rep = 1:n_repeticoes

            % Inicializacao reprodutivel.
            seed_atual = 42 + rep - 1;
            rng(seed_atual);

            % Criar pesos com Xavier.
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

            % Criar objetivos.
            [objective_train, theta0] = ...
                make_network_objective( ...
                    blocks, ...
                    @(b) loss_fn(b, Xtr, ytr));

            objective_val = make_network_objective( ...
                blocks, ...
                @(b) loss_fn(b, Xva, yva));

            % Criar objetivo dos mini-batches.
            if a == 3
                objective_batch = @(theta, indices) ...
                    batch_objective_deep( ...
                        theta, indices, blocks, Xtr, ytr);
            else
                objective_batch = @(theta, indices) ...
                    batch_objective( ...
                        theta, indices, blocks, Xtr, ytr);
            end

            % Configuracoes do treino.
            opts = struct();
            opts.learning_rate = learning_rate;
            opts.batch_size = batch_size;
            opts.max_epochs = max_epochs;
            opts.seed = seed_atual;

            for alg = 1:numel(algoritmos)

                % Ignorar experiencias ja realizadas.
                ja_existe = false;

                if ~isempty(resultados)
                    ja_existe = any( ...
                        resultados(:, 1) == a & ...
                        resultados(:, 2) == alg & ...
                        resultados(:, 3) == lr_idx & ...
                        resultados(:, 4) == rep);
                end

                if ja_existe
                    continue;
                end

                fprintf( ...
                    'Arquitetura %s | %s | LR %.6g | Rep %d\n', ...
                    arquiteturas{a}, ...
                    algoritmos{alg}, ...
                    learning_rate, rep);

                inicio = tic;

                if alg == 1
                    [~, hist] = train_adam( ...
                        theta0, objective_batch, ...
                        objective_train, objective_val, ...
                        size(Xtr, 1), opts);
                else
                    [~, hist] = train_gradient_descent( ...
                        theta0, objective_batch, ...
                        objective_train, objective_val, ...
                        size(Xtr, 1), opts);
                end

                tempo = toc(inicio);
                perda = hist.best_val_loss;

                % Guardar o resultado desta experiencia.
                resultados(end + 1, :) = [ ...
                    a, alg, lr_idx, rep, perda, tempo];

                % Guardar apos cada treino.
                save(ficheiro_resultados, ...
                    'resultados', ...
                    'learning_rates', ...
                    'arquiteturas', ...
                    'algoritmos');

                fprintf( ...
                    'MSE validacao: %.6f | Tempo: %.2f s\n', ...
                    perda, tempo);

                % Modo de teste: parar apos um treino.
                if modo_teste
                    fprintf('\nTeste inicial concluido.\n');
                    return;
                end
            end
        end
    end
end

% Calcular as medias por configuracao.
fprintf('\nRESUMO DA AFINACAO\n');

for a = 1:numel(arquiteturas)
    for alg = 1:numel(algoritmos)

        melhor_media = inf;
        melhor_lr = NaN;

        for lr_idx = 1:numel(learning_rates)

            linhas = resultados( ...
                resultados(:, 1) == a & ...
                resultados(:, 2) == alg & ...
                resultados(:, 3) == lr_idx, :);

            if size(linhas, 1) == n_repeticoes && ...
                    all(isfinite(linhas(:, 5)))

                media = mean(linhas(:, 5));

                if media < melhor_media
                    melhor_media = media;
                    melhor_lr = learning_rates(lr_idx);
                end
            end
        end

        fprintf('%s | %s | LR: %.6g | MSE: %.6f\n', ...
            arquiteturas{a}, algoritmos{alg}, ...
            melhor_lr, melhor_media);
    end
end

fprintf('\nAfinacao concluida.\n');
