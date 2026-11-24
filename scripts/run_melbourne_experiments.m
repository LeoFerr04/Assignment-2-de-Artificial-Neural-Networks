
% Reproduz as experiencias da MLP com BFGS em Melbourne.
% Compara diferentes neuronios escondidos e sementes.
% O conjunto de teste nao e utilizado na selecao.

clear;
clc;

% 1. Configurar os caminhos
addpath('src/melbourne');
addpath('src/networks');
addpath('src/optimizers');
addpath('src/evaluation');

output_dir = 'results/melbourne';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 2. Carregar o dataset
[dates, temperatures] = load_melbourne( ...
    'data/daily-minimum-temperatures-in-me.csv');

% 3. Construir as amostras
p = 7;

[X, y, target_dates] = create_melbourne_windows( ...
    dates, temperatures, p);

[X_seasonal, ~] = add_melbourne_seasonality( ...
    X, target_dates);

% 4. Dividir cronologicamente
[idx_train, idx_val, idx_test] = ...
    split_melbourne_chronological( ...
    target_dates, 0.70, 0.15);

% 5. Normalizar com estatisticas do treino
[Xtr, Xv, ~, ytr, ~, ~, params] = ...
    normalize_melbourne_data( ...
    X_seasonal(idx_train,:), ...
    X_seasonal(idx_val,:), ...
    X_seasonal(idx_test,:), ...
    y(idx_train), y(idx_val), y(idx_test));

% 6. Configurar as experiencias
hidden_sizes = [3, 5, 8];
seeds = [7, 42, 123];

max_iterations = 150;
gradient_tolerance = 1e-5;

rmse_results = zeros(numel(hidden_sizes), numel(seeds));

% Guarda os resultados sem substituir o CSV original.
csv_path = fullfile(output_dir, ...
    'arquiteturas_validacao_reproduzido.csv');

fid = fopen(csv_path, 'w');

if fid == -1
    error('Nao foi possivel criar o CSV.');
end

cleanup = onCleanup(@() fclose(fid));

fprintf(fid, ...
    'neuronios,semente,rmse,mae,r2,iteracoes,estado\n');

disp('===== EXPERIENCIAS MLP + BFGS =====');
fprintf('Treino: %d amostras\n', numel(idx_train));
fprintf('Validacao: %d amostras\n', numel(idx_val));
fprintf('Teste reservado: %d amostras\n', numel(idx_test));

disp('Neuronios | Semente | RMSE | MAE | R2 | Estado');

% 7. Executar os nove treinos
for i = 1:numel(hidden_sizes)

    nh = hidden_sizes(i);

    for j = 1:numel(seeds)

        randn('seed', seeds(j));

        % Inicializar a rede
        W1 = 0.1 * randn(nh, size(Xtr, 2));
        b1 = zeros(nh, 1);
        W2 = 0.1 * randn(1, nh);
        b2 = 0;

        % Preparar a funcao objetivo
        loss_fn = @(blocks) ...
            mlp_loss_gradient(blocks, Xtr, ytr);

        [objective_fn, theta0] = make_network_objective( ...
            {W1, b1, W2, b2}, loss_fn);

        % Treinar com BFGS
        options = struct( ...
            'max_iter', max_iterations, ...
            'grad_tol', gradient_tolerance);

        [theta, info] = bfgs_optimize( ...
            objective_fn, theta0, options);

        % Prever na validacao
        y_pred = mlp_predict(theta, Xv, nh);

        % Recuperar graus Celsius
        y_pred = y_pred * params.temp_std ...
            + params.temp_mean;

        % Calcular metricas
        metrics = regression_metrics( ...
            y(idx_val), y_pred);

        rmse_results(i,j) = metrics.rmse;

        % Apresentar no terminal
        fprintf('%9d | %7d | %.4f | %.4f | %.4f | %s\n', ...
            nh, seeds(j), metrics.rmse, metrics.mae, ...
            metrics.r2, info.status);

        % Guardar no CSV com maior precisao
        fprintf(fid, '%d,%d,%.8f,%.8f,%.8f,%d,%s\n', ...
            nh, seeds(j), metrics.rmse, metrics.mae, ...
            metrics.r2, info.iterations, info.status);

    end
end

% 8. Fechar o CSV antes de gerar o grafico
clear cleanup;

% 9. Resumo das arquiteturas
disp('===== RESUMO FINAL =====');
disp('Neuronios | RMSE medio | Desvio padrao');

for i = 1:numel(hidden_sizes)
    fprintf('%9d | %.4f | %.4f\n', ...
        hidden_sizes(i), ...
        mean(rmse_results(i,:)), ...
        std(rmse_results(i,:)));
end

[best_rmse, best_idx] = min(mean(rmse_results, 2));

fprintf('\nMelhor arquitetura provisoria: %d neuronios\n', ...
    hidden_sizes(best_idx));

fprintf('RMSE medio: %.4f\n', best_rmse);

% 10. Gerar o grafico comparativo
png_path = fullfile(output_dir, ...
    'comparacao_arquiteturas_reproduzido.png');

fig = plot_melbourne_architectures( ...
    csv_path, png_path);

assert(ishandle(fig));
assert(exist(csv_path, 'file') == 2);
assert(exist(png_path, 'file') == 2);

disp('===== EXPERIENCIAS CONCLUIDAS =====');
disp(csv_path);
disp(png_path);
