% Compara os limites de iterações do BFGS em três sementes de Melbourne.
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'src', 'melbourne'), ...
        fullfile(project_root, 'src', 'networks'), ...
        fullfile(project_root, 'src', 'optimizers'), ...
        fullfile(project_root, 'src', 'evaluation'));

p = 7;
n_hidden = 3;
seeds = [7, 42, 123];
iteration_limits = [150, 300, 600];
options = struct('grad_tol', 1e-5);

% Repete a preparação e a divisão cronológica das experiências anteriores.
csv_path = fullfile(project_root, 'data', 'daily-minimum-temperatures-in-me.csv');
[dates, temperatures] = load_melbourne(csv_path);
[X, y, target_dates] = create_melbourne_windows(dates, temperatures, p);
X_seasonal = add_melbourne_seasonality(X, target_dates);
[idx_train, idx_val] = split_melbourne_chronological(target_dates, 0.70, 0.15);
y_val = y(idx_val);

% A normalização usa os alvos de treino. O argumento de teste fica vazio.
[Xtr, Xv, ~, ytr, ~, ~, params] = normalize_melbourne_data( ...
    X_seasonal(idx_train, :), X_seasonal(idx_val, :), zeros(0, size(X_seasonal, 2)), ...
    y(idx_train), y_val, zeros(0, 1));
loss_fn = @(blocks) mlp_loss_gradient(blocks, Xtr, ytr);
rmse_results = zeros(numel(seeds), numel(iteration_limits));

fprintf('Comparação BFGS | %d neurónios | janela %d dias | tolerância %.1e\n', ...
    n_hidden, p, options.grad_tol);
fprintf('Treino %d amostras | Validação %d amostras\n', numel(idx_train), numel(idx_val));
fprintf('RMSE e MAE de validação em °C. Norma do gradiente calculada no treino.\n');
header = sprintf('%7s | %6s | %6s | %10s | %10s | %9s | %12s | %-18s | %s', ...
    'Semente', 'Limite', 'Iter.', 'RMSE', 'MAE', 'R2', 'Norma grad.', 'Estado', 'Convergiu');
fprintf('\n%s\n%s\n', header, repmat('-', 1, numel(header)));

for seed_index = 1:numel(seeds)
    seed = seeds(seed_index);

    % Cria uma inicialização por semente, partilhada pelos três limites.
    randn('seed', seed);
    W1 = 0.1 * randn(n_hidden, size(Xtr, 2));
    b1 = zeros(n_hidden, 1);
    W2 = 0.1 * randn(1, n_hidden);
    b2 = 0;
    [objective_fn, theta0] = make_network_objective({W1, b1, W2, b2}, loss_fn);

    for limit_index = 1:numel(iteration_limits)
        iteration_limit = iteration_limits(limit_index);
        options.max_iter = iteration_limit;
        [theta, info] = bfgs_optimize(objective_fn, theta0, options);

        y_pred = mlp_predict(theta, Xv, n_hidden) * params.temp_std + params.temp_mean;
        metrics = regression_metrics(y_val, y_pred);
        rmse_results(seed_index, limit_index) = metrics.rmse;
        converged = 'não';
        if info.converged
            converged = 'sim';
        end

        fprintf('%7d | %6d | %6d | %10.6f | %10.6f | %9.6f | %12.6e | %-18s | %s\n', ...
            seed, iteration_limit, info.iterations, metrics.rmse, metrics.mae, metrics.r2, ...
            info.grad_norm_history(end), info.status, converged);
    end
end

% Calcula o resumo do RMSE e o desvio-padrão amostral entre sementes.
rmse_means = mean(rmse_results, 1);
rmse_std = std(rmse_results, 0, 1);
[best_rmse, best_index] = min(rmse_means);
fprintf('\nResumo do RMSE de validação em °C\n');
fprintf('%6s | %12s | %12s\n', 'Limite', 'RMSE medio', 'Desvio-padrao');
for limit_index = 1:numel(iteration_limits)
    fprintf('%6d | %12.6f | %12.6f\n', ...
        iteration_limits(limit_index), rmse_means(limit_index), rmse_std(limit_index));
end
fprintf('\nMenor RMSE médio %.6f °C com limite de %d iterações.\n', ...
    best_rmse, iteration_limits(best_index));
