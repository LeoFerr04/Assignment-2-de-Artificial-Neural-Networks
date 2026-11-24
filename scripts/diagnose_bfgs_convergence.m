% Compara limites de iterações com a mesma configuração de Melbourne.
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'src', 'melbourne'), ...
        fullfile(project_root, 'src', 'networks'), ...
        fullfile(project_root, 'src', 'optimizers'), ...
        fullfile(project_root, 'src', 'evaluation'));

p = 7;
n_hidden = 3;
seed = 7;
iteration_limits = [150, 300, 600];
options = struct('grad_tol', 1e-5);

% Reproduz a preparação das experiências anteriores para treino e validação.
csv_path = fullfile(project_root, 'data', 'daily-minimum-temperatures-in-me.csv');
[dates, temperatures] = load_melbourne(csv_path);
[X, y, target_dates] = create_melbourne_windows(dates, temperatures, p);
X_seasonal = add_melbourne_seasonality(X, target_dates);
[idx_train, idx_val] = split_melbourne_chronological(target_dates, 0.70, 0.15);

% A normalização usa os alvos de treino. Passa um conjunto vazio no lugar do teste.
[Xtr, Xv, ~, ytr, ~, ~, params] = normalize_melbourne_data( ...
    X_seasonal(idx_train, :), X_seasonal(idx_val, :), zeros(0, size(X_seasonal, 2)), ...
    y(idx_train), y(idx_val), zeros(0, 1));

% Mantém o gerador e a sequência de inicialização das experiências anteriores.
randn('seed', seed);
W1 = 0.1 * randn(n_hidden, size(Xtr, 2));
b1 = zeros(n_hidden, 1);
W2 = 0.1 * randn(1, n_hidden);
b2 = 0;
[objective_fn, theta0] = make_network_objective({W1, b1, W2, b2}, ...
    @(blocks) mlp_loss_gradient(blocks, Xtr, ytr));

fprintf('Diagnóstico de convergência BFGS | %d neurónios | janela %d dias | semente %d\n', ...
    n_hidden, p, seed);
fprintf('Treino %d amostras | Validação %d amostras | Tolerância do gradiente %.1e\n', ...
    numel(idx_train), numel(idx_val), options.grad_tol);
fprintf('Perda e norma do gradiente no treino. RMSE e MAE de validação em °C.\n');
header = sprintf('%6s | %8s | %13s | %12s | %10s | %10s | %9s | %-18s | %s', ...
    'Limite', 'Iter.', 'Perda final', 'Norma grad.', 'RMSE', 'MAE', 'R2', 'Estado', 'Convergiu');
fprintf('\n%s\n%s\n', header, repmat('-', 1, numel(header)));

% Todos os treinos partem do mesmo vetor de parâmetros.
for iteration_limit = iteration_limits
    options.max_iter = iteration_limit;
    [theta, info] = bfgs_optimize(objective_fn, theta0, options);

    y_pred = mlp_predict(theta, Xv, n_hidden) * params.temp_std + params.temp_mean;
    metrics = regression_metrics(y(idx_val), y_pred);
    converged = 'não';
    if info.converged
        converged = 'sim';
    end

    fprintf('%6d | %8d | %13.10f | %12.6e | %10.6f | %10.6f | %9.6f | %-18s | %s\n', ...
        iteration_limit, info.iterations, info.loss_history(end), info.grad_norm_history(end), ...
        metrics.rmse, metrics.mae, metrics.r2, info.status, converged);
end

fprintf('\nPerda inicial de treino comum aos três limites %.10f\n', info.loss_history(1));
