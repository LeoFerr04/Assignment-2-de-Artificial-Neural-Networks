% Avalia um AR(7) com intercepto nas amostras de validação de Melbourne.
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'src', 'melbourne'), ...
        fullfile(project_root, 'src', 'baselines'), ...
        fullfile(project_root, 'src', 'evaluation'));

csv_path = fullfile(project_root, 'data', 'daily-minimum-temperatures-in-me.csv');
[dates, temperatures] = load_melbourne(csv_path);
p = 7;
[X, y, target_dates] = create_melbourne_windows(dates, temperatures, p);
[idx_train, idx_val] = split_melbourne_chronological(target_dates, 0.70, 0.15);

% Ajusta o modelo com os sete atrasos de treino, em graus Celsius.
[y_pred, model] = linear_regression_baseline( ...
    X(idx_train, :), y(idx_train), X(idx_val, :));
assert(model.n_features == p, 'O AR(7) deve usar exatamente sete características.');
metrics = regression_metrics(y(idx_val), y_pred);

fprintf('AR(%d) com intercepto | Melbourne\n', p);
fprintf('Treino %d amostras | Validação %d amostras | Características %d\n', ...
    numel(idx_train), numel(idx_val), model.n_features);
fprintf('\nMétricas de validação\n');
fprintf('  RMSE %.6f °C\n', metrics.rmse);
fprintf('  MAE  %.6f °C\n', metrics.mae);
fprintf('  R²   %.6f\n', metrics.r2);
