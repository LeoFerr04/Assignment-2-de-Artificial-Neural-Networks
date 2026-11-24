% Avalia uma regressão linear com sete atrasos e duas entradas sazonais.
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'src', 'melbourne'), ...
        fullfile(project_root, 'src', 'baselines'), ...
        fullfile(project_root, 'src', 'evaluation'));

csv_path = fullfile(project_root, 'data', 'daily-minimum-temperatures-in-me.csv');
[dates, temperatures] = load_melbourne(csv_path);
p = 7;
[X, y, target_dates] = create_melbourne_windows(dates, temperatures, p);
X_seasonal = add_melbourne_seasonality(X, target_dates);
[idx_train, idx_val] = split_melbourne_chronological(target_dates, 0.70, 0.15);

% Ajusta os coeficientes e o intercepto com as amostras de treino.
[y_pred, model] = linear_regression_baseline( ...
    X_seasonal(idx_train, :), y(idx_train), X_seasonal(idx_val, :));
assert(model.n_features == 9, 'O modelo deve usar exatamente nove características.');
metrics = regression_metrics(y(idx_val), y_pred);

fprintf('Regressão linear com sazonalidade e intercepto | Melbourne\n');
fprintf('Treino %d amostras | Validação %d amostras | Características %d\n', ...
    numel(idx_train), numel(idx_val), model.n_features);
fprintf('\nMétricas de validação\n');
fprintf('  RMSE %.6f °C\n', metrics.rmse);
fprintf('  MAE  %.6f °C\n', metrics.mae);
fprintf('  R²   %.6f\n', metrics.r2);

% Compara o resultado à precisão de quatro casas decimais do relatório.
documented_rmse = 2.3008;
rmse_reproduced = strcmp(sprintf('%.4f', metrics.rmse), sprintf('%.4f', documented_rmse));
fprintf('\nRMSE documentado %.4f °C | Diferença %+.8f °C\n', ...
    documented_rmse, metrics.rmse - documented_rmse);
if rmse_reproduced
    fprintf('O RMSE coincide com o valor documentado a quatro casas decimais.\n');
else
    fprintf('Os valores de RMSE arredondados a quatro casas decimais diferem.\n');
end
