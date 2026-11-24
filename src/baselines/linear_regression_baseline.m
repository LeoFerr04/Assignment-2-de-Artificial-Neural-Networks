function [y_pred, model] = linear_regression_baseline(X_train, y_train, X_eval)
    % Ajusta uma regressão linear com termo independente e prevê novas amostras.
    narginchk(3, 3);

    if ~isnumeric(X_train) || ~isreal(X_train) || ~ismatrix(X_train) || ...
            any(isnan(X_train(:)) | isinf(X_train(:)))
        error('linear_regression_baseline:InvalidTrainingInputs', ...
            'X_train deve ser uma matriz numérica real com valores finitos.');
    end
    if ~isnumeric(y_train) || ~isreal(y_train) || ~isvector(y_train) || isempty(y_train) || ...
            any(isnan(y_train(:)) | isinf(y_train(:)))
        error('linear_regression_baseline:InvalidTargets', ...
            'y_train deve ser um vetor numérico real, não vazio e com valores finitos.');
    end
    if ~isnumeric(X_eval) || ~isreal(X_eval) || ~ismatrix(X_eval) || ...
            any(isnan(X_eval(:)) | isinf(X_eval(:)))
        error('linear_regression_baseline:InvalidEvaluationInputs', ...
            'X_eval deve ser uma matriz numérica real com valores finitos.');
    end
    if size(X_train, 1) ~= numel(y_train)
        error('linear_regression_baseline:RowMismatch', ...
            'Cada linha de X_train deve ter um valor correspondente em y_train.');
    end
    n_features = size(X_train, 2);
    if size(X_eval, 2) ~= n_features
        error('linear_regression_baseline:ColumnMismatch', ...
            'X_train e X_eval devem ter o mesmo número de colunas.');
    end

    % A coluna de uns representa o termo independente e também conta para o posto.
    design = [ones(size(X_train, 1), 1), full(double(X_train))];
    n_parameters = n_features + 1;
    if size(design, 1) < n_parameters || rank(design) < n_parameters
        error('linear_regression_baseline:RankDeficient', ...
            'O posto da matriz de treino com o termo independente é insuficiente para identificar os %d parâmetros.', ...
            n_parameters);
    end

    % Ajusta os parâmetros só com o treino e aplica-os às entradas de avaliação.
    beta = design \ full(double(y_train(:)));
    model = struct('coefficients', beta(2:end, 1), 'intercept', beta(1), ...
        'n_features', n_features);
    y_pred = double(X_eval) * model.coefficients + model.intercept;
end
