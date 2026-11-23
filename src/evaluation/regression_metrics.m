function metrics = regression_metrics(y_true, y_pred)
    % Calcula as métricas de regressão para valores observados e previstos.
    narginchk(2, 2);
    if ~isnumeric(y_true) || ~isreal(y_true) || ~isvector(y_true) || isempty(y_true) || ...
            ~isnumeric(y_pred) || ~isreal(y_pred) || ~isvector(y_pred) || isempty(y_pred)
        error('regression_metrics:InvalidVectors', ...
            'Os valores observados e previstos devem formar vetores numéricos reais e não vazios.');
    end
    if numel(y_true) ~= numel(y_pred)
        error('regression_metrics:SizeMismatch', ...
            'Os valores observados e previstos devem ter o mesmo número de elementos.');
    end

    y_true = double(y_true(:));
    y_pred = double(y_pred(:));
    if any(isnan(y_true) | isinf(y_true) | isnan(y_pred) | isinf(y_pred))
        error('regression_metrics:NonfiniteValues', ...
            'Todos os valores observados e previstos devem ser finitos.');
    end

    errors = y_true - y_pred;
    mse = mean(errors.^2);
    mae = mean(abs(errors));

    % R² fica indefinido quando os valores observados são constantes.
    r2 = NaN;
    if any(y_true ~= y_true(1))
        deviations = y_true - mean(y_true);
        % O quadrado da razão das normas é a razão entre as somas dos quadrados.
        r2 = 1 - (norm(errors) / norm(deviations))^2;
    end

    metrics = struct('mse', mse, 'rmse', sqrt(mse), 'mae', mae, 'r2', r2);
end
