function y_pred = mlp_predict(theta, X, n_hidden)
    % Calcula as previsões de um MLP com uma camada escondida.
    narginchk(3, 3);

    if ~isnumeric(theta) || ~isreal(theta) || ~iscolumn(theta) || isempty(theta) || ...
            any(isnan(theta) | isinf(theta))
        error('mlp_predict:InvalidTheta', ...
            'theta deve ser um vetor coluna numérico real, não vazio e com valores finitos.');
    end
    if ~isnumeric(X) || ~isreal(X) || ~ismatrix(X) || size(X, 2) == 0 || ...
            any(isnan(X(:)) | isinf(X(:)))
        error('mlp_predict:InvalidInputs', ...
            'X deve ser uma matriz numérica real com valores finitos e pelo menos uma característica.');
    end
    if ~isnumeric(n_hidden) || ~isreal(n_hidden) || ~isscalar(n_hidden) || ...
            isnan(n_hidden) || isinf(n_hidden) || n_hidden < 1 || n_hidden ~= fix(n_hidden)
        error('mlp_predict:InvalidHiddenSize', 'n_hidden deve ser um inteiro positivo.');
    end

    n_features = size(X, 2);
    n_hidden = double(n_hidden);
    w1_end = n_hidden * n_features;
    b1_end = w1_end + n_hidden;
    w2_end = b1_end + n_hidden;
    n_parameters = w2_end + 1;
    if numel(theta) ~= n_parameters
        error('mlp_predict:ParameterCountMismatch', ...
            'Esperava %d parâmetros para %d características e %d neurónios escondidos.', ...
            n_parameters, n_features, n_hidden);
    end

    % Reconstrói os blocos pela ordem e pelo formato usados no empacotamento por colunas.
    theta = full(double(theta));
    W1 = reshape(theta(1:w1_end), n_hidden, n_features);
    b1 = theta(w1_end + 1:b1_end);
    W2 = theta(b1_end + 1:w2_end).';
    b2 = theta(end);

    % Aplica tanh na camada escondida e usa uma saída linear.
    hidden = tanh(bsxfun(@plus, double(X) * W1.', b1.'));
    y_pred = hidden * W2.' + b2;
    if any(isnan(y_pred) | isinf(y_pred))
        error('mlp_predict:NonfinitePredictions', 'A propagação direta produziu previsões não finitas.');
    end
end
