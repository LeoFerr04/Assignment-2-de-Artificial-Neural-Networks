function [loss, gradient_blocks] = mlp_loss_gradient(blocks, X, y)
    % Calcula a perda e os gradientes de um MLP com uma camada escondida.
    narginchk(3, 3);

    if ~iscell(blocks) || ~isvector(blocks) || numel(blocks) ~= 4
        error('mlp_loss_gradient:InvalidBlocks', ...
            'blocks deve conter quatro blocos pela ordem {W1, b1, W2, b2}.');
    end
    names = {'W1', 'b1', 'W2', 'b2'};
    for k = 1:numel(blocks)
        block = blocks{k};
        if ~isnumeric(block) || ~isreal(block) || ~ismatrix(block) || isempty(block) || ...
                any(isnan(block(:)) | isinf(block(:)))
            error('mlp_loss_gradient:InvalidParameter', ...
                'O bloco %s deve ser uma matriz numérica real, não vazia e com valores finitos.', names{k});
        end
    end
    if ~isnumeric(X) || ~isreal(X) || ~ismatrix(X) || isempty(X) || ...
            any(isnan(X(:)) | isinf(X(:)))
        error('mlp_loss_gradient:InvalidInputs', ...
            'X deve ser uma matriz numérica real, não vazia e com valores finitos.');
    end
    if ~isnumeric(y) || ~isreal(y) || ~isvector(y) || isempty(y) || ...
            any(isnan(y(:)) | isinf(y(:)))
        error('mlp_loss_gradient:InvalidTargets', ...
            'y deve ser um vetor numérico real, não vazio e com valores finitos.');
    end

    n_samples = size(X, 1);
    n_features = size(X, 2);
    if numel(y) ~= n_samples
        error('mlp_loss_gradient:RowMismatch', 'Cada linha de X deve ter um alvo correspondente em y.');
    end

    W1 = double(blocks{1});
    b1 = double(blocks{2});
    W2 = double(blocks{3});
    b2 = double(blocks{4});
    n_hidden = size(W1, 1);
    if size(W1, 2) ~= n_features || ~isequal(size(b1), [n_hidden, 1]) || ...
            ~isequal(size(W2), [1, n_hidden]) || ~isscalar(b2)
        error('mlp_loss_gradient:ParameterShapeMismatch', ...
            'Esperava W1 (%d por %d), b1 (%d por 1), W2 (1 por %d) e b2 escalar.', ...
            n_hidden, n_features, n_hidden, n_hidden);
    end

    % Propagação direta com tanh na camada escondida e saída linear.
    X = double(X);
    y = double(y(:));
    hidden = tanh(bsxfun(@plus, X * W1.', b1.'));
    y_pred = hidden * W2.' + b2;
    residuals = y_pred - y;
    loss = 0.5 * mean(residuals.^2);

    % A derivada da perda inclui a divisão pelo número de amostras.
    delta_output = residuals ./ n_samples;
    delta_hidden = (delta_output * W2) .* (1 - hidden.^2);
    dW1 = delta_hidden.' * X;
    db1 = sum(delta_hidden, 1).';
    dW2 = delta_output.' * hidden;
    db2 = sum(delta_output);
    gradient_blocks = reshape({dW1, db1, dW2, db2}, size(blocks));
end
