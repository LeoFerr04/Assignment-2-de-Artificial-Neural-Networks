
function y_pred = mlp_deep_predict(theta, X, layers)
    % Calcula as previsoes de uma rede neuronal profunda.
    % Utiliza tanh nas camadas escondidas e saida linear.
    % Compativel com MATLAB e GNU Octave.

    narginchk(3, 3);

    % Validar os dados de entrada.
    if ~isnumeric(X) || ~isreal(X) || ...
            ~ismatrix(X) || isempty(X) || ...
            any(~isfinite(X(:)))

        error('mlp_deep_predict:InvalidInputs', ...
            'X deve ser uma matriz numerica real e finita.');
    end

    % Validar a arquitetura.
    if ~isnumeric(layers) || ~isvector(layers) || ...
            numel(layers) < 3 || ...
            any(~isfinite(layers(:))) || ...
            any(layers(:) < 1) || ...
            any(layers(:) ~= fix(layers(:))) || ...
            layers(end) ~= 1 || ...
            size(X, 2) ~= layers(1)

        error('mlp_deep_predict:InvalidLayers', ...
            'A arquitetura da rede e invalida.');
    end

    % Validar os parametros.
    if ~isnumeric(theta) || ~isreal(theta) || ...
            ~isvector(theta) || isempty(theta) || ...
            any(~isfinite(theta(:)))

        error('mlp_deep_predict:InvalidTheta', ...
            'theta deve ser um vetor numerico real e finito.');
    end

    theta = double(theta(:));
    X = double(X);
    layers = double(layers(:).');

    n_layers = numel(layers) - 1;

    % Calcular o numero esperado de parametros.
    n_parameters = 0;

    for i = 1:n_layers
        n_parameters = n_parameters + ...
            layers(i + 1) * layers(i) + layers(i + 1);
    end

    if numel(theta) ~= n_parameters
        error('mlp_deep_predict:ParameterCountMismatch', ...
            'Esperava %d parametros, mas recebeu %d.', ...
            n_parameters, numel(theta));
    end

    % Propagacao direta.
    activations = X;
    pos = 1;

    for i = 1:n_layers

        n_in = layers(i);
        n_out = layers(i + 1);

        % Recuperar a matriz de pesos.
        n_weights = n_in * n_out;

        W = reshape( ...
            theta(pos:pos + n_weights - 1), ...
            n_out, n_in);

        pos = pos + n_weights;

        % Recuperar os bias.
        b = theta(pos:pos + n_out - 1);

        pos = pos + n_out;

        % Transformacao linear.
        Z = bsxfun(@plus, activations * W.', b.');

        if i < n_layers
            % Camadas escondidas.
            activations = tanh(Z);
        else
            % Camada de saida linear.
            activations = Z;
        end
    end

    y_pred = activations;

    if any(~isfinite(y_pred(:)))
        error('mlp_deep_predict:NonfinitePredictions', ...
            'A rede produziu previsoes invalidas.');
    end
end
