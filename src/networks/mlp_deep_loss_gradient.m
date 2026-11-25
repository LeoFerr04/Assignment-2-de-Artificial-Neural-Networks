
function [loss, gradient_blocks] = mlp_deep_loss_gradient(blocks, X, y)
    % Calcula a perda MSE e os gradientes de uma rede profunda.
    % Suporta qualquer numero de camadas escondidas.
    % Utiliza tanh nas camadas escondidas e saida linear.
    % Compativel com MATLAB e GNU Octave.

    n_layers = numel(blocks) / 2;

    if mod(numel(blocks), 2) ~= 0 || n_layers < 2
        error('Numero de blocos invalido.');
    end

    X = double(X);
    y = double(y(:));

    n_samples = size(X, 1);

    % Guardar as ativacoes de todas as camadas.
    activations = cell(n_layers + 1, 1);
    activations{1} = X;

    % Propagacao direta pelas camadas escondidas.
    for layer = 1:(n_layers - 1)

        W = blocks{2 * layer - 1};
        b = blocks{2 * layer};

        Z = bsxfun(@plus, activations{layer} * W.', b.');

        activations{layer + 1} = tanh(Z);
    end

    % Camada de saida linear.
    W_out = blocks{2 * n_layers - 1};
    b_out = blocks{2 * n_layers};

    y_pred = bsxfun(@plus, ...
        activations{n_layers} * W_out.', b_out.');

    % Funcao de perda: erro quadratico medio.
    residuals = y_pred - y;
    loss = mean(residuals .^ 2);

    % Inicializar os blocos dos gradientes.
    gradient_blocks = cell(size(blocks));

    % Gradiente da camada de saida.
    delta = (2 / n_samples) * residuals;

    gradient_blocks{2 * n_layers - 1} = ...
        delta.' * activations{n_layers};

    gradient_blocks{2 * n_layers} = sum(delta, 1).';

    % Retropropagacao pelas camadas escondidas.
    for layer = (n_layers - 1):-1:1

        W_next = blocks{2 * (layer + 1) - 1};

        delta = (delta * W_next) .* ...
            (1 - activations{layer + 1} .^ 2);

        gradient_blocks{2 * layer - 1} = ...
            delta.' * activations{layer};

        gradient_blocks{2 * layer} = sum(delta, 1).';
    end
end
