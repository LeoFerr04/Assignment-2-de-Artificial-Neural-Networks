
function lib = ann_lib()
    % ========================================================
    % ASSIGNMENT 2 - ARTIFICIAL NEURAL NETWORKS
    % Biblioteca principal das redes neuronais.
    %
    % Todas as funcoes serao integradas neste ficheiro.
    % ========================================================

    lib = struct();

    % Redes neuronais
    lib.mlp_loss_gradient = @mlp_loss_gradient;
    lib.mlp_deep_loss_gradient = @mlp_deep_loss_gradient;
    lib.mlp_predict = @mlp_predict;
    lib.mlp_deep_predict = @mlp_deep_predict;

    % Preparacao da funcao objetivo
    lib.make_network_objective = @make_network_objective;

    % Algoritmos de otimizacao
    lib.train_adam = @train_adam;
    lib.train_gradient_descent = @train_gradient_descent;
    lib.bfgs_optimize = @bfgs_optimize;
    lib.bfgs_line_search = @bfgs_line_search;

    % Pre-processamento
    lib.concrete_preprocess = @concrete_preprocess;
    lib.load_melbourne = @load_melbourne;
    lib.create_melbourne_windows = @create_melbourne_windows;
    lib.split_melbourne_chronological = ...
        @split_melbourne_chronological;
    lib.normalize_melbourne_data = ...
        @normalize_melbourne_data;
    lib.add_melbourne_seasonality = ...
        @add_melbourne_seasonality;

    % Avaliacao e baselines
    lib.regression_metrics = @regression_metrics;
    lib.linear_regression_baseline = ...
        @linear_regression_baseline;
    lib.persistence_baseline = @persistence_baseline;

end


% ========================================
% Origem: src\networks\mlp_loss_gradient.m
% ========================================
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
    loss = mean(residuals.^2);

    % A derivada da perda inclui a divisão pelo número de amostras.
    delta_output = (2 * residuals) ./ n_samples;
    delta_hidden = (delta_output * W2) .* (1 - hidden.^2);
    dW1 = delta_hidden.' * X;
    db1 = sum(delta_hidden, 1).';
    dW2 = delta_output.' * hidden;
    db2 = sum(delta_output);
    gradient_blocks = reshape({dW1, db1, dW2, db2}, size(blocks));
end


% ========================================
% Origem: src\networks\mlp_deep_loss_gradient.m
% ========================================

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


% ========================================
% Origem: src\networks\mlp_predict.m
% ========================================
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


% ========================================
% Origem: src\networks\mlp_deep_predict.m
% ========================================

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


% ========================================
% Origem: src\optimizers\make_network_objective.m
% ========================================
function [objective_fn, theta0] = make_network_objective(initial_blocks, loss_gradient_fn)
    % Prepara uma função de perda e gradiente para um otimizador vetorial.
    % initial_blocks é uma lista ordenada, por exemplo {W1, b1, W2, b2}.
    % loss_gradient_fn recebe os blocos e devolve a perda escalar e os
    % gradientes em blocos com as mesmas dimensões e pela mesma ordem.
    narginchk(2, 2);
    if ~iscell(initial_blocks) || ~isvector(initial_blocks) || isempty(initial_blocks)
        error('make_network_objective:InvalidBlocks', ...
            'Os parâmetros iniciais devem formar uma lista de blocos.');
    end
    if ~isa(loss_gradient_fn, 'function_handle')
        error('make_network_objective:InvalidEvaluator', ...
            'O avaliador deve ser um identificador de função.');
    end

    block_sizes = cell(size(initial_blocks));
    vectors = cell(size(initial_blocks));
    block_lengths = zeros(numel(initial_blocks), 1);
    for k = 1:numel(initial_blocks)
        block = initial_blocks{k};
        if ~isnumeric(block) || ~isreal(block) || isempty(block) || ...
                any(isnan(block(:)) | isinf(block(:)))
            error('make_network_objective:InvalidBlocks', ...
                'O bloco %d deve conter valores numéricos reais e finitos.', k);
        end
        block_sizes{k} = size(block);
        block_lengths(k) = numel(block);
        vectors{k} = full(double(block(:)));
    end

    % Cada bloco é empacotado por colunas, na ordem da lista.
    theta0 = vertcat(vectors{:});
    block_ends = cumsum(block_lengths);
    block_starts = [1; block_ends(1:end-1) + 1];
    objective_fn = @evaluate;

    function [loss, gradient] = evaluate(theta)
        if ~isnumeric(theta) || ~isreal(theta) || ~isequal(size(theta), size(theta0))
            error('make_network_objective:InvalidTheta', ...
                'theta deve ser um vetor coluna real com %d parâmetros.', numel(theta0));
        end

        blocks = cell(size(block_sizes));
        theta = full(double(theta));
        for block_index = 1:numel(block_sizes)
            indices = block_starts(block_index):block_ends(block_index);
            blocks{block_index} = reshape(theta(indices), block_sizes{block_index});
        end

        [loss, gradient_blocks] = loss_gradient_fn(blocks);
        if ~isnumeric(loss) || ~isreal(loss) || ~isscalar(loss)
            error('make_network_objective:InvalidLoss', 'A perda deve ser um escalar numérico real.');
        end
        if ~iscell(gradient_blocks) || ~isvector(gradient_blocks) || ...
                numel(gradient_blocks) ~= numel(block_sizes)
            error('make_network_objective:InvalidGradient', ...
                'O gradiente deve ter um bloco por cada bloco de parâmetros.');
        end

        % Mantém a ordem de theta ao empacotar os gradientes.
        % Preserva valores não finitos para a pesquisa linear poder rejeitar o passo.
        loss = full(double(loss));
        gradient = zeros(size(theta0));
        for block_index = 1:numel(block_sizes)
            block_gradient = gradient_blocks{block_index};
            if ~isnumeric(block_gradient) || ~isreal(block_gradient) || ...
                    ~isequal(size(block_gradient), block_sizes{block_index})
                error('make_network_objective:InvalidGradient', ...
                    'O bloco %d do gradiente deve ter as dimensões do parâmetro correspondente.', block_index);
            end
            indices = block_starts(block_index):block_ends(block_index);
            gradient(indices) = block_gradient(:);
        end
    end
end


% ========================================
% Origem: src\optimizers\train_adam.m
% ========================================

function [theta_best, history] = train_adam( ...
    theta0, objective_batch_fn, objective_train_fn, ...
    objective_val_fn, n_samples, opts)

    % Treino de uma rede neuronal com o algoritmo Adam.
    % objective_batch_fn(theta, indices) devolve a perda e o
    % gradiente calculados apenas nas amostras indicadas.
    % objective_train_fn(theta) devolve a perda do treino completo.
    % objective_val_fn(theta) devolve a perda da validação.

    if nargin < 6 || isempty(opts)
        opts = struct();
    end

    if ~isfield(opts, 'learning_rate')
        opts.learning_rate = 0.001;
    end

    if ~isfield(opts, 'batch_size')
        opts.batch_size = 32;
    end

    if ~isfield(opts, 'max_epochs')
        opts.max_epochs = 100;
    end

    if ~isfield(opts, 'seed')
        opts.seed = 42;
    end

    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;

    % Inicialização reprodutível.
    rng(opts.seed);

    theta = double(theta0(:));
    theta_best = theta;

    m = zeros(size(theta));
    v = zeros(size(theta));

    iteration = 0;
    best_val_loss = inf;

    history.train_loss = zeros(opts.max_epochs, 1);
    history.val_loss = zeros(opts.max_epochs, 1);
    history.epoch = (1:opts.max_epochs).';
    history.n_evals = 0;

    for epoch = 1:opts.max_epochs

        % Baralhar as amostras no início de cada época.
        order = randperm(n_samples);

        for start_idx = 1:opts.batch_size:n_samples

            end_idx = min(start_idx + opts.batch_size - 1, n_samples);
            indices = order(start_idx:end_idx);

            % Gradiente calculado sobre o mini-batch.
            [~, gradient] = objective_batch_fn(theta, indices);
            gradient = double(gradient(:));

            iteration = iteration + 1;

            % Primeiro momento.
            m = beta1 * m + (1 - beta1) * gradient;

            % Segundo momento.
            v = beta2 * v + (1 - beta2) * (gradient .^ 2);

            % Correção do enviesamento.
            m_hat = m / (1 - beta1 ^ iteration);
            v_hat = v / (1 - beta2 ^ iteration);

            % Atualização dos parâmetros.
            theta = theta - opts.learning_rate * ...
                m_hat ./ (sqrt(v_hat) + epsilon);

            history.n_evals = history.n_evals + ...
                numel(indices) / n_samples;
        end

        % Avaliação no conjunto de treino e validação.
        train_loss = objective_train_fn(theta);
        val_loss = objective_val_fn(theta);

        history.train_loss(epoch) = train_loss;
        history.val_loss(epoch) = val_loss;

        % Guardar os pesos com menor perda de validação.
        if val_loss < best_val_loss
            best_val_loss = val_loss;
            theta_best = theta;
        end
    end

    history.best_val_loss = best_val_loss;
    history.iterations = iteration;
end


% ========================================
% Origem: src\optimizers\train_gradient_descent.m
% ========================================

function [theta_best, history] = train_gradient_descent( ...
    theta0, objective_batch_fn, objective_train_fn, ...
    objective_val_fn, n_samples, opts)

    % Treino com descida do gradiente em mini-batches.
    % Compatível com MATLAB e GNU Octave.

    if nargin < 6 || isempty(opts)
        opts = struct();
    end

    if ~isfield(opts, 'learning_rate')
        opts.learning_rate = 0.01;
    end

    if ~isfield(opts, 'batch_size')
        opts.batch_size = 32;
    end

    if ~isfield(opts, 'max_epochs')
        opts.max_epochs = 100;
    end

    if ~isfield(opts, 'seed')
        opts.seed = 42;
    end

    % Inicialização reprodutível.
    rng(opts.seed);

    theta = double(theta0(:));
    theta_best = theta;
    best_val_loss = inf;

    history.train_loss = zeros(opts.max_epochs, 1);
    history.val_loss = zeros(opts.max_epochs, 1);
    history.epoch = (1:opts.max_epochs).';
    history.n_evals = 0;

    for epoch = 1:opts.max_epochs

        % Baralhar os dados no início de cada época.
        order = randperm(n_samples);

        for start_idx = 1:opts.batch_size:n_samples

            end_idx = min(start_idx + opts.batch_size - 1, n_samples);
            indices = order(start_idx:end_idx);

            % Calcular o gradiente do mini-batch.
            [~, gradient] = objective_batch_fn(theta, indices);
            gradient = double(gradient(:));

            % Atualizar os pesos.
            theta = theta - opts.learning_rate * gradient;

            % Contabilizar o custo computacional.
            history.n_evals = history.n_evals + ...
                numel(indices) / n_samples;
        end

        % Avaliar a perda de treino e validação.
        train_loss = objective_train_fn(theta);
        val_loss = objective_val_fn(theta);

        history.train_loss(epoch) = train_loss;
        history.val_loss(epoch) = val_loss;

        % Guardar os melhores pesos segundo a validação.
        if val_loss < best_val_loss
            best_val_loss = val_loss;
            theta_best = theta;
        end
    end

    history.best_val_loss = best_val_loss;
end


% ========================================
% Origem: src\optimizers\bfgs_optimize.m
% ========================================

function [theta, info] = bfgs_optimize( ...
    objective_fn, theta0, options, objective_val_fn)

    % Otimizacao BFGS com pesquisa linear de Armijo.
    % Compativel com MATLAB e GNU Octave.
    %
    % A funcao de validacao e opcional.
    % Quando fornecida, devolve os pesos com menor
    % perda de validacao durante o treino.
    %
    % Mantem o funcionamento anterior quando a
    % funcao de validacao nao e fornecida.

    narginchk(2, 4);

    if nargin < 3 || isempty(options)
        options = struct();
    end

    usar_validacao = nargin >= 4 && ...
        ~isempty(objective_val_fn);

    % Validacao das entradas.

    if ~isa(objective_fn, 'function_handle')
        error('bfgs_optimize:InvalidObjective', ...
            'objective_fn deve ser uma funcao.');
    end

    if ~isnumeric(theta0) || ...
            ~isreal(theta0) || ...
            ~iscolumn(theta0) || ...
            isempty(theta0) || ...
            any(~isfinite(theta0))

        error('bfgs_optimize:InvalidTheta', ...
            'theta0 deve ser um vetor coluna real e finito.');
    end

    if ~isstruct(options) || ~isscalar(options)
        error('bfgs_optimize:InvalidOptions', ...
            'options deve ser uma estrutura escalar.');
    end

    if usar_validacao && ...
            ~isa(objective_val_fn, 'function_handle')

        error('bfgs_optimize:InvalidValidation', ...
            'objective_val_fn deve ser uma funcao.');
    end

    % Configuracoes por defeito.

    settings = struct( ...
        'max_iter', 200, ...
        'grad_tol', 1e-6, ...
        'step_tol', 1e-10, ...
        'max_params', 2000);

    fields = fieldnames(settings);

    for k = 1:numel(fields)

        name = fields{k};

        if isfield(options, name)

            value = options.(name);

            if ~isnumeric(value) || ...
                    ~isreal(value) || ...
                    ~isscalar(value) || ...
                    ~isfinite(value)

                error('bfgs_optimize:InvalidOptions', ...
                    'O campo %s deve ser real e finito.', name);
            end

            settings.(name) = double(value);
        end
    end

    if settings.max_iter < 0 || ...
            settings.max_iter ~= fix(settings.max_iter)

        error('bfgs_optimize:InvalidOptions', ...
            'max_iter deve ser um inteiro nao negativo.');
    end

    if settings.grad_tol < 0 || settings.step_tol < 0
        error('bfgs_optimize:InvalidOptions', ...
            'As tolerancias devem ser nao negativas.');
    end

    if settings.max_params < 1 || ...
            settings.max_params ~= fix(settings.max_params)

        error('bfgs_optimize:InvalidOptions', ...
            'max_params deve ser um inteiro positivo.');
    end

    % Configuracao de mensagens.

    verbose = false;

    if isfield(options, 'verbose')

        value = options.verbose;

        if ~isscalar(value) || ...
                ~(islogical(value) || ...
                (isnumeric(value) && isreal(value) && ...
                (value == 0 || value == 1)))

            error('bfgs_optimize:InvalidOptions', ...
                'verbose deve ser um valor logico.');
        end

        verbose = logical(value);
    end

    % Configuracao da pesquisa linear.

    line_search_options = struct();

    if isfield(options, 'line_search')

        line_search_options = options.line_search;

        if ~isstruct(line_search_options) || ...
                ~isscalar(line_search_options)

            error('bfgs_optimize:InvalidOptions', ...
                'line_search deve ser uma estrutura escalar.');
        end
    end

    % Verificar o numero de parametros.

    n = numel(theta0);

    if n > settings.max_params
        error('bfgs_optimize:TooManyParameters', ...
            ['O modelo tem %d parametros e excede ' ...
             'o limite de %d.'], ...
            n, settings.max_params);
    end

    % Inicializacao.

    theta = full(double(theta0));

    [loss, gradient] = objective_fn(theta);

    if ~isnumeric(loss) || ...
            ~isreal(loss) || ...
            ~isscalar(loss) || ...
            ~isfinite(loss) || ...
            ~isnumeric(gradient) || ...
            ~isreal(gradient) || ...
            ~isequal(size(gradient), size(theta)) || ...
            any(~isfinite(gradient(:)))

        error('bfgs_optimize:InvalidObjectiveResult', ...
            'A perda e o gradiente iniciais devem ser finitos.');
    end

    loss = full(double(loss));
    gradient = full(double(gradient));

    grad_norm = norm(gradient);

    % Historico do treino.

    info = struct( ...
        'loss_history', loss, ...
        'grad_norm_history', grad_norm, ...
        'iterations', 0, ...
        'n_evals', 1, ...
        'converged', false, ...
        'status', 'max_iter', ...
        'curvature_skips', 0);

    % Novos campos para acompanhamento da validacao.

    info.val_loss_history = [];
    info.best_val_loss = NaN;
    info.best_iteration = NaN;
    info.theta_best = [];
    info.n_val_evals = 0;
    info.train_theta_final = [];

    % Avaliar a validacao inicial.

    if usar_validacao

        val_loss = objective_val_fn(theta);

        if ~isnumeric(val_loss) || ...
                ~isreal(val_loss) || ...
                ~isscalar(val_loss) || ...
                ~isfinite(val_loss)

            error('bfgs_optimize:InvalidValidationResult', ...
                'A perda de validacao inicial deve ser finita.');
        end

        val_loss = full(double(val_loss));

        info.val_loss_history = val_loss;
        info.best_val_loss = val_loss;
        info.best_iteration = 0;
        info.theta_best = theta;
        info.n_val_evals = 1;
    end

    % Verificar convergencia inicial.

    if grad_norm == 0 || grad_norm < settings.grad_tol

        info.converged = true;
        info.status = 'grad_tol';

        info.train_theta_final = theta;

        return;
    end

    % Matriz inicial da aproximacao inversa da Hessiana.

    H = eye(n);

    % Ciclo principal BFGS.

    while info.iterations < settings.max_iter

        % Direcao de descida.

        direction = -H * gradient;

        slope = gradient.' * direction;

        if ~isfinite(slope) || ...
                slope >= 0 || ...
                any(~isfinite(direction))

            % Recuperar uma direcao de descida valida.

            H = eye(n);
            direction = -gradient;
        end

        % Pesquisa linear de Armijo.

        [alpha, loss_new, gradient_new, ...
            accepted, n_evals] = bfgs_line_search( ...
                objective_fn, ...
                theta, ...
                loss, ...
                gradient, ...
                direction, ...
                line_search_options);

        info.n_evals = info.n_evals + n_evals;

        if ~accepted

            info.status = 'line_search_failed';

            break;
        end

        % Atualizacao dos parametros.

        theta_new = theta + alpha * direction;

        s = theta_new - theta;

        gradient_new = double(gradient_new);

        y = gradient_new - gradient;

        step_norm = norm(s);

        % Condicao de curvatura.

        curvature = y.' * s;

        curvature_limit = ...
            sqrt(eps) * step_norm * norm(y);

        updated = false;

        if isfinite(curvature) && ...
                curvature > curvature_limit

            rho = 1 / curvature;

            if isfinite(rho)

                % Atualizacao BFGS da inversa da Hessiana.

                scaled_step = rho * s;

                H_new = H - scaled_step * (y.' * H);

                H_new = H_new - ...
                    (H_new * y) * scaled_step.' + ...
                    scaled_step * s.';

                % Garantir simetria numerica.

                H_new = 0.5 * H_new + 0.5 * H_new.';

                if all(isfinite(H_new(:)))

                    H = H_new;
                    updated = true;
                end
            end
        end

        if ~updated
            info.curvature_skips = ...
                info.curvature_skips + 1;
        end

        % Aceitar a atualizacao.

        theta = theta_new;
        loss = double(loss_new);
        gradient = gradient_new;

        grad_norm = norm(gradient);

        info.iterations = info.iterations + 1;

        % Registar a perda de treino.

        info.loss_history(end + 1, 1) = loss;

        info.grad_norm_history(end + 1, 1) = grad_norm;

        % Avaliar e registar a perda de validacao.

        if usar_validacao

            val_loss = objective_val_fn(theta);

            if ~isnumeric(val_loss) || ...
                    ~isreal(val_loss) || ...
                    ~isscalar(val_loss) || ...
                    ~isfinite(val_loss)

                error('bfgs_optimize:InvalidValidationResult', ...
                    'A validacao produziu uma perda invalida.');
            end

            val_loss = full(double(val_loss));

            info.val_loss_history(end + 1, 1) = val_loss;

            info.n_val_evals = info.n_val_evals + 1;

            % Guardar os melhores pesos.

            if val_loss < info.best_val_loss

                info.best_val_loss = val_loss;

                info.best_iteration = info.iterations;

                info.theta_best = theta;
            end
        end

        % Mensagens de acompanhamento.

        if verbose

            fprintf( ...
                'Iteracao %d | treino %.6g | gradiente %.6g', ...
                info.iterations, loss, grad_norm);

            if usar_validacao
                fprintf(' | validacao %.6g', val_loss);
            end

            fprintf('\n');
        end

        % Verificar os criterios de paragem.

        if grad_norm == 0 || ...
                grad_norm < settings.grad_tol

            info.converged = true;

            info.status = 'grad_tol';

            break;
        end

        if step_norm <= settings.step_tol

            info.status = 'step_tol';

            break;
        end
    end

    % Guardar os parametros da ultima iteracao.

    info.train_theta_final = theta;

    % Devolver os melhores pesos de validacao.

    if usar_validacao

        theta = info.theta_best;
    end
end


% ========================================
% Origem: src\optimizers\bfgs_line_search.m
% ========================================
function [alpha, new_loss, new_gradient, success, n_evals] = bfgs_line_search(objective_fn, theta, loss, gradient, direction, options)
    % Procura um passo que satisfaça a condição de Armijo.
    % max_backtracks indica o número máximo de reduções após a tentativa inicial.
    narginchk(5, 6);
    if nargin < 6
        options = struct();
    end

    if ~isa(objective_fn, 'function_handle')
        error('bfgs_line_search:InvalidObjective', 'objective_fn deve ser um identificador de função.');
    end
    if ~isnumeric(theta) || ~isreal(theta) || ~iscolumn(theta) || isempty(theta) || ...
            any(isnan(theta) | isinf(theta))
        error('bfgs_line_search:InvalidTheta', 'theta deve ser um vetor coluna real, finito e não vazio.');
    end
    if ~isnumeric(loss) || ~isreal(loss) || ~isscalar(loss) || isnan(loss) || isinf(loss)
        error('bfgs_line_search:InvalidLoss', 'A perda atual deve ser um escalar real e finito.');
    end
    if ~isnumeric(gradient) || ~isreal(gradient) || ~isequal(size(gradient), size(theta)) || ...
            any(isnan(gradient) | isinf(gradient))
        error('bfgs_line_search:InvalidGradient', ...
            'O gradiente deve ser real, finito e ter as mesmas dimensões de theta.');
    end
    if ~isnumeric(direction) || ~isreal(direction) || ~isequal(size(direction), size(theta)) || ...
            any(isnan(direction) | isinf(direction))
        error('bfgs_line_search:InvalidDirection', ...
            'A direção deve ser real, finita e ter as mesmas dimensões de theta.');
    end

    if ~isstruct(options) || ~isscalar(options)
        error('bfgs_line_search:InvalidOptions', 'options deve ser uma estrutura escalar.');
    end
    settings = struct('alpha0', 1, 'rho', 0.5, 'c1', 1e-4, ...
        'max_backtracks', 30, 'min_alpha', 1e-12);
    fields = fieldnames(settings);
    for k = 1:numel(fields)
        name = fields{k};
        if isfield(options, name)
            value = options.(name);
            if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || isnan(value) || isinf(value)
                error('bfgs_line_search:InvalidOptions', 'O campo %s deve ser um escalar real e finito.', name);
            end
            settings.(name) = double(value);
        end
    end
    if settings.alpha0 <= 0 || settings.min_alpha <= 0
        error('bfgs_line_search:InvalidOptions', 'alpha0 e min_alpha devem ser positivos.');
    end
    if settings.rho <= 0 || settings.rho >= 1 || settings.c1 <= 0 || settings.c1 >= 1
        error('bfgs_line_search:InvalidOptions', 'rho e c1 devem estar estritamente entre 0 e 1.');
    end
    if settings.max_backtracks < 0 || settings.max_backtracks ~= fix(settings.max_backtracks)
        error('bfgs_line_search:InvalidOptions', 'max_backtracks deve ser um inteiro não negativo.');
    end

    % Mantém os valores atuais como resultado em caso de falha.
    alpha = 0;
    new_loss = loss;
    new_gradient = gradient;
    success = false;
    n_evals = 0;
    search_direction = double(direction);
    slope = double(gradient).' * search_direction;
    if isnan(slope) || isinf(slope) || slope >= 0
        return;
    end

    current_theta = double(theta);
    trial_alpha = settings.alpha0;
    n_backtracks = 0;
    while trial_alpha >= settings.min_alpha
        trial_theta = current_theta + trial_alpha * search_direction;
        if all(~isnan(trial_theta) & ~isinf(trial_theta))
            [trial_loss, trial_gradient] = objective_fn(trial_theta);
            n_evals = n_evals + 1;
            if ~isnumeric(trial_loss) || ~isreal(trial_loss) || ~isscalar(trial_loss) || ...
                    ~isnumeric(trial_gradient) || ~isreal(trial_gradient) || ...
                    ~isequal(size(trial_gradient), size(theta))
                error('bfgs_line_search:InvalidObjectiveResult', ...
                    'objective_fn deve devolver uma perda escalar real e um gradiente real com as dimensões de theta.');
            end

            % Verifica Armijo apenas para perdas e gradientes finitos.
            if ~isnan(trial_loss) && ~isinf(trial_loss) && ...
                    all(~isnan(trial_gradient) & ~isinf(trial_gradient)) && ...
                    double(trial_loss) <= double(loss) + settings.c1 * trial_alpha * slope
                alpha = trial_alpha;
                new_loss = trial_loss;
                new_gradient = trial_gradient;
                success = true;
                return;
            end
        end
        reduced_alpha = settings.rho * trial_alpha;
        if n_backtracks >= settings.max_backtracks || reduced_alpha >= trial_alpha
            break;
        end
        trial_alpha = reduced_alpha;
        n_backtracks = n_backtracks + 1;
    end
end


% ========================================
% Origem: src\concrete\concrete_preprocess.m
% ========================================

function data = concrete_preprocess(csv_path)
    % Preparação do dataset Concrete para as redes neuronais.
    % Divisão: 70% treino, 15% validação e 15% teste.
    % Normalização calculada apenas com os dados de treino.

    if nargin < 1
        csv_path = 'src/concrete/Concrete_Data.csv';
    end

    % Carregar o dataset.
    dados = dlmread(csv_path, ',', 1, 0);

    if size(dados, 2) ~= 9 || size(dados, 1) ~= 1030
        error('Dimensões inesperadas no dataset Concrete.');
    end

    if any(~isfinite(dados(:)))
        error('O dataset contém valores inválidos.');
    end

    % Separar entradas e variável de saída.
    X = dados(:, 1:8);
    y = dados(:, 9);

    % Divisão aleatória reprodutível.
    rng(42);

    n = size(X, 1);
    indices = randperm(n);

    n_train = floor(0.70 * n);
    n_val = floor(0.15 * n);

    idx_train = indices(1:n_train);
    idx_val = indices(n_train + 1:n_train + n_val);
    idx_test = indices(n_train + n_val + 1:end);

    % Separar os três conjuntos.
    Xtr = X(idx_train, :);
    Xva = X(idx_val, :);
    Xte = X(idx_test, :);

    ytr = y(idx_train);
    yva = y(idx_val);
    yte = y(idx_test);

    % Estatísticas calculadas exclusivamente no treino.
    media_X = mean(Xtr, 1);
    desvio_X = std(Xtr, 0, 1);

    media_y = mean(ytr);
    desvio_y = std(ytr);

    % Evitar divisões por zero.
    desvio_X(desvio_X == 0) = 1;

    if desvio_y == 0
        desvio_y = 1;
    end

    % Normalizar entradas.
    data.Xtr = bsxfun(@rdivide, ...
        bsxfun(@minus, Xtr, media_X), desvio_X);

    data.Xva = bsxfun(@rdivide, ...
        bsxfun(@minus, Xva, media_X), desvio_X);

    data.Xte = bsxfun(@rdivide, ...
        bsxfun(@minus, Xte, media_X), desvio_X);

    % Normalizar a variável de saída.
    data.ytr = (ytr - media_y) / desvio_y;
    data.yva = (yva - media_y) / desvio_y;
    data.yte = (yte - media_y) / desvio_y;

    % Guardar os valores originais para avaliação.
    data.ytr_original = ytr;
    data.yva_original = yva;
    data.yte_original = yte;

    % Guardar os parâmetros de normalização.
    data.media_X = media_X;
    data.desvio_X = desvio_X;
    data.media_y = media_y;
    data.desvio_y = desvio_y;

    % Guardar os índices para reproduzir a divisão.
    data.idx_train = idx_train;
    data.idx_val = idx_val;
    data.idx_test = idx_test;

    fprintf('Concrete preparado com sucesso.\n');
    fprintf('Treino: %d observações\n', numel(idx_train));
    fprintf('Validação: %d observações\n', numel(idx_val));
    fprintf('Teste: %d observações\n', numel(idx_test));
end


% ========================================
% Origem: src\melbourne\load_melbourne.m
% ========================================

function [dates, temperatures] = load_melbourne(csv_path)
    % Le o dataset Melbourne e devolve dois vetores coluna:
    % dates: datas no formato numerico MATLAB/Octave (datenum)
    % temperatures: temperaturas minimas diarias em graus Celsius

    % 1. Abrir o ficheiro CSV
    [file_id, message] = fopen(csv_path, 'r');

    if file_id == -1
        error('load_melbourne:OpenFailed', ...
            'Nao foi possivel abrir "%s": %s', csv_path, message);
    end

    % Garantir que o ficheiro e fechado automaticamente
    cleanup = onCleanup(@() fclose(file_id));

    % 2. Definir o cabecalho, rodape e formatos esperados
    expected_header = '"Date","Daily minimum temperatures in Melbourne, Australia, 1981-1990"';
    footer = 'Daily minimum temperatures in Melbourne, Australia, 1981-1990';

    date_pattern = '^[0-9]{4}-[0-9]{2}-[0-9]{2}$';
    number_pattern = '^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$';

    % 3. Inicializar os vetores e variaveis de controlo
    dates = zeros(0, 1);
    temperatures = zeros(0, 1);

    line_number = 0;
    found_header = false;
    found_footer = false;

    % 4. Ler o ficheiro linha a linha
    while true
        line = fgetl(file_id);

        % Terminar quando chegar ao fim do ficheiro
        if ~ischar(line)
            break;
        end

        line_number = line_number + 1;
        line = strtrim(line);

        % Ignorar linhas vazias
        if isempty(line)
            continue;
        end

        % 5. Validar o cabecalho
        if ~found_header
            if ~strcmp(line, expected_header)
                error('load_melbourne:InvalidHeader', ...
                    'Linha %d: cabecalho inesperado.', line_number);
            end

            found_header = true;
            continue;
        end

        % Nao permitir conteudo depois do rodape
        if found_footer
            error('load_melbourne:AfterFooter', ...
                'Linha %d: conteudo inesperado apos o rodape.', ...
                line_number);
        end

        % Identificar e ignorar o rodape
        if strcmp(line, footer)
            found_footer = true;
            continue;
        end

        % 6. Separar a data e a temperatura
        separator = find(line == ',');

        if numel(separator) ~= 1
            error('load_melbourne:InvalidRow', ...
                'Linha %d: esperava uma data e uma temperatura.', ...
                line_number);
        end

        date_text = strtrim(line(1:separator-1));
        temperature_text = strtrim(line(separator+1:end));

        % Remover as aspas exteriores, caso existam
        date_text = regexprep(date_text, '^"([^"]*)"$', '$1');
        temperature_text = regexprep(temperature_text, '^"([^"]*)"$', '$1');

        % 7. Validar o formato da data
        if isempty(regexp(date_text, date_pattern, 'once'))
            error('load_melbourne:InvalidDate', ...
                'Linha %d: data invalida "%s".', ...
                line_number, date_text);
        end

        % Extrair ano, mes e dia
        year = str2double(date_text(1:4));
        month = str2double(date_text(6:7));
        day = str2double(date_text(9:10));

        % Converter para o formato numerico de data
        serial_date = datenum(year, month, day);

        % Confirmar que a data existe no calendario
        calendar_date = datevec(serial_date);

        if ~isequal(calendar_date(1:3), [year, month, day])
            error('load_melbourne:InvalidDate', ...
                'Linha %d: data inexistente "%s".', ...
                line_number, date_text);
        end

        % 8. Validar a temperatura
        % Valores com '?' sao suspeitos e ficam representados por NaN.
        % O CSV original nao e alterado.

        if ~isempty(temperature_text) && temperature_text(1) == '?'

            % Retirar o sinal de duvida
            numeric_part = temperature_text(2:end);

            % Validar o numero que aparece depois do '?'
            if isempty(regexp(numeric_part, number_pattern, 'once')) || ...
                    ~isfinite(str2double(numeric_part))

                error('load_melbourne:InvalidTemperature', ...
                    'Linha %d: temperatura invalida "%s".', ...
                    line_number, temperature_text);
            end

            % Avisar que a observacao e suspeita
            warning('load_melbourne:SuspiciousTemperature', ...
                'Linha %d: temperatura suspeita "%s". Marcada como NaN.', ...
                line_number, temperature_text);

            temperature = NaN;

        else

            % Converter uma temperatura normal
            temperature = str2double(temperature_text);

            % Verificar se e um numero real finito
            if isempty(regexp(temperature_text, number_pattern, 'once')) || ...
                    ~isfinite(temperature)

                error('load_melbourne:InvalidTemperature', ...
                    'Linha %d: temperatura invalida "%s".', ...
                    line_number, temperature_text);
            end

        end

        % 9. Verificar se as datas estao ordenadas
        if ~isempty(dates) && serial_date <= dates(end)
            error('load_melbourne:DateOrder', ...
                'Linha %d: data repetida ou fora de ordem: %s.', ...
                line_number, date_text);
        end

        % 10. Guardar a observacao
        dates(end+1, 1) = serial_date;
        temperatures(end+1, 1) = temperature;

    end

    % 11. Verificar se houve erros de leitura
    if ~feof(file_id)
        error('load_melbourne:ReadFailed', ...
            'Erro ao ler o ficheiro "%s".', csv_path);
    end

    % Confirmar que existem observacoes
    if isempty(dates)
        error('load_melbourne:NoData', ...
            'O ficheiro nao contem observacoes.');
    end

    % 12. Detetar dias em falta na serie temporal
    day_steps = diff(dates);

    for index = find(day_steps > 1).'

        missing_days = day_steps(index) - 1;

        warning('load_melbourne:DateGap', ...
            'Existem %d dia(s) em falta entre %s e %s.', ...
            missing_days, ...
            datestr(dates(index), 'yyyy-mm-dd'), ...
            datestr(dates(index+1), 'yyyy-mm-dd'));

    end

end


% ========================================
% Origem: src\melbourne\create_melbourne_windows.m
% ========================================
function [X, y, target_dates] = create_melbourne_windows(dates, temperatures, p)
    % Constrói amostras com p temperaturas anteriores e o alvo do dia seguinte.
    narginchk(3, 3);

    if ~isnumeric(dates) || ~isreal(dates) || (~isvector(dates) && ~isempty(dates))
        error('create_melbourne_windows:InvalidDates', ...
            'As datas devem formar um vetor numérico real.');
    end
    if ~isnumeric(temperatures) || ~isreal(temperatures) || ...
            (~isvector(temperatures) && ~isempty(temperatures))
        error('create_melbourne_windows:InvalidTemperatures', ...
            'As temperaturas devem formar um vetor numérico real.');
    end
    if numel(dates) ~= numel(temperatures)
        error('create_melbourne_windows:SizeMismatch', ...
            'Os vetores de datas e temperaturas devem ter o mesmo comprimento.');
    end
    if ~isnumeric(p) || ~isreal(p) || ~isscalar(p) || ...
            isnan(p) || isinf(p) || p < 1 || p ~= fix(p)
        error('create_melbourne_windows:InvalidWindowSize', 'p deve ser um inteiro positivo.');
    end

    serial_dates = dates(:);
    values = temperatures(:);
    if any(isnan(serial_dates) | isinf(serial_dates) | serial_dates ~= fix(serial_dates))
        error('create_melbourne_windows:InvalidDates', ...
            'As datas devem ser números de série inteiros e finitos.');
    end
    if any(serial_dates(2:end) <= serial_dates(1:end-1))
        error('create_melbourne_windows:DateOrder', ...
            'As datas devem estar em ordem estritamente crescente.');
    end
    if any(isinf(values))
        error('create_melbourne_windows:InvalidTemperatures', ...
            'As temperaturas devem ser números reais finitos ou NaN.');
    end

    p = double(p);
    X = zeros(0, p, class(values));
    y = zeros(0, 1, class(values));
    target_dates = zeros(0, 1, class(serial_dates));
    if numel(values) <= p
        return;
    end

    % As contagens cumulativas permitem verificar a janela inteira, incluindo o alvo.
    gap_count = [0; cumsum(diff(serial_dates) ~= 1)];
    missing_count = [0; cumsum(isnan(values))];
    starts = (1:numel(values)-p).';
    targets = starts + p;
    valid = gap_count(targets) == gap_count(starts) & ...
        missing_count(targets + 1) == missing_count(starts);
    starts = starts(valid);
    targets = targets(valid);
    if isempty(starts)
        return;
    end

    % Preenche as colunas da temperatura mais antiga para a mais recente.
    X = zeros(numel(starts), p, class(values));
    for column = 1:p
        X(:, column) = values(starts + column - 1);
    end
    y = values(targets);
    target_dates = serial_dates(targets);
end


% ========================================
% Origem: src\melbourne\split_melbourne_chronological.m
% ========================================
function [idx_train, idx_val, idx_test] = split_melbourne_chronological(target_dates, train_ratio, val_ratio)
    % Divide os índices em treino, validação e teste por ordem cronológica.
    narginchk(3, 3);

    if ~isnumeric(target_dates) || ~isreal(target_dates) || ...
            (~isvector(target_dates) && ~isempty(target_dates))
        error('split_melbourne_chronological:InvalidDates', ...
            'As datas-alvo devem formar um vetor numérico real.');
    end
    serial_dates = target_dates(:);
    if any(isnan(serial_dates) | isinf(serial_dates) | serial_dates ~= fix(serial_dates))
        error('split_melbourne_chronological:InvalidDates', ...
            'As datas-alvo devem ser números de série inteiros e finitos.');
    end
    if any(serial_dates(2:end) <= serial_dates(1:end-1))
        error('split_melbourne_chronological:DateOrder', ...
            'As datas-alvo devem estar em ordem estritamente crescente.');
    end

    if ~isnumeric(train_ratio) || ~isreal(train_ratio) || ~isscalar(train_ratio) || ...
            ~isnumeric(val_ratio) || ~isreal(val_ratio) || ~isscalar(val_ratio)
        error('split_melbourne_chronological:InvalidRatios', ...
            'As proporções de treino e validação devem ser escalares numéricos reais.');
    end
    train_ratio = double(train_ratio);
    val_ratio = double(val_ratio);
    if isnan(train_ratio) || isinf(train_ratio) || train_ratio <= 0 || ...
            isnan(val_ratio) || isinf(val_ratio) || val_ratio <= 0 || ...
            train_ratio + val_ratio >= 1
        error('split_melbourne_chronological:InvalidRatios', ...
            'As proporções devem ser finitas e positivas, com soma inferior a 1.');
    end

    % Arredonda para baixo os tamanhos de treino e validação. O teste recebe o restante.
    n = numel(serial_dates);
    n_train = floor(n * train_ratio);
    n_val = floor(n * val_ratio);
    last_val = n_train + n_val;
    if n_train == 0 || n_val == 0 || last_val >= n
        error('split_melbourne_chronological:EmptySplit', ...
            'A divisão deve deixar pelo menos uma amostra em cada conjunto.');
    end

    idx_train = (1:n_train).';
    idx_val = (n_train + 1:last_val).';
    idx_test = (last_val + 1:n).';
end


% ========================================
% Origem: src\melbourne\normalize_melbourne_data.m
% ========================================
function [Xtr, Xv, Xte, ytr, yv, yte, params] = normalize_melbourne_data(X_train, X_val, X_test, y_train, y_val, y_test)
    % Normaliza as temperaturas com a média e o desvio-padrão dos alvos de treino.
    narginchk(6, 6);

    input_sets = {X_train, X_val, X_test};
    target_sets = {y_train, y_val, y_test};
    set_names = {'treino', 'validação', 'teste'};
    n_columns = size(X_train, 2);
    for k = 1:numel(input_sets)
        inputs = input_sets{k};
        targets = target_sets{k};
        if ~isnumeric(inputs) || ~isreal(inputs) || ~ismatrix(inputs) || ...
                any(isnan(inputs(:)) | isinf(inputs(:)))
            error('normalize_melbourne_data:InvalidInputs', ...
                'As entradas de %s devem formar uma matriz numérica real com valores finitos.', set_names{k});
        end
        if n_columns < 3 || size(inputs, 2) ~= n_columns
            error('normalize_melbourne_data:ColumnMismatch', ...
                'As matrizes X devem ter o mesmo número de colunas, com pelo menos três.');
        end
        if ~isnumeric(targets) || ~isreal(targets) || ...
                (~isvector(targets) && ~isequal(size(targets), [0, 0])) || ...
                any(isnan(targets(:)) | isinf(targets(:)))
            error('normalize_melbourne_data:InvalidTargets', ...
                'Os alvos de %s devem formar um vetor numérico real com valores finitos.', set_names{k});
        end
        if numel(targets) ~= size(inputs, 1)
            error('normalize_melbourne_data:RowMismatch', ...
                'Cada linha de X no conjunto de %s deve ter um alvo correspondente.', set_names{k});
        end
    end
    if isempty(y_train)
        error('normalize_melbourne_data:EmptyTrainingSet', ...
            'É necessário pelo menos um alvo de treino para calcular a normalização.');
    end

    % Usa o desvio-padrão amostral e aplica escala unitária a alvos constantes.
    training_targets = full(double(y_train(:)));
    if all(training_targets == training_targets(1))
        temp_mean = training_targets(1);
        temp_std = 1;
    else
        temp_mean = mean(training_targets);
        temp_std = std(training_targets);
        if isnan(temp_mean) || isinf(temp_mean) || isnan(temp_std) || isinf(temp_std) || temp_std == 0
            % Ajusta a escala quando os valores intermédios excedem a precisão numérica.
            magnitude = max(abs(training_targets));
            scaled_targets = training_targets ./ magnitude;
            temp_mean = mean(scaled_targets) * magnitude;
            temp_std = std(scaled_targets) * magnitude;
        end
        if temp_std == 0
            temp_std = 1;
        end
        if isnan(temp_mean) || isinf(temp_mean) || isnan(temp_std) || isinf(temp_std)
            error('normalize_melbourne_data:InvalidScale', ...
                'As estatísticas dos alvos de treino excedem o intervalo numérico suportado.');
        end
    end

    % Aplica os mesmos parâmetros aos três conjuntos e conserva o seno e o cosseno.
    p = n_columns - 2;
    Xtr = double(X_train);
    Xv = double(X_val);
    Xte = double(X_test);
    Xtr(:, 1:p) = (Xtr(:, 1:p) - temp_mean) ./ temp_std;
    Xv(:, 1:p) = (Xv(:, 1:p) - temp_mean) ./ temp_std;
    Xte(:, 1:p) = (Xte(:, 1:p) - temp_mean) ./ temp_std;
    ytr = (double(y_train) - temp_mean) ./ temp_std;
    yv = (double(y_val) - temp_mean) ./ temp_std;
    yte = (double(y_test) - temp_mean) ./ temp_std;
    params = struct('temp_mean', temp_mean, 'temp_std', temp_std, 'p', p);
end


% ========================================
% Origem: src\melbourne\add_melbourne_seasonality.m
% ========================================
function [X_seasonal, seasonal_features] = add_melbourne_seasonality(X, target_dates)
    % Acrescenta o seno e o cosseno do ciclo anual da data de cada alvo.
    narginchk(2, 2);

    if ~isnumeric(X) || ~isreal(X) || ~ismatrix(X) || ...
            any(isnan(X(:)) | isinf(X(:)))
        error('add_melbourne_seasonality:InvalidInputs', ...
            'X deve ser uma matriz numérica real com valores finitos.');
    end
    if ~isnumeric(target_dates) || ~isreal(target_dates) || ...
            (~isvector(target_dates) && ~isempty(target_dates))
        error('add_melbourne_seasonality:InvalidDates', ...
            'As datas-alvo devem formar um vetor numérico real.');
    end
    n_samples = size(X, 1);
    if n_samples ~= numel(target_dates)
        error('add_melbourne_seasonality:SizeMismatch', ...
            'O número de linhas de X deve corresponder ao número de datas-alvo.');
    end
    serial_dates = full(double(target_dates(:)));
    if any(isnan(serial_dates) | isinf(serial_dates) | serial_dates ~= fix(serial_dates))
        error('add_melbourne_seasonality:InvalidDates', ...
            'As datas-alvo devem ser números de série inteiros e finitos.');
    end

    seasonal_features = zeros(n_samples, 2);
    if n_samples > 0
        % O calendário de 2000 reserva um índice próprio para 29 de fevereiro.
        calendar_dates = datevec(serial_dates);
        reference_year = 2000;
        reference_dates = datenum([repmat(reference_year, n_samples, 1), calendar_dates(:, 2:3)]);
        seasonal_index = reference_dates - datenum(reference_year, 1, 1);
        phase = 2 * pi * seasonal_index / 366;
        seasonal_features = [sin(phase), cos(phase)];
    end

    % Usa precisão dupla para conservar as características fracionárias.
    X_seasonal = [double(X), seasonal_features];
end


% ========================================
% Origem: src\evaluation\regression_metrics.m
% ========================================
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


% ========================================
% Origem: src\baselines\linear_regression_baseline.m
% ========================================
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


% ========================================
% Origem: src\baselines\persistence_baseline.m
% ========================================
function y_pred = persistence_baseline(X, p)
    % Prevê o dia seguinte com a temperatura mais recente de cada janela.
    narginchk(2, 2);
    if ~isnumeric(X) || ~isreal(X) || ~ismatrix(X) || isempty(X) || ...
            any(isnan(X(:)) | isinf(X(:)))
        error('persistence_baseline:InvalidInputs', ...
            'X deve ser uma matriz numérica real, não vazia e com valores finitos.');
    end
    if ~isnumeric(p) || ~isreal(p) || ~isscalar(p) || ...
            isnan(p) || isinf(p) || p < 1 || p ~= fix(p)
        error('persistence_baseline:InvalidWindowSize', 'p deve ser um inteiro positivo.');
    end

    p = double(p);
    n_columns = size(X, 2);
    if n_columns ~= p && n_columns ~= p + 2
        error('persistence_baseline:ColumnMismatch', ...
            'X deve ter p colunas de temperaturas ou p+2 colunas com as características sazonais.');
    end

    % A coluna p contém a última temperatura, na escala original.
    y_pred = X(:, p);
end
