
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
