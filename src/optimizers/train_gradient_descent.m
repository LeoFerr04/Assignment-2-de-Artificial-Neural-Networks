
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
