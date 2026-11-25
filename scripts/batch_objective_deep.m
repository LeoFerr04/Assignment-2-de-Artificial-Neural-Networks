
function [loss, gradient] = batch_objective_deep( ...
    theta, indices, initial_blocks, Xtr, ytr)

    % Calcula a perda e os gradientes de um mini-batch
    % para uma rede neuronal profunda.
    % Compativel com MATLAB e GNU Octave.

    % Selecionar as amostras do mini-batch.
    X_batch = Xtr(indices, :);
    y_batch = ytr(indices);

    % Criar a funcao objetivo para o mini-batch.
    objective = make_network_objective( ...
        initial_blocks, ...
        @(blocks) mlp_deep_loss_gradient( ...
            blocks, X_batch, y_batch));

    % Calcular a perda e os gradientes.
    [loss, gradient] = objective(theta);

end
