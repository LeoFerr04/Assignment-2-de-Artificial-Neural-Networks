
function [loss, gradient] = batch_objective( ...
    theta, indices, initial_blocks, Xtr, ytr)

    % Calcula a perda e o gradiente de um mini-batch.

    objective = make_network_objective( ...
        initial_blocks, ...
        @(blocks) mlp_loss_gradient( ...
            blocks, Xtr(indices, :), ytr(indices)));

    [loss, gradient] = objective(theta);

end
