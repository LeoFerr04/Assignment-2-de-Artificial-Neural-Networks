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
