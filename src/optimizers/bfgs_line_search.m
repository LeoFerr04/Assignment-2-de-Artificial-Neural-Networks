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
