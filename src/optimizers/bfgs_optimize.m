function [theta, info] = bfgs_optimize(objective_fn, theta0, options)
    % Minimiza uma função com BFGS e pesquisa linear de Armijo.
    % max_params limita a dimensão da matriz densa. iterations conta passos aceites.
    % converged indica que foi atingida a tolerância do gradiente.
    narginchk(2, 3);
    if nargin < 3
        options = struct();
    end

    if ~isa(objective_fn, 'function_handle')
        error('bfgs_optimize:InvalidObjective', 'objective_fn deve ser um identificador de função.');
    end
    if ~isnumeric(theta0) || ~isreal(theta0) || ~iscolumn(theta0) || isempty(theta0) || ...
            any(isnan(theta0) | isinf(theta0))
        error('bfgs_optimize:InvalidTheta', 'theta0 deve ser um vetor coluna real, finito e não vazio.');
    end
    if ~isstruct(options) || ~isscalar(options)
        error('bfgs_optimize:InvalidOptions', 'options deve ser uma estrutura escalar.');
    end

    settings = struct('max_iter', 200, 'grad_tol', 1e-6, 'step_tol', 1e-10, 'max_params', 2000);
    fields = fieldnames(settings);
    for k = 1:numel(fields)
        name = fields{k};
        if isfield(options, name)
            value = options.(name);
            if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || isnan(value) || isinf(value)
                error('bfgs_optimize:InvalidOptions', 'O campo %s deve ser um escalar real e finito.', name);
            end
            settings.(name) = double(value);
        end
    end
    if settings.max_iter < 0 || settings.max_iter ~= fix(settings.max_iter)
        error('bfgs_optimize:InvalidOptions', 'max_iter deve ser um inteiro não negativo.');
    end
    if settings.grad_tol < 0 || settings.step_tol < 0
        error('bfgs_optimize:InvalidOptions', 'grad_tol e step_tol devem ser não negativos.');
    end
    if settings.max_params < 1 || settings.max_params ~= fix(settings.max_params)
        error('bfgs_optimize:InvalidOptions', 'max_params deve ser um inteiro positivo.');
    end

    verbose = false;
    if isfield(options, 'verbose')
        value = options.verbose;
        if ~isscalar(value) || ~(islogical(value) || (isnumeric(value) && isreal(value) && ...
                (value == 0 || value == 1)))
            error('bfgs_optimize:InvalidOptions', 'verbose deve ser um valor lógico ou 0 ou 1.');
        end
        verbose = logical(value);
    end
    line_search_options = struct();
    if isfield(options, 'line_search')
        line_search_options = options.line_search;
        if ~isstruct(line_search_options) || ~isscalar(line_search_options)
            error('bfgs_optimize:InvalidOptions', 'line_search deve ser uma estrutura escalar.');
        end
    end

    n = numel(theta0);
    if n > settings.max_params
        error('bfgs_optimize:TooManyParameters', ...
            'O modelo tem %d parâmetros e excede o limite de %d para a matriz densa BFGS.', ...
            n, settings.max_params);
    end
    theta = double(theta0);
    [loss, gradient] = objective_fn(theta);
    if ~isnumeric(loss) || ~isreal(loss) || ~isscalar(loss) || isnan(loss) || isinf(loss) || ...
            ~isnumeric(gradient) || ~isreal(gradient) || ~isequal(size(gradient), size(theta)) || ...
            any(isnan(gradient) | isinf(gradient))
        error('bfgs_optimize:InvalidObjectiveResult', ...
            'A avaliação inicial deve devolver uma perda escalar e um gradiente coluna reais e finitos.');
    end
    loss = double(loss);
    gradient = double(gradient);
    grad_norm = norm(gradient);
    info = struct('loss_history', loss, 'grad_norm_history', grad_norm, 'iterations', 0, ...
        'n_evals', 1, 'converged', false, 'status', 'max_iter', 'curvature_skips', 0);
    if grad_norm == 0 || grad_norm < settings.grad_tol
        info.converged = true;
        info.status = 'grad_tol';
        return;
    end

    H = eye(n);
    while info.iterations < settings.max_iter
        direction = -H * gradient;
        slope = gradient.' * direction;
        if isnan(slope) || isinf(slope) || slope >= 0 || any(isnan(direction) | isinf(direction))
            % Recupera uma direção de descida quando a aproximação perde essa propriedade.
            H = eye(n);
            direction = -gradient;
        end

        [alpha, loss_new, gradient_new, accepted, n_evals] = ...
            bfgs_line_search(objective_fn, theta, loss, gradient, direction, line_search_options);
        info.n_evals = info.n_evals + n_evals;
        if ~accepted
            info.status = 'line_search_failed';
            break;
        end

        theta_new = theta + alpha * direction;
        s = theta_new - theta;
        gradient_new = double(gradient_new);
        y = gradient_new - gradient;
        step_norm = norm(s);
        curvature = y.' * s;
        curvature_limit = sqrt(eps) * step_norm * norm(y);
        updated = false;
        if ~isnan(curvature) && ~isinf(curvature) && curvature > curvature_limit
            rho = 1 / curvature;
            if ~isinf(rho)
                % Aplica os dois fatores da fórmula BFGS com produtos de posto um.
                scaled_step = rho * s;
                H_new = H - scaled_step * (y.' * H);
                H_new = H_new - (H_new * y) * scaled_step.' + scaled_step * s.';
                H_new = 0.5 * H_new + 0.5 * H_new.';
                if all(~isnan(H_new(:)) & ~isinf(H_new(:)))
                    H = H_new;
                    updated = true;
                end
            end
        end
        if ~updated
            info.curvature_skips = info.curvature_skips + 1;
        end

        % Regista apenas pontos aceites, além da avaliação inicial.
        theta = theta_new;
        loss = double(loss_new);
        gradient = gradient_new;
        grad_norm = norm(gradient);
        info.iterations = info.iterations + 1;
        info.loss_history(end + 1, 1) = loss;
        info.grad_norm_history(end + 1, 1) = grad_norm;
        if verbose
            fprintf('Iteração %d | perda %.6g | norma do gradiente %.6g\n', ...
                info.iterations, loss, grad_norm);
        end

        if grad_norm == 0 || grad_norm < settings.grad_tol
            info.converged = true;
            info.status = 'grad_tol';
            break;
        end
        if step_norm <= settings.step_tol
            info.status = 'step_tol';
            break;
        end
    end
end
