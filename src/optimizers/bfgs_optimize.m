
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
