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
