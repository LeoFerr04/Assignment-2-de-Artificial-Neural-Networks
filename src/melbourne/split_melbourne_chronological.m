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
