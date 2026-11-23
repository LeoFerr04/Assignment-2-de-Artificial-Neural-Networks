function [X_seasonal, seasonal_features] = add_melbourne_seasonality(X, target_dates)
    % Acrescenta o seno e o cosseno do ciclo anual da data de cada alvo.
    narginchk(2, 2);

    if ~isnumeric(X) || ~isreal(X) || ~ismatrix(X) || ...
            any(isnan(X(:)) | isinf(X(:)))
        error('add_melbourne_seasonality:InvalidInputs', ...
            'X deve ser uma matriz numérica real com valores finitos.');
    end
    if ~isnumeric(target_dates) || ~isreal(target_dates) || ...
            (~isvector(target_dates) && ~isempty(target_dates))
        error('add_melbourne_seasonality:InvalidDates', ...
            'As datas-alvo devem formar um vetor numérico real.');
    end
    n_samples = size(X, 1);
    if n_samples ~= numel(target_dates)
        error('add_melbourne_seasonality:SizeMismatch', ...
            'O número de linhas de X deve corresponder ao número de datas-alvo.');
    end
    serial_dates = full(double(target_dates(:)));
    if any(isnan(serial_dates) | isinf(serial_dates) | serial_dates ~= fix(serial_dates))
        error('add_melbourne_seasonality:InvalidDates', ...
            'As datas-alvo devem ser números de série inteiros e finitos.');
    end

    seasonal_features = zeros(n_samples, 2);
    if n_samples > 0
        % O calendário de 2000 reserva um índice próprio para 29 de fevereiro.
        calendar_dates = datevec(serial_dates);
        reference_year = 2000;
        reference_dates = datenum([repmat(reference_year, n_samples, 1), calendar_dates(:, 2:3)]);
        seasonal_index = reference_dates - datenum(reference_year, 1, 1);
        phase = 2 * pi * seasonal_index / 366;
        seasonal_features = [sin(phase), cos(phase)];
    end

    % Usa precisão dupla para conservar as características fracionárias.
    X_seasonal = [double(X), seasonal_features];
end
