
function [lags, acf_values, n_pairs] = melbourne_acf(dates, temperatures, max_lag)
    % Calcula manualmente a autocorrelacao das temperaturas de Melbourne.
    % Cada atraso corresponde a dias reais do calendario.
    % Utiliza a correlacao de Pearson entre pares validos.

    % 1. Verificar o numero de argumentos
    narginchk(3, 3);

    % 2. Validar as datas
    if ~isnumeric(dates) || ~isreal(dates) || ...
            (~isvector(dates) && ~isempty(dates))
        error('melbourne_acf:InvalidDates', ...
            'As datas devem formar um vetor numerico real.');
    end

    % 3. Validar as temperaturas
    if ~isnumeric(temperatures) || ~isreal(temperatures) || ...
            (~isvector(temperatures) && ~isempty(temperatures))
        error('melbourne_acf:InvalidTemperatures', ...
            'As temperaturas devem formar um vetor numerico real.');
    end

    % 4. Verificar se os vetores possuem o mesmo comprimento
    if numel(dates) ~= numel(temperatures)
        error('melbourne_acf:SizeMismatch', ...
            'Os vetores de datas e temperaturas devem ter o mesmo comprimento.');
    end

    % 5. Validar o atraso maximo
    if ~isnumeric(max_lag) || ~isreal(max_lag) || ...
            ~isscalar(max_lag) || ~isfinite(max_lag) || ...
            max_lag < 0 || max_lag ~= fix(max_lag)
        error('melbourne_acf:InvalidMaxLag', ...
            'max_lag deve ser um inteiro nao negativo.');
    end

    % 6. Converter os dados para vetores coluna
    serial_dates = double(dates(:));
    values = double(temperatures(:));

    % Verificar se as datas sao inteiras e finitas
    if any(~isfinite(serial_dates) | serial_dates ~= fix(serial_dates))
        error('melbourne_acf:InvalidDates', ...
            'As datas devem ser numeros de serie inteiros e finitos.');
    end

    % Verificar se as datas estao ordenadas
    if any(diff(serial_dates) <= 0)
        error('melbourne_acf:DateOrder', ...
            'As datas devem estar em ordem estritamente crescente.');
    end

    % Permitir NaN, mas rejeitar valores infinitos
    if any(isinf(values))
        error('melbourne_acf:InvalidTemperatures', ...
            'As temperaturas devem ser numeros reais finitos ou NaN.');
    end

    % 7. Inicializar os resultados
    max_lag = double(max_lag);

    lags = (0:max_lag).';
    acf_values = NaN(size(lags));
    n_pairs = zeros(size(lags));

    % Terminar se nao existirem observacoes
    if isempty(serial_dates)
        return;
    end

    % 8. Construir uma sequencia diaria completa
    % Os dias em falta ficam representados por NaN.
    positions = serial_dates - serial_dates(1) + 1;
    n_days = positions(end);

    daily_temperatures = NaN(n_days, 1);
    daily_temperatures(positions) = values;

    % 9. Calcular a autocorrelacao para cada atraso
    for lag = 0:min(max_lag, n_days - 1)

        index = lag + 1;

        % Selecionar observacoes separadas por 'lag' dias
        earlier = daily_temperatures(1:n_days - lag);
        later = daily_temperatures(1 + lag:n_days);

        % Identificar pares sem valores em falta
        valid_pairs = ~isnan(earlier) & ~isnan(later);

        % Contar os pares validos
        n_pairs(index) = nnz(valid_pairs);

        % Sao necessarios pelo menos dois pares
        if n_pairs(index) < 2
            continue;
        end

        % Manter apenas os pares validos
        earlier = earlier(valid_pairs);
        later = later(valid_pairs);

        % Verificar se algum conjunto possui variancia nula
        if all(earlier == earlier(1)) || all(later == later(1))
            continue;
        end

        % 10. A autocorrelacao no atraso zero e igual a 1
        if lag == 0
            acf_values(index) = 1;
            continue;
        end

        % 11. Calcular manualmente a correlacao de Pearson
        % Subtrair a media de cada conjunto
        earlier = earlier - mean(earlier);
        later = later - mean(later);

        % Calcular o numerador e o denominador
        numerator = sum(earlier .* later);
        denominator = norm(earlier) * norm(later);

        % Evitar divisao por zero
        if denominator == 0
            continue;
        end

        % Calcular o coeficiente de correlacao
        coefficient = numerator / denominator;

        % Garantir que o resultado fica entre -1 e 1
        acf_values(index) = max(-1, min(1, coefficient));

    end

end
