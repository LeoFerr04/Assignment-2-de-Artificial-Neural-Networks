
function [fig, stats] = plot_melbourne_scatter(dates, temperatures, output_path)
    % Compara as temperaturas minimas de dois dias consecutivos em Melbourne.

    % 1. Verificar os argumentos
    narginchk(2, 3);

    if nargin < 3
        output_path = '';
    end

    output_path = char(output_path);

    % 2. Validar as datas
    if ~isnumeric(dates) || ~isreal(dates) || ...
            (~isvector(dates) && ~isempty(dates))
        error('plot_melbourne_scatter:InvalidDates', ...
            'As datas devem formar um vetor numerico real.');
    end

    % 3. Validar as temperaturas
    if ~isnumeric(temperatures) || ~isreal(temperatures) || ...
            (~isvector(temperatures) && ~isempty(temperatures))
        error('plot_melbourne_scatter:InvalidTemperatures', ...
            'As temperaturas devem formar um vetor numerico real.');
    end

    % 4. Verificar se os vetores possuem o mesmo comprimento
    if numel(dates) ~= numel(temperatures)
        error('plot_melbourne_scatter:SizeMismatch', ...
            'Os vetores de datas e temperaturas devem ter o mesmo comprimento.');
    end

    % 5. Converter os dados para vetores coluna
    serial_dates = double(dates(:));
    values = double(temperatures(:));

    % Verificar se as datas sao validas
    if any(~isfinite(serial_dates) | serial_dates ~= fix(serial_dates))
        error('plot_melbourne_scatter:InvalidDates', ...
            'As datas devem ser numeros de serie inteiros e finitos.');
    end

    % Verificar a ordem cronologica
    day_steps = diff(serial_dates);

    if any(day_steps <= 0)
        error('plot_melbourne_scatter:DateOrder', ...
            'As datas devem estar em ordem estritamente crescente.');
    end

    % Permitir NaN, mas rejeitar temperaturas infinitas
    if any(isinf(values))
        error('plot_melbourne_scatter:InvalidTemperatures', ...
            'As temperaturas devem ser numeros reais finitos ou NaN.');
    end

    % 6. Selecionar apenas pares separados por um dia
    pair_starts = find(day_steps == 1);

    today = values(pair_starts);
    next_day = values(pair_starts + 1);

    % 7. Excluir pares que contem NaN
    valid_pairs = ~isnan(today) & ~isnan(next_day);

    today = today(valid_pairs);
    next_day = next_day(valid_pairs);

    % 8. Inicializar as estatisticas
    stats = struct('n_pairs', numel(today), 'correlation', NaN);

    % 9. Calcular manualmente a correlacao de Pearson
    if stats.n_pairs >= 2 && ...
            any(today ~= today(1)) && any(next_day ~= next_day(1))

        % Subtrair a media de cada conjunto
        x = today - mean(today);
        y = next_day - mean(next_day);

        % Calcular o numerador e o denominador
        numerator = sum(x .* y);
        denominator = norm(x) * norm(y);

        % Evitar divisao por zero
        if denominator > 0
            coefficient = numerator / denominator;

            % Garantir que o coeficiente fica entre -1 e 1
            stats.correlation = max(-1, min(1, coefficient));
        end
    end

    % 10. Criar o grafico de dispersao
    fig = figure('Name', 'Temperaturas minimas em dias consecutivos', ...
        'NumberTitle', 'off', 'Color', 'w', ...
        'Position', [100, 100, 800, 550]);

    ax = axes('Parent', fig);

    scatter(ax, today, next_day, 12, ...
        [0.12, 0.36, 0.56], 'filled');

    % 11. Configurar o grafico
    xlabel(ax, 'Temperatura minima do dia t (°C)');
    ylabel(ax, 'Temperatura minima do dia t+1 (°C)');

    title(ax, 'Temperaturas minimas em dias consecutivos em Melbourne');

    grid(ax, 'on');

    % 12. Guardar a figura em PNG, se for indicado um caminho
    if ~isempty(output_path)
        print(fig, output_path, '-dpng', '-r150');
    end

end
