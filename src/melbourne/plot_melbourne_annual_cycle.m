function [fig, annual_stats] = plot_melbourne_annual_cycle(dates, temperatures, output_path)
    % Calcula o ciclo anual das temperaturas mínimas de Melbourne.
    if nargin < 3
        output_path = '';
    end
    output_path = char(output_path);

    if ~isequal(size(dates), size(temperatures))
        error('plot_melbourne_annual_cycle:SizeMismatch', ...
            'Os vetores de datas e temperaturas devem ter as mesmas dimensões.');
    end
    if ~isnumeric(dates) || ~isreal(dates) || (~isvector(dates) && ~isempty(dates))
        error('plot_melbourne_annual_cycle:InvalidDates', ...
            'As datas devem formar um vetor numérico real.');
    end
    if ~isnumeric(temperatures) || ~isreal(temperatures)
        error('plot_melbourne_annual_cycle:InvalidTemperatures', ...
            'As temperaturas devem formar um vetor numérico real.');
    end

    serial_dates = double(dates(:));
    values = double(temperatures(:));
    if any(isnan(serial_dates) | isinf(serial_dates))
        error('plot_melbourne_annual_cycle:InvalidDates', ...
            'As datas devem ser números de série finitos.');
    end
    if any(isinf(values))
        error('plot_melbourne_annual_cycle:InvalidTemperatures', ...
            'As temperaturas devem ser números reais finitos ou NaN.');
    end

    % O calendário de 2000 reserva a posição 60 para 29 de fevereiro.
    reference_year = 2000;
    reference_start = datenum(reference_year, 1, 1);
    n_days = datenum(reference_year + 1, 1, 1) - reference_start;
    annual_stats = struct('day_index', (1:n_days).', 'mean', NaN(n_days, 1), ...
        'std', NaN(n_days, 1), 'n_valid', zeros(n_days, 1));

    % Seleciona as observações válidas de 1981 a 1990 e agrupa-as por mês e dia.
    valid = serial_dates >= datenum(1981, 1, 1) & ...
        serial_dates < datenum(1991, 1, 1) & ~isnan(values);
    if any(valid)
        calendar_dates = datevec(serial_dates(valid));
        reference_dates = datenum([repmat(reference_year, size(calendar_dates, 1), 1), ...
            calendar_dates(:, 2:3)]);
        day_indices = reference_dates - reference_start + 1;
        valid_temperatures = values(valid);
        for day = 1:n_days
            day_temperatures = valid_temperatures(day_indices == day);
            annual_stats.n_valid(day) = numel(day_temperatures);
            if ~isempty(day_temperatures)
                annual_stats.mean(day) = mean(day_temperatures);
                annual_stats.std(day) = std(day_temperatures);
            end
        end
    end

    % Assinala o primeiro dia de cada mês no eixo horizontal.
    month_starts = datenum([repmat(reference_year, 12, 1), (1:12).', ones(12, 1)]) ...
        - reference_start + 1;
    month_labels = {'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', ...
        'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'};
    fig = figure('Name', 'Ciclo anual das temperaturas mínimas em Melbourne', ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [100, 100, 1050, 460]);
    ax = axes('Parent', fig);
    plot(ax, annual_stats.day_index, annual_stats.mean, '-', ...
        'Color', [0.12, 0.36, 0.56], 'LineWidth', 1, 'Marker', '.', 'MarkerSize', 4);
    set(ax, 'XTick', month_starts, 'XTickLabel', month_labels);
    xlim(ax, [1, n_days]);
    xlabel(ax, 'Mês');
    ylabel(ax, 'Temperatura mínima média (°C)');
    title(ax, 'Ciclo anual das temperaturas mínimas em Melbourne (1981–1990)');
    legend(ax, 'Média por dia do calendário', 'Location', 'southoutside');
    grid(ax, 'on');

    % Guarda a figura em PNG quando é indicado um caminho.
    if ~isempty(output_path)
        print(fig, output_path, '-dpng', '-r150');
    end
end
