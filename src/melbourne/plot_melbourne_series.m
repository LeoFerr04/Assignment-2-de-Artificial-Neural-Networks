function fig = plot_melbourne_series(dates, temperatures, output_path)
    % Representa as temperaturas mínimas diárias de Melbourne.
    if nargin < 3
        output_path = '';
    end
    output_path = char(output_path);

    if ~isequal(size(dates), size(temperatures))
        error('plot_melbourne_series:SizeMismatch', ...
            'Os vetores de datas e temperaturas devem ter as mesmas dimensões.');
    end
    if ~isnumeric(dates) || ~isreal(dates) || ~isvector(dates) || isempty(dates)
        error('plot_melbourne_series:InvalidDates', ...
            'As datas devem formar um vetor numérico real e não vazio.');
    end
    if ~isnumeric(temperatures) || ~isreal(temperatures) || any(isinf(temperatures))
        error('plot_melbourne_series:InvalidTemperatures', ...
            'As temperaturas devem ser números reais finitos ou NaN.');
    end

    serial_dates = double(dates(:));
    if any(isnan(serial_dates) | isinf(serial_dates) | serial_dates ~= fix(serial_dates))
        error('plot_melbourne_series:InvalidDates', ...
            'As datas devem ser números de série inteiros e finitos.');
    end
    if any(diff(serial_dates) <= 0)
        error('plot_melbourne_series:DateOrder', ...
            'As datas devem estar em ordem estritamente crescente.');
    end

    % Mantém os NaN originais e deixa NaN nos dias sem observações.
    date_range = [serial_dates(1), serial_dates(end)];
    daily_dates = (date_range(1):date_range(2)).';
    daily_temperatures = NaN(size(daily_dates));
    daily_temperatures(serial_dates - date_range(1) + 1) = temperatures(:);

    % Marca cada ano no início do respetivo período visível.
    calendar_limits = datevec(date_range);
    years = (calendar_limits(1, 1):calendar_limits(2, 1)).';
    year_ticks = datenum([years, ones(numel(years), 2)]);
    year_ticks(1) = date_range(1);

    fig = figure('Name', 'Temperaturas mínimas em Melbourne', ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [100, 100, 1050, 460]);
    ax = axes('Parent', fig);
    plot(ax, daily_dates, daily_temperatures, '-', ...
        'Color', [0.12, 0.36, 0.56], 'LineWidth', 0.8, 'Marker', '.', 'MarkerSize', 4);
    set(ax, 'XTick', year_ticks, 'XTickLabel', cellstr(num2str(years)));
    if numel(daily_dates) == 1
        date_range = date_range + [-0.5, 0.5];
    end
    xlim(ax, date_range);
    xlabel(ax, 'Ano');
    ylabel(ax, 'Temperatura mínima (°C)');
    title(ax, 'Temperaturas mínimas diárias em Melbourne');
    legend(ax, 'Temperatura mínima diária', 'Location', 'southoutside');
    grid(ax, 'on');

    % Guarda a figura em PNG quando é indicado um caminho.
    if ~isempty(output_path)
        print(fig, output_path, '-dpng', '-r150');
    end
end
