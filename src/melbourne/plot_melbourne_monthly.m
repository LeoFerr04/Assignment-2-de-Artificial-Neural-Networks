function [fig, monthly_stats] = plot_melbourne_monthly(dates, temperatures, output_path)
    % Agrupa as temperaturas mínimas pelo mês do calendário.
    if nargin < 3
        output_path = '';
    end
    output_path = char(output_path);

    if ~isequal(size(dates), size(temperatures))
        error('plot_melbourne_monthly:SizeMismatch', ...
            'Os vetores de datas e temperaturas devem ter as mesmas dimensões.');
    end
    if ~isnumeric(dates) || ~isreal(dates) || (~isvector(dates) && ~isempty(dates))
        error('plot_melbourne_monthly:InvalidDates', ...
            'As datas devem formar um vetor numérico real.');
    end
    if ~isnumeric(temperatures) || ~isreal(temperatures)
        error('plot_melbourne_monthly:InvalidTemperatures', ...
            'As temperaturas devem formar um vetor numérico real.');
    end

    serial_dates = double(dates(:));
    values = double(temperatures(:));
    if any(isnan(serial_dates) | isinf(serial_dates))
        error('plot_melbourne_monthly:InvalidDates', ...
            'As datas devem ser números de série finitos.');
    end
    if any(isinf(values))
        error('plot_melbourne_monthly:InvalidTemperatures', ...
            'As temperaturas devem ser números reais finitos ou NaN.');
    end

    % Seleciona as observações válidas de 1981 a 1990 para os cálculos.
    valid = serial_dates >= datenum(1981, 1, 1) & ...
        serial_dates < datenum(1991, 1, 1) & ~isnan(values);
    monthly_stats = struct('months', (1:12).', 'mean', NaN(12, 1), ...
        'std', NaN(12, 1), 'n_valid', zeros(12, 1));

    if any(valid)
        calendar_dates = datevec(serial_dates(valid));
        valid_temperatures = values(valid);
        for month = 1:12
            month_temperatures = valid_temperatures(calendar_dates(:, 2) == month);
            monthly_stats.n_valid(month) = numel(month_temperatures);
            if ~isempty(month_temperatures)
                monthly_stats.mean(month) = mean(month_temperatures);
                monthly_stats.std(month) = std(month_temperatures);
            end
        end
    end

    % Os meses sem observações válidas ficam sem barra.
    month_labels = {'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', ...
        'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'};
    fig = figure('Name', 'Temperaturas mínimas mensais em Melbourne', ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [100, 100, 920, 460]);
    ax = axes('Parent', fig);
    bar(ax, monthly_stats.months, monthly_stats.mean, 0.75, ...
        'FaceColor', [0.12, 0.36, 0.56], 'EdgeColor', 'none');
    set(ax, 'XTick', monthly_stats.months, 'XTickLabel', month_labels);
    xlim(ax, [0.5, 12.5]);
    xlabel(ax, 'Mês');
    ylabel(ax, 'Temperatura mínima média (°C)');
    title(ax, 'Temperaturas mínimas mensais em Melbourne (1981–1990)');
    legend(ax, 'Média das temperaturas mínimas diárias', 'Location', 'southoutside');
    grid(ax, 'on');

    % Guarda a figura em PNG quando é indicado um caminho.
    if ~isempty(output_path)
        print(fig, output_path, '-dpng', '-r150');
    end
end
