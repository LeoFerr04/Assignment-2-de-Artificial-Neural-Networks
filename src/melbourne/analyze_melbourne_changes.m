function [fig, stats] = analyze_melbourne_changes(dates, temperatures, output_path)
    % Analisa as diferenças entre temperaturas mínimas de dias consecutivos.
    % O limiar de 5 °C é exploratório. A confirmação de erros exige evidência adicional.
    narginchk(2, 3);
    if nargin < 3
        output_path = '';
    end
    output_path = char(output_path);

    if ~isnumeric(dates) || ~isreal(dates) || (~isvector(dates) && ~isempty(dates))
        error('analyze_melbourne_changes:InvalidDates', ...
            'As datas devem formar um vetor numérico real.');
    end
    if ~isnumeric(temperatures) || ~isreal(temperatures) || ...
            (~isvector(temperatures) && ~isempty(temperatures))
        error('analyze_melbourne_changes:InvalidTemperatures', ...
            'As temperaturas devem formar um vetor numérico real.');
    end
    if numel(dates) ~= numel(temperatures)
        error('analyze_melbourne_changes:SizeMismatch', ...
            'Os vetores de datas e temperaturas devem ter o mesmo comprimento.');
    end

    serial_dates = double(dates(:));
    values = double(temperatures(:));
    if any(isnan(serial_dates) | isinf(serial_dates) | serial_dates ~= fix(serial_dates))
        error('analyze_melbourne_changes:InvalidDates', ...
            'As datas devem ser números de série inteiros e finitos.');
    end
    day_steps = diff(serial_dates);
    if any(day_steps <= 0)
        error('analyze_melbourne_changes:DateOrder', ...
            'As datas devem estar em ordem estritamente crescente.');
    end
    if any(isinf(values))
        error('analyze_melbourne_changes:InvalidTemperatures', ...
            'As temperaturas devem ser números reais finitos ou NaN.');
    end

    % Atribui cada variação ao segundo dia e deixa NaN onde falta um par válido.
    missing = isnan(values);
    pair_starts = find(day_steps == 1);
    pair_starts = pair_starts(~missing(pair_starts) & ~missing(pair_starts + 1));
    pair_ends = pair_starts + 1;
    valid_changes = values(pair_ends) - values(pair_starts);
    changes = NaN(size(values));
    changes(pair_ends) = valid_changes;

    threshold = 5;
    magnitudes = abs(valid_changes);
    % Compensa o arredondamento binário de diferenças exatamente no limiar.
    abrupt = magnitudes >= threshold - eps(threshold);
    stats = struct('n_pairs', numel(valid_changes), 'threshold', threshold, ...
        'n_abrupt', nnz(abrupt), 'mean_abs_change', NaN, 'max_abs_change', NaN, ...
        'n_nan', nnz(missing), 'top_dates', zeros(0, 1), 'top_changes', zeros(0, 1));
    if stats.n_pairs > 0
        stats.mean_abs_change = mean(magnitudes);
        stats.max_abs_change = max(magnitudes);
        [~, order] = sort(magnitudes, 'descend');
        top_indices = order(1:min(10, stats.n_pairs));
        stats.top_dates = serial_dates(pair_ends(top_indices));
        stats.top_changes = valid_changes(top_indices);
    end

    if isempty(serial_dates)
        date_limits = [datenum(1981, 1, 1), datenum(1990, 12, 31)];
    elseif isscalar(serial_dates)
        date_limits = serial_dates(1) + [-0.5, 0.5];
    else
        date_limits = [serial_dates(1), serial_dates(end)];
    end

    % Mostra anos em séries longas e datas completas em intervalos curtos.
    if diff(date_limits) >= 365
        calendar_limits = datevec(date_limits);
        year_limits = calendar_limits(:, 1);
        year_step = max(1, ceil(diff(year_limits) / 10));
        years = (year_limits(1):year_step:year_limits(2)).';
        date_ticks = datenum([years, ones(numel(years), 2)]);
        date_ticks(1) = date_limits(1);
        date_labels = cellstr(num2str(years));
    else
        first_day = ceil(date_limits(1));
        last_day = floor(date_limits(2));
        tick_count = min(7, last_day - first_day + 1);
        date_ticks = round(linspace(first_day, last_day, tick_count));
        date_labels = cellstr(datestr(date_ticks, 'dd/mm/yyyy'));
    end

    fig = figure('Name', 'Variações diárias de temperatura em Melbourne', ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [100, 100, 1050, 460]);
    ax = axes('Parent', fig);
    daily_line = line(ax, serial_dates, changes, 'Color', [0.12, 0.36, 0.56], ...
        'LineWidth', 0.8, 'Marker', '.', 'MarkerSize', 4);
    line(ax, date_limits, [0, 0], 'Color', [0.38, 0.38, 0.38]);
    threshold_color = [0.65, 0.28, 0.18];
    upper_line = line(ax, date_limits, [threshold, threshold], ...
        'Color', threshold_color, 'LineStyle', '--');
    line(ax, date_limits, [-threshold, -threshold], ...
        'Color', threshold_color, 'LineStyle', '--');

    abrupt_days = pair_ends(abrupt);
    abrupt_dates = serial_dates(abrupt_days);
    abrupt_changes = changes(abrupt_days);
    abrupt_points = line(ax, abrupt_dates(:), abrupt_changes(:), ...
        'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', 3, ...
        'Color', [0.77, 0.24, 0.12], 'MarkerFaceColor', [0.77, 0.24, 0.12]);

    xlim(ax, date_limits);
    set(ax, 'XTick', date_ticks, 'XTickLabel', date_labels);
    xlabel(ax, 'Data do segundo dia');
    ylabel(ax, 'Variação da temperatura mínima (°C)');
    title(ax, 'Variações diárias da temperatura mínima em Melbourne');
    grid(ax, 'on');
    legend([daily_line, upper_line, abrupt_points], ...
        {'Variação diária', 'Limiar exploratório (±5 °C)', 'Mudança abrupta'}, ...
        'Location', 'southoutside', 'Orientation', 'horizontal');

    % Guarda a figura em PNG quando é indicado um caminho.
    if ~isempty(output_path)
        print(fig, output_path, '-dpng', '-r150');
    end
end
