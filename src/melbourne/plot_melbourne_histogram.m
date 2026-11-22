function [fig, stats] = plot_melbourne_histogram(temperatures, output_path)
    % Mostra a distribuição das temperaturas mínimas de Melbourne.
    if nargin < 2
        output_path = '';
    end
    output_path = char(output_path);

    if ~isnumeric(temperatures) || ~isreal(temperatures) || ...
            (~isvector(temperatures) && ~isempty(temperatures))
        error('plot_melbourne_histogram:InvalidTemperatures', ...
            'As temperaturas devem formar um vetor numérico real.');
    end
    values = double(temperatures(:));
    if any(isinf(values))
        error('plot_melbourne_histogram:InfiniteTemperatures', ...
            'As temperaturas não podem incluir valores infinitos.');
    end

    % Usa uma cópia dos valores válidos e contabiliza os NaN em separado.
    missing = isnan(values);
    valid_temperatures = values(~missing);
    stats = struct('n_valid', numel(valid_temperatures), 'n_missing', nnz(missing), ...
        'mean', NaN, 'median', NaN, 'std', NaN, 'min', NaN, 'max', NaN);

    n_bins = 20;
    limits = [-0.5, 0.5];
    if stats.n_valid > 0
        stats.mean = mean(valid_temperatures);
        stats.median = median(valid_temperatures);
        stats.std = std(valid_temperatures);
        stats.min = min(valid_temperatures);
        stats.max = max(valid_temperatures);
        limits = [stats.min, stats.max];

        % Dá largura às barras quando os valores são iguais ou quase iguais.
        minimum_span = 2 * n_bins * eps(max(abs(limits)));
        if diff(limits) <= minimum_span
            padding = max(0.5, minimum_span);
            limits = limits + [-padding, padding];
        end
    end

    % Cada intervalo inclui o limite esquerdo. O último inclui também o máximo.
    edges = linspace(limits(1), limits(2), n_bins + 1);
    counts = zeros(1, n_bins);
    for bin = 1:n_bins
        counts(bin) = nnz(valid_temperatures >= edges(bin) & ...
            valid_temperatures < edges(bin+1));
    end
    counts(end) = counts(end) + nnz(valid_temperatures == edges(end));
    centres = edges(1:end-1) + diff(edges) / 2;

    fig = figure('Name', 'Histograma das temperaturas de Melbourne', ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [100, 100, 850, 460]);
    ax = axes('Parent', fig);
    bar(ax, centres, counts, 1, 'FaceColor', [0.12, 0.36, 0.56], ...
        'EdgeColor', 'w');
    xlim(ax, limits);
    xlabel(ax, 'Temperatura mínima (°C)');
    ylabel(ax, 'Número de observações');
    title(ax, 'Histograma das temperaturas mínimas diárias em Melbourne');
    grid(ax, 'on');

    % Guarda a figura em PNG quando é indicado um caminho.
    if ~isempty(output_path)
        print(fig, output_path, '-dpng', '-r150');
    end
end
