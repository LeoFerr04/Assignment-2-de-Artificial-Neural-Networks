function fig = plot_melbourne_acf(lags, acf_values, output_path)
    % Representa a autocorrelação das temperaturas mínimas de Melbourne.
    narginchk(2, 3);
    if nargin < 3
        output_path = '';
    end
    output_path = char(output_path);

    if ~isnumeric(lags) || ~isreal(lags) || ~isvector(lags) || isempty(lags)
        error('plot_melbourne_acf:InvalidLags', ...
            'Os atrasos devem formar um vetor numérico real e não vazio.');
    end
    if ~isnumeric(acf_values) || ~isreal(acf_values) || ~isvector(acf_values)
        error('plot_melbourne_acf:InvalidCoefficients', ...
            'Os coeficientes devem formar um vetor numérico real.');
    end
    if numel(lags) ~= numel(acf_values)
        error('plot_melbourne_acf:SizeMismatch', ...
            'Os vetores de atrasos e coeficientes devem ter o mesmo comprimento.');
    end

    lag_days = double(lags(:));
    coefficients = double(acf_values(:));
    if any(isnan(lag_days) | isinf(lag_days) | lag_days < 0 | lag_days ~= fix(lag_days))
        error('plot_melbourne_acf:InvalidLags', ...
            'Os atrasos devem ser inteiros finitos e não negativos.');
    end
    if any(diff(lag_days) <= 0)
        error('plot_melbourne_acf:LagOrder', 'Os atrasos devem estar em ordem estritamente crescente.');
    end
    if any(abs(coefficients) > 1)
        error('plot_melbourne_acf:InvalidCoefficients', ...
            'Os coeficientes devem estar entre -1 e 1 ou ser NaN.');
    end

    % O eixo inclui o atraso zero e o maior atraso, mesmo se faltarem coeficientes.
    max_lag = lag_days(end);
    margin = max(0.5, 0.015 * max_lag);
    x_limits = [-margin, max_lag + margin];
    lag_ticks = round(linspace(0, max_lag, min(max_lag + 1, 11)));

    fig = figure('Name', 'Autocorrelação das temperaturas em Melbourne', ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [100, 100, 1050, 460]);
    ax = axes('Parent', fig);
    line(ax, x_limits, [0, 0], 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1);

    % Desenha hastes apenas para coeficientes definidos.
    valid = ~isnan(coefficients);
    if any(valid)
        hold(ax, 'on');
        stems = stem(ax, lag_days(valid), coefficients(valid), 'filled');
        set(stems, 'ShowBaseLine', 'off', 'Color', [0.12, 0.36, 0.56], ...
            'LineWidth', 0.9, 'MarkerSize', 4);
        hold(ax, 'off');
    end

    xlim(ax, x_limits);
    ylim(ax, [-1.05, 1.05]);
    set(ax, 'XTick', lag_ticks, 'YTick', -1:0.25:1);
    xlabel(ax, 'Atraso (dias)');
    ylabel(ax, 'Coeficiente de autocorrelação');
    title(ax, 'Autocorrelação das temperaturas mínimas em Melbourne');
    grid(ax, 'on');

    % Guarda a figura em PNG quando é indicado um caminho.
    if ~isempty(output_path)
        print(fig, output_path, '-dpng', '-r150');
    end
end
