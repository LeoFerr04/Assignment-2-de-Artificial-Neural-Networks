
% Gera os gráficos de validação a partir dos resultados guardados de Melbourne.
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'src', 'evaluation'));
results_dir = fullfile(project_root, 'results', 'melbourne');
csv_path = fullfile(results_dir, 'arquiteturas_validacao_reproduzido.csv');
architecture_path = fullfile(results_dir, 'rmse_validacao_mlp_media_desvio.png');
boxplot_path = fullfile(results_dir, 'boxplots_rmse_validacao_mlp.png');
models_path = fullfile(results_dir, 'comparacao_rmse_validacao_modelos.png');

% Reutiliza a leitura do CSV e as séries do gráfico de arquiteturas existente.
architecture_fig = plot_melbourne_architectures(csv_path, architecture_path);
architecture_ax = findobj(architecture_fig, 'Type', 'axes', '-not', 'Tag', 'legend');
mean_line = findobj(architecture_ax, 'Type', 'line', 'DisplayName', 'RMSE medio');
assert(isscalar(mean_line), 'Falta a série de RMSE médio no gráfico de arquiteturas.');

architectures = get(mean_line, 'XData');
assert(isequal(architectures(:), [3; 5; 8]), ...
    'Esperava as arquiteturas com 3, 5 e 8 neurónios.');

seeds = [7, 42, 123];
offsets = [-0.16, 0, 0.16];
seed_labels = cell(1, numel(seeds));
seed_lines = cell(1, numel(seeds));
seed_colours = zeros(numel(seeds), 3);
rmse_samples = zeros(numel(architectures), numel(seeds));

for k = 1:numel(seeds)
    seed_labels{k} = sprintf('Semente %d', seeds(k));
    seed_lines{k} = findobj(architecture_ax, 'Type', 'line', ...
        'DisplayName', seed_labels{k});

    assert(isscalar(seed_lines{k}), ...
        sprintf('Falta a série de resultados da semente %d.', seeds(k)));

    samples = get(seed_lines{k}, 'YData');
    rmse_samples(:, k) = samples(:);
    seed_colours(k, :) = get(seed_lines{k}, 'Color');

    % Separa os marcadores de sementes próximas no eixo horizontal.
    set(seed_lines{k}, 'XData', architectures + offsets(k));
end

mean_rmse = mean(rmse_samples, 2);
std_rmse = std(rmse_samples, 0, 2);

% Acrescenta as barras de desvio-padrão amostral ao gráfico existente.
legend(architecture_ax, 'off');
hold(architecture_ax, 'on');

std_bars = errorbar(architecture_ax, architectures(:), ...
    mean_rmse, std_rmse, 'k');
set(std_bars, 'LineStyle', 'none', 'LineWidth', 1.2);

hold(architecture_ax, 'off');
title(architecture_ax, 'RMSE de validação das arquiteturas MLP com BFGS');
xlabel(architecture_ax, 'Neurónios escondidos');
ylabel(architecture_ax, 'RMSE de validação (°C)');

legend(architecture_ax, [seed_lines{:}, mean_line], ...
    [seed_labels, {'Média ± 1 desvio-padrão'}], ...
    'Location', 'southoutside', 'Orientation', 'horizontal');

print(architecture_fig, architecture_path, '-dpng', '-r150');

% Desenha as caixas com funções base do Octave e mostra cada observação.
boxplot_fig = figure('Name', 'Boxplots do RMSE de validação', ...
    'NumberTitle', 'off', 'Color', 'w', ...
    'Position', [100, 100, 950, 520]);

boxplot_ax = axes('Parent', boxplot_fig);
hold(boxplot_ax, 'on');

positions = 1:numel(architectures);
sorted_rmse = sort(rmse_samples, 2);

% Para três valores, os quartis lineares ficam a meio dos pares adjacentes.
lower_quartile = mean(sorted_rmse(:, 1:2), 2);
upper_quartile = mean(sorted_rmse(:, 2:3), 2);
box_colour = [0.40, 0.47, 0.55];

for k = positions
    low = sorted_rmse(k, 1);
    centre = sorted_rmse(k, 2);
    high = sorted_rmse(k, 3);
    q1 = lower_quartile(k);
    q3 = upper_quartile(k);

    line(boxplot_ax, [k, k], [low, high], ...
        'Color', box_colour, 'LineWidth', 1.2);

    line(boxplot_ax, k + [-0.10, 0.10], [low, low], ...
        'Color', box_colour, 'LineWidth', 1.2);

    line(boxplot_ax, k + [-0.10, 0.10], [high, high], ...
        'Color', box_colour, 'LineWidth', 1.2);

    patch(boxplot_ax, k + [-0.23, 0.23, 0.23, -0.23], ...
        [q1, q1, q3, q3], [0.91, 0.94, 0.97], ...
        'EdgeColor', box_colour, 'LineWidth', 1.2);

    line(boxplot_ax, k + [-0.23, 0.23], [centre, centre], ...
        'Color', 'k', 'LineWidth', 1.5);
end

% Desloca os pontos apenas no eixo horizontal para evitar sobreposições.
point_handles = cell(1, numel(seeds));

for k = 1:numel(seeds)
    point_handles{k} = plot(boxplot_ax, ...
        positions(:) + offsets(k), rmse_samples(:, k), ...
        'o', 'Color', seed_colours(k, :), ...
        'MarkerFaceColor', seed_colours(k, :), 'MarkerSize', 6);
end

hold(boxplot_ax, 'off');

set(boxplot_ax, 'XTick', positions, ...
    'XTickLabel', cellstr(num2str(architectures(:))));

xlim(boxplot_ax, [0.5, numel(architectures) + 0.5]);
xlabel(boxplot_ax, 'Neurónios escondidos');
ylabel(boxplot_ax, 'RMSE de validação (°C)');

title(boxplot_ax, {'RMSE de validação por arquitetura MLP', ...
    'n = 3 por arquitetura | Caixa Q1–Q3 | Traço mediana | Hastes mínimo–máximo'});

grid(boxplot_ax, 'on');
set(boxplot_ax, 'XGrid', 'off');

legend(boxplot_ax, [point_handles{:}], seed_labels, ...
    'Location', 'southoutside', 'Orientation', 'horizontal');

print(boxplot_fig, boxplot_path, '-dpng', '-r150');

% Os três valores de referência vêm de docs/BFGS_Melbourne.md.
% O valor da MLP é calculado a partir das três sementes do CSV.
model_labels = {'Persistência', 'AR(7) com intercepto', ...
    'Regressão linear sazonal', ...
    'MLP BFGS (3 neurónios, média de 3 sementes)'};

model_rmse = [2.5925, 2.359835, 2.300801, ...
    mean_rmse(architectures == 3)];

models_fig = figure('Name', 'Comparação dos modelos de Melbourne', ...
    'NumberTitle', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1050, 440]);

models_ax = axes('Parent', models_fig, ...
    'Position', [0.42, 0.16, 0.54, 0.71]);

barh(models_ax, 1:numel(model_rmse), model_rmse, 0.60, ...
    'FaceColor', [0.12, 0.36, 0.56], 'EdgeColor', 'none');

set(models_ax, 'YDir', 'reverse', ...
    'YTick', 1:numel(model_rmse), 'YTickLabel', model_labels);

xlim(models_ax, [0, 3.15]);
xlabel(models_ax, 'RMSE de validação (°C)');
title(models_ax, 'RMSE de validação dos modelos de Melbourne');

grid(models_ax, 'on');
set(models_ax, 'YGrid', 'off', 'Layer', 'bottom');

% ALTERAÇÃO: apresenta o RMSE com 4 casas decimais em vez de 6.
for k = 1:numel(model_rmse)
    label = strrep(sprintf('%.4f', model_rmse(k)), '.', ',');
    text(models_ax, model_rmse(k) + 0.045, k, label, ...
        'VerticalAlignment', 'middle');
end

print(models_fig, models_path, '-dpng', '-r150');

fprintf('\nGráficos guardados em\n%s\n%s\n%s\n', ...
    architecture_path, boxplot_path, models_path);
