
% ============================================================
% ASSIGNMENT 2 - ARTIFICIAL NEURAL NETWORKS
% Boxplots comparativos - Dataset Concrete
%
% 3 arquiteturas x 3 algoritmos x 30 repeticoes.
% Nao requer Statistics Toolbox.
% ============================================================

clear;
clc;
close all;

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);

addpath(genpath(project_dir));

input_file = fullfile( ...
    project_dir, 'results', 'concrete', ...
    'evaluation', 'avaliacao_completa.mat');

output_dir = fullfile( ...
    project_dir, 'results', 'concrete', 'plots');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

if ~exist(input_file, 'file')
    error('Ficheiro de avaliacao nao encontrado.');
end

dados = load(input_file);
metricas = dados.metricas;

if size(metricas, 1) ~= 270
    error('Esperadas 270 experiencias.');
end

% Cada conjunto de 30 linhas corresponde a uma
% combinacao arquitetura-algoritmo.
n_grupos = 9;
n_repeticoes = 30;

nomes = { ...
    'Ref-A', 'Ref-GD', 'Ref-B', ...
    'Wide-A', 'Wide-GD', 'Wide-B', ...
    'Deep-A', 'Deep-GD', 'Deep-B' ...
};

titulos = { ...
    'MSE no conjunto de teste', ...
    'RMSE no conjunto de teste', ...
    'MAE no conjunto de teste', ...
    'R2 no conjunto de teste' ...
};

eixos_y = { ...
    'MSE (MPa^2)', ...
    'RMSE (MPa)', ...
    'MAE (MPa)', ...
    'R2' ...
};

ficheiros = { ...
    'boxplot_mse.png', ...
    'boxplot_rmse.png', ...
    'boxplot_mae.png', ...
    'boxplot_r2.png' ...
};

% Cores para os tres algoritmos.
cores = [ ...
    0.20, 0.45, 0.80; ...
    0.90, 0.55, 0.20; ...
    0.25, 0.65, 0.45 ...
];

fprintf('\n');
fprintf('====================================\n');
fprintf('GRAFICOS BOXPLOT - CONCRETE\n');
fprintf('====================================\n');

for m = 1:4

    fig = figure('visible', 'off', ...
        'position', [100, 100, 1200, 650]);

    hold on;
    grid on;
    box on;

    for g = 1:n_grupos

        indices = (g - 1) * n_repeticoes + ...
                  (1:n_repeticoes);

        valores = sort(metricas(indices, m));

        n = numel(valores);

        % Quartis com interpolacao linear.
        percentis = [25, 50, 75];
        quartis = zeros(1, 3);

        for p = 1:3

            posicao = 1 + (n - 1) * ...
                      percentis(p) / 100;

            inferior = floor(posicao);
            superior = ceil(posicao);

            if inferior == superior

                quartis(p) = valores(inferior);

            else

                fracao = posicao - inferior;

                quartis(p) = ...
                    valores(inferior) * (1 - fracao) + ...
                    valores(superior) * fracao;

            end

        end

        q1 = quartis(1);
        mediana = quartis(2);
        q3 = quartis(3);

        iqr_valor = q3 - q1;

        limite_inferior = q1 - 1.5 * iqr_valor;
        limite_superior = q3 + 1.5 * iqr_valor;

        valores_validos = valores( ...
            valores >= limite_inferior & ...
            valores <= limite_superior);

        whisker_inf = min(valores_validos);
        whisker_sup = max(valores_validos);

        outliers = valores( ...
            valores < limite_inferior | ...
            valores > limite_superior);

        % Algoritmos:
        % 1 = Adam, 2 = Gradient Descent, 3 = BFGS.
        indice_cor = mod(g - 1, 3) + 1;
        cor = cores(indice_cor, :);

        largura = 0.60;

        % Caixa Q1-Q3.
        patch( ...
            [g-largura/2, g+largura/2, ...
             g+largura/2, g-largura/2], ...
            [q1, q1, q3, q3], ...
            cor, ...
            'facealpha', 0.35, ...
            'edgecolor', cor, ...
            'linewidth', 1.5);

        % Mediana.
        line( ...
            [g-largura/2, g+largura/2], ...
            [mediana, mediana], ...
            'color', cor, ...
            'linewidth', 2.5);

        % Bigode inferior.
        line([g, g], [whisker_inf, q1], ...
            'color', cor, 'linewidth', 1.3);

        line( ...
            [g-0.15, g+0.15], ...
            [whisker_inf, whisker_inf], ...
            'color', cor, 'linewidth', 1.3);

        % Bigode superior.
        line([g, g], [q3, whisker_sup], ...
            'color', cor, 'linewidth', 1.3);

        line( ...
            [g-0.15, g+0.15], ...
            [whisker_sup, whisker_sup], ...
            'color', cor, 'linewidth', 1.3);

        % Valores atipicos.
        if ~isempty(outliers)

            plot( ...
                g * ones(size(outliers)), ...
                outliers, ...
                'ko', ...
                'markersize', 5, ...
                'markerfacecolor', 'none');

        end

    end

    % Separar visualmente as arquiteturas.
    line([3.5, 3.5], ylim(), ...
        'color', [0.6, 0.6, 0.6], ...
        'linestyle', '--');

    line([6.5, 6.5], ylim(), ...
        'color', [0.6, 0.6, 0.6], ...
        'linestyle', '--');

    xlim([0.3, 9.7]);

    set(gca, ...
        'xtick', 1:9, ...
        'xticklabel', nomes, ...
        'fontsize', 11);

    xlabel('Arquitetura e algoritmo');
    ylabel(eixos_y{m});
    title(titulos{m});

    % Legenda com linhas auxiliares.
    h1 = plot(NaN, NaN, '-', ...
        'color', cores(1, :), 'linewidth', 4);

    h2 = plot(NaN, NaN, '-', ...
        'color', cores(2, :), 'linewidth', 4);

    h3 = plot(NaN, NaN, '-', ...
        'color', cores(3, :), 'linewidth', 4);

    legend( ...
        [h1, h2, h3], ...
        {'Adam', 'Gradient Descent', 'BFGS'}, ...
        'location', 'northeast');

    caminho = fullfile(output_dir, ficheiros{m});

    print(fig, caminho, '-dpng', '-r160');

    close(fig);

    fprintf('Guardado: %s\n', ficheiros{m});

end

fprintf('\n====================================\n');
fprintf('4 BOXPLOTS GERADOS COM SUCESSO\n');
fprintf('Pasta: %s\n', output_dir);
fprintf('====================================\n');
