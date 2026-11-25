
% ============================================================
% ASSIGNMENT 2 - ARTIFICIAL NEURAL NETWORKS
% Curvas de convergencia - Concrete
%
% 3 arquiteturas x 3 algoritmos x 30 repeticoes
% Mediana e intervalo interquartil (Q1-Q3).
% Nao utiliza Statistics Toolbox.
% ============================================================

clear;
clc;
close all;

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);

final_dir = fullfile( ...
    project_dir, 'results', 'concrete', 'final');

output_dir = fullfile( ...
    project_dir, 'results', 'concrete', 'plots');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

arquiteturas = {'Reference', 'Wide', 'Deep'};

algoritmos = { ...
    'Adam', ...
    'Gradient_Descent', ...
    'BFGS' ...
};

titulos = { ...
    'Adam', ...
    'Gradient Descent', ...
    'BFGS' ...
};

cores = [ ...
    0.20, 0.45, 0.80; ...
    0.90, 0.55, 0.20; ...
    0.25, 0.65, 0.45 ...
];

n_repeticoes = 30;

fprintf('\n');
fprintf('====================================\n');
fprintf('CONVERGENCIA - CONCRETE\n');
fprintf('====================================\n');

for a = 1:numel(arquiteturas)

    arquitetura = arquiteturas{a};

    fig = figure( ...
        'visible', 'off', ...
        'position', [100, 100, 1500, 500]);

    for alg = 1:numel(algoritmos)

        algoritmo = algoritmos{alg};

        % Carregar o primeiro historico para determinar
        % o comprimento das curvas.
        ficheiro_1 = fullfile( ...
            final_dir, ...
            sprintf('%s_%s_rep01.mat', ...
                arquitetura, algoritmo));

        primeiro = load(ficheiro_1);

        n_pontos = numel(primeiro.history.val_loss);

        perdas = zeros(n_pontos, n_repeticoes);

        for rep = 1:n_repeticoes

            caminho = fullfile( ...
                final_dir, ...
                sprintf('%s_%s_rep%02d.mat', ...
                    arquitetura, algoritmo, rep));

            if ~exist(caminho, 'file')
                error('Ficheiro nao encontrado: %s', caminho);
            end

            modelo = load(caminho);

            valores = modelo.history.val_loss(:);

            if numel(valores) ~= n_pontos
                error( ...
                    'Historico com comprimento diferente: %s', ...
                    caminho);
            end

            if any(~isfinite(valores))
                error( ...
                    'Historico contem valores invalidos: %s', ...
                    caminho);
            end

            perdas(:, rep) = valores;

        end

        % Ordenar cada linha para calcular os percentis.
        perdas_ordenadas = sort(perdas, 2);

        mediana = zeros(n_pontos, 1);
        q1 = zeros(n_pontos, 1);
        q3 = zeros(n_pontos, 1);

        % Percentil com interpolacao linear:
        % posicao = 1 + (n-1)*p/100
        percentis = [25, 50, 75];

        for p = 1:3

            posicao = 1 + ...
                (n_repeticoes - 1) * percentis(p) / 100;

            inferior = floor(posicao);
            superior = ceil(posicao);
            fracao = posicao - inferior;

            valores_percentil = ...
                perdas_ordenadas(:, inferior) * (1 - fracao) + ...
                perdas_ordenadas(:, superior) * fracao;

            if p == 1
                q1 = valores_percentil;
            elseif p == 2
                mediana = valores_percentil;
            else
                q3 = valores_percentil;
            end

        end

        % Eixo horizontal:
        % Adam/GD: epocas completas.
        % BFGS: iteracoes, incluindo o estado inicial.
        if strcmp(algoritmo, 'BFGS')

            x = (0:n_pontos-1)';

            nome_eixo = 'Iteracoes BFGS';

        else

            x = (1:n_pontos)';

            nome_eixo = 'Epocas';

        end

        subplot(1, 3, alg);

        hold on;
        grid on;
        box on;

        cor = cores(alg, :);

        % Faixa interquartil.
        h_faixa = fill( ...
            [x; flipud(x)], ...
            [q1; flipud(q3)], ...
            cor, ...
            'edgecolor', 'none', ...
            'facealpha', 0.22);

        % Curva da mediana.
        h_mediana = plot( ...
            x, mediana, ...
            'color', cor, ...
            'linewidth', 2);

        title(sprintf('%s - %s', ...
            arquitetura, titulos{alg}));

        xlabel(nome_eixo);
        ylabel('MSE de validacao (normalizado)');

        legend( ...
            [h_mediana, h_faixa], ...
            {'Mediana', 'IQR (Q1-Q3)'}, ...
            'location', 'northeast');

        xlim([min(x), max(x)]);

        set(gca, 'fontsize', 10);

    end

    nome_saida = sprintf( ...
        'convergencia_%s.png', lower(arquitetura));

    caminho_saida = fullfile(output_dir, nome_saida);

    print(fig, caminho_saida, '-dpng', '-r160');

    close(fig);

    fprintf('Guardado: %s\n', nome_saida);

end

fprintf('\n====================================\n');
fprintf('3 GRAFICOS DE CONVERGENCIA GERADOS\n');
fprintf('Pasta: %s\n', output_dir);
fprintf('====================================\n');
