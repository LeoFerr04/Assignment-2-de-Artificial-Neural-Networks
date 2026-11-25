
% ============================================================
% ASSIGNMENT 2 - ARTIFICIAL NEURAL NETWORKS
% Avaliacao final do dataset Concrete
%
% 3 arquiteturas x 3 algoritmos x 30 repeticoes = 270 modelos
% Sem novos treinos.
% ============================================================

clear;
clc;

% Localizar a pasta principal do projeto.
script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);

addpath(genpath(project_dir));

% Pasta com os modelos guardados.
final_dir = fullfile( ...
    project_dir, 'results', 'concrete', 'final');

% Pasta dos resultados da avaliacao.
output_dir = fullfile( ...
    project_dir, 'results', 'concrete', 'evaluation');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Carregar os dados com a divisao e normalizacao originais.
data = concrete_preprocess();

% Valores reais no conjunto de teste (MPa).
y_real = data.yte_original(:);

% Configuracoes.
arquiteturas = {'Reference', 'Wide', 'Deep'};

algoritmos = { ...
    'Adam', ...
    'Gradient_Descent', ...
    'BFGS' ...
};

n_repeticoes = 30;
n_total = numel(arquiteturas) * ...
          numel(algoritmos) * n_repeticoes;

% Cada linha:
% [MSE, RMSE, MAE, R2, tempo, MSE_validacao]
metricas = zeros(n_total, 6);

resultados = struct( ...
    'arquitetura', {}, ...
    'algoritmo', {}, ...
    'repeticao', {}, ...
    'MSE', {}, ...
    'RMSE', {}, ...
    'MAE', {}, ...
    'R2', {}, ...
    'tempo', {}, ...
    'mse_validacao', {}, ...
    'y_pred', {} ...
);

fprintf('\n');
fprintf('========================================\n');
fprintf('AVALIACAO FINAL - CONCRETE\n');
fprintf('========================================\n');

fprintf('Observacoes de teste: %d\n', numel(y_real));
fprintf('Modelos previstos: %d\n\n', n_total);

indice = 0;

for a = 1:numel(arquiteturas)

    arquitetura = arquiteturas{a};

    for alg = 1:numel(algoritmos)

        algoritmo = algoritmos{alg};

        fprintf('\n%s | %s\n', arquitetura, algoritmo);
        fprintf('----------------------------------------\n');

        for rep = 1:n_repeticoes

            nome_ficheiro = sprintf( ...
                '%s_%s_rep%02d.mat', ...
                arquitetura, algoritmo, rep);

            caminho = fullfile(final_dir, nome_ficheiro);

            if ~exist(caminho, 'file')
                error('Ficheiro nao encontrado: %s', caminho);
            end

            modelo = load(caminho);

            % Recuperar as previsoes normalizadas.
            if strcmp(arquitetura, 'Deep')

                y_normalizado = mlp_deep_predict( ...
                    modelo.theta_best, ...
                    data.Xte, ...
                    modelo.layers);

            else

                y_normalizado = mlp_predict( ...
                    modelo.theta_best, ...
                    data.Xte, ...
                    modelo.layers(2));

            end

            % Converter novamente para MPa.
            y_pred = ...
                y_normalizado(:) * data.desvio_y + ...
                data.media_y;

            if numel(y_pred) ~= numel(y_real)
                error('Numero incorreto de previsoes.');
            end

            if any(~isfinite(y_pred))
                error('Previsoes invalidas em %s.', nome_ficheiro);
            end

            % Calcular metricas nas unidades originais.
            erros = y_real - y_pred;

            mse = mean(erros .^ 2);
            rmse = sqrt(mse);
            mae = mean(abs(erros));

            ss_res = sum(erros .^ 2);
            ss_tot = sum((y_real - mean(y_real)) .^ 2);

            r2 = 1 - ss_res / ss_tot;

            indice = indice + 1;

            metricas(indice, :) = [ ...
                mse, ...
                rmse, ...
                mae, ...
                r2, ...
                modelo.tempo, ...
                modelo.mse_validacao ...
            ];

            resultados(indice).arquitetura = arquitetura;
            resultados(indice).algoritmo = algoritmo;
            resultados(indice).repeticao = rep;

            resultados(indice).MSE = mse;
            resultados(indice).RMSE = rmse;
            resultados(indice).MAE = mae;
            resultados(indice).R2 = r2;

            resultados(indice).tempo = modelo.tempo;

            resultados(indice).mse_validacao = ...
                modelo.mse_validacao;

            resultados(indice).y_pred = y_pred;

        end

        fprintf('30 repeticoes avaliadas.\n');

    end

end

% ============================================================
% RESUMO POR ARQUITETURA E ALGORITMO
% ============================================================

% Colunas:
% MSE_medio, MSE_desvio, RMSE_medio, RMSE_desvio,
% MAE_medio, MAE_desvio, R2_medio, R2_desvio,
% Tempo_medio, MSE_validacao_medio

resumo = zeros(9, 10);

linha = 0;

fprintf('\n');
fprintf('========================================\n');
fprintf('RESULTADOS MEDIOS NO CONJUNTO DE TESTE\n');
fprintf('========================================\n\n');

for a = 1:numel(arquiteturas)

    for alg = 1:numel(algoritmos)

        linha = linha + 1;

        indices = (linha - 1) * n_repeticoes + ...
                  (1:n_repeticoes);

        valores = metricas(indices, :);

        resumo(linha, :) = [ ...
            mean(valores(:, 1)), std(valores(:, 1)), ...
            mean(valores(:, 2)), std(valores(:, 2)), ...
            mean(valores(:, 3)), std(valores(:, 3)), ...
            mean(valores(:, 4)), std(valores(:, 4)), ...
            mean(valores(:, 5)), mean(valores(:, 6)) ...
        ];

        fprintf('%s | %s\n', ...
            arquiteturas{a}, algoritmos{alg});

        fprintf('MSE:  %.4f +/- %.4f MPa^2\n', ...
            resumo(linha, 1), resumo(linha, 2));

        fprintf('RMSE: %.4f +/- %.4f MPa\n', ...
            resumo(linha, 3), resumo(linha, 4));

        fprintf('MAE:  %.4f +/- %.4f MPa\n', ...
            resumo(linha, 5), resumo(linha, 6));

        fprintf('R2:   %.4f +/- %.4f\n', ...
            resumo(linha, 7), resumo(linha, 8));

        fprintf('Tempo medio: %.2f segundos\n\n', ...
            resumo(linha, 9));

    end

end

% ============================================================
% GUARDAR RESULTADOS MAT
% ============================================================

save( ...
    fullfile(output_dir, 'avaliacao_completa.mat'), ...
    'resultados', ...
    'metricas', ...
    'resumo', ...
    'arquiteturas', ...
    'algoritmos', ...
    'y_real' ...
);

% ============================================================
% GUARDAR CSV INDIVIDUAL
% ============================================================

ficheiro = fullfile( ...
    output_dir, 'metricas_270_experiencias.csv');

fid = fopen(ficheiro, 'w');

if fid < 0
    error('Nao foi possivel criar o CSV individual.');
end

fprintf(fid, ...
    'Arquitetura,Algoritmo,Repeticao,MSE,RMSE,MAE,R2,Tempo,MSE_Validacao\n');

for i = 1:n_total

    r = resultados(i);

    fprintf(fid, ...
        '%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.4f,%.8f\n', ...
        r.arquitetura, ...
        r.algoritmo, ...
        r.repeticao, ...
        r.MSE, ...
        r.RMSE, ...
        r.MAE, ...
        r.R2, ...
        r.tempo, ...
        r.mse_validacao ...
    );

end

fclose(fid);

% ============================================================
% GUARDAR CSV RESUMO
% ============================================================

ficheiro = fullfile( ...
    output_dir, 'resumo_9_combinacoes.csv');

fid = fopen(ficheiro, 'w');

if fid < 0
    error('Nao foi possivel criar o CSV resumo.');
end

fprintf(fid, ...
    ['Arquitetura,Algoritmo,' ...
     'MSE_Media,MSE_Desvio,' ...
     'RMSE_Media,RMSE_Desvio,' ...
     'MAE_Media,MAE_Desvio,' ...
     'R2_Media,R2_Desvio,' ...
     'Tempo_Medio,MSE_Validacao_Medio\n']);

linha = 0;

for a = 1:numel(arquiteturas)

    for alg = 1:numel(algoritmos)

        linha = linha + 1;

        fprintf(fid, '%s,%s', ...
            arquiteturas{a}, algoritmos{alg});

        for j = 1:size(resumo, 2)
            fprintf(fid, ',%.8f', resumo(linha, j));
        end

        fprintf(fid, '\n');

    end

end

fclose(fid);

fprintf('========================================\n');
fprintf('AVALIACAO CONCLUIDA\n');
fprintf('Modelos avaliados: %d\n', indice);
fprintf('Resultados: %s\n', output_dir);
fprintf('========================================\n');
