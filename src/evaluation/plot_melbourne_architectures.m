
function fig = plot_melbourne_architectures(csv_path, output_path)
    % Compara o RMSE de validacao das arquiteturas da MLP.
    % Os resultados sao lidos do CSV, sem repetir os treinos.
    narginchk(2, 2);

    % 1. Abrir o CSV
    [fid, message] = fopen(csv_path, 'r');
    if fid == -1
        error('Nao foi possivel abrir o CSV: %s', message);
    end
    cleanup = onCleanup(@() fclose(fid));

    header = fgetl(fid);
    expected_header = 'neuronios,semente,rmse,mae,r2,iteracoes,estado';

    if ~ischar(header) || ~strcmp(strtrim(header), expected_header)
        error('O cabecalho do CSV nao corresponde ao esperado.');
    end

    % 2. Ler as colunas
    data = textscan(fid, '%f%f%f%f%f%f%s', ...
        'Delimiter', ',');

    neuronios = data{1};
    sementes = data{2};
    rmse = data{3};

    if isempty(rmse) || any(~isfinite(rmse))
        error('O CSV nao contem valores RMSE validos.');
    end

    % 3. Organizar os resultados
    arquiteturas = unique(neuronios);
    sementes_unicas = unique(sementes);

    n_arq = numel(arquiteturas);
    n_seeds = numel(sementes_unicas);

    resultados = NaN(n_arq, n_seeds);

    for i = 1:n_arq
        for j = 1:n_seeds
            indices = neuronios == arquiteturas(i) & ...
                      sementes == sementes_unicas(j);

            if nnz(indices) ~= 1
                error('Cada combinacao de arquitetura e semente deve aparecer uma vez.');
            end

            resultados(i,j) = rmse(indices);
        end
    end

    medias = mean(resultados, 2);
    desvios = std(resultados, 0, 2);

    % 4. Criar o grafico
    fig = figure('Name', 'Comparacao das arquiteturas MLP', ...
        'NumberTitle', 'off', 'Color', 'w', ...
        'Position', [100, 100, 950, 520]);

    hold on;

    cores = lines(n_seeds);
    legendas = cell(1, n_seeds + 1);

    for j = 1:n_seeds
        plot(arquiteturas, resultados(:,j), '-o', ...
            'Color', cores(j,:), ...
            'LineWidth', 1.2, ...
            'MarkerSize', 7);

        legendas{j} = sprintf('Semente %d', sementes_unicas(j));
    end

    plot(arquiteturas, medias, 'k-s', ...
        'LineWidth', 2.5, ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', 'k');

    legendas{end} = 'RMSE medio';

    xlabel('Numero de neuronios escondidos');
    ylabel('RMSE de validacao (graus Celsius)');
    title('Comparacao das arquiteturas MLP com BFGS');

    set(gca, 'XTick', arquiteturas);
    grid on;
    legend(legendas, 'Location', 'southeast');

    hold off;

    % 5. Guardar em PNG
    print(fig, output_path, '-dpng', '-r150');

    % 6. Apresentar o resumo no terminal
    disp('===== RESUMO DAS ARQUITETURAS =====');
    fprintf('Neuronios | RMSE medio | Desvio padrao\n');

    for i = 1:n_arq
        fprintf('%9d | %.4f | %.4f\n', ...
            arquiteturas(i), medias(i), desvios(i));
    end

    [~, melhor] = min(medias);
    fprintf('\nMelhor arquitetura: %d neuronios\n', ...
        arquiteturas(melhor));

    disp('Grafico criado com sucesso!');
end
