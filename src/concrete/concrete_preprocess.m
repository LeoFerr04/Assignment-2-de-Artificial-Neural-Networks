
function data = concrete_preprocess(csv_path)
    % Preparação do dataset Concrete para as redes neuronais.
    % Divisão: 70% treino, 15% validação e 15% teste.
    % Normalização calculada apenas com os dados de treino.

    if nargin < 1
        csv_path = 'src/concrete/Concrete_Data.csv';
    end

    % Carregar o dataset.
    dados = dlmread(csv_path, ',', 1, 0);

    if size(dados, 2) ~= 9 || size(dados, 1) ~= 1030
        error('Dimensões inesperadas no dataset Concrete.');
    end

    if any(~isfinite(dados(:)))
        error('O dataset contém valores inválidos.');
    end

    % Separar entradas e variável de saída.
    X = dados(:, 1:8);
    y = dados(:, 9);

    % Divisão aleatória reprodutível.
    rng(42);

    n = size(X, 1);
    indices = randperm(n);

    n_train = floor(0.70 * n);
    n_val = floor(0.15 * n);

    idx_train = indices(1:n_train);
    idx_val = indices(n_train + 1:n_train + n_val);
    idx_test = indices(n_train + n_val + 1:end);

    % Separar os três conjuntos.
    Xtr = X(idx_train, :);
    Xva = X(idx_val, :);
    Xte = X(idx_test, :);

    ytr = y(idx_train);
    yva = y(idx_val);
    yte = y(idx_test);

    % Estatísticas calculadas exclusivamente no treino.
    media_X = mean(Xtr, 1);
    desvio_X = std(Xtr, 0, 1);

    media_y = mean(ytr);
    desvio_y = std(ytr);

    % Evitar divisões por zero.
    desvio_X(desvio_X == 0) = 1;

    if desvio_y == 0
        desvio_y = 1;
    end

    % Normalizar entradas.
    data.Xtr = bsxfun(@rdivide, ...
        bsxfun(@minus, Xtr, media_X), desvio_X);

    data.Xva = bsxfun(@rdivide, ...
        bsxfun(@minus, Xva, media_X), desvio_X);

    data.Xte = bsxfun(@rdivide, ...
        bsxfun(@minus, Xte, media_X), desvio_X);

    % Normalizar a variável de saída.
    data.ytr = (ytr - media_y) / desvio_y;
    data.yva = (yva - media_y) / desvio_y;
    data.yte = (yte - media_y) / desvio_y;

    % Guardar os valores originais para avaliação.
    data.ytr_original = ytr;
    data.yva_original = yva;
    data.yte_original = yte;

    % Guardar os parâmetros de normalização.
    data.media_X = media_X;
    data.desvio_X = desvio_X;
    data.media_y = media_y;
    data.desvio_y = desvio_y;

    % Guardar os índices para reproduzir a divisão.
    data.idx_train = idx_train;
    data.idx_val = idx_val;
    data.idx_test = idx_test;

    fprintf('Concrete preparado com sucesso.\n');
    fprintf('Treino: %d observações\n', numel(idx_train));
    fprintf('Validação: %d observações\n', numel(idx_val));
    fprintf('Teste: %d observações\n', numel(idx_test));
end
