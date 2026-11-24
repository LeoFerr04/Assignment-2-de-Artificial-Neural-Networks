function y_pred = persistence_baseline(X, p)
    % Prevê o dia seguinte com a temperatura mais recente de cada janela.
    narginchk(2, 2);
    if ~isnumeric(X) || ~isreal(X) || ~ismatrix(X) || isempty(X) || ...
            any(isnan(X(:)) | isinf(X(:)))
        error('persistence_baseline:InvalidInputs', ...
            'X deve ser uma matriz numérica real, não vazia e com valores finitos.');
    end
    if ~isnumeric(p) || ~isreal(p) || ~isscalar(p) || ...
            isnan(p) || isinf(p) || p < 1 || p ~= fix(p)
        error('persistence_baseline:InvalidWindowSize', 'p deve ser um inteiro positivo.');
    end

    p = double(p);
    n_columns = size(X, 2);
    if n_columns ~= p && n_columns ~= p + 2
        error('persistence_baseline:ColumnMismatch', ...
            'X deve ter p colunas de temperaturas ou p+2 colunas com as características sazonais.');
    end

    % A coluna p contém a última temperatura, na escala original.
    y_pred = X(:, p);
end
