function [Xtr, Xv, Xte, ytr, yv, yte, params] = normalize_melbourne_data(X_train, X_val, X_test, y_train, y_val, y_test)
    % Normaliza as temperaturas com a média e o desvio-padrão dos alvos de treino.
    narginchk(6, 6);

    input_sets = {X_train, X_val, X_test};
    target_sets = {y_train, y_val, y_test};
    set_names = {'treino', 'validação', 'teste'};
    n_columns = size(X_train, 2);
    for k = 1:numel(input_sets)
        inputs = input_sets{k};
        targets = target_sets{k};
        if ~isnumeric(inputs) || ~isreal(inputs) || ~ismatrix(inputs) || ...
                any(isnan(inputs(:)) | isinf(inputs(:)))
            error('normalize_melbourne_data:InvalidInputs', ...
                'As entradas de %s devem formar uma matriz numérica real com valores finitos.', set_names{k});
        end
        if n_columns < 3 || size(inputs, 2) ~= n_columns
            error('normalize_melbourne_data:ColumnMismatch', ...
                'As matrizes X devem ter o mesmo número de colunas, com pelo menos três.');
        end
        if ~isnumeric(targets) || ~isreal(targets) || ...
                (~isvector(targets) && ~isequal(size(targets), [0, 0])) || ...
                any(isnan(targets(:)) | isinf(targets(:)))
            error('normalize_melbourne_data:InvalidTargets', ...
                'Os alvos de %s devem formar um vetor numérico real com valores finitos.', set_names{k});
        end
        if numel(targets) ~= size(inputs, 1)
            error('normalize_melbourne_data:RowMismatch', ...
                'Cada linha de X no conjunto de %s deve ter um alvo correspondente.', set_names{k});
        end
    end
    if isempty(y_train)
        error('normalize_melbourne_data:EmptyTrainingSet', ...
            'É necessário pelo menos um alvo de treino para calcular a normalização.');
    end

    % Usa o desvio-padrão amostral e aplica escala unitária a alvos constantes.
    training_targets = full(double(y_train(:)));
    if all(training_targets == training_targets(1))
        temp_mean = training_targets(1);
        temp_std = 1;
    else
        temp_mean = mean(training_targets);
        temp_std = std(training_targets);
        if isnan(temp_mean) || isinf(temp_mean) || isnan(temp_std) || isinf(temp_std) || temp_std == 0
            % Ajusta a escala quando os valores intermédios excedem a precisão numérica.
            magnitude = max(abs(training_targets));
            scaled_targets = training_targets ./ magnitude;
            temp_mean = mean(scaled_targets) * magnitude;
            temp_std = std(scaled_targets) * magnitude;
        end
        if temp_std == 0
            temp_std = 1;
        end
        if isnan(temp_mean) || isinf(temp_mean) || isnan(temp_std) || isinf(temp_std)
            error('normalize_melbourne_data:InvalidScale', ...
                'As estatísticas dos alvos de treino excedem o intervalo numérico suportado.');
        end
    end

    % Aplica os mesmos parâmetros aos três conjuntos e conserva o seno e o cosseno.
    p = n_columns - 2;
    Xtr = double(X_train);
    Xv = double(X_val);
    Xte = double(X_test);
    Xtr(:, 1:p) = (Xtr(:, 1:p) - temp_mean) ./ temp_std;
    Xv(:, 1:p) = (Xv(:, 1:p) - temp_mean) ./ temp_std;
    Xte(:, 1:p) = (Xte(:, 1:p) - temp_mean) ./ temp_std;
    ytr = (double(y_train) - temp_mean) ./ temp_std;
    yv = (double(y_val) - temp_mean) ./ temp_std;
    yte = (double(y_test) - temp_mean) ./ temp_std;
    params = struct('temp_mean', temp_mean, 'temp_std', temp_std, 'p', p);
end
