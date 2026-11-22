
function [dates, temperatures] = load_melbourne(csv_path)
    % Le o dataset Melbourne e devolve dois vetores coluna:
    % dates: datas no formato numerico MATLAB/Octave (datenum)
    % temperatures: temperaturas minimas diarias em graus Celsius

    % 1. Abrir o ficheiro CSV
    [file_id, message] = fopen(csv_path, 'r');

    if file_id == -1
        error('load_melbourne:OpenFailed', ...
            'Nao foi possivel abrir "%s": %s', csv_path, message);
    end

    % Garantir que o ficheiro e fechado automaticamente
    cleanup = onCleanup(@() fclose(file_id));

    % 2. Definir o cabecalho, rodape e formatos esperados
    expected_header = '"Date","Daily minimum temperatures in Melbourne, Australia, 1981-1990"';
    footer = 'Daily minimum temperatures in Melbourne, Australia, 1981-1990';

    date_pattern = '^[0-9]{4}-[0-9]{2}-[0-9]{2}$';
    number_pattern = '^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$';

    % 3. Inicializar os vetores e variaveis de controlo
    dates = zeros(0, 1);
    temperatures = zeros(0, 1);

    line_number = 0;
    found_header = false;
    found_footer = false;

    % 4. Ler o ficheiro linha a linha
    while true
        line = fgetl(file_id);

        % Terminar quando chegar ao fim do ficheiro
        if ~ischar(line)
            break;
        end

        line_number = line_number + 1;
        line = strtrim(line);

        % Ignorar linhas vazias
        if isempty(line)
            continue;
        end

        % 5. Validar o cabecalho
        if ~found_header
            if ~strcmp(line, expected_header)
                error('load_melbourne:InvalidHeader', ...
                    'Linha %d: cabecalho inesperado.', line_number);
            end

            found_header = true;
            continue;
        end

        % Nao permitir conteudo depois do rodape
        if found_footer
            error('load_melbourne:AfterFooter', ...
                'Linha %d: conteudo inesperado apos o rodape.', ...
                line_number);
        end

        % Identificar e ignorar o rodape
        if strcmp(line, footer)
            found_footer = true;
            continue;
        end

        % 6. Separar a data e a temperatura
        separator = find(line == ',');

        if numel(separator) ~= 1
            error('load_melbourne:InvalidRow', ...
                'Linha %d: esperava uma data e uma temperatura.', ...
                line_number);
        end

        date_text = strtrim(line(1:separator-1));
        temperature_text = strtrim(line(separator+1:end));

        % Remover as aspas exteriores, caso existam
        date_text = regexprep(date_text, '^"([^"]*)"$', '$1');
        temperature_text = regexprep(temperature_text, '^"([^"]*)"$', '$1');

        % 7. Validar o formato da data
        if isempty(regexp(date_text, date_pattern, 'once'))
            error('load_melbourne:InvalidDate', ...
                'Linha %d: data invalida "%s".', ...
                line_number, date_text);
        end

        % Extrair ano, mes e dia
        year = str2double(date_text(1:4));
        month = str2double(date_text(6:7));
        day = str2double(date_text(9:10));

        % Converter para o formato numerico de data
        serial_date = datenum(year, month, day);

        % Confirmar que a data existe no calendario
        calendar_date = datevec(serial_date);

        if ~isequal(calendar_date(1:3), [year, month, day])
            error('load_melbourne:InvalidDate', ...
                'Linha %d: data inexistente "%s".', ...
                line_number, date_text);
        end

        % 8. Validar a temperatura
        % Valores com '?' sao suspeitos e ficam representados por NaN.
        % O CSV original nao e alterado.

        if ~isempty(temperature_text) && temperature_text(1) == '?'

            % Retirar o sinal de duvida
            numeric_part = temperature_text(2:end);

            % Validar o numero que aparece depois do '?'
            if isempty(regexp(numeric_part, number_pattern, 'once')) || ...
                    ~isfinite(str2double(numeric_part))

                error('load_melbourne:InvalidTemperature', ...
                    'Linha %d: temperatura invalida "%s".', ...
                    line_number, temperature_text);
            end

            % Avisar que a observacao e suspeita
            warning('load_melbourne:SuspiciousTemperature', ...
                'Linha %d: temperatura suspeita "%s". Marcada como NaN.', ...
                line_number, temperature_text);

            temperature = NaN;

        else

            % Converter uma temperatura normal
            temperature = str2double(temperature_text);

            % Verificar se e um numero real finito
            if isempty(regexp(temperature_text, number_pattern, 'once')) || ...
                    ~isfinite(temperature)

                error('load_melbourne:InvalidTemperature', ...
                    'Linha %d: temperatura invalida "%s".', ...
                    line_number, temperature_text);
            end

        end

        % 9. Verificar se as datas estao ordenadas
        if ~isempty(dates) && serial_date <= dates(end)
            error('load_melbourne:DateOrder', ...
                'Linha %d: data repetida ou fora de ordem: %s.', ...
                line_number, date_text);
        end

        % 10. Guardar a observacao
        dates(end+1, 1) = serial_date;
        temperatures(end+1, 1) = temperature;

    end

    % 11. Verificar se houve erros de leitura
    if ~feof(file_id)
        error('load_melbourne:ReadFailed', ...
            'Erro ao ler o ficheiro "%s".', csv_path);
    end

    % Confirmar que existem observacoes
    if isempty(dates)
        error('load_melbourne:NoData', ...
            'O ficheiro nao contem observacoes.');
    end

    % 12. Detetar dias em falta na serie temporal
    day_steps = diff(dates);

    for index = find(day_steps > 1).'

        missing_days = day_steps(index) - 1;

        warning('load_melbourne:DateGap', ...
            'Existem %d dia(s) em falta entre %s e %s.', ...
            missing_days, ...
            datestr(dates(index), 'yyyy-mm-dd'), ...
            datestr(dates(index+1), 'yyyy-mm-dd'));

    end

end
