
% Análise exploratória do dataset Concrete
% Redes Neuronais Artificiais - Assignment 2

clear;
clc;
close all;

% Carregar os dados
dados = dlmread('src/concrete/Concrete_Data.csv', ',', 1, 0);

% Nomes das variáveis
nomes = {
    'Cimento',
    'Escória',
    'Cinzas volantes',
    'Água',
    'Superplastificante',
    'Agregado grosso',
    'Agregado fino',
    'Idade',
    'Resistência'
};

% Apresentar informações gerais
fprintf('Número de observações: %d\n', size(dados, 1));
fprintf('Número de variáveis: %d\n', size(dados, 2));
fprintf('Valores em falta: %d\n', sum(isnan(dados(:))));

% Criar histogramas das nove variáveis
figure('Name', 'Histogramas - Concrete');

for i = 1:9
    subplot(3, 3, i);

    hist(dados(:, i), 20);

    title(nomes{i});
    xlabel('Valor');
    ylabel('Frequência');
    grid on;
end

% Guardar a figura
if ~exist('results/concrete', 'dir')
    mkdir('results/concrete');
end

print('results/concrete/histogramas_concrete.png', '-dpng', '-r150');

fprintf('Histogramas gerados com sucesso.\n');


% Matriz de correlação entre as variáveis
matriz_correlacao = corrcoef(dados);

figure('Name', 'Matriz de Correlação - Concrete');

imagesc(matriz_correlacao);
colorbar;
caxis([-1, 1]);
colormap(jet);

set(gca, ...
    'XTick', 1:9, ...
    'YTick', 1:9, ...
    'XTickLabel', nomes, ...
    'YTickLabel', nomes, ...
    'XTickLabelRotation', 45);

title('Matriz de Correlação - Concrete');

% Guardar o gráfico
print('results/concrete/matriz_correlacao.png', '-dpng', '-r150');

fprintf('Matriz de correlação gerada com sucesso.\n');


% Gráficos de dispersão das entradas face à resistência
figure('Name', 'Dispersão - Concrete');

for i = 1:8
    subplot(2, 4, i);

    scatter(dados(:, i), dados(:, 9), 8, 'filled');

    xlabel(nomes{i});
    ylabel('Resistência (MPa)');
    title(nomes{i});
    grid on;
end

% Guardar a figura
print('results/concrete/dispersao_concrete.png', ...
      '-dpng', '-r150');

fprintf('Gráficos de dispersão gerados com sucesso.\n');
