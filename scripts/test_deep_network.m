
% Teste da arquitetura C - Deep
% 10 camadas escondidas com 10 neuronios cada
% Compativel com GNU Octave e MATLAB

clear;
clc;

addpath(genpath(pwd));

rng(42);

% Criar dados artificiais
X = randn(20, 8);
y = randn(20, 1);

% Definir a arquitetura da rede
layers = [8, repmat(10, 1, 10), 1];

% Inicializar os pesos com Xavier
blocks = cell(1, 2 * (numel(layers) - 1));

for i = 1:(numel(layers) - 1)

    n_in = layers(i);
    n_out = layers(i + 1);

    limite = sqrt(6 / (n_in + n_out));

    W = (2 * rand(n_out, n_in) - 1) * limite;
    b = zeros(n_out, 1);

    blocks{2 * i - 1} = W;
    blocks{2 * i} = b;
end

% Calcular a perda e os gradientes
[loss, gradients] = mlp_deep_loss_gradient(blocks, X, y);

% Verificar os resultados
fprintf('Arquitetura Deep\n');
fprintf('Camadas escondidas: 10\n');
fprintf('Neuronios por camada: 10\n');
fprintf('Perda MSE: %.6f\n', loss);
fprintf('Numero de blocos: %d\n', numel(gradients));

% Verificar se os gradientes sao finitos
gradientes_validos = true;

for i = 1:numel(gradients)
    if any(~isfinite(gradients{i}(:)))
        gradientes_validos = false;
    end
end

if gradientes_validos
    fprintf('Gradientes validos: SIM\n');
else
    fprintf('Gradientes validos: NAO\n');
end

% Verificar as dimensoes dos gradientes
dimensoes_validas = true;

for i = 1:numel(blocks)
    if ~isequal(size(blocks{i}), size(gradients{i}))
        dimensoes_validas = false;
    end
end

if dimensoes_validas
    fprintf('Dimensoes dos gradientes: CORRETAS\n');
else
    fprintf('Dimensoes dos gradientes: INCORRETAS\n');
end


% Verificacao numerica dos gradientes
epsilon = 1e-5;

% Converter os blocos num vetor
[objective_fn, theta0] = make_network_objective( ...
    blocks, ...
    @(b) mlp_deep_loss_gradient(b, X, y));

% Calcular o gradiente analitico
[~, grad_analitico] = objective_fn(theta0);

% Calcular o gradiente por diferencas finitas
grad_numerico = zeros(size(theta0));

for i = 1:numel(theta0)

    theta_mais = theta0;
    theta_menos = theta0;

    theta_mais(i) = theta_mais(i) + epsilon;
    theta_menos(i) = theta_menos(i) - epsilon;

    perda_mais = objective_fn(theta_mais);
    perda_menos = objective_fn(theta_menos);

    grad_numerico(i) = ...
        (perda_mais - perda_menos) / (2 * epsilon);
end

% Calcular o erro relativo
erro_relativo = norm(grad_analitico - grad_numerico) / ...
    max(1e-12, norm(grad_analitico) + norm(grad_numerico));

fprintf('Erro relativo dos gradientes: %.10e\n', erro_relativo);

if erro_relativo < 1e-5
    fprintf('Verificacao numerica: APROVADA\n');
else
    fprintf('Verificacao numerica: REVER GRADIENTES\n');
end
