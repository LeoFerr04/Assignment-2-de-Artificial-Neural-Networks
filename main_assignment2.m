
% ============================================================
% ASSIGNMENT 2 - ARTIFICIAL NEURAL NETWORKS
% Programa principal
%
% Universidade da Madeira - 2026/2027
% ============================================================

clear;
clc;
close all;

% Localizar a pasta do projeto.
project_dir = fileparts(mfilename('fullpath'));

if isempty(project_dir)
    project_dir = pwd;
end

% Carregar a biblioteca de funcoes.
lib = ann_lib();

fprintf('\n');
fprintf('========================================\n');
fprintf('ASSIGNMENT 2 - NEURAL NETWORKS\n');
fprintf('========================================\n');

fprintf('Biblioteca carregada: %d funcoes\n', ...
    numel(fieldnames(lib)));

% ============================================================
% DATASET 1 - CONCRETE
% ============================================================

fprintf('\nDATASET 1 - CONCRETE\n');

concrete_path = fullfile( ...
    project_dir, 'src', 'concrete', 'Concrete_Data.csv');

if exist(concrete_path, 'file')

    concrete = lib.concrete_preprocess(concrete_path);

    fprintf('Concrete carregado com sucesso.\n');

else

    warning('Dataset Concrete nao encontrado.');

end

% ============================================================
% DATASET 2 - MELBOURNE
% ============================================================

fprintf('\nDATASET 2 - MELBOURNE\n');

melbourne_path = fullfile( ...
    project_dir, 'data', ...
    'daily-minimum-temperatures-in-me.csv');

if exist(melbourne_path, 'file')

    [dates, temperatures] = ...
        lib.load_melbourne(melbourne_path);

    fprintf('Melbourne carregado com sucesso.\n');
    fprintf('Observacoes: %d\n', numel(temperatures));

else

    warning('Dataset Melbourne nao encontrado.');

end

fprintf('\n========================================\n');
fprintf('CARREGAMENTO INICIAL CONCLUIDO\n');
fprintf('========================================\n');
