function main_assignment2()
% ============================================================
% ARTIFICIAL NEURAL NETWORKS - ASSIGNMENT 2 (2026/27)
% GNU Octave / MATLAB - um unico script + ann_lib.m
%
% USO:
%   1) Colocar este ficheiro e ann_lib.m na raiz do repositorio.
%   2) Manter os dois CSVs nas pastas data/ e src/concrete/.
%   3) Executar: main_assignment2
%
% MODOS:
%   'reuse' (defeito): avalia 540 modelos ja treinados em
%       results/concrete/final e results/melbourne/final.
%       Nao volta a treinar; exporta tabelas, baselines, figuras.
%       Os checkpoints historicos de Melbourne usaram 100 epocas
%       para Adam/GD por uma incompatibilidade de configuracao.
%       O BFGS antigo nao foi limitado a 200 avaliaces completas.
%       Por isso, 'reuse' NAO demonstra equivalencia de orÃ§amento.
%
%   'full': executa novo tuning e treinos finais para ambos os
%       datasets com max_epochs=200, batch=32 e BFGS limitado
%       a 200 avaliaces full-batch (incluindo pesquisa linear).
%       Guarda separadamente em results_compliant/; usa checkpoints
%       para retomar execucoes. Este e o modo reproduzivel.
% ============================================================

clear;
clc;
close all;

cfg = struct();
cfg.mode = 'full';            % entrega reproduzivel; 'reuse' so para checkpoints antigos
cfg.budget = 200;            % full-data epoch equivalents
cfg.batch_size = 32;
cfg.n_repeticoes_tuning = 5;
cfg.n_repeticoes_final = 30;
cfg.learning_rates = logspace(-4,-1,6);

root = fileparts(mfilename('fullpath'));
if isempty(root), root = pwd; end
addpath(root);
L = ann_lib();

fprintf('\n==============================================\n');
fprintf('ASSIGNMENT 2 - REDES NEURONAIS ARTIFICIAIS\n');
fprintf('Modo: %s | OrÃ§amento: %d epoch-equivalents\n', ...
    cfg.mode,cfg.budget);
fprintf('==============================================\n');

if strcmp(cfg.mode,'full')
    base_results = fullfile(root,'results_compliant');
elseif strcmp(cfg.mode,'reuse')
    base_results = fullfile(root,'results');
    fprintf(['AVISO: checkpoints historicos de Melbourne usam 100 ' ...
        'epocas de Adam/GD; BFGS nao respeitou um limite comum.\n']);
else
    error('cfg.mode deve ser reuse ou full.');
end

names = {'concrete','melbourne'};
archs = {'Reference','Wide','Deep'};
methods = {'Adam','Gradient_Descent','BFGS'};

for ds=1:numel(names)
    name = names{ds};
    fprintf('\n========== DATASET: %s ==========\n',upper(name));
    D = L.prepare_dataset(L,name,root);
    workdir = fullfile(base_results,name);
    if ~exist(workdir,'dir'),mkdir(workdir);end
    plotdir = fullfile(workdir,'plots');
    if ~exist(plotdir,'dir'),mkdir(plotdir);end

    L.run_eda(D,plotdir);

    if strcmp(cfg.mode,'full')
        % A selecao do hiperparametro so utiliza treino/validacao.
        best_rates = full_tuning(L,D,cfg,workdir,archs);
        full_final(L,D,cfg,workdir,archs,methods,best_rates);
    end

    modelsdir = fullfile(workdir,'final');
    evaldir = fullfile(workdir,'evaluation');
    if ~exist(modelsdir,'dir')
        error('Pasta dos modelos inexistente: %s',modelsdir);
    end
    if strcmp(cfg.mode,'reuse') || cfg.n_repeticoes_final==30
        L.evaluate_final(L,D,modelsdir,evaldir);
        L.plot_convergence(D,modelsdir,plotdir);
    else
        fprintf('Teste reduzido: avaliacao de 30 seeds ignorada.\n');
    end
end

fprintf('\n==============================================\n');
fprintf('ESTUDO CONCLUIDO: tabelas, baselines e graficos.\n');
fprintf('Resultados: %s\n',base_results);
fprintf('==============================================\n');

% ============================================================
% Selecionar seis learning rates para Adam e GD, com cinco
% repeticoes independentes e modelos identicos por semente.
% ============================================================
end

function best_rates = full_tuning(L,D,cfg,workdir,archs)
    outdir = fullfile(workdir,'tuning');
    if ~exist(outdir,'dir'),mkdir(outdir);end
    checkpoint = fullfile(outdir,'tuning_200ep.mat');
    nrep = cfg.n_repeticoes_tuning;
    values = NaN(3,2,numel(cfg.learning_rates),nrep);
    if exist(checkpoint,'file')
        saved=load(checkpoint);
        if isfield(saved,'values') && isequal(size(saved.values),size(values))
            values=saved.values;
            fprintf('Checkpoint de tuning carregado.\n');
        end
    end
    for a=1:3
        layers=architecture_layers(size(D.Xtr,2),a);
        for m=1:2
            for lr=1:numel(cfg.learning_rates)
                for rep=1:nrep
                    if isfinite(values(a,m,lr,rep)),continue;end
                    [theta0,~]=L.init_theta(layers,rep);
                    batch_fn=@(theta,indices) L.objective_batch( ...
                        L,theta,layers,D.Xtr(indices,:),D.ytr(indices));
                    train_fn=@(theta) L.objective_loss( ...
                        L,theta,layers,D.Xtr,D.ytr);
                    val_fn=@(theta) L.objective_loss( ...
                        L,theta,layers,D.Xva,D.yva);
                    opts=struct('learning_rate',cfg.learning_rates(lr), ...
                        'batch_size',cfg.batch_size, ...
                        'max_epochs',cfg.budget,'seed',42);
                    if m==1
                        [theta_best,~]=L.train_adam(theta0,batch_fn, ...
                            train_fn,val_fn,size(D.Xtr,1),opts);
                    else
                        [theta_best,~]=L.train_gradient_descent(theta0, ...
                            batch_fn,train_fn,val_fn,size(D.Xtr,1),opts);
                    end
                    score=val_fn(theta_best);
                    if ~isfinite(score),score=realmax;end
                    values(a,m,lr,rep)=score;
                    save(checkpoint,'values','cfg');
                    fprintf('TUNING %s %s lr %.6g rep %02d: %.5f\n', ...
                        archs{a},algorithm_name(m), ...
                        cfg.learning_rates(lr),rep,score);
                end
            end
        end
    end
    best_rates=zeros(3,2);
    fid=fopen(fullfile(outdir,'selected_learning_rates.csv'),'w');
    fprintf(fid,'architecture,algorithm,learning_rate,validation_MSE_mean\n');
    for a=1:3
        for m=1:2
            scores=zeros(numel(cfg.learning_rates),1);
            for k=1:numel(scores)
                x=reshape(values(a,m,k,:),[],1);
                scores(k)=mean(x);
            end
            [v,idx]=min(scores);
            best_rates(a,m)=cfg.learning_rates(idx);
            fprintf(fid,'%s,%s,%.10g,%.10g\n', ...
                archs{a},algorithm_name(m),best_rates(a,m),v);
            fprintf('LR otimizado %s %s = %.7g\n', ...
                archs{a},algorithm_name(m),best_rates(a,m));
        end
    end
    fclose(fid);
    save(fullfile(outdir,'selected_learning_rates.mat'),'best_rates');
end

% ============================================================
% Comparacao final: 3 arquiteturas x 3 algoritmos x 30 seeds.
% O ficheiro por modelo guarda pesos iniciais, melhores pesos,
% historico, configuracao, tempo e perda de validacao.
% ============================================================
function full_final(L,D,cfg,workdir,archs,methods,best_rates)
    outdir=fullfile(workdir,'final');
    if ~exist(outdir,'dir'),mkdir(outdir);end
    for a=1:3
        layers=architecture_layers(size(D.Xtr,2),a);
        for m=1:3
            for rep=1:cfg.n_repeticoes_final
                dest=fullfile(outdir,sprintf('%s_%s_rep%02d.mat', ...
                    archs{a},methods{m},rep));
                if exist(dest,'file'),continue;end
                [theta0,blocks]=L.init_theta(layers,rep);
                batch_fn=@(theta,indices) L.objective_batch( ...
                    L,theta,layers,D.Xtr(indices,:),D.ytr(indices));
                train_fn=@(theta) L.objective_loss( ...
                    L,theta,layers,D.Xtr,D.ytr);
                val_fn=@(theta) L.objective_loss( ...
                    L,theta,layers,D.Xva,D.yva);
                opts=struct();
                tic;
                if m<3
                    opts=struct('learning_rate',best_rates(a,m), ...
                        'batch_size',cfg.batch_size, ...
                        'max_epochs',cfg.budget,'seed',42);
                    if m==1
                        [theta_best,history]=L.train_adam( ...
                            theta0,batch_fn,train_fn,val_fn, ...
                            size(D.Xtr,1),opts);
                    else
                        [theta_best,history]=L.train_gradient_descent( ...
                            theta0,batch_fn,train_fn,val_fn, ...
                            size(D.Xtr,1),opts);
                    end
                    info=struct();
                else
                    % Todos os pontos avaliados por Armijo contam.
                    % max_evals=200: mesmo orÃ§amento de gradiente
                    % em termos de equivalentes ao dataset completo.
                    loss_fn=@(b) L.mlp_deep_loss_gradient(b,D.Xtr,D.ytr);
                    [obj_bfgs,checktheta]=L.make_network_objective( ...
                        blocks,loss_fn);
                    assert(norm(theta0-checktheta)<1e-10);
                    opts=struct('max_iter',cfg.budget*3, ...
                        'max_evals',cfg.budget,'max_params',2000, ...
                        'grad_tol',1e-6,'step_tol',1e-10, ...
                        'verbose',false);
                    [theta_best,info]=L.bfgs_optimize( ...
                        obj_bfgs,theta0,opts,val_fn);
                    history=struct('train_loss',info.loss_history, ...
                        'val_loss',info.val_loss_history, ...
                        'epoch_equivalents',info.epoch_equivalents, ...
                        'n_evals',info.n_evals);
                end
                tempo=toc;
                mse_validacao=val_fn(theta_best);
                if ~isfinite(mse_validacao)
                    error('Loss invalida %s / %s / seed %d', ...
                        D.name,methods{m},rep);
                end
                arquitetura=archs{a};
                algoritmo=methods{m};
                save(dest,'theta_best','theta0','history','info', ...
                    'tempo','mse_validacao','opts','layers','rep', ...
                    'arquitetura','algoritmo');
                fprintf('FINAL %s %s rep%02d MSEval %.6g time %.1fs\n', ...
                    arquitetura,algoritmo,rep,mse_validacao,tempo);
            end
        end
    end
end

function layers=architecture_layers(n_inputs,a)
    switch a
        case 1
            layers=[n_inputs,10,1];
        case 2
            layers=[n_inputs,100,1];
        case 3
            layers=[n_inputs,repmat(10,1,10),1];
        otherwise
            error('Arquitetura invalida.');
    end
end

function name=algorithm_name(m)
    if m==1,name='Adam';else name='Gradient_Descent';end
end



