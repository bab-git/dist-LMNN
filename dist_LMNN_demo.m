% This is the demo of distance-based LMNN algoeithm
% which is the modified version of LMNN method of Kilian Q. Weinberger

clc;clear; close all
main_path=fileparts(mfilename('fullpath'));
addpath(genpath(main_path));
cd(main_path)
%% loading Data
% 1- alldistances: a NxNxd squared-distance marix for pairwise distance of 
%                  N data points. For each of the d dimensions the distance
%                  are computed individually.
%                  e.g., alldistances(i,j,k)=(x(k,i)-x(k,j))^2
% 2- list_labels:  ground truth labels

% file_name='DTW_Squat_class9';
file_name='DTW_Words_25_class';
% file_name='DTW_Cricket_12_class';


load(file_name,'list_labels','alldistances','alldistances_feats')
if exist('alldistances_feats')
    alldistances=alldistances_feats;
end

if ~exist ('list_labels')
    disp('-- Please put labeling data into "list_labels"')
    disp('-- If you use labeling, please set all list_labels(i)=1')
end

%% calculate DTW of the data
if ~exist('alldistances')
    disp('--Please set distance matrix --> "alldistances"')
    disp('--or')
    disp('--To compute pairwise distances save each motion sequence "i" in "data_all{i}.data"')
    disp('--and run the following DTW_calc')
    return
    [~,alldistances]=DTW_calc(data_all,'DTW',0,1);
end
D_path=fullfile(main_path,'data/DTW_result');
save(D_path, 'alldistances')

%% ========= Main Paramters
Tr_rate=0.75; % poriton of trainig samples
pars.Kg=3;   %  LMNN algorithm neighborhood size
pars.Knn=1;  %  K for k-NN algorithm for evaluating the results
pars.mu=0.5; % trade-off between push and pull loss terms  
pars.outdim=size(alldistances,3);
% ========= other Paramters
pars.maxiter=3000;
pars.quiet=1;
pars.validation=0;   % a value in (0,1]: a portion of training used to estimate an efficient maxiter
                     % 0: uses the manual maxiter value
pars.subsample=1;
pars.classsplit=1;
pars.stepsize=1e-16;

%% test/train selection
N = size(alldistances,1);   
perm = randperm(N);
r_tr=perm(1:floor(N*Tr_rate));
r_te=perm(floor(N*Tr_rate)+1:end);
D_tr = alldistances(r_tr,r_tr,:);
D_te = alldistances([r_tr r_te],r_te,:);
% D_te = alldistances(r_te,r_te,:);
yTr = list_labels(r_tr)';
yTe = list_labels(r_te)';

%% distance-based LMNN
Nf=size(alldistances,3);
L0=rand(pars.outdim,Nf); % Initial L0
[L,Details] = lmnn_dist(D_tr,yTr,L0,pars);

%% Analysing the results
Acc_knn=knn_eval_dis(1,alldistances,r_tr,yTr,r_te,yTe,pars.Knn);
Acc_LMNN=knn_eval_dis(L,alldistances,r_tr,yTr,r_te,yTe,pars.Knn);

disp(sprintf('Accuracy of kNN classifier=%2.2f%%',(Acc_knn)))
disp(sprintf('Accuracy of LMNN classifier=%2.2f%%',(Acc_LMNN)))