clc
clear
close all

load('Scene15.mat');

% obtain the label vectors of training and test data
trls = vec2ind(H_train);
ttls = vec2ind(H_test);

trnX = training_feats';
trnY = trls';
tstX = testing_feats';
tstY = ttls';
nTst = length(ttls); % total number of test samples

% number of classes
ClassNum = length(unique(trnY));

numAtom = 450; %total number of atoms

% numer of atoms in each class
eachAtom = numAtom/ClassNum;

% label vector of atoms
Dlabel = repelem(1:ClassNum,eachAtom);

% label matrix of atoms
dictLabel = full(ind2vec(Dlabel,ClassNum));
dictsize = numAtom;

% construct the block diagonal matrix
Q = zeros(dictsize,length(trnY)); % energy matrix
for frameid=1:length(trnY)
    label_training = H_train(:,frameid);
    [maxv1,maxid1] = max(label_training);
    for itemid=1:dictsize
        label_item = dictLabel(:,itemid);
        [maxv2,maxid2] = max(label_item);
        if(maxid1==maxid2)
            Q(itemid,frameid) = 1;
        else
            Q(itemid,frameid) = 0;
        end
    end
end

% visualize Q
% imagesc(mat2gray(Q))

% parameters
epsilon = 0.05;
sqrt_alpha = 0.1;
mu = 1;
lambda = 1e-3;

% training stage
tic;
[T, Z, A] =lcTransformLearning(trnX',numAtom,Q, epsilon,sqrt_alpha,mu,lambda);
toc;

tic;
Z_train = T*trnX';
eta = 1e-5;
% learn a linear classifier
W = H_train*Z_train'/(Z_train*Z_train'+eta*eye(size(Z_train,1)));

% classification
Z_test = T*tstX';
Label_test_pred= W * Z_test;
[~, pred] = max(Label_test_pred);
pred=pred';
toc;

acc = sum(pred==tstY)/nTst*100;
fprintf('Recognition accuracy is %.2f%%\n',acc)


