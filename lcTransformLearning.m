function [T, Z, A] = lcTransformLearning (X, labels, numOfAtoms,...
    Q, epsilon, sqrt_alpha, mu)

% solves ||TX - Z||_Fro - mu*logdet(T) + eps*mu||T||_Fro + lambda||Q-WZ||_Fro

% Inputs
% X          - Training Data
% labels     - Class labels
% numOfAtoms - dimensionaity after Transform
% mu         - regularizer for Tranform
% lambda     - regularizer for coefficient
% eps        - regularizer for Transform
% type       - 'soft' or 'hard' update: default is 'soft'
% Output
% T          - learnt Transform
% Z          - learnt sparse coefficients
% W          - linear map
if nargin < 7
    mu = 1;
end
if nargin < 6
    sqrt_alpha = 1;
end
if nargin < 5
    epsilon = 0.1;
end

maxIter = 10;
type = 'soft'; % default 'soft'

rng(1); % repeatable
T = randn(numOfAtoms, size(X,1));
Z = T*X;

numOfSamples = length(labels);
if min(labels) == 0
    labels = labels + 1;
end

numOfClass = max(labels);
% Q = zeros(numOfClass,numOfSamples);
% for i = 1:numOfSamples
%     Q(labels(i),i) = 1;
% end
% W = Q / Z;
A = Q*Z'/(Z*Z'+eye(size(Z,1)));

invL = (X*X' + epsilon*mu*eye(size(X,1)))^(-0.5);
% scene15, sparseL=30
sparseL = 30;
lambda = 1e-3;
for i = 1:maxIter
    
    % update Transform T
    [U,S,V] = svd(invL*X*Z');
    if numOfAtoms>size(X,1)
        D = [diag(diag(S) + (diag(S).^2 + 2*epsilon).^0.5);zeros(numOfAtoms-size(X,1),size(X,1))];
    else
        %         D1 = diag(diag(S) + (diag(S).^2 + 2*epsilon).^0.5);
        %         D2 = zeros(numOfAtoms,size(X,1)-numOfAtoms);
        D = [diag(diag(S) + (diag(S).^2 + 2*epsilon).^0.5) zeros(numOfAtoms,size(X,1)-numOfAtoms)];
    end
    T = 0.5*V*D*U'*invL;
    
    % update Coefficients Z
    %     TX = T*X;
    %     Y = [TX;sqrt_alpha*Q];
    %     D = [eye(size(TX,1),size(A,2));sqrt_alpha*A];
    %
    %     Y = normc(Y);
    %     D = normc(D);
    %     G = D'*D;
    %     Z = omp(D'*Y,G,sparseL);
    
    Z = ((lambda+1)*eye(size(A,2)) + sqrt_alpha*(A'*A))\(T*X + sqrt_alpha*A'*Q);
    
    %     Z = (eye(size(A,2)) + lambda*(A'*A))\(T*X + lambda*A'*Q);
    
    % update map
    %     W = Q / Z;
    A = Q*Z'/(Z*Z'+eye(size(Z,1)));
end
