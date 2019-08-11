function [T, Z, A] = lcTransformLearning (X, numOfAtoms,...
    Q, epsilon, sqrt_alpha, mu, lambda)

% solves min_{T,Z,A}||TX - Z||_F^2 - mu*logdet(T) + eps*mu||T||_F^2
% + alpha||Q-AZ||_F^2+lambda||Z||_1

% Inputs
% X          - Training Data
% numOfAtoms - dimensionaity after Transform
% Q          - the ideal representation matrix
% epsilon    - regularizer for Transform
% sqrt_alpha - regularizer for discriminative sparse code error
% mu         - regularizer for Tranform
% lambda     - regularizer for sparse coefficient

% Output
% T          - learned Transform
% Z          - learned sparse coefficients
% A          - learned transform matrix

if nargin < 7
    mu = 1;
end
if nargin < 6
    sqrt_alpha = 1;
end
if nargin < 5
    epsilon = 0.1;
end

% number of maximum iteration
maxIter = 10;

rng(1); % repeatable
T = randn(numOfAtoms, size(X,1));
Z = T*X;

% initialization for A
A = Q*Z'/(Z*Z'+eye(size(Z,1)));

invL = (X*X' + epsilon*mu*eye(size(X,1)))^(-0.5);
% scene15, sparseL=30
sparseL = 30;

for i = 1:maxIter
    
    % update Transform T
    [U,S,V] = svd(invL*X*Z');
    if numOfAtoms>size(X,1)
        D = [diag(diag(S) + (diag(S).^2 + 2*epsilon).^0.5);zeros(numOfAtoms-size(X,1),size(X,1))];
    else
        D = [diag(diag(S) + (diag(S).^2 + 2*epsilon).^0.5) zeros(numOfAtoms,size(X,1)-numOfAtoms)];
    end
    T = 0.5*V*D*U'*invL;
    
    % update Coefficients Z
    %%method 1, replace lambda||Z||_1 with ||Z||_0<=sparseL, use OMP
    %     TX = T*X;
    %     Y = [TX;sqrt_alpha*Q];
    %     D = [eye(size(TX,1),size(A,2));sqrt_alpha*A];
    %
    %     Y = normc(Y);
    %     D = normc(D);
    %     G = D'*D;
    %     Z = omp(D'*Y,G,sparseL);
    
    
    %%method 2,replace lambda||Z||_1 with lambda||Z||_F^2, closed-form
    %%solution
    Z = ((lambda+1)*eye(size(A,2)) + sqrt_alpha*(A'*A))\(T*X + sqrt_alpha*A'*Q);
    
    % update A
    A = Q*Z'/(Z*Z'+eye(size(Z,1)));
end
