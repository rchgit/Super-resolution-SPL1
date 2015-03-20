function [X,V,spherical_conf] = learnSphericalClusters(X,V,n,varargin)
% V = (cluster centers = dictionary atoms) OR num of dict atoms
% A & B = least-squares statistics
% n = training count
% alpha = sparse coefficients
% beta = learning factor
fprintf('\tLearn spherical clusters\n');
minargs = 2;
maxargs = 5;
narginchk(minargs, maxargs)

% Define the mini-batch size
batchSize = 10;
learnRate = 1000;

% Ensure that the signals are zero-mean
X = X - repmat(mean(X),size(X,1),1);

% Remove the columns with negligible norms 
%X = X(:,sqrt(sum(X .^ 2)) >= 0.1);

% Normalize the remaining signals
X = X ./ repmat(sqrt(sum(X .^ 2)),size(X,1),1);

% Extract the signal information
numCols = size(X,2);
dictSize = size(X,1);
spherical_conf.dictsize = dictSize;
spherical_conf.batchsize = batchSize;
% Check if the input statistics have been provided
if length(V(:)) == 1
    % The value stored in V is actually the number of atoms
    numAtoms = V;
    
    % Create blank statistics
    A = sparse(numAtoms,numAtoms);
    B = sparse(dictSize,numAtoms);
    n = 1;
    
    % Initialize a random dictionary
    V = randn(dictSize,numAtoms);
    V = V ./ repmat(sqrt(sum(V .^ 2)),dictSize,1);
else
    % Check if the dictionary size is consistent with the data
    assert(size(V,1) == dictSize,'The dictionary and data have different sizes');
    
    % Determine the number of atoms present in the dictionary
    numAtoms = size(V,2);
    A = varargin{1};
    B = varargin{2};
    % Verify the sizes of the statistics
    assert(size(A,1) == numAtoms && size(A,2) == numAtoms,'The statistics matrix A has an invalid size');
    %assert(size(A,1) == dictSize && size(A,2) == numAtoms,'The statistics matrix A has an invalid size');
    assert(size(B,1) == dictSize && size(B,2) == numAtoms,'The statistics matrix B has an invalid size');
    assert(n > 0,'The training count is invalid');
end
fprintf('\tTraining spherical clusters...\n');
for i = 1:batchSize:numCols-batchSize+1
    %fprintf('\t\tTraining count: %d of %d\n',n,numCols-batchSize+1);
    % Copy the batch to a local variable
    Xbatch = X(:,i:i+batchSize-1); 
    
    % Find the closest cluster
    alpha = V' * Xbatch;
    alpha(alpha < repmat(max(alpha),numAtoms,1)) = 0;
    alpha = sparse(alpha);
    % Calculate the learning factor
    beta = (1 - 1 / n) ^ learnRate;

    % Update the statistics
    A = beta * A + alpha * alpha' / batchSize;
    B = beta * B + Xbatch * alpha' / batchSize;
    
    % Update the dictionary
    for j = 1:numAtoms
        if A(j,j) >= 1e-6
            u = (B(:,j) - V * A(:,j)) / A(j,j) + V(:,j);
            V(:,j) = u / norm(u);
        end
    end

    % Replace unused atoms with random atoms
    j = diag(A) < 1e-6;
    V(:,j) = randn(dictSize,nnz(j));
    V(:,j) = V(:,j) ./ repmat(sqrt(sum(V(:,j) .^ 2)),dictSize,1);

    % Update the training count
    n = n + batchSize;
    
end
spherical_conf.A = A;
spherical_conf.B = B;
spherical_conf.traincount = n;
end
