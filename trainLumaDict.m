function D = trainLumaDict(bsize,numAtoms)

% Define the mini-batch size
miniBatchSize = 100;
batchSize = 10000;
learnRate = 1000;
numIter = 10;

% Define constants
dictSize = bsize * bsize;

% Create blank statistics
A = zeros(numAtoms,numAtoms);
B = zeros(dictSize,numAtoms);
C = 1;

% Initialize a random dictionary
D = randn(dictSize,numAtoms);
D = D ./ repmat(sqrt(sum(D .^ 2)),dictSize,1);

% Train for several epochs
for n = 1:numIter
    % Inform the user of the progress
    fprintf('Iteration %d of %d\n',n,numIter);
    
    for i = 1:24
        % Inform user of the progress
        fprintf('  Processing image %d of %d',i,24);

        % Load the image to memory
        X = double(imread(sprintf('kodim%02d.png',i))) / 255;
        
        % Retain only the luminance information
        X = X(:,:,1) / 4 + X(:,:,2) / 2 + X(:,:,3) / 4;
        
        % Convert the image to column form and remove the mean
        X = im2col(X,[bsize bsize],'sliding');
        X = X - repmat(mean(X),dictSize,1);
        
        % Select a random subset of all the patches
        ind = randperm(size(X,2),batchSize);
        X = X(:,ind);
        
        % Process each mini-batch
        for j = 1:miniBatchSize:batchSize-miniBatchSize+1
            % Copy the batch to a local variable
            Xbatch = X(:,j:j+miniBatchSize-1); 

            % Compute for the sparse coefficients using OMP
            alpha = D' * Xbatch;
            alpha = alpha .* (alpha == repmat(max(alpha),numAtoms,1));

            % Calculate the learning factor
            beta = (1 - 1 / C) ^ learnRate;

            % Update the statistics
            A = beta * A + alpha * alpha' / miniBatchSize;
            B = beta * B + Xbatch * alpha' / miniBatchSize;

            % Update the dictionary
            for k = 1:numAtoms
                if A(k,k) >= 1e-6
                    u = (B(:,k) - D * A(:,k)) / A(k,k) + D(:,k);
                    D(:,k) = u / norm(u);
                end
            end

            % Replace unused atoms with random atoms
            k = diag(A) < 1e-6;
            D(:,k) = randn(dictSize,nnz(k));
            D(:,k) = D(:,k) ./ repmat(sqrt(sum(D(:,k) .^ 2)),dictSize,1);

            % Update the training count
            C = C + miniBatchSize;
        end
        
        % Test the current dictionary
        alpha = D' * X;
        alpha = alpha .* (alpha == repmat(max(alpha),numAtoms,1));
        Xhat = D * alpha;
        
        fprintf(' (PSNR: %f)\n',psnr(Xhat,X));
    end
end

close all;
imshow(0.5+col2im(D,[bsize bsize],8*[bsize bsize],'distinct'));


