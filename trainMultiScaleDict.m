function D = trainMultiScaleDict(bsize,numAtoms)

% Reset the random number generator
rng(0);

% Define the mini-batch size
miniBatchSize = 100;
batchSize = 10000;
learnRate = 1000;
numIter = 10;

% Define constants
dictSize = bsize * bsize;

% Create blank statistics
A = cell(1,7);
B = cell(1,7);
D = cell(1,7);

% Initialize the statistics and the initial dictionary
parfor i = 1:7
    A{i} = zeros(numAtoms,numAtoms);
    B{i} = zeros(dictSize,numAtoms);
    
    D{i} = randn(dictSize,numAtoms);
    D{i} = D{i} ./ repmat(sqrt(sum(D{i} .^ 2)),dictSize,1);
end

C = ones(1,7);

% Train for several epochs
for n = 1:numIter
    % Inform the user of the progress
    fprintf('Iteration %d of %d\n',n,numIter);

    for i = 1:24
        % Inform user of the progress
        fprintf('  Processing image %d of %d\n',i,24);

        % Load the image to memory
        X = double(imread(sprintf('kodim%02d.png',i))) / 255;

        % Convert to the YCbCr color space and retain only the luminance
        X = rgb2ycbcr(X);
        X = X(:,:,1);

        % Decompose into a multiscale set
        M = multiScaleDecomposition(X);
        
        % Train a dictionary for each scale
        for s = 1:7
            % Convert the image to column form and remove the mean
            X = im2col(M{s},[bsize bsize],'sliding');
            X = X - repmat(mean(X),dictSize,1);

            % Select a random subset of all the patches
            ind = randperm(size(X,2),batchSize);
            X = X(:,ind);

            % Process each mini-batch
            for j = 1:miniBatchSize:batchSize-miniBatchSize+1
                % Copy the batch to a local variable
                Xbatch = X(:,j:j+miniBatchSize-1); 

                % Compute for the sparse coefficients using OMP
                alpha = D{s}' * Xbatch;
                alpha = alpha .* (alpha == repmat(max(alpha),numAtoms,1));

                % Calculate the learning factor
                beta = (1 - 1 / C(s)) ^ learnRate;

                % Update the statistics
                A{s} = beta * A{s} + alpha * alpha' / miniBatchSize;
                B{s} = beta * B{s} + Xbatch * alpha' / miniBatchSize;

                % Update the dictionary
                for k = 1:numAtoms
                    if A{s}(k,k) >= 1e-6
                        u = (B{s}(:,k) - D{s} * A{s}(:,k)) / A{s}(k,k) + D{s}(:,k);
                        D{s}(:,k) = u / norm(u);
                    end
                end

                % Replace unused atoms with random atoms
                k = diag(A{s}) < 1e-6;
                D{s}(:,k) = randn(dictSize,nnz(k));
                D{s}(:,k) = D{s}(:,k) ./ repmat(sqrt(sum(D{s}(:,k) .^ 2)),dictSize,1);

                % Update the training count
                C(s) = C(s) + miniBatchSize;
            end
        end
    end
end



try
    close all;
    subplot(2,4,1); imshow(0.5+col2im(D{1},[bsize bsize],bsize*[8 8],'distinct'));
    subplot(2,4,2); imshow(0.5+col2im(D{2},[bsize bsize],bsize*[8 8],'distinct'));
    subplot(2,4,3); imshow(0.5+col2im(D{3},[bsize bsize],bsize*[8 8],'distinct'));
    subplot(2,4,4); imshow(0.5+col2im(D{4},[bsize bsize],bsize*[8 8],'distinct'));
    subplot(2,4,5); imshow(0.5+col2im(D{5},[bsize bsize],bsize*[8 8],'distinct'));
    subplot(2,4,6); imshow(0.5+col2im(D{6},[bsize bsize],bsize*[8 8],'distinct'));
    subplot(2,4,7); imshow(0.5+col2im(D{7},[bsize bsize],bsize*[8 8],'distinct'));
catch
   
end


end