function V = trainLSDictSpherical(bsize,numAtoms,M)
% X = cell containing signals
% M = cell containing multi-scale decomposition of signals

% Reset the random number generator
rng(0);

% Define the mini-batch size
%miniBatchSize = 100;
%batchSize = 10000;
%learnRate = 1000;
numIter = 10;

% Define constants
dictSize = bsize * bsize;

% Create blank statistics
A = cell(1,7);
B = cell(1,7);
V = cell(1,7);

% Initialize the statistics and the initial dictionary
for i = 1:7
    A{i} = zeros(numAtoms,numAtoms);
    B{i} = zeros(dictSize,numAtoms);
    
    V{i} = randn(dictSize,numAtoms);
    % Normalize initial atoms
    V{i} = V{i} ./ repmat(sqrt(sum(V{i} .^ 2)),dictSize,1);
end

C = ones(1,7);

% Load all Kodak images
%m = matfile('kodimMat.mat');
%X = m.kodimMat;

% Train for several epochs
for n = 1:numIter
    % Inform the user of the progress
    fprintf('Iteration %d of %d\n',n,numIter);

    for i = 1:24
        % Inform user of the progress
        fprintf('  Processing image %d of %d\n',i,24);
        Ms = M{i};
        % Learn spherical clusters for each scale
        parfor s = 1:7
            % Convert the image to column form and remove the mean
            X_sig = im2col(Ms{s},[bsize bsize],'sliding');
            X_sig = X_sig - repmat(mean(X_sig),dictSize,1);
            
            % Send the scale for spherical training
            [V{s},A{s},B{s},C(s)] = learnSphericalClusters(X_sig,V{s},A{s},B{s},C(s));
            
%             % Select a random subset of all the patches
%             ind = randperm(size(X,2),batchSize);
%             X = X(:,ind);
% 
%             % Process each mini-batch
%             for j = 1:miniBatchSize:batchSize-miniBatchSize+1
%                 % Copy the batch to a local variable
%                 Xbatch = X(:,j:j+miniBatchSize-1); 
% 
%                 % Compute for the sparse coefficients using OMP
%                 alpha = D{s}' * Xbatch;
%                 alpha = alpha .* (alpha == repmat(max(alpha),numAtoms,1));
% 
%                 % Calculate the learning factor
%                 beta = (1 - 1 / C(s)) ^ learnRate;
% 
%                 % Update the statistics
%                 A{s} = beta * A{s} + alpha * alpha' / miniBatchSize;
%                 B{s} = beta * B{s} + Xbatch * alpha' / miniBatchSize;
% 
%                 % Update the dictionary
%                 for k = 1:numAtoms
%                     if A{s}(k,k) >= 1e-6
%                         u = (B{s}(:,k) - D{s} * A{s}(:,k)) / A{s}(k,k) + D{s}(:,k);
%                         D{s}(:,k) = u / norm(u);
%                     end
%                 end
% 
%                 % Replace unused atoms with random atoms
%                 k = diag(A{s}) < 1e-6;
%                 D{s}(:,k) = randn(dictSize,nnz(k));
%                 D{s}(:,k) = D{s}(:,k) ./ repmat(sqrt(sum(D{s}(:,k) .^ 2)),dictSize,1);
% 
%                 % Update the training count
%                 C(s) = C(s) + miniBatchSize;
%             end
        end
        
    end
end



try
    close all;
%     for i=1:7
%         subplot(2,4,i); imshow(0.5+col2im(V{i},[bsize bsize],bsize*[8 8],'distinct'));
%     end
    
    subplot(2,4,1); imshow(0.5+col2im(V{1},[bsize bsize],bsize*[sqrt(numAtoms) sqrt(numAtoms)],'distinct'));
    subplot(2,4,2); imshow(0.5+col2im(V{2},[bsize bsize],bsize*[sqrt(numAtoms) sqrt(numAtoms)],'distinct'));
    subplot(2,4,3); imshow(0.5+col2im(V{3},[bsize bsize],bsize*[sqrt(numAtoms) sqrt(numAtoms)],'distinct'));
    subplot(2,4,4); imshow(0.5+col2im(V{4},[bsize bsize],bsize*[sqrt(numAtoms) sqrt(numAtoms)],'distinct'));
    subplot(2,4,5); imshow(0.5+col2im(V{5},[bsize bsize],bsize*[sqrt(numAtoms) sqrt(numAtoms)],'distinct'));
    subplot(2,4,6); imshow(0.5+col2im(V{6},[bsize bsize],bsize*[sqrt(numAtoms) sqrt(numAtoms)],'distinct'));
    subplot(2,4,7); imshow(0.5+col2im(V{7},[bsize bsize],bsize*[sqrt(numAtoms) sqrt(numAtoms)],'distinct'));
        
catch
   
end     
save('trainSpherical.mat','V','A','B','bsize','numAtoms');

end