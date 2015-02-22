function P = trainRegressors(D,K)

% Extract information from the dictionary
bsize = round(sqrt(size(D{1},1)));
dictSize = size(D{1},1);
numAtoms = size(D{1},2);

% Initialize the neighborhood patches
NX = cell(7,numAtoms);
NY = cell(7,numAtoms);
ndist = cell(7,numAtoms);

for i = 1:7
    for j = 1:numAtoms
        NX{i,j} = zeros(4*dictSize,K);
        NY{i,j} = zeros(dictSize,K);
        ndist{i,j} = inf(1,K);
    end
end

% Process each of the Kodak images
for i = 1:24
    % Inform user of the progress
    fprintf('Processing image %d of %d\n',i,24);

    % Load the image to memory and retain only the luminance channel
    X = double(imread(sprintf('kodim%02d.png',i))) / 255;
    X = rgb2ycbcr(X);
    X = X(:,:,1);

    % Extract image dimensions
    h = size(X,1);
    w = size(X,2);
    
    % Calculate half dimensions
    hh = floor(h / 2);
    hw = floor(w / 2);
    
    % Calculate the multiscale decomposition of the image
    M = multiScaleDecomposition(X);
    
    % Process each scale
    for s = 1:7
        % Inform user of the progress
        fprintf('  Scale %d of %d\n',s,7);
        
        % Store the high-resolution copy of the image at the given scale
        X = M{s};
        
        % Create a bicubic resampled low-resolution image
        Y = imresize(X,0.5,'bicubic');

        % Create a high-resolution mask
        mask = repmat([1 0; 0 0],hh,hw);
        mask = mask(1:end-2*bsize+1,1:end-2*bsize+1);
        mask = logical(mask(:));
        
        % Process first and second derivative features
%         deriv1 = [-1 0 1]; % 1st derivative
%        deriv2 = [-1 0 2 0 -1]; % 2nd derivative
%          F1 = imfilter(Y,deriv1,'same','symmetric');
%         F2 = imfilter(Y,deriv1','same','symmetric');
%        F3 = imfilter(Y,deriv2,'same','symmetric');
%        F4 = imfilter(Y,deriv2','same','symmetric');

%        Y = imresize([F1 F2],size(Y));
%             Y=F1;
        
        % Convert the images to column form
        X = im2col(X,2*[bsize bsize],'sliding');
        Y = im2col(Y,[bsize bsize],'sliding');
        X = X(:,mask);
        
        % Remove the patch means
         n = mean(Y);
         X = X - repmat(n,size(X,1),1);
         Y = Y - repmat(n,size(Y,1),1);        
        
        % Remove patches with low norms
        ind = sqrt(sum(Y .^ 2)) >= 0.01;
        X = X(:,ind);
         Y = Y(:,ind);
        
        % Normalize the patches to ensure that the low-resolution patches
        % have unit norms
         n = sqrt(sum(Y .^ 2));
        X = X ./ repmat(n,size(X,1),1);
        Y = Y ./ repmat(n,size(Y,1),1);
        
       
        % Process each atom
        numPatches = size(X,2);
        for j = 1:numAtoms
            % Calculate the distance of the candidate patches from the current atom
            dist = sqrt(sum((Y - repmat(D{s}(:,j),1,numPatches)) .^ 2));

            % Locate the K nearest neighbors
            [~,ind] = sort(dist,'ascend');
            ind = ind(1:K);

            % Retain the color channels corresponding to the neighbors
            TX = [NX{s,j} X(:,ind)]; 
            TY = [NY{s,j} Y(:,ind)];
            dist = [ndist{s,j} dist(ind)];

            % Combine with the current neighborhood and select the new K
            % neighbors
            [~,ind] = sort(dist,'ascend');
            ind = ind(1:K);
            NX{s,j} = TX(:,ind);
            NY{s,j} = TY(:,ind);
            ndist{s,j} = dist(ind);
        end
    end
end

% Calculate the final projection matrices
P = cell(7,numAtoms);
for i = 1:7
    for j = 1:numAtoms
        % Determine the low-resolution and high-resolution pairs
        L = NY{i,j};
        H = NX{i,j};

        % Calculate the projection matrix
        P{i,j} = H * inv(L' * L + 0.001 * eye(K)) * L';
    end
end
