function V = trainPCA(H,bsize)

% Preallocate the covariance and eigenvector matrices for each scale
V = cell(1,7);
for i = 1:7
    V{i} = zeros(4*bsize*bsize);
end

% Process each image in the Kodak set
for i = 1:24
    % Inform user of the progress
    fprintf('Processing image %d of %d\n',i,24);
    
    % Load the image to memory and convert to grayscale
    load(sprintf('multiscale_images/kodim%02d',i));
    
    % Process each scale
    for j = 1:7
        % Extract features from the image
        F1 = imfilter(ML{j},H{j}(:,:,1),'same','symmetric');
        F2 = imfilter(ML{j},H{j}(:,:,2),'same','symmetric');
        F3 = imfilter(ML{j},H{j}(:,:,3),'same','symmetric');
        F4 = imfilter(ML{j},H{j}(:,:,4),'same','symmetric');
        
        % Convert the features to column form
        F1 = im2col(F1,[bsize bsize],'sliding');
        F2 = im2col(F2,[bsize bsize],'sliding');
        F3 = im2col(F3,[bsize bsize],'sliding');
        F4 = im2col(F4,[bsize bsize],'sliding');
        
        % Combine the features
        F = [F1; F2; F3; F4];
        
        % Update the covariance matrix
        V{j} = V{j} + F * F';
    end
end

for i = 1:7
    % Ensure that the matrix is recognized as a SPD
    V{i} = nearestSPD(V{i});

    % Calculate the eigenvectors
    [V{i},lambda] = eig(V{i},'vector');

    % Reorder the eigenvectors by importance
    [~,ind] = sort(lambda,'descend');
    V{i} = V{i}(:,ind);
end
