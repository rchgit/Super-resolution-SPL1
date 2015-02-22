function Y = imSR(X,D,P)

% Determine the block size of the patches
bsize = round(sqrt(size(D{1},1)));

% Extract image dimensions
h = size(X,1);
w = size(X,2);

% Separate the luminance channel from the chrominance channels
X = rgb2ycbcr(X);
L = X(:,:,1);
C = X(:,:,2:3);

% Find the multiscale representation of the image
M = multiScaleDecomposition(L);

% Process each scale
Y = zeros(2*h,2*w);
for i = 1:7
    % Divide the image into patches
    X = im2col(M{i},[bsize bsize],'distinct');
    
    % Remove the patch means
    m = mean(X);
    X = X - repmat(m,size(X,1),1);
    
    % Find the nearest dictionary atom
    [~,ind] = max(abs(D{i}' * X));
    
    % Apply the projection matrix to each patch
    T = zeros(2*size(X,1),size(X,2));
    for j = 1:size(X,2)
        T(:,j) = P{i,ind(j)} * X(:,j);
    end
    
    % Restore the patch means
    T = T + repmat(m,size(T,1),1);
    Y = Y + col2im(T,2*[bsize bsize],2*[h w],'distinct');
end

% Interpolate the color channels
C = imresize(C,2,'lanczos3');

% Restore the color of the image
Y = ycbcr2rgb(cat(3,Y,C));