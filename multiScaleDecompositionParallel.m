function [M,C] = multiScaleDecompositionParallel(X)
tic;
% Define the initial lambda
lambda = zeros(1,6);
lamb = 0.1;
alpha = 0.8;
for i=1:6
    lambda(i) = lamb * 2^i;
end

% Convert the image to YCbCr and isolate the luminance from the chrominances
if size(X,3) == 3
    X = rgb2ycbcr(X);
    Y = X(:,:,1);
    C = X(:,:,2:3);
else
    Y = X;
    C = zeros(size(Y));
end

% Define the output cells
M = cell(1,7);

% Filter the image using a weighted least-squares filter
TC = cell(1,6);
TF = cell(1,7);
TF{1} = Y;

for i = 2:7
    % Apply the WLS filter
    TC{i-1} = wlsFilter(Y,lambda(i-1),alpha);
    
    % Find the difference between the previous and current scales
    M{i-1} = TF{i-1} - TC{i-1};
    TF{i} = TC{i-1};
    
end

% Store the final approximation
M{7} = TF{7};
toc;
end
