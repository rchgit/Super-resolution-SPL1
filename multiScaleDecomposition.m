function [M,C] = multiScaleDecomposition(X)

% Define the initial lambda
lambda = 0.1;
alpha = 0.8;

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
TF = Y;
for i = 1:6
    % Apply the WLS filter
    TC = wlsFilter(Y,lambda,alpha);
    
    % Find the difference between the previous and current scales
    M{i} = TF - TC;
    TF = TC;
    
    lambda = 2 * lambda;
end

% Store the final approximation
M{7} = TC;

end
