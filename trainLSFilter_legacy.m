function [hHighRes] = trainLSFilter(numIter,wsize,filename)
files = dir('*.jpg');
% Define the Gaussian size
gsize = 11;

% Calculate the pad size
gpad = (gsize - 1) / 2;
wpad = (wsize - 1) / 2;

% Initialize the statistics
A = zeros(gsize*gsize);
B = zeros(1,gsize*gsize);
AH = zeros(wsize*wsize);
AV = zeros(wsize*wsize);
BH1 = zeros(1,wsize*wsize);
BH2 = zeros(1,wsize*wsize);
BH3 = zeros(1,wsize*wsize);
BH4 = zeros(1,wsize*wsize);
BV1 = zeros(1,wsize*wsize);
BV2 = zeros(1,wsize*wsize);
BV3 = zeros(1,wsize*wsize);
BV4 = zeros(1,wsize*wsize);

%% Initial pass

% Define the initial filter set
hHighRes.L = fspecial('gaussian',[gsize gsize],2);
% First derivative filters
hHighRes.H1 = [-1 0 1];
hHighRes.H2 = [-1 0 1]';
% Second derivative filters
hHighRes.H3 = [-1 0 2 0 -1];
hHighRes.H4 = [-1 0 2 0 -1]';

% Normalize the filters
hHighRes.L = hHighRes.L / sum(abs(hHighRes.L(:)));
hHighRes.H1 = hHighRes.H1 / sum(abs(hHighRes.H1(:)));
hHighRes.H2 = hHighRes.H2 / sum(abs(hHighRes.H2(:)));
hHighRes.H3 = hHighRes.H3 / sum(abs(hHighRes.H3(:)));
hHighRes.H4 = hHighRes.H4 / sum(abs(hHighRes.H4(:)));

% Process each of the images

fprintf('Performing the initial pass...\n');
i = 0;
for file = files'
    i = i + 1;
    % Inform user of the progress
    fprintf('  Processing image %d\n',i);

    % Load image to memory
    X = rgb2gray(double(imread(file.name)) / 255);
    
    % Create a downsampled and approximate version of the image
    Y = imresize(imresize(X,0.5,'bicubic'),2,'lanczos3');

    % Obtain features from the high-resolution image
    FL = imfilter(X,hHighRes.L,'same','symmetric');
    FH1 = imfilter(X,hHighRes.H1,'same','symmetric');
    FH2 = imfilter(X,hHighRes.H2,'same','symmetric');
    FH3 = imfilter(X,hHighRes.H3,'same','symmetric');
    FH4 = imfilter(X,hHighRes.H4,'same','symmetric');
    
    % Calculate the gradient of the approximate image
    [GX,GY] = imgradientxy(Y);
    GX = GX(17:end-16,17:end-16) .^ 2;
    GY = GY(17:end-16,17:end-16) .^ 2;
    GX = imfilter(GX,ones(5),'same','symmetric');
    GY = imfilter(GY,ones(5),'same','symmetric');
    dirVar = (GX < GY);
    
    % Crop the images to remove unnecessary borders
    FL = FL(17:end-16,17:end-16);
    FH1 = FH1(17:end-16,17:end-16);
    FH2 = FH2(17:end-16,17:end-16);
    FH3 = FH3(17:end-16,17:end-16);
    FH4 = FH4(17:end-16,17:end-16);
    G = Y(17-gpad:end-16+gpad,17-gpad:end-16+gpad);
    Y = Y(17-wpad:end-16+wpad,17-wpad:end-16+wpad);
    
    % Convert the images and their features into column form
    FL = reshape(FL,1,[]);
    FH1 = reshape(FH1,1,[]);
    FH2 = reshape(FH2,1,[]);
    FH3 = reshape(FH3,1,[]);
    FH4 = reshape(FH4,1,[]);
    dirVar = reshape(dirVar,1,[]);
    G = im2col(G,[gsize gsize],'sliding');
    Y = im2col(Y,[wsize wsize],'sliding');
    
    % Collect statistics
    A = A + G * G';
    B = B + FL * G';
    
    AH = AH + Y(:,dirVar) * Y(:,dirVar)';
    AV = AV + Y(:,~dirVar) * Y(:,~dirVar)';

    BH1 = BH1 + FH1(:,dirVar) * Y(:,dirVar)';
    BH2 = BH2 + FH2(:,dirVar) * Y(:,dirVar)';
    BH3 = BH3 + FH3(:,dirVar) * Y(:,dirVar)';
    BH4 = BH4 + FH4(:,dirVar) * Y(:,dirVar)';
    
    BV1 = BV1 + FH1(:,~dirVar) * Y(:,~dirVar)';
    BV2 = BV2 + FH2(:,~dirVar) * Y(:,~dirVar)';
    BV3 = BV3 + FH3(:,~dirVar) * Y(:,~dirVar)';
    BV4 = BV4 + FH4(:,~dirVar) * Y(:,~dirVar)';
end

% Extract the filter coefficients
hLowRes.L = reshape(B / A,[],gsize);
hLowRes.H1 = reshape(BH1 / AH,[],wsize);
hLowRes.H2 = reshape(BH2 / AH,[],wsize);
hLowRes.H3 = reshape(BH3 / AH,[],wsize);
hLowRes.H4 = reshape(BH4 / AH,[],wsize);
hLowRes.V1 = reshape(BV1 / AV,[],wsize);
hLowRes.V2 = reshape(BV2 / AV,[],wsize);
hLowRes.V3 = reshape(BV3 / AV,[],wsize);
hLowRes.V4 = reshape(BV4 / AV,[],wsize);

%% Refinement passes

% Iteratively update the two filters to promote convergence
for n = 1:numIter
    % Reset statistics
    A = zeros(gsize*gsize);
    B = zeros(1,gsize*gsize);
    AH = zeros(wsize*wsize);
    AV = zeros(wsize*wsize);
    BH1 = zeros(1,wsize*wsize);
    BH2 = zeros(1,wsize*wsize);
    BH3 = zeros(1,wsize*wsize);
    BH4 = zeros(1,wsize*wsize);
    BV1 = zeros(1,wsize*wsize);
    BV2 = zeros(1,wsize*wsize);
    BV3 = zeros(1,wsize*wsize);
    BV4 = zeros(1,wsize*wsize);

    % Update the high-resolution filters
    fprintf('Updating high-resolution filters %d of %d\n',n,numIter);
    i = 0;
    for file = files'
        i = i + 1;
        % Inform user of the progress
        fprintf('  Processing image %d\n',i);

        % Load image to memory
        X = rgb2gray(double(imread(file.name)) / 255);

        % Create a downsampled and approximate version of the image
        Y = imresize(imresize(X,0.5,'bicubic'),2,'lanczos3');

        % Obtain features from the high-resolution image
        FL = imfilter(Y,hLowRes.L,'same','symmetric');
        FH1 = imfilter(Y,hLowRes.H1,'same','symmetric');
        FH2 = imfilter(Y,hLowRes.H2,'same','symmetric');
        FH3 = imfilter(Y,hLowRes.H3,'same','symmetric');
        FH4 = imfilter(Y,hLowRes.H4,'same','symmetric');
        FV1 = imfilter(Y,hLowRes.V1,'same','symmetric');
        FV2 = imfilter(Y,hLowRes.V2,'same','symmetric');
        FV3 = imfilter(Y,hLowRes.V3,'same','symmetric');
        FV4 = imfilter(Y,hLowRes.V4,'same','symmetric');

        % Calculate the gradient of the approximate image
        [GX,GY] = imgradientxy(Y);
        GX = GX(17:end-16,17:end-16) .^ 2;
        GY = GY(17:end-16,17:end-16) .^ 2;
        GX = imfilter(GX,ones(5),'same','symmetric');
        GY = imfilter(GY,ones(5),'same','symmetric');
        dirVar = (GX < GY);

        % Crop the images to remove unnecessary borders
        FL = FL(17:end-16,17:end-16);
        FH1 = FH1(17:end-16,17:end-16);
        FH2 = FH2(17:end-16,17:end-16);
        FH3 = FH3(17:end-16,17:end-16);
        FH4 = FH4(17:end-16,17:end-16);
        FV1 = FV1(17:end-16,17:end-16);
        FV2 = FV2(17:end-16,17:end-16);
        FV3 = FV3(17:end-16,17:end-16);
        FV4 = FV4(17:end-16,17:end-16);
        G = X(17-gpad:end-16+gpad,17-gpad:end-16+gpad);
        X = X(17-wpad:end-16+wpad,17-wpad:end-16+wpad);

        % Convert the images and their features into column form
        FL = reshape(FL,1,[]);
        FH1 = reshape(FH1,1,[]);
        FH2 = reshape(FH2,1,[]);
        FH3 = reshape(FH3,1,[]);
        FH4 = reshape(FH4,1,[]);
        FV1 = reshape(FV1,1,[]);
        FV2 = reshape(FV2,1,[]);
        FV3 = reshape(FV3,1,[]);
        FV4 = reshape(FV4,1,[]);
        dirVar = reshape(dirVar,1,[]);
        G = im2col(G,[gsize gsize],'sliding');
        X = im2col(X,[wsize wsize],'sliding');

        % Collect statistics
        A = A + G * G';
        B = B + FL * G';

        AH = AH + X(:,dirVar) * X(:,dirVar)';
        AV = AV + X(:,~dirVar) * X(:,~dirVar)';

        BH1 = BH1 + FH1(:,dirVar) * X(:,dirVar)';
        BH2 = BH2 + FH2(:,dirVar) * X(:,dirVar)';
        BH3 = BH3 + FH3(:,dirVar) * X(:,dirVar)';
        BH4 = BH4 + FH4(:,dirVar) * X(:,dirVar)';

        BV1 = BV1 + FV1(:,~dirVar) * X(:,~dirVar)';
        BV2 = BV2 + FV2(:,~dirVar) * X(:,~dirVar)';
        BV3 = BV3 + FV3(:,~dirVar) * X(:,~dirVar)';
        BV4 = BV4 + FV4(:,~dirVar) * X(:,~dirVar)';
    end
    
    % Extract the filter coefficients
    hHighRes.L = reshape(B / A,[],gsize);
    hHighRes.H1 = reshape(BH1 / AH,[],wsize);
    hHighRes.H2 = reshape(BH2 / AH,[],wsize);
    hHighRes.H3 = reshape(BH3 / AH,[],wsize);
    hHighRes.H4 = reshape(BH4 / AH,[],wsize);
    hHighRes.V1 = reshape(BV1 / AV,[],wsize);
    hHighRes.V2 = reshape(BV2 / AV,[],wsize);
    hHighRes.V3 = reshape(BV3 / AV,[],wsize);
    hHighRes.V4 = reshape(BV4 / AV,[],wsize);
    
    % Normalize the filters
    hHighRes.L = hHighRes.L / sum(abs(hHighRes.L(:)));
    hHighRes.H1 = hHighRes.H1 / sum(abs(hHighRes.H1(:)));
    hHighRes.H2 = hHighRes.H2 / sum(abs(hHighRes.H2(:)));
    hHighRes.H3 = hHighRes.H3 / sum(abs(hHighRes.H3(:)));
    hHighRes.H4 = hHighRes.H4 / sum(abs(hHighRes.H4(:)));
    hHighRes.V1 = hHighRes.V1 / sum(abs(hHighRes.V1(:)));
    hHighRes.V2 = hHighRes.V2 / sum(abs(hHighRes.V2(:)));
    hHighRes.V3 = hHighRes.V3 / sum(abs(hHighRes.V3(:)));
    hHighRes.V4 = hHighRes.V4 / sum(abs(hHighRes.V4(:)));
    
    % Reset statistics
    A = zeros(gsize*gsize);
    B = zeros(1,gsize*gsize);
    AH = zeros(wsize*wsize);
    AV = zeros(wsize*wsize);
    BH1 = zeros(1,wsize*wsize);
    BH2 = zeros(1,wsize*wsize);
    BH3 = zeros(1,wsize*wsize);
    BH4 = zeros(1,wsize*wsize);
    BV1 = zeros(1,wsize*wsize);
    BV2 = zeros(1,wsize*wsize);
    BV3 = zeros(1,wsize*wsize);
    BV4 = zeros(1,wsize*wsize);

    % Update the low-resolution filters
    fprintf('Updating low-resolution filters %d of %d\n',n,numIter);
    
 
    i = 0;
    for file = files'
        i = i + 1;
        % Inform user of the progress
        fprintf('  Processing image %d\n',i);

        % Load image to memory
        X = rgb2gray(double(imread(file.name)) / 255);
        % Create a downsampled and approximate version of the image
        Y = imresize(imresize(X,0.5,'bicubic'),2,'lanczos3');

        % Obtain features from the high-resolution image
        FL = imfilter(X,hHighRes.L,'same','symmetric');
        FH1 = imfilter(X,hHighRes.H1,'same','symmetric');
        FH2 = imfilter(X,hHighRes.H2,'same','symmetric');
        FH3 = imfilter(X,hHighRes.H3,'same','symmetric');
        FH4 = imfilter(X,hHighRes.H4,'same','symmetric');
        FV1 = imfilter(X,hHighRes.V1,'same','symmetric');
        FV2 = imfilter(X,hHighRes.V2,'same','symmetric');
        FV3 = imfilter(X,hHighRes.V3,'same','symmetric');
        FV4 = imfilter(X,hHighRes.V4,'same','symmetric');

        % Calculate the gradient of the approximate image
        [GX,GY] = imgradientxy(Y);
        GX = GX(17:end-16,17:end-16) .^ 2;
        GY = GY(17:end-16,17:end-16) .^ 2;
        GX = imfilter(GX,ones(5),'same','symmetric');
        GY = imfilter(GY,ones(5),'same','symmetric');
        dirVar = (GX < GY);

        % Crop the images to remove unnecessary borders
        FL = FL(17:end-16,17:end-16);
        FH1 = FH1(17:end-16,17:end-16);
        FH2 = FH2(17:end-16,17:end-16);
        FH3 = FH3(17:end-16,17:end-16);
        FH4 = FH4(17:end-16,17:end-16);
        FV1 = FV1(17:end-16,17:end-16);
        FV2 = FV2(17:end-16,17:end-16);
        FV3 = FV3(17:end-16,17:end-16);
        FV4 = FV4(17:end-16,17:end-16);
        G = Y(17-gpad:end-16+gpad,17-gpad:end-16+gpad);
        Y = Y(17-wpad:end-16+wpad,17-wpad:end-16+wpad);

        % Convert the images and their features into column form
        FL = reshape(FL,1,[]);
        FH1 = reshape(FH1,1,[]);
        FH2 = reshape(FH2,1,[]);
        FH3 = reshape(FH3,1,[]);
        FH4 = reshape(FH4,1,[]);
        FV1 = reshape(FV1,1,[]);
        FV2 = reshape(FV2,1,[]);
        FV3 = reshape(FV3,1,[]);
        FV4 = reshape(FV4,1,[]);
        dirVar = reshape(dirVar,1,[]);
        G = im2col(G,[gsize gsize],'sliding');
        Y = im2col(Y,[wsize wsize],'sliding');

        % Collect statistics
        A = A + G * G';
        B = B + FL * G';

        AH = AH + Y(:,dirVar) * Y(:,dirVar)';
        AV = AV + Y(:,~dirVar) * Y(:,~dirVar)';

        BH1 = BH1 + FH1(:,dirVar) * Y(:,dirVar)';
        BH2 = BH2 + FH2(:,dirVar) * Y(:,dirVar)';
        BH3 = BH3 + FH3(:,dirVar) * Y(:,dirVar)';
        BH4 = BH4 + FH4(:,dirVar) * Y(:,dirVar)';

        BV1 = BV1 + FV1(:,~dirVar) * Y(:,~dirVar)';
        BV2 = BV2 + FV2(:,~dirVar) * Y(:,~dirVar)';
        BV3 = BV3 + FV3(:,~dirVar) * Y(:,~dirVar)';
        BV4 = BV4 + FV4(:,~dirVar) * Y(:,~dirVar)';
    end
    
    % Extract the filter coefficients
    hLowRes.L = reshape(B / A,[],gsize);
    hLowRes.H1 = reshape(BH1 / AH,[],wsize);
    hLowRes.H2 = reshape(BH2 / AH,[],wsize);
    hLowRes.H3 = reshape(BH3 / AH,[],wsize);
    hLowRes.H4 = reshape(BH4 / AH,[],wsize);
    hLowRes.V1 = reshape(BV1 / AV,[],wsize);
    hLowRes.V2 = reshape(BV2 / AV,[],wsize);
    hLowRes.V3 = reshape(BV3 / AV,[],wsize);
    hLowRes.V4 = reshape(BV4 / AV,[],wsize);
end

save(filename);

close all;
figure(1);
subplot(1,2,1); surf(hHighRes.L); view(2);
subplot(1,2,2); surf(hLowRes.L); view(2);
figure(2);
subplot(1,2,1); surf(hHighRes.H1); view(2);
subplot(1,2,2); surf(hLowRes.H1); view(2);
figure(3);
subplot(1,2,1); surf(hHighRes.H2); view(2);
subplot(1,2,2); surf(hLowRes.H2); view(2);
figure(4);
subplot(1,2,1); surf(hHighRes.H3); view(2);
subplot(1,2,2); surf(hLowRes.H3); view(2);
figure(5);
subplot(1,2,1); surf(hHighRes.H4); view(2);
subplot(1,2,2); surf(hLowRes.H4); view(2);
figure(6);
subplot(1,2,1); surf(hHighRes.V1); view(2);
subplot(1,2,2); surf(hLowRes.V1); view(2);
figure(7);
subplot(1,2,1); surf(hHighRes.V2); view(2);
subplot(1,2,2); surf(hLowRes.V2); view(2);
figure(8);
subplot(1,2,1); surf(hHighRes.V3); view(2);
subplot(1,2,2); surf(hLowRes.V3); view(2);
figure(9);
subplot(1,2,1); surf(hHighRes.V4); view(2);
subplot(1,2,2); surf(hLowRes.V4); view(2);
end
