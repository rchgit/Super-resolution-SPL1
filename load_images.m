function [imgs, imgsCB, imgsCR] = load_images(paths,params)

imgs = cell(size(paths));
imgsCB = cell(size(paths));
imgsCR = cell(size(paths));
for i = 1:numel(paths)
    X = imread(paths{i});
    if size(X, 3) == 3 % we extract our features from Y channel
        X = rgb2ycbcr(X);                       
        imgsCB{i} = im2double(X(:,:,2)); 
        imgsCR{i} = im2double(X(:,:,3));
        X = X(:, :, 1);
    end
%     X = im2double(X);
    X = im2colrand(X,params.window, 1024);
%     ind = randperm(size(X,2),uint32(patch_ratio*size(X,2)));
%     X = X(:,ind);
    imgs{i} = X;
end
