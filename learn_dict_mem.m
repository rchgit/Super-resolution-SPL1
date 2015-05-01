function [params] = learn_dict_mem(params,simparams)
%% learn_dict.m
% Sample patches (from high-res. images) and extract features (from low-res.)
% for the Super Resolution algorithm training phase, using specified scale
% factor between high-res. and low-res.
% inds = cell(numel(paths),1);
%% Load training high-res. image set and resample it
ksvd_params = [];
params.ksvd_conf = ksvd_params;
params_spherical.A = [];
params_spherical.B = [];
params.dict_lores = params.dictsize;
if exists('C.mat','file') % File that stores features and lo-res dictionary
    load('C.mat');
    result = glob(simparams.train_dir,simparams.train_pattern);
    for i = 1:numel(result)
        hires = rgb2ycbcr(imread(result{i}));
        hires = im2double(hires(:,:,1)); % Y channel only
        hires = modcrop_single(hires,params.scale); % to avoid scaling prob
        % Do we still need to get the hi-res features to get the
        % hi-res version of the dictionary?
        features = extract(params, hires, params.scale, params.filters);
        % Do we need to remove the high frequencies from the
        % patches?
        patches = hires - midres;
        patches = extract(params, patches, params.scale, {});
        
        fprintf('\tTrain count: %d\n',i);
        if i == 1
            C_h = features * features';
        else
            C_h = C_h + features * features';
        end
        
    end
    % then PCA
    C = nearestSPD(C);
    [C, lambda] = eig(C,'vector');
    lambda = diag(lambda); % perform PCA on features matrix
    lambda = cumsum(lambda) / sum(lambda);
    k = find(lambda >= 1e-3, 1); % ignore 0.1% energy
    [~,ind] = sort(lambda,'descend');
    C = C(:,ind);
    params.V_pca = C;
    
    fprintf('Computing sparse coefficients\n');
    gamma = params.dict_lores' * C;
    gamma = gamma .* (repmat(max(gamma),size(gamma,1),1) == gamma);
    fprintf('Computing high-res. dictionary from low-res. dictionary\n');
    dict_hires = (C * gamma') / (gamma * gamma');
    params.dict_hires = dict_hires;
    params.spherical_conf = params_spherical;
else
    result = glob(simparams.train_dir,simparams.train_pattern);
    features = cell(numel(result),1);
    %     features_pca = zeros(numel(result),1);
    features_pca = [];
    patches = cell(numel(result),1);
    params.V_pca = zeros(numel(result),1);
    pruneBnW(result); % Removes all non-conformant images to YCbCr color space
    result = glob(simparams.train_dir,simparams.train_pattern);
    %% TRAINING
    if strcmp(params.train_method,'spherical')
        for i = 1:numel(result)
            hires = rgb2ycbcr(imread(result{i}));
            hires = im2double(hires(:,:,1)); % Y channel only
            hires = modcrop_single(hires,params.scale);
            lores = imresize(hires,1/params.scale,params.interpolate_kernel);
            midres = imresize(lores,params.scale,params.interpolate_kernel);
            % NOTE: features and patches are taken per image to save on RAM
            features = extract(params, midres, params.scale, params.filters);
            patches = hires - midres;
            patches = extract(params, patches, params.scale, {});
            
            fprintf('\tTrain count: %d\n',i);
            if i == 1
                C = features * features';
            else
                C = C + features * features';
            end
        end
        % PCA dimensionality reduction
        C = nearestSPD(C);
        [V, lambda] = eig(C,'vector');
        lambda = diag(lambda); % perform PCA on features matrix
        lambda = cumsum(lambda) / sum(lambda);
        k = find(lambda >= 1e-3, 1); % ignore 0.1% energy
        [~,ind] = sort(lambda,'descend');
        V = V(:,ind);
        params.V_pca = V;
        feat_pca = V' * features;
        [C,params.dict_lores,params_spherical] = ...
            learnSphericalClusters(C,params.dict_lores,params_spherical);
        fprintf('Computing sparse coefficients\n');
        gamma = params.dict_lores' * C;
        gamma = gamma .* (repmat(max(gamma),size(gamma,1),1) == gamma);
        fprintf('Computing high-res. dictionary from low-res. dictionary\n');
        dict_hires = (patches * gamma') / (gamma * gamma');
        params.dict_hires = dict_hires;
        params.spherical_conf = params_spherical;
        
        
        %%%%%%% NO NEED TO USE THIS. KEPT HERE FOR REFERENCE PURPOSES %%%%%%%
    elseif strcmp(params.train_method,'ksvd')
        % Set KSVD configuration
        ksvd_params.iternum = 20; % TBD
        ksvd_params.memusage = 'normal'; % higher usage doesn't fit...
        ksvd_params.dictsize = dictsize; % TBD
        ksvd_params.Tdata = 3; % maximal sparsity: TBD
        ksvd_params.samples = size(patches,2);
        
        % PCA dimensionality reduction
        C = double(features * features');
        [V, lambda] = eig(C);
        lambda = diag(lambda); % perform PCA on features matrix
        lambda = cumsum(lambda) / sum(lambda);
        k = find(lambda >= 1e-3, 1); % ignore 0.1% energy
        params.V_pca = V(:, k:end); % choose the largest eigenvectors' projection
        
        features_pca = params.V_pca' * features;
        
        % Combine into one large training set
        clear C D V
        ksvd_params.data = double(features_pca);
        clear features_pca
        
        % Training process (will take a while)
        tic;
        fprintf('Training [%d x %d] dictionary on %d vectors using K-SVD\n', ...
            size(ksvd_params.data, 1), ksvd_params.dictsize, size(ksvd_params.data, 2))
        [params.dict_lores, gamma] = ksvd(ksvd_params);
        toc;
        
        fprintf('Computing high-res. dictionary from low-res. dictionary\n');
        % dict_hires = patches / full(gamma); % Takes too much memory...
        % patches = double(patches); % Since it is saved in single-precision.
        dict_hires = (patches * gamma') / full((gamma * gamma'));
        
        params.dict_hires = double(dict_hires);
    end
end
end
