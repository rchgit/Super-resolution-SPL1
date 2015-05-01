function [params] = learn_dict_puresphere(params,simparams)
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
    
    result = glob(simparams.train_dir,simparams.train_pattern);
%     features = cell(numel(result),1);
    %     features_pca = zeros(numel(result),1);
    features_pca = [];
    patches = cell(numel(result),1);
    params.V_pca = zeros(numel(result),1);
    pruneBnW(result); % Removes all non-conformant images to YCbCr color space
    result = glob(simparams.train_dir,simparams.train_pattern);
    %% TRAINING
    for i = 1:numel(result)
        hires = rgb2ycbcr(imread(result{i}));
        hires = im2double(hires(:,:,1)); % Y channel only
        hires = modcrop_single(hires,params.scale);
        lores = imresize(hires,1/params.scale,params.interpolate_kernel);
        midres = imresize(lores,params.scale,params.interpolate_kernel);
        % NOTE: features and patches are taken per image to save on RAM
        features = extract(params, midres, params.scale, params.filters);
        patches = hires - midres; % remove hi frequencies
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
%     k = find(lambda >= 1e-3, 1); % ignore 0.1% energy
    [~,ind] = sort(lambda,'descend');
    V = V(:,ind);
    params.V_pca = V;
%     feat_pca = V' * features;
    [C,params.dict_lores,params_spherical] = ...
        learnSphericalClusters(C,params.dict_lores,params_spherical);
    fprintf('Computing sparse coefficients\n');
    gamma = params.dict_lores' * C;
    gamma = gamma .* (repmat(max(gamma),size(gamma,1),1) == gamma);
    fprintf('Computing high-res. dictionary from low-res. dictionary\n');
    
    % FIXME: Do we still need all the patches from the hi-res images? (RAM hog)
    dict_hires = (patches * gamma') / (gamma * gamma');
    
    params.dict_hires = dict_hires;
    params.spherical_conf = params_spherical;
    
    
end
