%% Anchored Neighborhood Regression for Fast Example-Based Super-Resolution
% Example code
%
% March 22, 2013. Radu Timofte, VISICS @ KU Leuven
%
% Revised version: (includes all [1] methods)
% October 3, 2013. Radu Timofte, CVL @ ETH Zurich
%
% Updated version: (adds A+ methods [2])
% September 5, 2014. Radu Timofte, CVL @ ETH Zurich
% %
% Please reference to both:
% [1] Radu Timofte, Vincent De Smet, Luc Van Gool.
% Anchored Neighborhood Regression for Fast Example-Based Super-Resolution.
% International Conference on Computer Vision (ICCV), 2013. 
%
% [2] Radu Timofte, Vincent De Smet, Luc Van Gool.
% A+: Adjusted Anchored Neighborhood Regression for Fast Super-Resolution.
% Asian Conference on Computer Vision (ACCV), 2014. 
%
% For any questions, email me by timofter@vision.ee.ethz.ch
%

% LS Filter & Spherical learning revisions 
% February 21, 2015. Reich Canlas, DLSU Manila

function [params,simparams] = main_expt(simparams)
p = pwd;
addpath(fullfile(p, '/methods'));  % the upscaling methods

% Uncomment these if ksvdbox and ompbox are not yet installed in the path
% addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
% addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm

% Initializing dictionary sizes and neighbors
dict_sizes = [2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536];
neighbors = [1:1:12, 16:4:32, 40:8:64, 80:16:128, 256, 512, 1024];

%% RUN SETTINGS
imgscale = simparams.imgscale;
% imgscale = 1; % the scale reference we work with
flag = simparams.flag;
% flag = 0;       % flag = 0 - only GR, ANR, A+, and bicubic methods, the other get the bicubic result by default
                % flag = 1 - all the methods are applied
upscaling = simparams.upscaling;
% upscaling = 4; % the magnification factor x2, x3, x4...
filter = simparams.filter;
% filter = 'default';
% filter = 'LS'; % the filter 
input_dir = simparams.input_dir;
% input_dir = 'Set5'; % Directory with input images from Set5 image dataset
% input_dir = 'Set14'; % Directory with input images from Set14 image dataset
train_method = simparams.train_method;
%train_method = 'ksvd';
% train_method = 'spherical'; % use spherical training method
input_pattern = simparams.input_pattern;
% pattern = '*.bmp'; % Pattern to process
retrain = simparams.retrain;
% retrain = 1; % 0 to reload pre-trained files, 1 to force retrain
if retrain
    train_dir = simparams.train_dir;
    train_pattern = simparams.train_pattern;
end
% Number of atoms (in powers of 2)
d = simparams.d;

patch_ratio = simparams.patch_ratio;

%% Introduction


%d = 7
%for nn=1:28
%nn= 28

clusterszA = 2048; % neighborhood size for A+

disp('The experiment corresponds to the results from Table 2 in the referenced [1] and [2] papers.');

disp(['The experiment uses ' input_dir ' dataset and aims at a magnification of factor x' num2str(upscaling) '.']);

if flag==1
    disp('All methods are employed : Bicubic, Yang et al., Zeyde et al., GR, ANR, NE+LS, NE+NNLS, NE+LLE, A+ (0.5 mil), A+, A+ (16 atoms).');    
else
    disp('We run only for Bicubic, GR, ANR and A+ methods, the other get the Bicubic result by default.');
end

fprintf('\n\n');
if strcmp(filter,'LS')
    load('matfiles/filter_trainLS_YCbCr.mat','hHighRes'); 
end

%% MAIN CODE
tag = ['matfiles/' input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];
disp(['Upscaling x' num2str(upscaling) ' ' input_dir ' with Zeyde dictionary of size = ' num2str(dict_sizes(d))]);
zeyde_file = ['matfiles/conf_Zeyde_' num2str(dict_sizes(d)) '_final_x' num2str(upscaling)];    

if exist([zeyde_file '.mat'],'file') && retrain == 0

    fprintf('\t');
    disp(['Load trained dictionary...' zeyde_file]);
    load(zeyde_file, 'params');

else
    if exist([zeyde_file '.mat'],'file') && retrain == 1
        % Delete pre-trained data
        fprintf('\tDeleting pre-trained dictionary...');
        delete([zeyde_file '.mat']);
    end
    % Training
    fprintf('\t');
    disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using Zeyde approach...']);
    % Simulation settings
    params.scale = upscaling; % scale-up factor
    params.level = 1; % # of scale-ups to perform
    params.window = [3 3]; % low-res. window size
    params.border = [1 1]; % border of the image (to ignore)

    % High-pass filters for feature extraction (defined for upsampled 
    % low-res.)
    params.train_method = train_method;

    if strcmp(filter,'LS')
        params.filters = {hHighRes.L,hHighRes.H1,hHighRes.H2,hHighRes.H3,hHighRes.H4,...
            hHighRes.V1,hHighRes.V2,hHighRes.V3,hHighRes.V4};

    else
        params.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, params.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        params.filters = {G, G.', L, L.'}; % 2D versions
    end
    params.interpolate_kernel = 'bicubic';

    params.overlap = [1 1]; % partial overlap (for faster training)
    if upscaling <= 2
        params.overlap = [1 1]; % partial overlap (for faster training)
    end

    startt = tic;

    % Learn dictionaries via Spherical
    params = learn_dict_puresphere(params,simparams);   
    params.overlap = params.window - [1 1]; % full overlap scheme (for better reconstruction)    
    params.trainingtime = toc(startt);
    toc(startt)

    save(zeyde_file, 'params');                       
    % train call        
end

if dict_sizes(d) < 1024
    lambda = 0.01;
elseif dict_sizes(d) < 2048
    lambda = 0.1;
elseif dict_sizes(d) < 8192
    lambda = 1;
else
    lambda = 5;
end
    
    %% GR
    fprintf('GR\n');
    if dict_sizes(d) < 10000
        params.ProjM = (params.dict_lores'*params.dict_lores+lambda*eye(size(params.dict_lores,2)))\params.dict_lores';    
        params.PP = (1+lambda)*params.dict_hires*params.ProjM;
    else
        % here should be an approximation
        params.PP = zeros(size(params.dict_hires,1), size(params.V_pca,2));
        params.ProjM = [];
    end
     
    params.filenames = glob(input_dir, input_pattern); % Cell array      
    params.desc = {'Original', 'Bicubic', 'Yang et al.', ...
         'Zeyde et al.', 'Our GR', 'Our ANR', ...
         'NE+LS','NE+NNLS','NE+LLE','Our A+ (0.5mil)','Our A+', ...
         'Our A+ (16atoms)'};
    params.results = {};
     
    %params.points = [1:10:size(params.dict_lores,2)];
    params.points = 1:1:size(params.dict_lores,2);
    params.pointslo = params.dict_lores(:,params.points);
    params.pointsloPCA = params.pointslo'*params.V_pca';
    
    % precompute for ANR the anchored neighborhoods and the projection
    % matrices for the dictionary 
    
    params.PPs = [];    
    if size(params.dict_lores,2) < 40
        clustersz = size(params.dict_lores,2);
    else
        clustersz = 40;
    end
    D = abs(params.pointslo'*params.dict_lores);    
    
    for i = 1:length(params.points)
        [~, idx] = sort(D(i,:), 'descend');
        if (clustersz >= size(params.dict_lores,2)/2)
            params.PPs{i} = params.PP;
        else
            Lo = params.dict_lores(:, idx(1:clustersz));        
            params.PPs{i} = 1.01*params.dict_hires(:,idx(1:clustersz))/(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';    
        end
    end    
     
    ANR_PPs = params.PPs; % store the ANR regressors
    
    save([zeyde_file '_ANR_projections_imgscale_' num2str(imgscale)],'params');
    
    %% A+ computing the regressors
    fprintf('A+ (5 mil)\n');
    Aplus_PPs = [];
    aplus_5mil_file = ['matfiles/Aplus_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszA) 'nn_5mil.mat'];
    
    if exist(aplus_5mil_file,'file') && retrain == 0
        fprintf('Load A+ 5mil file\n');
        load(aplus_5mil_file);
         
    else
        if exist(aplus_5mil_file,'file') && retrain == 1
            % Delete pre-trained data
            delete(aplus_5mil_file);
        end
        fprintf('\tCompute A+ regressors');
        ttime = tic;
        tic
        [plores, phires] = collectSamplesScales(params, load_images(...            
            glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  

        if size(plores,2) > 5000000                
            plores = plores(:,1:5000000);
            phires = phires(:,1:5000000);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n
        fprintf('\tA+: L2 normalized and scaled\n');

        llambda = 0.1;
        Aplus_PPs = cell(size(params.dict_lores,2),1);
        for i = 1:size(params.dict_lores,2)
            D = pdist2(single(plores'),single(params.dict_lores(:,i)'));
            [~, idx] = sort(D);                
            Lo = plores(:, idx(1:clusterszA));                                    
            Hi = phires(:, idx(1:clusterszA));
            Aplus_PPs{i} = Hi/(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo'; 
            fprintf('\tA+ regressors %.3f%% complete \n',i*100/size(params.dict_lores,2));
        end        
        clear plores
        clear phires
        
        ttime = toc(ttime);        
        save(aplus_5mil_file,'Aplus_PPs','ttime', 'number_samples');   
        toc
    end    

    %% A+ (0.5mil) computing the regressors with 0.5 milion training samples
    fprintf('A+ (0.5 mil)\n');
    Aplus05_PPs = [];    
    aplus_05mil_file = ['matfiles/Aplus_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszA) 'nn_05mil.mat'];    
    
    if exist(aplus_05mil_file,'file') && retrain == 0
       
        fprintf('Load A+ 0.5mil file\n');
        load(aplus_05mil_file);

    else
        if exist(aplus_05mil_file,'file') && retrain == 1
            delete(aplus_05mil_file);
        end
        disp('Compute A+ (0.5 mil) regressors');
        ttime = tic;
        tic
        [plores, phires] = collectSamplesScales(params, load_images(...            
        glob('CVPR08-SR/Data/Training', '*.bmp')), 1,1);  

        if size(plores,2) > 500000                
            plores = plores(:,1:500000);
            phires = phires(:,1:500000);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);      
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        fprintf('\tA+ (0.5 mil): L2 normalized and scaled\n');
        clear l2
        clear l2n

        llambda = 0.1;
        Aplus05_PPs = cell(size(params.dict_lores,2));
        for i = 1:size(params.dict_lores,2)
            D = pdist2(single(plores'),single(params.dict_lores(:,i)'));
            [~, idx] = sort(D);                
            Lo = plores(:, idx(1:clusterszA));                                    
            Hi = phires(:, idx(1:clusterszA));
            Aplus05_PPs{i} = Hi/(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo'; 
            fprintf('\tA+ (0.5 mil) regressors %.3f%% complete \n',i*100/size(params.dict_lores,2));
        end        
        clear plores;
        clear phires;
        
        ttime = toc(ttime);        
        save(aplus_05mil_file,'Aplus05_PPs','ttime', 'number_samples');   
        toc
    end            
    
    %% load the A+ (16 atoms) for comparison results
    conf16 = [];       
    fprintf('Loading A+ (16 atoms) for comparison\n');
    aplus_05mil_file = ['matfiles/Aplus_x' num2str(upscaling) '_16atoms' num2str(clusterszA) 'nn_05mil.mat'];
    fnamec = ['matfiles/Set14_x' num2str(upscaling) '_16atoms_conf_Zeyde_16_finalx' num2str(upscaling) '_ANR_projections_imgscale_' num2str(imgscale) '.mat']; 
    if exist(aplus_05mil_file,'file') && exist(fnamec,'file')
       kk = load(fnamec);
       conf16 = kk.params;       
       kk = load(aplus_05mil_file);       
       conf16.PPs = kk.Aplus05_PPs;
       clear kk
    end
    %% Save folders
    params.result_dirImages = qmkdir([input_dir '/results_' tag]);
    params.result_dirImagesRGB = qmkdir([input_dir '/results_' tag 'RGB']);
    params.result_dir = qmkdir(['results/Results-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
    params.result_dirRGB = qmkdir(['results/ResultsRGB-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
    
    %% Progress Monitoring
    t = cputime;    
    params.countedtime = zeros(numel(params.desc),numel(params.filenames));
    res =[];
    for i = 1:numel(params.filenames)
        f = params.filenames{i};
        [p, n, x] = fileparts(f);
        [img, imgCB, imgCR] = load_images({f}); 
        if imgscale<1
            img = resize(img, imgscale, params.interpolate_kernel);
            imgCB = resize(imgCB, imgscale, params.interpolate_kernel);
            imgCR = resize(imgCR, imgscale, params.interpolate_kernel);
        end
        sz = size(img{1});
        
        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(params.filenames), f, sz(1), sz(2));
    
        img = modcrop(img, params.scale^params.level);
        imgCB = modcrop(imgCB, params.scale^params.level);
        imgCR = modcrop(imgCR, params.scale^params.level);

            low = resize(img, 1/params.scale^params.level, params.interpolate_kernel);
            if ~isempty(imgCB{1})
                lowCB = resize(imgCB, 1/params.scale^params.level, params.interpolate_kernel);
                lowCR = resize(imgCR, 1/params.scale^params.level, params.interpolate_kernel);
            end
            
        interpolated = resize(low, params.scale^params.level, params.interpolate_kernel);
        if ~isempty(imgCB{1})
            interpolatedCB = resize(lowCB, params.scale^params.level, params.interpolate_kernel);    
            interpolatedCR = resize(lowCR, params.scale^params.level, params.interpolate_kernel);    
        end
        
        res{1} = interpolated;
                        
        if (flag == 1) && (dict_sizes(d) == 1024) && (upscaling==3)
            startt = tic;
            res{2} = {yima(low{1}, upscaling)};                        
            toc(startt)
            params.countedtime(2,i) = toc(startt);
        else
            res{2} = interpolated;
        end
        
        % Zeyde
        if (flag == 1)
            startt = tic;
            fprintf('Zeyde\n');
            res{3} = scaleup_Zeyde(params, low);
            toc(startt)
            params.countedtime(3,i) = toc(startt);    
        else
            res{3} = interpolated;
        end
        
        % GR
        fprintf('GR\n');
        startt = tic;
        res{4} = scaleup_GR(params, low);
        toc(startt)
        params.countedtime(4,i) = toc(startt);    
        
        % ANR
        fprintf('ANR\n');
        startt = tic;
        params.PPs = ANR_PPs;
        res{5} = scaleup_ANR(params, low);
        toc(startt)
        params.countedtime(5,i) = toc(startt);    
        
        % NE+LS
        if flag == 1
            startt = tic;
            fprintf('NE+LS\n');
            if 12 < dict_sizes(d)
                res{6} = scaleup_NE_LS(params, low, 12);
            else
                res{6} = scaleup_NE_LS(params, low, dict_sizes(d));
            end
            toc(startt)
            params.countedtime(6,i) = toc(startt);    
        else
            res{6} = interpolated;
        end
        
        % NE+NNLS
        if flag == 1
            startt = tic;
            fprintf('NE+NNLS\n');
            if 24 < dict_sizes(d)
                res{7} = scaleup_NE_NNLS(params, low, 24);
            else
                res{7} = scaleup_NE_NNLS(params, low, dict_sizes(d));
            end
            toc(startt)
            params.countedtime(7,i) = toc(startt);    
        else
            res{7} = interpolated;
        end
        % NE+LLE
        if flag == 1
            startt = tic;
            fprintf('NE+LLE\n');
            if 24 < dict_sizes(d)
                res{8} = scaleup_NE_LLE(params, low, 24);
            else
                res{8} = scaleup_NE_LLE(params, low, dict_sizes(d));
            end
            toc(startt)
            params.countedtime(8,i) = toc(startt);    
        else
            res{8} = interpolated;
        end
            
        % A+ (0.5 mil)
        if flag == 1 && ~isempty(Aplus05_PPs)
            fprintf('A+ (0.5mil)\n');
            params.PPs = Aplus05_PPs;
            startt = tic;
            res{9} = scaleup_ANR(params, low);
            toc(startt)
            params.countedtime(9,i) = toc(startt);    
        else
            res{9} = interpolated;
        end
        
        % A+
        if ~isempty(Aplus_PPs)
            fprintf('A+\n');
            params.PPs = Aplus_PPs;
            startt = tic;
            res{10} = scaleup_ANR(params, low);
            toc(startt)
            params.countedtime(10,i) = toc(startt);    
        else
            res{10} = interpolated;
        end        
        % A+ 16atoms
        if flag == 1 && ~isempty(conf16)
            fprintf('A+ 16atoms\n');
            startt = tic;
            res{11} = scaleup_ANR(conf16, low);
            toc(startt)
            params.countedtime(11,i) = toc(startt);    
        else
            res{11} = interpolated;
        end
        
        result = cat(3, img{1}, interpolated{1}, res{2}{1}, res{3}{1}, ...
            res{4}{1}, res{5}{1}, res{6}{1}, res{7}{1}, res{8}{1}, ...
            res{9}{1}, res{10}{1}, res{11}{1});
        
        result = shave(uint8(result * 255), params.border * params.scale);
        
        if ~isempty(imgCB{1})
            resultCB = interpolatedCB{1};
            resultCR = interpolatedCR{1};           
            resultCB = shave(uint8(resultCB * 255), params.border * params.scale);
            resultCR = shave(uint8(resultCR * 255), params.border * params.scale);
        end

        params.results{i} = {};
        for j = 1:numel(params.desc)            
            params.results{i}{j} = fullfile(params.result_dirImages, [n sprintf('[%d-%s]', j, params.desc{j}) x]);            
            imwrite(result(:, :, j), params.results{i}{j});

            params.resultsRGB{i}{j} = fullfile(params.result_dirImagesRGB, [n sprintf('[%d-%s]', j, params.desc{j}) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
            end
            
            imwrite(rgbImg, params.resultsRGB{i}{j});
        end        
        params.filenames{i} = f;
    end   
    params.duration = cputime - t;

    % Test performance
    scores = run_comparison(params);
    process_scores_Tex(params, scores,length(params.filenames));
    
    run_comparisonRGB(params); % provides color images and HTML summary
    % Save
    save([tag '_results_imgscale_' num2str(imgscale)],'params','scores','simconf');

%
end
