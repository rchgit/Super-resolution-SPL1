% Prune Black & white
% Rename all images nonconformant to YCbCr space with *.bak extension

function pruneBnW(filenames)

for i = 1:numel(filenames);
    try
        rgb2ycbcr(imread(filenames{i}));
    catch ME
        if (strcmp(ME.identifier,'images:rgb2ycbcr:invalidSizeForColormap'))
            [~,message,messageid] = movefile(filenames{i},[filenames{i} '.bak']);
            fprintf('%s %s',messageid,message);
        end
    end
end


end