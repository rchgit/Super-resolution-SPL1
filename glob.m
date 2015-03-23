function [result] = glob(directory, pattern)
files = [];
for i =1:numel(pattern)
    d = fullfile(directory, pattern{i});
    files = [files; dir(d)];
end

result = cell(numel(files), 1);
for i = 1:numel(result)
    result{i} = fullfile(directory, files(i).name);
end

end
