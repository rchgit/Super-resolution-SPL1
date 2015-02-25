function dir = qmkdir(dir)
% Quiet MKDIR (does not emit warning if DIR exists)
[~, message] = mkdir(dir);  %#ok<NASGU>
