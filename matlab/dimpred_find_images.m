% function files = dimpred_find_images(folder)
%
% All image files in a folder, sorted by file name. Use this to get the
% images of a folder for dimpred_extract_features, which only takes files,
% so that the order of the rows of the features is always the order of the
% files you passed.
%
% Only files directly in the folder are returned (subfolders are not
% searched), with the extensions .jpg .jpeg .png .bmp .tif .tiff .webp in
% upper or lower case. Hidden files (names starting with '.') are left out,
% e.g. the '._' files that macOS writes on external drives, and so are
% links to files that no longer exist. The names are sorted by their
% characters, so upper case comes before lower case and 'img10.jpg' before
% 'img2.jpg' (the same order as sorted() in Python). If you need another
% order, sort the list yourself.
%
% Input:
%   folder: path of the folder, absolute or relative to the current folder
%
% Output:
%   files: cell column with the absolute paths of the images
%
% Example:
%   files = dimpred_find_images('my_images');
%   features = dimpred_extract_features(files);
%
% Martin Hebart, 2026/09/30
%
% See also DIMPRED_EXTRACT_FEATURES

% History:
% 2026/09/30: written for the first release of the package

function files = dimpred_find_images(folder)

% Check input. An empty name is an error (as in Python), not the current
% folder.
folder = char(folder);
if isempty(folder)
    error('dimpred:folderNotFound', 'No folder given.')
end
folder = full_path(folder);
if exist(folder, 'dir') ~= 7
    if exist(folder, 'file')
        error('dimpred:folderNotFound', ['%s is a file, not a folder. dimpred_find_images needs a folder, ' ...
            'single images can be passed to dimpred_extract_features directly.'], folder)
    end
    error('dimpred:folderNotFound', 'The folder %s does not exist.', folder)
end

% Get the image files, sorted by name. We compare the extensions in lower
% case instead of searching with dir('*.jpg'), because on Linux that would
% miss a.JPG, while on the Mac it finds a.JPG also with *.jpg.
extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'};
entries = dir(folder);
if ~isempty(entries)
    folder = entries(1).folder; % without '..' or '.' in it, as os.path.abspath in Python
end
% A link to a file that no longer exists is listed by dir as a file, but
% without size. We leave it out, as Python does.
is_file = ~[entries.isdir] & ~cellfun(@isempty, {entries.bytes});
names = {entries(is_file).name}';
[~, ~, ext] = cellfun(@fileparts, names, 'UniformOutput', false);
is_image = ismember(lower(ext), extensions) & ~strncmp(names, '.', 1);
names = sort(names(is_image));

if isempty(names)
    error('dimpred:noImages', 'No images found in %s (looked for files ending in %s).', folder, strjoin(extensions, ' '))
end
files = fullfile(folder, names);


%% Subfunctions

function fname = full_path(fname)
% Absolute path of fname. A relative path is taken relative to the current
% folder, as in Python. We do not use exist for this, because it would also
% find folders of the same name on the MATLAB path.
if isempty(regexp(fname, '^([A-Za-z]:)?[\\/]', 'once'))
    fname = fullfile(pwd, fname);
end
