% function result = dimpred_rise(images, model, cfg)
%
% Heatmaps that show which parts of each image drive its predicted
% dimensions, with RISE (Petsiuk et al., 2018), as in Figure 7 of the
% DimPred paper. Each image is multiplied with many random masks (6000 by
% default, the same masks for every image), the dimensions of each masked
% image are predicted, and the map of a dimension is the average of its
% predictions, weighted by the masks. The relevance map combines the maps
% of all dimensions, weighted by the predicted dimension values of the
% image without masks and divided by their sum. It shows which parts of
% the image matter for the dimensions that describe it best.
%
% The networks run in Python, so this function runs the command line tool
% of the Python version of dimpred,
%
%   python -m dimpred <images> --rise --n-masks <n> --model <model file> --out <temporary .mat file>
%
% and loads the file that it writes (a temporary file, deleted at the
% end). Python needs numpy, scipy, torch, open_clip_torch, timm (1.0.15 or
% newer) and pillow, as for dimpred_extract_features (see there for
% cfg.python and DIMPRED_PYTHON). What Python prints, e.g. one line per
% image, is shown while it runs.
%
% The masks are those of the DimPred paper and of the original RISE code:
% an 8 x 8 grid of cells, each kept with probability 0.1, upsampled
% bilinearly and shifted at random. Unlike the original code, the maps are
% normalized at each pixel by the sum of the masks (in Python,
% normalization="original" gives the normalization of the original code).
% More settings (grid, p, normalization, seed, size of the maps) are in
% Python, help(dimpred.rise).
%
% Time: each image and each mask is one pass through the network. With
% 6000 masks this takes about 10 min per image with RN50x64 and less than
% 1 min with AligNet on the GPU of an Apple M1 Max, much longer on the cpu.
% Fewer masks (e.g. cfg.n_masks = 2000) still give stable maps.
%
% Number of images: Python returns the maps in a .mat file, which holds at
% most 2 GB per variable, so one call takes at most 162 images with 66
% dimensions (218 with 49). The maps of one image need about 13 MB in
% MATLAB. On Windows, the length of the command line (at most 8191
% characters) can limit the number further. Too many images give the
% error dimpred:tooManyImages before Python starts. For more images, call
% dimpred_rise in a loop, e.g. with cfg.png to keep only the PNG files.
%
% Model: for heatmaps we recommend rn50x64_66d_ridge (a convolutional
% network; of the 66d models we compared, its maps were the closest to
% Figure 7 of the DimPred paper). Figure 7 itself was made with
% rn50x64_49d_ridge. Any model works; the default model (AligNet) is the
% fast option.
%
% The maps cover what the network sees: for the CLIP models (e.g.
% rn50x64_66d_ridge) the central square of the image, for AligNet the
% whole image, resized to a square. They are computed at the input size of
% the network (448 x 448 for RN50x64) and returned at 224 x 224. The field
% view is this image, so that the maps can be shown on top of it.
%
% Input:
%   images: cell array of image files, or one file as text (no folders,
%           use dimpred_find_images)
%   model:  model name, path of a model file, or a model struct, e.g.
%           from dimpred_load_model, which Python gets as it is, also if
%           it was changed by hand (default: [], the default model
%           alignet_siglip2b_66d_kernel)
%   cfg:    optional struct with the fields
%     cfg.n_masks:    number of masks (default: 6000)
%     cfg.png:        folder for PNG files (default: '', no PNG files): for
%                     each image <name>_relevance.png and, for the 3
%                     dimensions with the largest values,
%                     <name>_top<rank>_dim<k>.png, each map on the image
%     cfg.python:     Python program (default: the environment variable
%                     DIMPRED_PYTHON if it is set, else 'python3')
%     cfg.device:     'cuda', 'mps' or 'cpu' (default: '', i.e. the first
%                     of these that is available)
%     cfg.batch_size: number of masked images passed through the network
%                     at once (default: 32); use a smaller value if you run
%                     out of memory
%
% Output:
%   result: struct with the fields
%     relevance:      single, n_images x 224 x 224, the relevance map of
%                     each image
%     dimension_maps: single, n_images x n_dims x 224 x 224, the map of
%                     each dimension
%     embedding:      double, n_images x n_dims, the predicted dimension
%                     values of the images without masks (as
%                     dimpred_predict(dimpred_extract_features(images, model), model))
%     labels:         n_dims x 1 cell, the names of the dimensions
%     files:          the given image files as a cell column
%     view:           uint8, n_images x 224 x 224 x 3, the image as the
%                     network sees it (RGB)
%     model:          name of the model
%     settings:       struct with n_masks, grid, p, seed, normalization,
%                     input_size and map_size
%   The images come first, as in Python, so use squeeze for one image.
%
% Example:
%   cfg.n_masks = 2000;
%   result = dimpred_rise({'cat.jpg'}, 'rn50x64_66d_ridge', cfg);
%   [~, top] = max(result.embedding(1, :));
%   subplot(1, 3, 1), image(squeeze(result.view(1, :, :, :))), axis image off
%   subplot(1, 3, 2), imagesc(squeeze(result.relevance(1, :, :))), axis image off, title('relevance')
%   subplot(1, 3, 3), imagesc(squeeze(result.dimension_maps(1, top, :, :))), axis image off, title(result.labels{top})
%   colormap(jet)
%
% Hebartlab, 2026/10/02
%
% See also DIMPRED_EXTRACT_FEATURES, DIMPRED_PREDICT, DIMPRED_FIND_IMAGES

% History:
% 2026/10/04: new default model alignet_siglip2b_66d_kernel (help text); a model
%   struct reaches Python with its kernel part and close-pair settings
% 2026/10/02: after review: a model struct reaches Python as it is; error
%   dimpred:tooManyImages for more than 2 GB of maps; trailing backslashes
%   on Windows; no second start line
% 2026/10/02: written for dimpred 1.1.0

function result = dimpred_rise(images, model, cfg)

% Check the images before Python is started
if ischar(images), images = {images}; end
files = cellstr(images); % also accepts strings ("...")
files = files(:);
if isempty(files)
    error('dimpred:noImages', 'No images given.')
end
full_files = cellfun(@full_path, files, 'UniformOutput', false);
for i_file = 1:numel(files)
    if exist(full_files{i_file}, 'dir') == 7
        error('dimpred:fileNotFound', ['%s is a folder. dimpred_rise only takes image files. Please get ' ...
            'the images of a folder with dimpred_find_images first, e.g. ' ...
            'dimpred_rise(dimpred_find_images(folder)).'], files{i_file})
    end
    if exist(full_files{i_file}, 'file') ~= 2
        error('dimpred:fileNotFound', 'Image not found: %s', files{i_file})
    end
end

% Model (an unknown model gives an error here, before Python is started).
% Python gets the model as the path of its file. A model struct is saved
% to a temporary file below, so that Python uses it as it is, also if it
% was changed by hand or has no file (as rise in Python with a model dict).
if ~exist('model', 'var'), model = []; end
model_variables = {'weights', 'feature_mean', 'feature_scale', 'target_mean', 'labels', 'info'};
model_is_struct = isstruct(model);
if model_is_struct
    missing = model_variables(~isfield(model, model_variables));
    if ~isempty(missing)
        error('dimpred:inconsistentModel', 'The model struct has no field %s (see help dimpred_load_model).', ...
            strjoin(missing, ', '))
    end
end
model = dimpred_load_model(model);

% Python returns the maps in a .mat file, which holds at most 2 GB per
% variable. The largest variable is dimension_maps, 4 bytes for each of
% n_images x n_dims x 224 x 224 values (Python checks the same).
n_dims = size(model.weights, 2);
max_images = floor((2^31 - 1) / (4 * n_dims * 224 * 224));
if numel(files) > max_images
    error('dimpred:tooManyImages', ['The maps of %i images with %i dimensions are too large for one call of ' ...
        'dimpred_rise: Python returns them in a .mat file, which holds at most 2 GB per variable, i.e. ' ...
        '%i images. Please call dimpred_rise for fewer images at a time, e.g. in a loop.'], ...
        numel(files), n_dims, max_images)
end

% Set defaults
if ~exist('cfg', 'var') || isempty(cfg), cfg = struct; end
if ~isfield(cfg, 'python') || isempty(cfg.python)
    cfg.python = getenv('DIMPRED_PYTHON');
    if isempty(cfg.python), cfg.python = 'python3'; end
end
if ~isfield(cfg, 'n_masks') || isempty(cfg.n_masks), cfg.n_masks = 6000; end
if ~isfield(cfg, 'png'), cfg.png = ''; end
if ~isfield(cfg, 'device'), cfg.device = ''; end % Python picks cuda, mps or cpu
if ~isfield(cfg, 'batch_size') || isempty(cfg.batch_size), cfg.batch_size = 32; end
cfg.python = char(cfg.python); % also accepts strings ("...")
cfg.png = char(cfg.png);
cfg.device = char(cfg.device);

% Put the dimpred repository first on the Python path, so that
% "python -m dimpred" finds the package next to this folder. The old
% PYTHONPATH is restored when we leave this function (also after an error).
repo = fileparts(fileparts(mfilename('fullpath')));
old_pythonpath = getenv('PYTHONPATH');
restore_pythonpath = onCleanup(@() setenv('PYTHONPATH', old_pythonpath));
if isempty(old_pythonpath)
    setenv('PYTHONPATH', repo);
else
    setenv('PYTHONPATH', [repo pathsep old_pythonpath]);
end

% On Linux, MATLAB's own libraries would be found by torch and pillow, see
% dimpred_extract_features. We remove them while Python runs.
if isunix && ~ismac
    old_ld_path = getenv('LD_LIBRARY_PATH');
    restore_ld_path = onCleanup(@() setenv('LD_LIBRARY_PATH', old_ld_path));
    ld_folders = strsplit(old_ld_path, pathsep);
    ld_folders = ld_folders(~strncmp(ld_folders, matlabroot, numel(matlabroot)));
    setenv('LD_LIBRARY_PATH', strjoin(ld_folders, pathsep));
end

% Temporary folder for the output of Python, deleted when we leave
out_folder = tempname;
mkdir(out_folder);
remove_out_folder = onCleanup(@() rmdir(out_folder, 's'));
out_file = fullfile(out_folder, 'heatmaps.mat');
if model_is_struct
    model_file = fullfile(out_folder, 'model.mat');
    % with the kernel part and the close-pair settings, if the model has them
    optional = {'kernel_features', 'kernel_coefficients', 'kernel_tau', 'close_pairs_weight', 'close_pairs_threshold'};
    optional = optional(isfield(model, optional));
    optional = optional(cellfun(@(v) ~isempty(model.(v)), optional));
    save(model_file, '-struct', 'model', model_variables{:}, optional{:}, '-v7');
else
    model_file = model.file;
end

% The command. Unlike dimpred_extract_features, we do not split the images
% into several Python runs (the 2 GB limit above allows only a few hundred
% images anyway). On Windows, a command can have at most 8191 characters,
% so there we give an error for a command that is too long.
quoted_files = cellfun(@quote, full_files, 'UniformOutput', false);
command = [quote(cfg.python) ' -m dimpred ' strjoin(quoted_files', ' ') ' --rise' ...
    sprintf(' --n-masks %i --batch-size %i', cfg.n_masks, cfg.batch_size) ' --model ' quote(model_file) ...
    ' --out ' quote(out_file)];
if ~isempty(cfg.png)
    command = [command ' --png ' quote(full_path(cfg.png))];
end
if ~isempty(cfg.device)
    command = [command ' --device ' quote(cfg.device)];
end
if ispc
    if numel(command) > 8000
        error('dimpred:tooManyImages', ['The command line for %i images is too long for Windows. ' ...
            'Please run dimpred_rise on fewer images at a time.'], numel(files))
    end
    % cmd.exe removes the first and the last quote of a command, so we add one more pair
    command = ['"' command '"'];
end

% Run Python. With -echo, what Python prints (the images, the model and
% the masks, then one line per image) is shown while it runs, which can
% take hours, and output still holds it for the error.
fprintf('Running RISE in Python (%s)\n', cfg.python);
[status, output] = system(command, '-echo');
if status ~= 0 || exist(out_file, 'file') ~= 2
    error('dimpred:pythonFailed', ['RISE in Python failed (exit status %i, Python: %s). cfg.python or the ' ...
        'environment variable DIMPRED_PYTHON has to be a Python with numpy, scipy, torch, open_clip_torch, ' ...
        'timm (1.0.15 or newer) and pillow. Python printed:\n%s'], status, cfg.python, strtrim(output))
end

% Load the result, in the order of the help
saved = load(out_file);
result = struct();
result.relevance = saved.relevance;
result.dimension_maps = saved.dimension_maps;
result.embedding = saved.embedding;
result.labels = saved.labels(:);
result.files = files; % as given (Python got the full paths)
result.view = saved.view;
result.model = saved.model;
result.settings = saved.settings;
% Python's last line names the temporary file, which is deleted now
fprintf('Loaded the heatmaps into MATLAB and deleted the temporary file\n');


%% Subfunctions

function fname = full_path(fname)
% Absolute path of fname. A relative path is taken relative to the current
% folder, as in Python (as in dimpred_extract_features).
if isempty(regexp(fname, '^([A-Za-z]:)?[\\/]', 'once'))
    fname = fullfile(pwd, fname);
end

function arg = quote(arg)
% Quotes arg for the command line, so that it arrives in Python as one
% argument, unchanged, also with spaces, quotes or & in it. On Windows,
% backslashes at the end are doubled, because \" would be read as a quote
% (as subprocess.list2cmdline in Python does it); cmd.exe still replaces
% %NAME% by the environment variable NAME.
if ispc
    arg = ['"' regexprep(arg, '(\\+)$', '$1$1') '"']; % names on Windows cannot contain "
else
    arg = ['''' strrep(arg, '''', '''\''''') '''']; % in single quotes, each ' becomes '\''
end
