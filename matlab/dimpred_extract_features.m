% function [features, files] = dimpred_extract_features(images, model, cfg)
%
% Network features of images, as needed by dimpred_predict. The networks
% run in Python (AligNet with dimpred/alignet.py, the CLIP networks with
% open_clip), which gives the same features the models were trained on, so
% this function runs the command line tool of the Python version of
% dimpred,
%
%   python -m dimpred <images> --features-only --model <model file> --out <temporary .mat file>
%
% and reads the features from the file that it writes. Python needs numpy,
% scipy, torch, open_clip_torch, timm (1.0.15 or newer) and pillow (pip
% install numpy scipy torch open_clip_torch "timm>=1.0.15" pillow). The
% Python package dimpred itself does not have to be installed: we put the
% dimpred repository (the folder above the folder of this function) on the
% Python path, so that Python and MATLAB use the same code and models.
%
% In Python, each image is read with PIL, converted to RGB, preprocessed as
% the network expects it (AligNet SigLIP2-B: the whole image resized to
% 224 x 224; RN50x64: resize to 448 px and center crop; ViT-B/32: 224 px),
% and passed through the network in float32. The features are pre_logits
% for AligNet and the output of the image encoder for the CLIP networks,
% not normalized, as used for training the models. The first time AligNet
% is used, Python downloads its weights (378 MB, into ~/.cache/dimpred);
% if you cannot download them there, set the environment variable
% DIMPRED_ALIGNET_WEIGHTS to the path of alignet_siglip2_b.safetensors
% (setenv in MATLAB is passed on to Python). The network is set by the
% model (model.info.network), so the features always fit the model you use
% for dimpred_predict. Unlike extract_features in Python, there is no option
% for another network or other pretrained weights: the network always comes
% from the model.
%
% Folders are not accepted, list their images with dimpred_find_images
% first. In this way, row i of the features always belongs to file i. All
% files are checked before Python is started, so a wrong path gives an
% error at once. Starting Python and loading the network take a few
% seconds (RN50x64 longer), and the first time a network is used, open_clip
% downloads its weights. The length of a command line is limited, so for
% many images Python is started several times, each time with as many
% images as fit into one command.
%
% Input:
%   images: cell array of image files, or one file as text
%   model:  model name, path of a model file, or a model from
%           dimpred_load_model; its network is used (default: [], the
%           default model alignet_siglip2b_66d_ridge with the network
%           AligNet SigLIP2-B). Python gets a model struct as it is,
%           also if it was changed or made by hand.
%   cfg:    optional struct with the fields
%     cfg.python:     Python program, e.g. '/home/me/envs/dimpred/bin/python'
%                     (default: the environment variable DIMPRED_PYTHON if
%                     it is set, else 'python3')
%     cfg.device:     'cuda', 'mps' or 'cpu' (default: '', i.e. the first
%                     of these that is available). Different devices give
%                     slightly different features (differences of about 1e-4).
%     cfg.batch_size: number of images passed through the network at once
%                     (default: 32); use a smaller value if you run out of
%                     memory
%
% Output:
%   features: single, n_images x n_features (one row per image, in the
%             order of images; AligNet SigLIP2-B: 768 features, RN50x64:
%             1024, ViT-B-32-quickgelu: 512)
%   files:    the given image files as a cell column (file i belongs to
%             row i of the features)
%
% Example:
%   files = dimpred_find_images('my_images');
%   cfg.python = '/path/to/python'; % or setenv('DIMPRED_PYTHON', '/path/to/python')
%   features = dimpred_extract_features(files, 'rn50x64_49d_ridge', cfg);
%   embedding = dimpred_predict(features, 'rn50x64_49d_ridge');
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_FIND_IMAGES, DIMPRED_PREDICT, DIMPRED_LOAD_MODEL

% History:
% 2026/10/02: after review: a model struct reaches Python as it is; the
%   length of the command line is counted in bytes; trailing backslashes
%   on Windows
% 2026/10/02: new default model alignet_siglip2b_66d_ridge (help text, timm version, error message)
% 2026/09/30: written for the first release of the package

function [features, files] = dimpred_extract_features(images, model, cfg)

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
        error('dimpred:fileNotFound', ['%s is a folder. dimpred_extract_features only takes image files. Please get ' ...
            'the images of a folder with dimpred_find_images first, e.g. ' ...
            'dimpred_extract_features(dimpred_find_images(folder)).'], files{i_file})
    end
    if exist(full_files{i_file}, 'file') ~= 2
        error('dimpred:fileNotFound', 'Image not found: %s', files{i_file})
    end
end

% Model (an unknown model gives an error here, before Python is started).
% Python gets the model as the path of its file, which also works for model
% files outside of dimpred/models, and reads the network from that file. A
% model struct is saved to a temporary file below, so that Python uses it
% as it is, also if it was changed by hand or has no file (as
% extract_features in Python with a model dict).
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

% Set defaults
if ~exist('cfg', 'var') || isempty(cfg), cfg = struct; end
if ~isfield(cfg, 'python') || isempty(cfg.python)
    cfg.python = getenv('DIMPRED_PYTHON');
    if isempty(cfg.python), cfg.python = 'python3'; end
end
if ~isfield(cfg, 'device'), cfg.device = ''; end % Python picks cuda, mps or cpu
if ~isfield(cfg, 'batch_size') || isempty(cfg.batch_size), cfg.batch_size = 32; end
cfg.python = char(cfg.python); % also accepts strings ("...")
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

% On Linux, MATLAB puts its own library folders into LD_LIBRARY_PATH, and
% Python inherits them. The libraries of torch or pillow would then find
% the older libstdc++ of MATLAB instead of the one of the system, which
% gives errors such as "GLIBCXX_3.4.29 not found". We remove these folders
% while Python runs (restored in the same way as PYTHONPATH).
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
if model_is_struct
    model_file = fullfile(out_folder, 'model.mat');
    save(model_file, '-struct', 'model', model_variables{:}, '-v7');
else
    model_file = model.file;
end

% Split the images into parts that fit into one command. A command can
% have at most 8191 characters on Windows and 128 kB on Linux (macOS
% allows more). Linux counts bytes, and a character such as a Chinese one
% takes 3 bytes in UTF-8, so we count bytes (on Windows, this gives
% smaller parts than needed). Each part starts Python and loads the
% network again.
if ispc, max_bytes = 6000; else, max_bytes = 100000; end
quoted_files = cellfun(@quote, full_files, 'UniformOutput', false);
n_bytes = cellfun(@(arg) numel(unicode2native(arg, 'UTF-8')), quoted_files);
part = 1 + floor(cumsum(n_bytes + 1) / max_bytes);
parts = unique(part);

% Options of the command line tool, the same for all parts
options = [' --features-only --model ' quote(model_file) sprintf(' --batch-size %i', cfg.batch_size)];
if ~isempty(cfg.device)
    options = [options ' --device ' quote(cfg.device)];
end

% Run Python for each part
if numel(files) == 1
    images_text = '1 image';
else
    images_text = sprintf('%i images', numel(files));
end
fprintf('Extracting the features of %s with %s in Python (%s)\n', images_text, model.info.network, cfg.python);
features = cell(numel(parts), 1);
for i_part = 1:numel(parts)
    in_part = part == parts(i_part);
    if numel(parts) > 1
        fprintf('  Python run %i of %i (%i images)\n', i_part, numel(parts), nnz(in_part));
    end

    out_file = fullfile(out_folder, sprintf('features_%i.mat', i_part));
    command = [quote(cfg.python) ' -m dimpred ' strjoin(quoted_files(in_part)', ' ') options ' --out ' quote(out_file)];
    if ispc
        % cmd.exe removes the first and the last quote of a command, so we add one more pair
        command = ['"' command '"']; %#ok<AGROW>
    end
    [status, output] = system(command);

    if status ~= 0 || exist(out_file, 'file') ~= 2
        error('dimpred:pythonFailed', ['Feature extraction in Python failed (exit status %i, Python: %s). ' ...
            'cfg.python or the environment variable DIMPRED_PYTHON has to be a Python with numpy, scipy, ' ...
            'torch, open_clip_torch, timm (1.0.15 or newer) and pillow. Python printed:\n%s'], status, ...
            cfg.python, strtrim(output))
    end
    result = load(out_file, 'features');
    if size(result.features, 1) ~= nnz(in_part)
        error('dimpred:pythonFailed', 'Python returned the features of %i images instead of %i. Python printed:\n%s', ...
            size(result.features, 1), nnz(in_part), strtrim(output))
    end
    features{i_part} = result.features;
end
% Python saves float32, i.e. single; single() only guards against double
features = single(cat(1, features{:}));


%% Subfunctions

function fname = full_path(fname)
% Absolute path of fname. A relative path is taken relative to the current
% folder, as in Python. We do not use exist for this, because it would also
% find files of the same name on the MATLAB path.
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
