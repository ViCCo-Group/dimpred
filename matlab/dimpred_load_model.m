% function model = dimpred_load_model(model)
%
% Load a model, i.e. everything that is needed to predict the SPoSE
% dimensions from network features: the regression weights, the mean and
% std of the features in the 1854 training images (for z-scoring new
% features), the mean of each dimension in the training images, the labels
% of the dimensions and a description of the model (info).
%
% The mean of each dimension (target_mean) is needed for every prediction:
% the regressions were fit on centered dimension values, so it has to be
% added back (see dimpred_predict). Without it, the predictions are too
% small and most of them are 0. For this reason, a model file that lacks
% one of the variables gives an error and is never filled with defaults.
%
% Model files are .mat files with the variables below. The Python version
% of dimpred uses the same files.
%   weights        n_features x n_dims
%   feature_mean   1 x n_features
%   feature_scale  1 x n_features (std of the features, all > 0)
%   target_mean    1 x n_dims (mean of each dimension)
%   labels         n_dims x 1 cell array of char
%   info           struct with the fields name, network, pretrained,
%                  layer, preprocessing, embedding, regression,
%                  training_images, source, note, created (text) and
%                  n_features, n_dims (numbers)
% You can build your own model files in the same format and pass their
% path.
%
% Input:
%   model: one of
%          [] or omitted: the default model, alignet_siglip2b_66d_ridge
%          name:   a model that comes with dimpred (see dimpred_list_models)
%          file:   path of a model file (.mat), absolute or relative to the
%                  current folder (the MATLAB path is not searched)
%          struct: a model that was already loaded, returned unchanged
%
% Output:
%   model: struct with the fields
%          weights        double, n_features x n_dims
%          feature_mean   double, 1 x n_features
%          feature_scale  double, 1 x n_features
%          target_mean    double, 1 x n_dims
%          labels         n_dims x 1 cell array of char
%          info           struct, as in the file
%          file           absolute path of the model file
%
% Example:
%   model = dimpred_load_model('rn50x64_49d_ridge');
%   disp(model.info.network)  % RN50x64
%   disp(size(model.weights)) % 1024 49
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_LIST_MODELS, DIMPRED_PREDICT

% History:
% 2026/10/02: new default model alignet_siglip2b_66d_ridge
% 2026/09/30: written for the first release of the package

function model = dimpred_load_model(model)

% The model that is used when no model is given (DEFAULT_MODEL in Python).
% The other functions get the default from here.
default_model = 'alignet_siglip2b_66d_ridge';

% A model that was already loaded
if exist('model', 'var') && isstruct(model)
    return
end

% Find the file: first among the shipped models, then as a path
if ~exist('model', 'var') || isempty(model)
    model = default_model;
end
model = char(model); % also accepts a string ("...")
names = dimpred_list_models;
if any(strcmp(names, model))
    fname = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'dimpred', 'models', [model '.mat']);
else
    fname = full_path(model);
    if exist(fname, 'file') ~= 2
        error('dimpred:unknownModel', ['Unknown model ''%s'': this is neither the name of a model that comes ' ...
            'with dimpred nor an existing model file. Available models: %s'], model, strjoin(names', ', '))
    end
end

% Load the file and check that all variables are there
data = load(fname, '-mat');
variables = {'weights', 'feature_mean', 'feature_scale', 'target_mean', 'labels', 'info'};
missing = variables(~isfield(data, variables));
if ~isempty(missing)
    error('dimpred:inconsistentModel', ['The model file %s has no variable %s. A model file needs the variables ' ...
        '%s (see help dimpred_load_model).'], fname, strjoin(missing, ', '), strjoin(variables, ', '))
end

model = struct;
model.weights = double(data.weights);
model.feature_mean = double(data.feature_mean(:)'); % as rows, also if the file has columns
model.feature_scale = double(data.feature_scale(:)');
model.target_mean = double(data.target_mean(:)');
model.labels = data.labels(:);
model.info = data.info;
model.file = fname;

% Check that the sizes fit together
[n_features, n_dims] = size(model.weights);
problems = {};
if numel(model.feature_mean) ~= n_features
    problems{end+1} = sprintf('feature_mean has %i values', numel(model.feature_mean));
end
if numel(model.feature_scale) ~= n_features
    problems{end+1} = sprintf('feature_scale has %i values', numel(model.feature_scale));
end
if numel(model.target_mean) ~= n_dims
    problems{end+1} = sprintf('target_mean has %i values', numel(model.target_mean));
end
if numel(model.labels) ~= n_dims
    problems{end+1} = sprintf('there are %i labels', numel(model.labels));
end
% info does not need n_features and n_dims, but if it has them they have to fit
if isfield(model.info, 'n_features') && ~isequal(double(model.info.n_features), n_features)
    problems{end+1} = sprintf('info.n_features is %s', num2str(model.info.n_features));
end
if isfield(model.info, 'n_dims') && ~isequal(double(model.info.n_dims), n_dims)
    problems{end+1} = sprintf('info.n_dims is %s', num2str(model.info.n_dims));
end
if ~isempty(problems)
    error('dimpred:inconsistentModel', 'The model file %s is inconsistent: weights are %i x %i (n_features x n_dims), but %s.', ...
        fname, n_features, n_dims, strjoin(problems, ', '))
end


%% Subfunctions

function fname = full_path(fname)
% Absolute path of fname. A relative path is taken relative to the current
% folder, as in Python. We do not use exist or which for this, because they
% would also find files of the same name on the MATLAB path.
if isempty(regexp(fname, '^([A-Za-z]:)?[\\/]', 'once'))
    fname = fullfile(pwd, fname);
end
