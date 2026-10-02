% function names = dimpred_list_models
%
% Names of the models that come with dimpred. Each model is a .mat file in
% the folder dimpred/models of the repository, and its name is the file
% name without .mat. Any of these names can be passed as model to
% dimpred_load_model, dimpred_predict, dimpred_extract_features and
% dimpred_rise.
%
% The models are found relative to the folder of this function (in
% ../dimpred/models), not relative to the current folder. For this reason,
% the folder matlab has to stay in the dimpred repository, next to the
% folder dimpred.
%
% Input:
%   none
%
% Output:
%   names: cell column with the names of the models, sorted alphabetically,
%          e.g. {'alignet_siglip2b_66d_ridge'; 'rn50x64_49d_ridge'; ...
%                'rn50x64_66d_elastic'; 'rn50x64_66d_ridge'; 'vitb32_66d_elastic'}
%
% Example:
%   names = dimpred_list_models;
%   for i_model = 1:numel(names)
%       model = dimpred_load_model(names{i_model});
%       fprintf('%s: %s\n', names{i_model}, model.info.note)
%   end
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_LOAD_MODEL

% History:
% 2026/10/02: dimpred_rise in the help text
% 2026/10/02: new model alignet_siglip2b_66d_ridge in the help text
% 2026/09/30: written for the first release of the package

function names = dimpred_list_models

models_folder = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'dimpred', 'models');
if exist(models_folder, 'dir') ~= 7
    error('dimpred:unknownModel', ['The folder with the dimpred models was not found (%s). The folder matlab ' ...
        'has to stay in the dimpred repository, next to the folder dimpred.'], models_folder)
end

% The name of a model is the name of its file without .mat
model_files = dir(fullfile(models_folder, '*.mat'));
[~, names] = cellfun(@fileparts, {model_files.name}', 'UniformOutput', false);
names = sort(names);
