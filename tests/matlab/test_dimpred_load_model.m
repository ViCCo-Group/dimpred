% function tests = test_dimpred_load_model
%
% Tests for dimpred_load_model and dimpred_list_models.
%
% We check that the five shipped models are found by name, that the default
% model is alignet_siglip2b_66d_ridge, that every model has all fields with sizes
% that fit together, that each model names the network it was trained with,
% and that wrong input gives an error with a message that helps the user.
%
% The network matters more than it seems: for OpenAI's ViT-B/32 the
% open_clip name has to be ViT-B-32-quickgelu. open_clip's plain ViT-B-32
% loads the same weights but uses GELU instead of QuickGELU, and its
% features correlate only about 0.98 with the ones the model was trained
% on. The predictions then look plausible but are off.
%
% The Python tests in tests/ check the same behavior.
%
% Run with
%   runtests('test_dimpred_load_model')
% or all MATLAB tests with run_dimpred_tests.
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_LOAD_MODEL, DIMPRED_LIST_MODELS, RUN_DIMPRED_TESTS

% History:
% 2026/10/04: new default model alignet_siglip2b_66d_kernel; its kernel part and
%   close-pair settings
% 2026/10/02: new default model alignet_siglip2b_66d_ridge (AligNet SigLIP2-B)
% 2026/09/30: after review: numbers compared with the file, models found
%   from any current folder, relative path to a model file, NaN and Inf in
%   feature_scale and target_mean
% 2026/09/30: written before the code (test-driven development)

function tests = test_dimpred_load_model
tests = functiontests(localfunctions);
end


%% Setup

function setupOnce(testCase)

% Add the dimpred MATLAB functions to the path (restored in teardownOnce)
here = fileparts(mfilename('fullpath')); % .../dimpred/tests/matlab
testCase.TestData.repo = fileparts(fileparts(here));
testCase.TestData.old_path = addpath(fullfile(testCase.TestData.repo, 'matlab'));

end

function teardownOnce(testCase)
path(testCase.TestData.old_path);
end


%% Listing and default model

function test_list_models_gives_shipped_models(testCase)
names = dimpred_list_models;
testCase.verifyEqual(names, shipped_models(), ...
    'dimpred_list_models should return the names of the 6 shipped models as a sorted cell column');
end

function test_default_model_is_alignet_siglip2b_66d_kernel(testCase)
model = dimpred_load_model;
testCase.verifyEqual(model.info.name, 'alignet_siglip2b_66d_kernel', ...
    'Without input, dimpred_load_model should load alignet_siglip2b_66d_kernel');
model = dimpred_load_model([]);
testCase.verifyEqual(model.info.name, 'alignet_siglip2b_66d_kernel', ...
    'dimpred_load_model([]) should load alignet_siglip2b_66d_kernel');
end


%% Content of the shipped models

function test_models_have_all_fields_with_fitting_sizes(testCase)
names = shipped_models();
for i_model = 1:numel(names)
    name = names{i_model};
    model = dimpred_load_model(name);

    missing = setdiff({'weights', 'feature_mean', 'feature_scale', 'target_mean', 'labels', 'info', 'file'}, fieldnames(model));
    testCase.assertEmpty(missing, sprintf('Model %s lacks the fields: %s', name, strjoin(missing(:)', ', ')));

    % all sizes follow from the weights (n_features x n_dims)
    [n_features, n_dims] = size(model.weights);
    msg = sprintf('model %s', name);
    testCase.verifyClass(model.weights, 'double', msg);
    testCase.verifyClass(model.feature_mean, 'double', msg);
    testCase.verifyClass(model.feature_scale, 'double', msg);
    testCase.verifyClass(model.target_mean, 'double', msg);
    testCase.verifySize(model.feature_mean, [1 n_features], [msg ': feature_mean should be 1 x n_features']);
    testCase.verifySize(model.feature_scale, [1 n_features], [msg ': feature_scale should be 1 x n_features']);
    testCase.verifySize(model.target_mean, [1 n_dims], [msg ': target_mean should be 1 x n_dims']);
    testCase.verifySize(model.labels, [n_dims 1], [msg ': labels should be n_dims x 1']);
    testCase.verifyTrue(iscellstr(model.labels), [msg ': labels should be a cell array of text']);
    testCase.verifyTrue(isstruct(model.info), [msg ': info should be a struct']);
    testCase.verifyTrue(ischar(model.file), [msg ': file should be text']);
end
end

function test_models_have_complete_info(testCase)
text_fields = {'name', 'network', 'pretrained', 'layer', 'preprocessing', 'embedding', ...
    'regression', 'training_images', 'source', 'note', 'created'};
names = shipped_models();
for i_model = 1:numel(names)
    name = names{i_model};
    model = dimpred_load_model(name);
    info = model.info;

    missing = setdiff([text_fields, {'n_features', 'n_dims'}], fieldnames(info));
    testCase.assertEmpty(missing, sprintf('info of model %s lacks the fields: %s', name, strjoin(missing(:)', ', ')));

    for i_field = 1:numel(text_fields)
        testCase.verifyTrue(ischar(info.(text_fields{i_field})), ...
            sprintf('info.%s of model %s should be text', text_fields{i_field}, name));
    end
    testCase.verifyEqual(info.name, name, sprintf('info.name of model %s', name));

    % numbers may come as double or integer from the file, so we compare the values
    testCase.verifyEqual(double(info.n_features), size(model.weights, 1), ...
        sprintf('info.n_features of model %s should be the number of rows of weights', name));
    testCase.verifyEqual(double(info.n_dims), size(model.weights, 2), ...
        sprintf('info.n_dims of model %s should be the number of columns of weights', name));
end
end

function test_models_use_their_network(testCase)
% Table of the shipped models (as in README.md and docs/models.md). The
% CLIP networks are open_clip networks with the openai weights, AligNet
% SigLIP2-B is run by dimpred/alignet.py with its own weights.
%   name                          network               pretrained                       n_features  n_dims
expected = {
    'alignet_siglip2b_66d_kernel', 'AligNet SigLIP2-B', 'alignet_siglip2_b.safetensors',  768,       66
    'alignet_siglip2b_66d_ridge', 'AligNet SigLIP2-B',  'alignet_siglip2_b.safetensors',  768,       66
    'rn50x64_49d_ridge',          'RN50x64',            'openai',                        1024,       49
    'rn50x64_66d_elastic',        'RN50x64',            'openai',                        1024,       66
    'rn50x64_66d_ridge',          'RN50x64',            'openai',                        1024,       66
    'vitb32_66d_elastic',         'ViT-B-32-quickgelu', 'openai',                         512,       66
    };
for i_model = 1:size(expected, 1)
    [name, network, pretrained, n_features, n_dims] = expected{i_model, :};
    model = dimpred_load_model(name);
    testCase.verifyEqual(model.info.network, network, ...
        sprintf('Model %s should use the network %s', name, network));
    testCase.verifyTrue(startsWith(model.info.pretrained, pretrained), ...
        sprintf('Model %s should use the weights %s, info.pretrained is %s', name, pretrained, model.info.pretrained));
    testCase.verifySize(model.weights, [n_features n_dims], ...
        sprintf('Model %s should map %i features to %i dimensions', name, n_features, n_dims));
end
end

function test_models_have_positive_scale_and_target_mean(testCase)
% feature_scale is a standard deviation (we divide by it) and target_mean
% the mean of a SPoSE dimension, which is never negative, so both have to
% be positive. Nothing in a model may be NaN or Inf.
names = shipped_models();
for i_model = 1:numel(names)
    name = names{i_model};
    model = dimpred_load_model(name);
    testCase.verifyTrue(all(model.feature_scale > 0), sprintf('feature_scale of model %s has values <= 0', name));
    testCase.verifyTrue(all(model.target_mean > 0), sprintf('target_mean of model %s has values <= 0', name));
    fields = {'weights', 'feature_mean', 'feature_scale', 'target_mean'}; % Inf would pass the > 0 above
    for i_field = 1:numel(fields)
        values = model.(fields{i_field});
        testCase.verifyTrue(all(isfinite(values(:))), sprintf('%s of model %s is not all finite', fields{i_field}, name));
    end
end
end

function test_loaded_numbers_are_the_numbers_in_the_file(testCase)
% We read each model file directly with load. Nothing may be transposed,
% rescaled or reordered on the way.
models_folder = fullfile(testCase.TestData.repo, 'dimpred', 'models');
names = shipped_models();
for i_model = 1:numel(names)
    name = names{i_model};
    raw = load(fullfile(models_folder, [name '.mat']));
    model = dimpred_load_model(name);
    fields = {'weights', 'feature_mean', 'feature_scale', 'target_mean', 'labels'};
    for i_field = 1:numel(fields)
        testCase.verifyEqual(model.(fields{i_field}), raw.(fields{i_field}), ...
            sprintf('Model %s: %s differs from the file', name, fields{i_field}));
    end
end
end

function test_labels_are_spose_labels(testCase)
% The labels have to be those of the embedding the model was trained on
% (training/data/labels_49d.txt or labels_66d.txt, one label per line)
names = shipped_models();
for i_model = 1:numel(names)
    name = names{i_model};
    model = dimpred_load_model(name);
    n_dims = size(model.weights, 2);
    label_file = fullfile(testCase.TestData.repo, 'training', 'data', sprintf('labels_%id.txt', n_dims));
    testCase.assertTrue(exist(label_file, 'file') == 2, sprintf('Label file %s not found', label_file));
    testCase.verifyEqual(model.labels, read_lines(label_file), ...
        sprintf('Labels of model %s differ from %s', name, label_file));
end
end

function test_file_is_the_shipped_model_file(testCase)
% The MATLAB functions find the models in dimpred/models of the repository
models_folder = fullfile(testCase.TestData.repo, 'dimpred', 'models');
names = shipped_models();
for i_model = 1:numel(names)
    name = names{i_model};
    model = dimpred_load_model(name);
    testCase.verifyTrue(is_absolute_path(model.file), sprintf('model.file should be an absolute path, got %s', model.file));
    testCase.assertTrue(exist(model.file, 'file') == 2, sprintf('model.file %s does not exist', model.file));
    [folder, base, ext] = fileparts(model.file);
    testCase.verifyEqual([base ext], [name '.mat'], sprintf('model.file of model %s should be %s.mat', name, name));
    testCase.verifyEqual(resolve_folder(folder), resolve_folder(models_folder), ...
        sprintf('Model %s should be loaded from %s', name, models_folder));
end
end

function test_models_are_found_from_any_current_folder(testCase)
% The models are found relative to the folder of the MATLAB functions,
% not relative to the current folder
change_folder(testCase, make_temp_folder(testCase));
testCase.verifyEqual(dimpred_list_models, shipped_models(), ...
    'dimpred_list_models should find the shipped models from any current folder');
model = dimpred_load_model('rn50x64_49d_ridge');
testCase.verifyEqual(model.info.name, 'rn50x64_49d_ridge', ...
    'dimpred_load_model should find the shipped models from any current folder');
end


%% Other ways to give the model

function test_load_from_file_path(testCase)
% A model file somewhere else is loaded by its path
original = dimpred_load_model('rn50x64_49d_ridge');
folder = make_temp_folder(testCase);
fname = fullfile(folder, 'my_model.mat');
copyfile(original.file, fname);

model = dimpred_load_model(fname);
fields = {'weights', 'feature_mean', 'feature_scale', 'target_mean', 'labels', 'info'};
for i_field = 1:numel(fields)
    testCase.verifyEqual(model.(fields{i_field}), original.(fields{i_field}), ...
        sprintf('%s differs when the model is loaded from a copy of its file', fields{i_field}));
end
[model_folder, base, ext] = fileparts(model.file);
testCase.verifyEqual([base ext], 'my_model.mat', 'model.file should be the file we loaded');
testCase.verifyEqual(resolve_folder(model_folder), resolve_folder(folder), 'model.file should be the file we loaded');
end

function test_load_from_relative_path(testCase)
% A path relative to the current folder is loaded from there (not from
% dimpred/models), and model.file is the absolute path of that file
original = dimpred_load_model('rn50x64_49d_ridge');
parent = make_temp_folder(testCase);
mkdir(fullfile(parent, 'copies'));
copyfile(original.file, fullfile(parent, 'copies', 'my_model.mat'));
change_folder(testCase, parent);

model = dimpred_load_model(fullfile('copies', 'my_model.mat'));
testCase.verifyEqual(model.weights, original.weights, 'The weights of the copied model file differ from the original');
testCase.verifyTrue(is_absolute_path(model.file), sprintf('model.file should be an absolute path, got %s', model.file));
[model_folder, base, ext] = fileparts(model.file);
testCase.verifyEqual([base ext], 'my_model.mat', 'model.file should be the file we loaded');
testCase.verifyEqual(resolve_folder(model_folder), resolve_folder(fullfile(parent, 'copies')), ...
    'model.file should be the file we loaded');
end

function test_struct_is_returned_unchanged(testCase)
% A model that is already loaded is passed through, not loaded again. We
% add a field to see that it is really the same struct.
model = dimpred_load_model('rn50x64_66d_ridge');
model.my_comment = 'kept';
testCase.verifyEqual(dimpred_load_model(model), model, ...
    'dimpred_load_model(model) should return a loaded model unchanged');
end


%% Errors

function test_unknown_model_gives_error(testCase)
testCase.verifyError(@() dimpred_load_model('vitb32_66d_elsatic'), 'dimpred:unknownModel', ...
    'A misspelled model name should give an error');
end

function test_missing_model_file_gives_error(testCase)
% dimpred_load_model cannot know whether a name or a file was meant, so a
% missing file gives the same error as an unknown name
testCase.verifyError(@() dimpred_load_model(fullfile(tempname, 'no_model.mat')), 'dimpred:unknownModel', ...
    'A model file that does not exist should give an error');
end

function test_unknown_model_message_lists_models(testCase)
% The user should see which models there are
try
    dimpred_load_model('no_such_model');
    testCase.verifyFail('dimpred_load_model(''no_such_model'') should give an error');
catch err
    testCase.verifyEqual(err.identifier, 'dimpred:unknownModel', err.message);
    names = shipped_models();
    for i_model = 1:numel(names)
        testCase.verifySubstring(err.message, names{i_model}, ...
            'The error message for an unknown model should list the available models');
    end
end
end

function test_inconsistent_model_file_gives_error(testCase)
% Each variant makes one variable one element too short, so that the sizes
% do not fit together any more
good = rmfield(dimpred_load_model('rn50x64_49d_ridge'), 'file');
folder = make_temp_folder(testCase);
variants = {'weights', 'feature_mean', 'feature_scale', 'target_mean', 'labels'};
for i_variant = 1:numel(variants)
    bad = good;
    switch variants{i_variant}
        case 'weights'
            bad.weights = bad.weights(1:end-1, :); % one feature less than feature_mean
        case 'feature_mean'
            bad.feature_mean = bad.feature_mean(1:end-1);
        case 'feature_scale'
            bad.feature_scale = bad.feature_scale(1:end-1);
        case 'target_mean'
            bad.target_mean = bad.target_mean(1:end-1); % one dimension less than weights
        case 'labels'
            bad.labels = bad.labels(1:end-1);
    end
    fname = fullfile(folder, ['short_' variants{i_variant} '.mat']);
    save(fname, '-struct', 'bad');
    testCase.verifyError(@() dimpred_load_model(fname), 'dimpred:inconsistentModel', ...
        sprintf('A model file with %s one element too short should give an error', variants{i_variant}));
end
end

function test_model_file_without_target_mean_gives_error(testCase)
% A model without target_mean cannot give correct predictions (they would be
% far too small and mostly 0), so we do not accept such a file
bad = rmfield(dimpred_load_model('rn50x64_49d_ridge'), {'file', 'target_mean'});
fname = fullfile(make_temp_folder(testCase), 'no_target_mean.mat');
save(fname, '-struct', 'bad');
testCase.verifyError(@() dimpred_load_model(fname), 'dimpred:inconsistentModel', ...
    'A model file without target_mean should give an error');
end


%% Helpers

function test_kernel_model_has_its_kernel_part_and_close_pair_settings(testCase)
model = dimpred_load_model('alignet_siglip2b_66d_kernel');
testCase.verifySize(model.kernel_features, [1854 768], 'kernel_features should be 1854 x 768');
testCase.verifySize(model.kernel_coefficients, [1854 66], 'kernel_coefficients should be 1854 x 66');
testCase.verifyClass(model.kernel_features, 'double', 'kernel_features should be double after loading');
testCase.verifyEqual(model.kernel_tau, 0.5, 'kernel_tau should be 0.5');
testCase.verifyEqual(model.close_pairs_weight, 8, 'close_pairs_weight should be 8');
testCase.verifyGreaterThan(model.close_pairs_threshold, 0, 'close_pairs_threshold should be > 0');
testCase.verifyLessThan(model.close_pairs_threshold, 1, 'close_pairs_threshold should be < 1');
norms = sqrt(sum(model.kernel_features .^ 2, 2));
testCase.verifyLessThan(max(abs(norms - 1)), 1e-6, 'kernel_features should have length 1');
end

function test_linear_models_have_empty_kernel_fields(testCase)
names = setdiff(shipped_models(), {'alignet_siglip2b_66d_kernel'});
fields = {'kernel_features', 'kernel_coefficients', 'kernel_tau', 'close_pairs_weight', 'close_pairs_threshold'};
for i_model = 1:numel(names)
    model = dimpred_load_model(names{i_model});
    for i_field = 1:numel(fields)
        testCase.verifyTrue(isfield(model, fields{i_field}) && isempty(model.(fields{i_field})), ...
            sprintf('Model %s: field %s should exist and be empty', names{i_model}, fields{i_field}));
    end
end
end

function test_incomplete_kernel_variables_give_error(testCase)
data = load(dimpred_load_model('alignet_siglip2b_66d_kernel').file);
data = rmfield(data, 'kernel_tau');
fname = [tempname '.mat'];
cleanup = onCleanup(@() delete(fname));
save(fname, '-struct', 'data');
testCase.verifyError(@() dimpred_load_model(fname), 'dimpred:inconsistentModel', ...
    'A model file with kernel_features but without kernel_tau should give an error');
end

function names = shipped_models()
% Names of the shipped models, sorted alphabetically. The same list is in
% test_dimpred_predict.m and test_dimpred_fixtures.m, so that each test
% file can be read on its own. test_list_models_gives_shipped_models fails
% when a model is added and the lists need updating.
names = {'alignet_siglip2b_66d_kernel'; 'alignet_siglip2b_66d_ridge'; 'rn50x64_49d_ridge'; ...
    'rn50x64_66d_elastic'; 'rn50x64_66d_ridge'; 'vitb32_66d_elastic'};
end

function lines = read_lines(fname)
% Non-empty lines of a text file as a cell column
lines = splitlines(fileread(fname)); % also works with Windows line endings
lines = lines(~cellfun(@isempty, lines));
end

function folder = make_temp_folder(testCase)
% Temporary folder that is deleted after the test
folder = tempname;
mkdir(folder);
testCase.addTeardown(@rmdir, folder, 's');
end

function change_folder(testCase, folder)
% cd to folder for this test only. Teardowns run in reverse order, so we
% are back in the old folder before the temporary folder is deleted.
old_folder = cd(folder);
testCase.addTeardown(@cd, old_folder);
end

function folder = resolve_folder(folder)
% Full path of a folder without '..' and symbolic links, so that two
% spellings of the same folder compare equal (on macOS, /var is a link to
% /private/var)
old_folder = cd(folder);
folder = pwd;
cd(old_folder);
end

function out = is_absolute_path(fname)
out = startsWith(fname, '/') || ~isempty(regexp(fname, '^[A-Za-z]:[\\/]', 'once'));
end
