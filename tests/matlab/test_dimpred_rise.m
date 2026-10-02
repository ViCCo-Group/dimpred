% function tests = test_dimpred_rise
%
% Tests for dimpred_rise that really compute heatmaps. dimpred_rise runs
% the command line tool of the Python version (python -m dimpred ... --rise)
% and loads the maps that it saves. Here we use vitb32_66d_elastic, which
% is small, two of the CC0 test images and only 20 masks, so the maps are
% noisy, but their sizes, the order of the dimensions and the predictions
% of the images without masks can be checked:
%   - the fields of the result and their sizes (images first, as in Python)
%   - the relevance map is the average of the dimension maps, weighted by
%     the predicted dimension values (computed here in MATLAB, which also
%     shows that the order of the dimensions of the arrays is right)
%   - the predictions of the images without masks are those of
%     dimpred_predict(dimpred_extract_features(...))
%   - cfg.png gives 4 PNG files per image
%
% All tests here need Python with torch and open_clip. Set the environment
% variable DIMPRED_PYTHON to that Python, e.g. in MATLAB
%   setenv('DIMPRED_PYTHON', '/path/to/python')
% Without it, these tests are skipped (with the reason). They take about a
% minute. run_dimpred_tests('fast') leaves this file out. The tests that
% need no network (help text, wrong input, the command that is sent to
% Python) are in test_dimpred_rise_errors.m and always run.
%
% Run with
%   runtests('test_dimpred_rise')
% or all MATLAB tests with run_dimpred_tests.
%
% Hebartlab, 2026/10/02
%
% See also DIMPRED_RISE, TEST_DIMPRED_RISE_ERRORS, RUN_DIMPRED_TESTS

% History:
% 2026/10/02: written before dimpred_rise.m (test-driven)

function tests = test_dimpred_rise
tests = functiontests(localfunctions);
end


%% Setup

function setupOnce(testCase)

% Add the dimpred MATLAB functions to the path (restored in teardownOnce)
here = fileparts(mfilename('fullpath')); % .../dimpred/tests/matlab
repo = fileparts(fileparts(here));
testCase.TestData.old_path = addpath(fullfile(repo, 'matlab'));

% Two of the CC0 test images, in the order 2, 1 of cc0_files (not sorted)
fixture_folder = fullfile(repo, 'tests', 'fixtures');
ref = load(fullfile(fixture_folder, 'reference_data.mat'), 'cc0_files');
testCase.TestData.images = fullfile(fixture_folder, 'images', ref.cc0_files([2 1]));
testCase.TestData.images = testCase.TestData.images(:);
testCase.TestData.model = 'vitb32_66d_elastic';

% Check whether we have a Python that can run the networks
python = getenv('DIMPRED_PYTHON');
python_ok = false;
python_problem = 'the environment variable DIMPRED_PYTHON is not set';
if ~isempty(python)
    [status, output] = system(sprintf('"%s" -c "import torch, open_clip"', python));
    python_ok = status == 0;
    if ~python_ok
        python_problem = sprintf('%s cannot import torch and open_clip (%s)', python, strtrim(output));
    end
end
testCase.TestData.python = python;
testCase.TestData.python_ok = python_ok;
testCase.TestData.python_problem = python_problem;

% The result of dimpred_rise, computed once by the first test that needs it
% (containers.Map is a handle object, so what one test stores, the others see)
testCase.TestData.results = containers.Map;

end

function teardownOnce(testCase)
path(testCase.TestData.old_path);
results = testCase.TestData.results;
if isKey(results, 'png_folder')
    rmdir(results('png_folder'), 's');
end
end


%% Tests

function test_fields_and_sizes(testCase)
r = rise_once(testCase);
testCase.verifySize(r.relevance, [2 224 224], 'relevance should be n_images x 224 x 224');
testCase.verifySize(r.dimension_maps, [2 66 224 224], 'dimension_maps should be n_images x n_dims x 224 x 224');
testCase.verifySize(r.embedding, [2 66], 'embedding should be n_images x n_dims');
testCase.verifySize(r.view, [2 224 224 3], 'view should be n_images x 224 x 224 x 3');
testCase.verifyClass(r.view, 'uint8', 'view should be uint8 (RGB)');
testCase.verifyClass(r.relevance, 'single', 'relevance should be single');
testCase.verifyClass(r.embedding, 'double', 'embedding should be double, as from dimpred_predict');
testCase.verifyTrue(all(isfinite(r.relevance(:))) && all(isfinite(r.dimension_maps(:))), 'The maps should be finite');
end

function test_labels_files_model_and_settings(testCase)
r = rise_once(testCase);
model = dimpred_load_model(testCase.TestData.model);
testCase.verifyEqual(r.labels, model.labels, 'labels should be those of the model');
testCase.verifyEqual(r.files, testCase.TestData.images, 'files should be the given files, in the given order');
testCase.verifyEqual(r.model, testCase.TestData.model, 'model should be the name of the model');
testCase.verifyEqual(r.settings.n_masks, 20, 'settings.n_masks should be cfg.n_masks');
testCase.verifyEqual(r.settings.normalization, 'pixel', 'The default normalization is pixel');
testCase.verifyEqual(r.settings.input_size, [224 224], 'ViT-B/32 has an input of 224 x 224 pixels');
end

function test_relevance_is_the_weighted_average_of_the_dimension_maps(testCase)
% computed here in MATLAB: if the dimensions of the arrays were in another
% order than documented, this would fail
r = rise_once(testCase);
for i_image = 1:2
    maps = double(reshape(r.dimension_maps(i_image, :, :, :), 66, []));
    weights = r.embedding(i_image, :);
    expected = reshape(weights * maps / sum(weights), 224, 224);
    actual = double(squeeze(r.relevance(i_image, :, :)));
    testCase.verifyLessThan(max(abs(actual(:) - expected(:))), 1e-4 * max(abs(expected(:))), ...
        sprintf('The relevance map of image %i is not the weighted average of its dimension maps', i_image));
end
end

function test_embedding_is_the_prediction_of_the_images(testCase)
r = rise_once(testCase);
cfg.python = testCase.TestData.python;
features = dimpred_extract_features(testCase.TestData.images, testCase.TestData.model, cfg);
expected = dimpred_predict(features, testCase.TestData.model);
testCase.verifyEqual(r.embedding, expected, 'AbsTol', 2e-3, ...
    'The predictions of the images without masks should be those of dimpred_predict(dimpred_extract_features(...))');
end

function test_png_files(testCase)
r = rise_once(testCase);
results = testCase.TestData.results;
found = dir(fullfile(results('png_folder'), '*.png'));
testCase.verifyEqual(numel(found), 8, sprintf('cfg.png should give 4 PNG files per image, found %i', numel(found)));
[~, stem] = fileparts(r.files{1});
testCase.verifyEqual(exist(fullfile(results('png_folder'), [stem '_relevance.png']), 'file'), 2, ...
    'The PNG file of the relevance map is missing');
end


%% Helpers

function r = rise_once(testCase)
% Runs dimpred_rise once for all tests (skips the test without Python)
testCase.assumeTrue(testCase.TestData.python_ok, ...
    sprintf('Skipped because %s. Set DIMPRED_PYTHON to a Python with torch and open_clip to run it.', ...
    testCase.TestData.python_problem));
results = testCase.TestData.results;
if ~isKey(results, 'rise')
    cfg.python = testCase.TestData.python;
    cfg.n_masks = 20;
    cfg.png = tempname;
    results('png_folder') = cfg.png;
    results('rise') = dimpred_rise(testCase.TestData.images, testCase.TestData.model, cfg);
end
r = results('rise');
end
