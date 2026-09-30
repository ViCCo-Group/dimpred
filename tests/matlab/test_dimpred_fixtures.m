% function tests = test_dimpred_fixtures
%
% Checks the test fixtures themselves (tests/fixtures/reference_data.mat and
% tests/fixtures/images/), before the other tests rely on them.
%
% The other tests are only as good as the numbers they compare against. If
% a variable is missing or has the wrong size, these tests say so directly,
% instead of letting other tests fail in a confusing way. We also check a
% few facts the other tests depend on: the first 48 images are the 48nonref
% set, the published model reaches r = 0.810 with human similarity, and the
% three CC0 images are different enough from each other that the order
% tests of the extraction can tell them apart.
%
% make_fixtures.py computes expected_* and human_r_48nonref from the
% shipped model files. For the rebuilt models (all except
% rn50x64_49d_ridge) there are no published predictions, so a model file
% that is wrong but consistent with itself (e.g. with a wrong target_mean)
% would pass the tests of dimpred_predict. The checks of the zeros and of
% the correlation with human similarity below catch such model files
% without needing the model files as reference.
%
% The fixtures are made by tests/fixtures/make_fixtures.py.
%
% Run with
%   runtests('test_dimpred_fixtures')
% or all MATLAB tests with run_dimpred_tests.
%
% Martin Hebart, 2026/09/30
%
% See also RUN_DIMPRED_TESTS

% History:
% 2026/09/30: after review: text variables are cells of text, published
%   predictions equal expected_rn50x64_49d_ridge, CC0 images differ, not
%   too many zeros, r with human similarity of at least 0.80 for every
%   model, fixture numbers for every model file
% 2026/09/30: written before the fixtures (test-driven development)

function tests = test_dimpred_fixtures
tests = functiontests(localfunctions);
end


%% Setup

function setupOnce(testCase)

here = fileparts(mfilename('fullpath')); % .../dimpred/tests/matlab
repo = fileparts(fileparts(here));
testCase.TestData.repo = repo;
testCase.TestData.fixture_folder = fullfile(repo, 'tests', 'fixtures');

% Load the reference numbers
fixture_file = fullfile(testCase.TestData.fixture_folder, 'reference_data.mat');
testCase.assertTrue(exist(fixture_file, 'file') == 2, ...
    sprintf('Fixture file %s not found. It is made by tests/fixtures/make_fixtures.py.', fixture_file));
testCase.TestData.ref = load(fixture_file);

end


%% Content of reference_data.mat

function test_has_all_variables_with_sizes(testCase)
ref = testCase.TestData.ref;
%   variable                           size
expected = {
    'features_rn50x64',                [168 1024]
    'features_vitb32',                 [168 512]
    'files',                           [168 1]
    'image_set',                       [168 1]
    'published_rn50x64_49d_ridge',     [168 49]
    'expected_rn50x64_49d_ridge',      [168 49]
    'expected_rn50x64_66d_elastic',    [168 66]
    'expected_rn50x64_66d_ridge',      [168 66]
    'expected_vitb32_66d_elastic',     [168 66]
    'human_similarity_48nonref',       [48 48]
    'human_r_48nonref',                [1 1]
    'cc0_files',                       [3 1]
    'cc0_features_rn50x64',            [3 1024]
    'cc0_features_vitb32',             [3 512]
    'cc0_published_rn50x64_49d_ridge', [3 49]
    };
for i_var = 1:size(expected, 1)
    [name, sz] = expected{i_var, :};
    testCase.verifyTrue(isfield(ref, name), sprintf('Variable %s is missing in reference_data.mat', name));
    if isfield(ref, name)
        testCase.verifySize(ref.(name), sz, sprintf('Variable %s has the wrong size', name));
    end
end
end

function test_numbers_are_finite(testCase)
ref = testCase.TestData.ref;
names = {'features_rn50x64', 'features_vitb32', 'published_rn50x64_49d_ridge', ...
    'expected_rn50x64_49d_ridge', 'expected_rn50x64_66d_elastic', 'expected_rn50x64_66d_ridge', ...
    'expected_vitb32_66d_elastic', 'cc0_features_rn50x64', 'cc0_features_vitb32', 'cc0_published_rn50x64_49d_ridge'};
for i_var = 1:numel(names)
    values = ref.(names{i_var});
    testCase.verifyTrue(isnumeric(values) && all(isfinite(values(:))), ...
        sprintf('Variable %s should contain only finite numbers', names{i_var}));
end
end

function test_text_variables_are_cells_of_text(testCase)
ref = testCase.TestData.ref;
names = {'files', 'image_set', 'cc0_files'};
for i_var = 1:numel(names)
    testCase.verifyTrue(iscellstr(ref.(names{i_var})), ...
        sprintf('Variable %s should be a cell array of text', names{i_var}));
end
end

function test_first_48_images_are_48nonref(testCase)
% The human similarity matrix belongs to the 48nonref images, in this order
ref = testCase.TestData.ref;
expected = [repmat({'48nonref'}, 48, 1); repmat({'peterson-animals'}, 120, 1)];
testCase.verifyEqual(ref.image_set, expected, 'Rows 1-48 should be 48nonref, rows 49-168 the Peterson animals');
end

function test_file_names_are_unique(testCase)
% Two rows of the same image would point to a mistake in collecting the features
ref = testCase.TestData.ref;
testCase.verifyEqual(numel(unique(strcat(ref.image_set, '/', ref.files))), 168, ...
    'Each of the 168 rows should be a different image');
end

function test_human_similarity_is_symmetric(testCase)
H = testCase.TestData.ref.human_similarity_48nonref;
off_diagonal = ~eye(size(H));
testCase.verifyTrue(all(isfinite(H(off_diagonal))), 'Human similarity should be finite outside the diagonal');
Ht = H';
testCase.verifyEqual(H(off_diagonal), Ht(off_diagonal), 'AbsTol', 1e-12, 'Human similarity should be symmetric');
end

%% Checks of the model files

function test_every_model_file_has_fixture_numbers(testCase)
% Each model file in dimpred/models needs expected predictions and a
% correlation with human similarity here. When a model is added, the
% fixtures and the model lists in the test files have to be updated.
ref = testCase.TestData.ref;
model_files = dir(fullfile(testCase.TestData.repo, 'dimpred', 'models', '*.mat'));
testCase.assertNotEmpty(model_files, 'No model files found in dimpred/models');
for i_model = 1:numel(model_files)
    [~, name] = fileparts(model_files(i_model).name);
    testCase.verifyTrue(isfield(ref, ['expected_' name]), ...
        sprintf('reference_data.mat has no expected_%s for the model file %s', name, model_files(i_model).name));
    testCase.verifyTrue(isfield(ref.human_r_48nonref, name), ...
        sprintf('human_r_48nonref has no value for the model file %s', model_files(i_model).name));
end
end

function test_published_predictions_equal_expected_predictions(testCase)
% rn50x64_49d_ridge holds Philipp's published weights, so its expected
% predictions have to be his published predictions. If this fails, the
% model file or the fixture is wrong, not the dimpred code.
ref = testCase.TestData.ref;
testCase.verifyEqual(ref.expected_rn50x64_49d_ridge, ref.published_rn50x64_49d_ridge, 'AbsTol', 1e-10, ...
    'expected_rn50x64_49d_ridge should equal published_rn50x64_49d_ridge');
end

function test_predictions_are_not_mostly_zero(testCase)
% SPoSE dimensions are sparse: 26% of Philipp's published predictions are
% 0, and 15-26% of the expected predictions of the shipped models. A model
% without target_mean gives about 70% zeros, so we require less than 40%.
ref = testCase.TestData.ref;
names = {'published_rn50x64_49d_ridge', 'expected_rn50x64_49d_ridge', 'expected_rn50x64_66d_elastic', ...
    'expected_rn50x64_66d_ridge', 'expected_vitb32_66d_elastic'};
for i_var = 1:numel(names)
    values = ref.(names{i_var});
    zero_fraction = mean(values(:) == 0);
    testCase.verifyLessThan(zero_fraction, 0.4, ...
        sprintf('%.0f%% of %s are 0. Was target_mean added?', 100 * zero_fraction, names{i_var}));
end
end

function test_human_r_of_all_models(testCase)
% Each shipped model should predict human similarity at least about as
% well as the model of the DimPred paper (r = 0.810; the rebuilt models
% reach 0.82 to 0.83). The human data are independent of the model files,
% so this catches model files that are wrong but consistent with
% themselves: without target_mean, r drops to 0.76-0.78, without the
% scaling of the features to 0.70-0.77.
human_r = testCase.TestData.ref.human_r_48nonref;
names = {'rn50x64_49d_ridge'; 'rn50x64_66d_elastic'; 'rn50x64_66d_ridge'; 'vitb32_66d_elastic'};
for i_model = 1:numel(names)
    name = names{i_model};
    testCase.verifyTrue(isfield(human_r, name), sprintf('human_r_48nonref has no value for model %s', name));
    if isfield(human_r, name)
        r = human_r.(name);
        testCase.verifyTrue(isscalar(r) && r >= 0.80 && r < 1, ...
            sprintf('Model %s correlates r = %.3f with human similarity (48nonref), expected at least 0.80', name, r));
    end
end
end

function test_human_r_of_published_model_is_0810(testCase)
testCase.verifyEqual(testCase.TestData.ref.human_r_48nonref.rn50x64_49d_ridge, 0.810, 'AbsTol', 0.002, ...
    'The model of the DimPred paper should correlate r = 0.810 with human similarity of the 48nonref images');
end


%% CC0 images

function test_cc0_images_exist(testCase)
cc0_files = testCase.TestData.ref.cc0_files;
for i_file = 1:numel(cc0_files)
    fname = fullfile(testCase.TestData.fixture_folder, 'images', cc0_files{i_file});
    testCase.verifyTrue(exist(fname, 'file') == 2, sprintf('CC0 image %s not found', fname));
end
end

function test_cc0_files_are_unique(testCase)
cc0_files = testCase.TestData.ref.cc0_files;
testCase.verifyEqual(numel(unique(cc0_files)), numel(cc0_files), 'Each CC0 image should be listed once');
end

function test_cc0_files_are_not_sorted(testCase)
% Most extraction tests use the images in the order of cc0_files. With an
% unsorted list, they also fail for an implementation that sorts the
% files. The order tests themselves do not depend on this.
cc0_files = testCase.TestData.ref.cc0_files;
testCase.verifyFalse(isequal(cc0_files, sort(cc0_files)), ...
    'cc0_files should not be in sorted order (see make_fixtures.py)');
end

function test_cc0_features_differ_between_images(testCase)
% The order tests of the extraction can only tell the rows apart if the
% images have clearly different features. The tests require r > 0.99999
% with the right row, different images here correlate about 0.4 to 0.5.
ref = testCase.TestData.ref;
names = {'cc0_features_rn50x64', 'cc0_features_vitb32'};
for i_var = 1:numel(names)
    r = corrcoef(ref.(names{i_var})'); % between images
    r_between = r(~eye(size(r)));
    testCase.verifyLessThan(max(r_between), 0.9, ...
        sprintf('Two CC0 images in %s have almost the same features (r = %.4f)', names{i_var}, max(r_between)));
end
end
