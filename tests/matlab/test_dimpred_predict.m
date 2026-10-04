% function tests = test_dimpred_predict
%
% Tests for dimpred_predict, the step from image features to predicted
% SPoSE dimensions:
%
%   embedding = max(((features - feature_mean) ./ feature_scale) * weights + target_mean, 0)
%
% The main tests compare the predictions with numbers in
% tests/fixtures/reference_data.mat. Philipp Kaniuth's published
% predictions of the DimPred paper (model rn50x64_49d_ridge) are fully
% independent of dimpred. The expected predictions of the other models
% (expected_*) were computed by tests/fixtures/make_fixtures.py with its
% own code, but from the shipped model files, so they check the MATLAB
% code and not the model files (test_dimpred_fixtures and
% test_dimpred_validation_human check the model files against human
% data). The other tests check single parts of the formula, so that a
% failure points to the part that is wrong. We wrote them with the bugs of
% earlier versions in mind:
%   - forgetting to add target_mean: the predictions are about half as
%     large and ~60% of them become 0
%   - z-scoring with the wrong numbers, e.g. subtracting the mean twice,
%     dividing by the variance, or z-scoring new images with their own
%     mean and std instead of those of the training images
%
% The Python tests in tests/ check the same behavior with the same numbers
% and tolerances.
%
% Run with
%   runtests('test_dimpred_predict')
% or all MATLAB tests with run_dimpred_tests.
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_PREDICT, DIMPRED_LOAD_MODEL, RUN_DIMPRED_TESTS

% History:
% 2026/10/04: new default model alignet_siglip2b_66d_kernel; tests of the kernel part
% 2026/10/02: new default model alignet_siglip2b_66d_ridge (768 AligNet features)
% 2026/09/30: NaN and Inf give an error, as in Python
% 2026/09/30: after review: column vector gives an error, failure messages
%   for all checks, header says where expected_* come from
% 2026/09/30: written before the code (test-driven development)

function tests = test_dimpred_predict
tests = functiontests(localfunctions);
end


%% Setup

function setupOnce(testCase)

% Add the dimpred MATLAB functions to the path (restored in teardownOnce)
here = fileparts(mfilename('fullpath')); % .../dimpred/tests/matlab
repo = fileparts(fileparts(here));
testCase.TestData.old_path = addpath(fullfile(repo, 'matlab'));

% Load the reference numbers (made by tests/fixtures/make_fixtures.py)
fixture_file = fullfile(repo, 'tests', 'fixtures', 'reference_data.mat');
testCase.assertTrue(exist(fixture_file, 'file') == 2, ...
    sprintf('Fixture file %s not found. It is made by tests/fixtures/make_fixtures.py.', fixture_file));
testCase.TestData.ref = load(fixture_file);

end

function teardownOnce(testCase)
path(testCase.TestData.old_path);
end


%% Comparison with reference numbers

function test_reproduces_published_predictions(testCase)
% Philipp's RN50x64 features of 168 images (48nonref and Peterson animals)
% with rn50x64_49d_ridge have to give his published predictions (OSF)
ref = testCase.TestData.ref;
embedding = dimpred_predict(ref.features_rn50x64, 'rn50x64_49d_ridge');
testCase.verifyEqual(embedding, ref.published_rn50x64_49d_ridge, 'AbsTol', 1e-10, ...
    ['rn50x64_49d_ridge does not reproduce Philipp''s published predictions. ' ...
    'If the predictions are too small and many are 0, target_mean was probably not added.']);
end

function test_cc0_reference_features_give_published_predictions(testCase)
% Philipp's published predictions of the CC0 images come from his own
% feature extraction, our reference features from a separate extraction of
% the same images, so small differences are expected
ref = testCase.TestData.ref;
embedding = dimpred_predict(ref.cc0_features_rn50x64, 'rn50x64_49d_ridge');
testCase.verifyEqual(embedding, ref.cc0_published_rn50x64_49d_ridge, 'AbsTol', 2e-3, ...
    'Predictions of the CC0 reference features differ from Philipp''s published predictions');
end

function test_shipped_models_give_expected_predictions(testCase)
ref = testCase.TestData.ref;
names = shipped_models();
for i_model = 1:numel(names)
    name = names{i_model};
    embedding = dimpred_predict(features_for_model(ref, name), name);
    testCase.verifyEqual(embedding, ref.(['expected_' name]), 'AbsTol', 1e-10, ...
        sprintf('Model %s does not give the expected predictions (expected_%s)', name, name));
end
end

function test_default_model_is_used_without_model(testCase)
ref = testCase.TestData.ref;
expected = ref.expected_alignet_siglip2b_66d_kernel;
testCase.verifyEqual(dimpred_predict(ref.features_alignet), expected, 'AbsTol', 1e-10, ...
    'dimpred_predict(features) should use the default model alignet_siglip2b_66d_kernel');
testCase.verifyEqual(dimpred_predict(ref.features_alignet, []), expected, 'AbsTol', 1e-10, ...
    'dimpred_predict(features, []) should use the default model alignet_siglip2b_66d_kernel');
end


%% Parts of the formula

function test_feature_mean_gives_target_mean(testCase)
% An image with average features (z = 0 for all features) has to get the
% average value of each dimension. This fails if target_mean is not added,
% the bug of earlier unofficial wrappers. target_mean is > 0, so clipping
% at 0 plays no role here.
names = setdiff(shipped_models(), {'alignet_siglip2b_66d_kernel'}); % the kernel part adds to the linear part
for i_model = 1:numel(names)
    model = dimpred_load_model(names{i_model});
    testCase.verifyEqual(dimpred_predict(model.feature_mean, model), model.target_mean, 'AbsTol', 1e-12, ...
        sprintf('Model %s: features equal to feature_mean should give target_mean', names{i_model}));
end
end

function test_scale_step_adds_weight_row(testCase)
% Moving feature j away from its mean by c times feature_scale(j) (i.e. by
% c standard deviations) has to change the dimensions by c times row j of
% the weights. This fails if features are not z-scored with feature_mean
% and feature_scale of the model, e.g. if we divide by the variance,
% subtract the mean twice, or use the mean and std of the new images.
model = dimpred_load_model('vitb32_66d_elastic');
n_features = size(model.weights, 1);
for j = [1, round(n_features/2), n_features]
    for c = [1, -2]
        features = model.feature_mean;
        features(j) = features(j) + c * model.feature_scale(j);
        expected = max(model.target_mean + c * model.weights(j, :), 0);
        testCase.verifyEqual(dimpred_predict(features, model), expected, 'AbsTol', 1e-12, ...
            sprintf('Feature %i moved by %g standard deviations', j, c));
    end
end
end

function test_matches_formula_on_new_features(testCase)
% New features around the training distribution (z-scores from a standard
% normal distribution), compared with the formula written out here. Unlike
% the fixture features, these change all 1024 features independently.
model = dimpred_load_model('rn50x64_66d_ridge');
stream = RandStream('mt19937ar', 'Seed', 1);
z = randn(stream, 20, size(model.weights, 1));
features = model.feature_mean + z .* model.feature_scale;
expected = max(((features - model.feature_mean) ./ model.feature_scale) * model.weights + model.target_mean, 0);
testCase.verifyEqual(dimpred_predict(features, model), expected, 'AbsTol', 1e-10, ...
    'Predictions of random features differ from the formula');
end

function test_toy_model_by_hand(testCase)
% A model with 3 features and 2 dimensions that we can compute by hand:
%   feature_mean  = [1 2 3], feature_scale = [2 1 0.5]
%   weights       = [1 0; 0 2; -1 1], target_mean = [0.5 0.25]
model = toy_model();
features = [
    3  2  3.5   % z = [1 0 1], z*weights = [0 1], + target_mean = [0.5 1.25]
    1  2  3     % z = 0, gives target_mean
   -1  2  3     % z = [-1 0 0], z*weights = [-1 0], + target_mean = [-0.5 0.25], clipped
    1  3  3     % z = [0 1 0], z*weights = [0 2], + target_mean = [0.5 2.25]
    ];
expected = [
    0.5  1.25
    0.5  0.25
    0    0.25
    0.5  2.25
    ];
testCase.verifyEqual(dimpred_predict(features, model), expected, 'AbsTol', 1e-12, ...
    'Predictions of the toy model differ from the values computed by hand (see the comments above)');
end

function test_negative_values_are_set_to_zero(testCase)
% SPoSE dimensions are never negative, so negative predictions become 0.
% We use large z-scores to get many negative values before clipping.
model = dimpred_load_model('vitb32_66d_elastic');
stream = RandStream('mt19937ar', 'Seed', 2);
z = 3 * randn(stream, 50, size(model.weights, 1));
features = model.feature_mean + z .* model.feature_scale;
before_clipping = z * model.weights + model.target_mean;
negative = before_clipping < 0;
testCase.assertTrue(any(negative(:)) && any(~negative(:)), ...
    'The test data should give both negative and positive values before clipping');

embedding = dimpred_predict(features, model);
testCase.verifyTrue(all(embedding(:) >= 0), 'Predictions should never be negative');
testCase.verifyEqual(embedding(negative), zeros(nnz(negative), 1), 'AbsTol', 1e-10, ...
    'Negative values should be set to 0');
testCase.verifyEqual(embedding(~negative), before_clipping(~negative), 'AbsTol', 1e-10, ...
    'Positive values should stay as they are');
end


%% Input and output format

function test_single_row_gives_one_row(testCase)
ref = testCase.TestData.ref;
row = 7;
embedding = dimpred_predict(ref.features_alignet(row, :));
testCase.verifySize(embedding, [1 66], 'One image (1 x n_features) should give 1 x n_dims');
testCase.verifyEqual(embedding, ref.expected_alignet_siglip2b_66d_kernel(row, :), 'AbsTol', 1e-10, ...
    sprintf('Row %i predicted alone differs from expected_alignet_siglip2b_66d_kernel', row));
end

function test_column_vector_gives_error(testCase)
% One image is one row (1 x n_features), like one row of a feature
% matrix. A column (n_features x 1) means n_features images with one
% feature each, so we give an error instead of guessing that it was meant
% as one image. (Python accepts a 1D vector, which has no orientation.)
ref = testCase.TestData.ref;
testCase.verifyError(@() dimpred_predict(ref.features_alignet(1, :)'), 'dimpred:wrongFeatureCount', ...
    'A column vector (n_features x 1) should give an error, one image has to be a row');
end

function test_rows_are_predicted_independently(testCase)
% The prediction for an image must not depend on the other images that are
% passed with it. This fails e.g. if new features are z-scored with their
% own mean and std.
ref = testCase.TestData.ref;
name = 'rn50x64_66d_ridge';
features = ref.features_rn50x64(1:20, :);
all_rows = dimpred_predict(features, name);
for i_row = [1 10 20]
    testCase.verifyEqual(dimpred_predict(features(i_row, :), name), all_rows(i_row, :), 'AbsTol', 1e-12, ...
        sprintf('Row %i predicted alone differs from row %i predicted with the others', i_row, i_row));
end
testCase.verifyEqual(dimpred_predict(features(1:5, :), name), all_rows(1:5, :), 'AbsTol', 1e-12, ...
    'The first 5 rows predicted alone differ from the same rows predicted with 20 rows');
order = 20:-1:1;
testCase.verifyEqual(dimpred_predict(features(order, :), name), all_rows(order, :), 'AbsTol', 1e-12, ...
    'Reversing the order of the images should only reverse the order of the predictions');
end

function test_single_precision_input_is_computed_in_double(testCase)
% Features from the extraction are single (float32). They have to be
% converted to double before the computation, and the result is double.
ref = testCase.TestData.ref;
features = single(ref.features_alignet(1:5, :));
embedding = dimpred_predict(features);
testCase.verifyClass(embedding, 'double', 'Predictions should be double, also for single features');
testCase.verifyEqual(embedding, dimpred_predict(double(features)), 'AbsTol', 1e-12, ...
    'Single features should give the same result as the same features converted to double');
end

function test_model_given_as_name_file_or_struct(testCase)
ref = testCase.TestData.ref;
name = 'rn50x64_49d_ridge';
model = dimpred_load_model(name);
features = ref.features_rn50x64(1:10, :);
by_name = dimpred_predict(features, name);
testCase.verifyEqual(dimpred_predict(features, model.file), by_name, 'AbsTol', 1e-12, ...
    'The model given as file should give the same result as given by name');
testCase.verifyEqual(dimpred_predict(features, model), by_name, 'AbsTol', 1e-12, ...
    'The model given as struct should give the same result as given by name');
end


%% Errors

function test_wrong_feature_count_gives_error(testCase)
ref = testCase.TestData.ref;
testCase.verifyError(@() dimpred_predict(ref.features_rn50x64(1:2, :)), 'dimpred:wrongFeatureCount', ...
    'RN50x64 features (1024) with the default model (768 features) should give an error');
testCase.verifyError(@() dimpred_predict(ref.features_vitb32(1:2, :)), 'dimpred:wrongFeatureCount', ...
    'ViT-B-32-quickgelu features (512) with the default model (768 features) should give an error');
testCase.verifyError(@() dimpred_predict(ref.features_vitb32(1:2, :), 'rn50x64_49d_ridge'), 'dimpred:wrongFeatureCount', ...
    'ViT-B-32-quickgelu features (512) with an RN50x64 model (1024 features) should give an error');
testCase.verifyError(@() dimpred_predict(ref.features_alignet(1:2, :)'), 'dimpred:wrongFeatureCount', ...
    'Transposed features (features x images) should give an error');
end

function test_nan_or_inf_gives_error(testCase)
% NaN would give NaN predictions, and -Inf would give predictions of 0 that
% look like real values, so both have to give an error
ref = testCase.TestData.ref;
for value = [NaN Inf -Inf]
    features = ref.features_alignet(1:4, :);
    features(3, 11) = value;
    testCase.verifyError(@() dimpred_predict(features), 'dimpred:notFinite', ...
        sprintf('Features with %g should give an error', value));
end
end

function test_wrong_feature_count_message_names_model(testCase)
% The message should tell the user which model and network the features
% have to come from and how many features it expects
ref = testCase.TestData.ref;
try
    dimpred_predict(ref.features_rn50x64(1:2, :), 'vitb32_66d_elastic');
    testCase.verifyFail('Features with the wrong number of columns should give an error');
catch err
    testCase.verifyEqual(err.identifier, 'dimpred:wrongFeatureCount', err.message);
    testCase.verifySubstring(err.message, 'vitb32_66d_elastic', 'The error message should name the model');
    testCase.verifySubstring(err.message, 'ViT-B-32-quickgelu', 'The error message should name the network');
    testCase.verifySubstring(err.message, '512', 'The error message should give the expected number of features');
end
end


function test_kernel_prediction_matches_formula(testCase)
% The kernel model: the linear part plus exp((cos - 1) / tau) * kernel_coefficients, written out here
ref = testCase.TestData.ref;
model = dimpred_load_model('alignet_siglip2b_66d_kernel');
features = ref.features_alignet;
unit = features ./ sqrt(sum(features .^ 2, 2));
linear = ((features - model.feature_mean) ./ model.feature_scale) * model.weights + model.target_mean;
expected = max(linear + exp((unit * model.kernel_features' - 1) / model.kernel_tau) * model.kernel_coefficients, 0);
testCase.verifyEqual(dimpred_predict(features, model), expected, 'AbsTol', 1e-10, ...
    'Predictions of the kernel model differ from the formula');
end

function test_kernel_model_with_zero_features_gives_error(testCase)
features = testCase.TestData.ref.features_alignet(1:3, :);
features(2, :) = 0;
testCase.verifyError(@() dimpred_predict(features), 'dimpred:zeroFeatures', ...
    'Features that are all 0 should give an error with a kernel model (their cosine is not defined)');
end


%% Helpers

function names = shipped_models()
% Names of the shipped models, sorted alphabetically (same list in
% test_dimpred_load_model.m and test_dimpred_fixtures.m)
names = {'alignet_siglip2b_66d_kernel'; 'alignet_siglip2b_66d_ridge'; 'rn50x64_49d_ridge'; ...
    'rn50x64_66d_elastic'; 'rn50x64_66d_ridge'; 'vitb32_66d_elastic'};
end

function features = features_for_model(ref, name)
% The RN50x64 models need RN50x64 features, the ViT model ViT-B-32-quickgelu
% features, the AligNet model AligNet SigLIP2-B features (same function in
% test_dimpred_validation_human.m)
if startsWith(name, 'rn50x64')
    features = ref.features_rn50x64;
elseif startsWith(name, 'vitb32')
    features = ref.features_vitb32;
elseif startsWith(name, 'alignet')
    features = ref.features_alignet;
else
    error('No fixture features for model %s', name)
end
end

function model = toy_model()
% Small model in the format of dimpred_load_model, numbers in test_toy_model_by_hand
model.weights = [1 0; 0 2; -1 1];
model.feature_mean = [1 2 3];
model.feature_scale = [2 1 0.5];
model.target_mean = [0.5 0.25];
model.labels = {'dim 1'; 'dim 2'};
model.info = struct('name', 'toy', 'network', 'none', 'pretrained', 'none', 'layer', 'none', ...
    'preprocessing', 'none', 'embedding', 'none', 'regression', 'none', 'training_images', 'none', ...
    'source', 'test_dimpred_predict.m', 'note', 'toy model for testing', 'created', '2026-09-30', ...
    'n_features', 3, 'n_dims', 2);
model.file = '';
end
