% function tests = test_dimpred_validation_human
%
% Tests the whole chain from features to predicted similarity against
% human behavior, for each shipped model.
%
% For the 48nonref images we have a human similarity matrix from
% odd-one-out judgments. We predict the SPoSE dimensions of these images
% from the fixture features, compute the predicted SPoSE similarity, and
% correlate it with the human similarity (Pearson r of the lower triangle,
% i.e. each pair of images once, without the diagonal). The result has to
% match the correlation stored in the fixture (human_r_48nonref) within
% 0.002. For rn50x64_49d_ridge, the model of the DimPred paper, this is
% r = 0.810.
%
% If one of these tests fails while test_dimpred_predict and
% test_dimpred_similarity pass, look at the order of the images and at
% which features go with which model.
%
% The Python tests in tests/ check the same numbers.
%
% Run with
%   runtests('test_dimpred_validation_human')
% or all MATLAB tests with run_dimpred_tests.
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_PREDICT, DIMPRED_SIMILARITY, RUN_DIMPRED_TESTS

% History:
% 2026/10/04: new model alignet_siglip2b_66d_kernel
% 2026/10/02: new model alignet_siglip2b_66d_ridge
% 2026/09/30: written before the code (test-driven development)

function tests = test_dimpred_validation_human
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


%% One test per model

function test_human_similarity_alignet_siglip2b_66d_kernel(testCase)
verify_human_r(testCase, 'alignet_siglip2b_66d_kernel');
end

function test_human_similarity_alignet_siglip2b_66d_ridge(testCase)
verify_human_r(testCase, 'alignet_siglip2b_66d_ridge');
end

function test_human_similarity_rn50x64_49d_ridge(testCase)
verify_human_r(testCase, 'rn50x64_49d_ridge');
end

function test_human_similarity_rn50x64_66d_elastic(testCase)
verify_human_r(testCase, 'rn50x64_66d_elastic');
end

function test_human_similarity_rn50x64_66d_ridge(testCase)
verify_human_r(testCase, 'rn50x64_66d_ridge');
end

function test_human_similarity_vitb32_66d_elastic(testCase)
verify_human_r(testCase, 'vitb32_66d_elastic');
end


%% Helpers

function verify_human_r(testCase, name)
ref = testCase.TestData.ref;
is_48nonref = strcmp(ref.image_set, '48nonref');
features = features_for_model(ref, name);

embedding = dimpred_predict(features(is_48nonref, :), name);
S = dimpred_similarity(embedding);

pairs = tril(true(size(S)), -1); % each pair once, without the diagonal
r = corrcoef(S(pairs), ref.human_similarity_48nonref(pairs));
testCase.verifyEqual(r(1, 2), ref.human_r_48nonref.(name), 'AbsTol', 0.002, ...
    sprintf('Model %s: correlation of predicted and human similarity (48nonref) differs from the fixture', name));
end

function features = features_for_model(ref, name)
% The RN50x64 models need RN50x64 features, the ViT model ViT-B-32-quickgelu
% features, the AligNet model AligNet SigLIP2-B features (same function in
% test_dimpred_predict.m)
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
