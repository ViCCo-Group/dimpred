% function tests = test_dimpred_materials
%
% Tests the models of the 36 material dimensions (STUFF, Schmidt et al.,
% 2025) and the 2AFC similarity of dimpred_similarity (method '2afc').
%
% The fixture materials_data.mat (made by tests/fixtures/make_fixtures.py
% --materials) has, for the 48 out-of-sample images of the STUFF paper, the
% RN50x64 features from which the paper computed its predictions, the
% AligNet SigLIP2-B features, the published predictions, the human
% similarity (choice probabilities of the triplet 2AFC task), and for each
% model the expected predictions and the correlation of their 2AFC
% similarity with the human similarity (stuff48_human_r).
%
% The Python tests in tests/test_materials.py check the same numbers.
%
% Run with
%   runtests('test_dimpred_materials')
% or all MATLAB tests with run_dimpred_tests.
%
% Hebartlab, 2026/10/26
%
% See also DIMPRED_PREDICT, DIMPRED_SIMILARITY, RUN_DIMPRED_TESTS

% History:
% 2026/10/26: written for the models of the 36 material dimensions

function tests = test_dimpred_materials
tests = functiontests(localfunctions);
end


%% Setup

function setupOnce(testCase)
here = fileparts(mfilename('fullpath')); % .../dimpred/tests/matlab
repo = fileparts(fileparts(here));
testCase.TestData.old_path = addpath(fullfile(repo, 'matlab'));
fixture_file = fullfile(repo, 'tests', 'fixtures', 'materials_data.mat');
testCase.assertTrue(exist(fixture_file, 'file') == 2, ...
    sprintf('Fixture file %s not found. It is made by tests/fixtures/make_fixtures.py --materials.', fixture_file));
testCase.TestData.stuff = load(fixture_file);
end

function teardownOnce(testCase)
path(testCase.TestData.old_path);
end


%% The models

function test_paper_model_gives_the_published_predictions(testCase)
% rn50x64_36d_ridge is rebuilt from the MATLAB code of the STUFF paper; the
% published predictions are single precision
stuff = testCase.TestData.stuff;
embedding = dimpred_predict(stuff.stuff48_features_rn50x64, 'rn50x64_36d_ridge');
testCase.verifyEqual(embedding, stuff.stuff48_published_rn50x64_36d_ridge, 'AbsTol', 1e-5, ...
    'rn50x64_36d_ridge should give the published predictions of the 48 images');
end

function test_models_give_the_expected_predictions(testCase)
stuff = testCase.TestData.stuff;
names = material_models();
for i_model = 1:numel(names)
    name = names{i_model};
    embedding = dimpred_predict(features_for_model(stuff, name), name);
    testCase.verifyEqual(embedding, stuff.(['stuff48_expected_' name]), 'AbsTol', 1e-10, ...
        sprintf('Model %s does not give the expected predictions (stuff48_expected_%s)', name, name));
end
end

function test_similarity_correlates_with_human_similarity_as_expected(testCase)
% the paper model gives 0.818, the AligNet models 0.835 (ridge) and 0.842
% (kernel); all have to give more than 0.80
stuff = testCase.TestData.stuff;
names = material_models();
pairs = tril(true(48), -1); % each pair once, without the diagonal
for i_model = 1:numel(names)
    name = names{i_model};
    S = dimpred_similarity(dimpred_predict(features_for_model(stuff, name), name), '2afc');
    r = corrcoef(S(pairs), stuff.stuff48_human_similarity(pairs));
    testCase.verifyEqual(r(1, 2), stuff.stuff48_human_r.(name), 'AbsTol', 0.002, ...
        sprintf('Model %s: correlation of predicted and human similarity differs from the fixture', name));
    testCase.verifyGreaterThan(r(1, 2), 0.80, sprintf('Model %s: r with human similarity below 0.80', name));
end
end

function test_similarity_with_features_gives_error_for_the_material_kernel_model(testCase)
% the material kernel model has no close-pair settings
stuff = testCase.TestData.stuff;
features = stuff.stuff48_features_alignet;
embedding = dimpred_predict(features, 'alignet_siglip2b_36d_kernel');
testCase.verifyError(@() dimpred_similarity(embedding, '2afc', features, 'alignet_siglip2b_36d_kernel'), ...
    'dimpred:noClosePairs', 'The material kernel model has no close pairs, so features should give an error');
end


%% The 2AFC similarity

function test_2afc_similarity_follows_its_definition(testCase)
stream = RandStream('mt19937ar', 'Seed', 0);
for scale = [2 100] % 100: dot products up to about 10000, exp would overflow
    embedding = scale * rand(stream, 7, 4);
    testCase.verifyEqual(dimpred_similarity(embedding, '2afc'), similarity_2afc_by_definition(embedding), ...
        'AbsTol', 1e-12, sprintf('2afc similarity differs from its definition (scale %i)', scale));
end
end

function test_2afc_similarity_is_symmetric_with_diagonal_1_and_mean_one_half(testCase)
stream = RandStream('mt19937ar', 'Seed', 1);
S = dimpred_similarity(rand(stream, 30, 36), '2afc');
testCase.verifyEqual(S, S', 'AbsTol', 0, '2afc similarity should be symmetric');
testCase.verifyEqual(diag(S), ones(30, 1), '2afc similarity should have 1 on the diagonal');
testCase.verifyEqual(mean(S(~eye(30))), 0.5, 'AbsTol', 1e-12, 'The mean off the diagonal should be 1/2');
end

function test_2afc_method_name_ignores_case(testCase)
stream = RandStream('mt19937ar', 'Seed', 2);
embedding = rand(stream, 5, 3);
testCase.verifyEqual(dimpred_similarity(embedding, '2AFC'), dimpred_similarity(embedding, '2afc'), ...
    'The method name should not depend on case');
end

function test_2afc_similarity_needs_three_objects(testCase)
testCase.verifyError(@() dimpred_similarity(ones(2, 3), '2afc'), 'dimpred:tooFewObjects', ...
    'The 2afc similarity of 2 objects should give an error');
end


%% Helpers

function names = material_models()
% Names of the models of the 36 material dimensions, sorted alphabetically
names = {'alignet_siglip2b_36d_kernel'; 'alignet_siglip2b_36d_ridge'; 'rn50x64_36d_ridge'};
end

function features = features_for_model(stuff, name)
if startsWith(name, 'alignet')
    features = stuff.stuff48_features_alignet;
else
    features = stuff.stuff48_features_rn50x64;
end
end

function S = similarity_2afc_by_definition(embedding)
% The definition written out: P(i,j) is the mean over all k other than i
% and j of exp(e_i*e_j') / (exp(e_i*e_j') + exp(e_i*e_k')), with the larger
% of the two dot products subtracted before exp; S = (P + P') / 2.
n = size(embedding, 1);
dots = embedding * embedding';
P = zeros(n);
for i = 1:n
    for j = 1:n
        if i == j, continue, end
        p = [];
        for k = 1:n
            if k == i || k == j, continue, end
            larger = max(dots(i, j), dots(i, k));
            p(end+1) = exp(dots(i, j) - larger) / (exp(dots(i, j) - larger) + exp(dots(i, k) - larger)); %#ok<AGROW>
        end
        P(i, j) = mean(p);
    end
end
S = (P + P') / 2;
S(1:n+1:end) = 1;
end
