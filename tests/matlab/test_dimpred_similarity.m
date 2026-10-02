% function tests = test_dimpred_similarity
%
% Tests for dimpred_similarity, which turns an embedding (objects x
% dimensions) into a similarity matrix.
%
% For method 'spose' (default), S(i,j) is the probability that i and j are
% picked as the most similar pair in an odd-one-out triplet with a random
% third object k, averaged over all k other than i and j:
%
%   S(i,j) = mean over k of exp(e_i*e_j') / (exp(e_i*e_j') + exp(e_i*e_k') + exp(e_j*e_k'))
%
% The diagonal is 1. Because the three pair probabilities of each triplet
% add up to 1, the mean of all off-diagonal values is exactly 1/3, which
% is a simple check that uses no reference numbers. Method 'dot' gives
% embedding * embedding'.
%
% We compare against a small example computed by hand and against a slow
% loop that follows the definition (as in embedding2sim_stable.m). The
% tests need no fixture and run in a fraction of a second.
%
% The Python tests in tests/ check the same behavior.
%
% Run with
%   runtests('test_dimpred_similarity')
% or all MATLAB tests with run_dimpred_tests.
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_SIMILARITY, RUN_DIMPRED_TESTS

% History:
% 2026/09/30: NaN and Inf give an error, as in Python
% 2026/09/30: after the second review: large values in a small range
% 2026/09/30: after review: failure messages for all checks, no global
%   random numbers
% 2026/09/30: written before the code (test-driven development)

function tests = test_dimpred_similarity
tests = functiontests(localfunctions);
end


%% Setup

function setupOnce(testCase)

% Add the dimpred MATLAB functions to the path (restored in teardownOnce)
here = fileparts(mfilename('fullpath')); % .../dimpred/tests/matlab
repo = fileparts(fileparts(here));
testCase.TestData.old_path = addpath(fullfile(repo, 'matlab'));

end

function teardownOnce(testCase)
path(testCase.TestData.old_path);
end


%% Values

function test_three_objects_by_hand(testCase)
% With 3 objects there is only one triplet. For the embedding below the dot
% products are e1*e2' = 0, e1*e3' = 1, e2*e3' = 1, so
%   S(1,2) = exp(0) / (exp(0) + exp(1) + exp(1)) = 1 / (1 + 2e) = 0.1554
%   S(1,3) = exp(1) / (exp(1) + exp(0) + exp(1)) = e / (1 + 2e) = 0.4223
%   S(2,3) = e / (1 + 2e) as well
embedding = [1 0; 0 1; 1 1];
a = 1 / (1 + 2*exp(1));
b = exp(1) / (1 + 2*exp(1));
expected = [
    1  a  b
    a  1  b
    b  b  1
    ];
testCase.verifyEqual(dimpred_similarity(embedding), expected, 'AbsTol', 1e-12, ...
    'SPoSE similarity of 3 objects differs from the values computed by hand (see the comments above)');
end

function test_matches_definition(testCase)
embedding = random_embedding(10, 5, 1);
testCase.verifyEqual(dimpred_similarity(embedding, 'spose'), similarity_by_definition(embedding), 'AbsTol', 1e-12, ...
    'SPoSE similarity differs from the loop over all triplets');
end

function test_large_values_give_finite_result(testCase)
% Dot products of several thousand make exp() overflow to Inf. The result
% still has to be correct, which needs the maximum of the three dot
% products to be subtracted before exp().
embedding = 30 * random_embedding(8, 10, 2);
S = dimpred_similarity(embedding);
testCase.verifyTrue(all(isfinite(S(:))), 'Large embedding values should not give NaN or Inf');
testCase.verifyEqual(S, similarity_by_definition(embedding), 'AbsTol', 1e-12, ...
    'For large embedding values, SPoSE similarity differs from the loop over all triplets');
end

function test_large_values_in_a_small_range_give_finite_result(testCase)
% Dot products of about 1100, all within a range of 700. For these,
% dimpred_similarity computes exp() only once for all pairs (the test
% above uses the other way, triplet by triplet), which needs the largest
% dot product to be subtracted first, because exp(710) is already Inf.
stream = RandStream('mt19937ar', 'Seed', 11);
embedding = 10 + rand(stream, 8, 10);
dots = embedding * embedding';
testCase.assertLessThan(max(dots(:)) - min(dots(:)), 700, 'The test needs dot products within a range of 700');
testCase.assertGreaterThan(min(dots(:)), 710, 'The test needs dot products for which exp() is Inf');
S = dimpred_similarity(embedding);
testCase.verifyTrue(all(isfinite(S(:))), 'Large embedding values in a small range should not give NaN or Inf');
testCase.verifyEqual(S, similarity_by_definition(embedding), 'AbsTol', 1e-12, ...
    'For large embedding values in a small range, SPoSE similarity differs from the loop over all triplets');
end

function test_mean_off_diagonal_is_one_third(testCase)
embedding = random_embedding(12, 6, 3);
S = dimpred_similarity(embedding);
off_diagonal = ~eye(size(S));
testCase.verifyEqual(mean(S(off_diagonal)), 1/3, 'AbsTol', 1e-12, ...
    'The mean of all off-diagonal values of the SPoSE similarity has to be 1/3');
end

function test_is_symmetric(testCase)
S = dimpred_similarity(random_embedding(10, 5, 4));
testCase.verifyEqual(S, S', 'AbsTol', 1e-12, 'The similarity matrix should be symmetric');
end

function test_diagonal_is_one(testCase)
S = dimpred_similarity(random_embedding(10, 5, 5));
testCase.verifyEqual(diag(S), ones(10, 1), 'The diagonal of the similarity matrix should be exactly 1');
end

function test_values_between_zero_and_one(testCase)
S = dimpred_similarity(random_embedding(10, 5, 6));
testCase.verifyTrue(all(S(:) >= 0 & S(:) <= 1), 'Similarities are probabilities and have to be between 0 and 1');
end

function test_permuting_objects_permutes_matrix(testCase)
embedding = random_embedding(10, 5, 7);
order = [3 9 1 10 2 8 4 7 5 6];
S = dimpred_similarity(embedding);
testCase.verifyEqual(dimpred_similarity(embedding(order, :)), S(order, order), 'AbsTol', 1e-12, ...
    'Reordering the objects should only reorder rows and columns of the similarity matrix');
end

function test_output_size(testCase)
testCase.verifySize(dimpred_similarity(random_embedding(7, 3, 8)), [7 7], ...
    'An embedding of 7 objects should give a 7 x 7 similarity matrix');
end


%% Methods

function test_default_method_is_spose(testCase)
embedding = random_embedding(6, 4, 9);
testCase.verifyEqual(dimpred_similarity(embedding), dimpred_similarity(embedding, 'spose'), ...
    'Without method, dimpred_similarity should use ''spose''');
end

function test_dot_method(testCase)
embedding = random_embedding(6, 4, 10);
testCase.verifyEqual(dimpred_similarity(embedding, 'dot'), embedding * embedding', 'AbsTol', 1e-12, ...
    'Method ''dot'' should give embedding * embedding''');
end


%% Errors

function test_too_few_objects_gives_error(testCase)
% SPoSE similarity needs a third object, so at least 3 objects
testCase.verifyError(@() dimpred_similarity(ones(2, 5)), 'dimpred:tooFewObjects', ...
    'An embedding of 2 objects should give an error');
testCase.verifyError(@() dimpred_similarity(ones(1, 5)), 'dimpred:tooFewObjects', ...
    'An embedding of 1 object should give an error');
end

function test_unknown_method_gives_error(testCase)
testCase.verifyError(@() dimpred_similarity(ones(4, 3), 'cosine'), 'dimpred:unknownMethod', ...
    'A method other than ''spose'' or ''dot'' should give an error');
end

function test_non_matrix_input_gives_error(testCase)
testCase.verifyError(@() dimpred_similarity(ones(4, 3, 2)), 'dimpred:notMatrix', ...
    'A 3D array should give an error');
end

function test_nan_or_inf_gives_error(testCase)
% With spose, one such object would make every value NaN, because it is
% the third object of every pair
for value = [NaN Inf]
    embedding = random_embedding(6, 3, 19);
    embedding(5, 2) = value;
    testCase.verifyError(@() dimpred_similarity(embedding), 'dimpred:notFinite', ...
        sprintf('An embedding with %g should give an error', value));
    testCase.verifyError(@() dimpred_similarity(embedding, 'dot'), 'dimpred:notFinite', ...
        sprintf('An embedding with %g should give an error also with method dot', value));
end
end


%% Helpers

function embedding = random_embedding(n_objects, n_dims, seed)
% Non-negative random embedding like SPoSE dimensions. We use our own
% random stream so that the tests do not change the global random state.
stream = RandStream('mt19937ar', 'Seed', seed);
embedding = rand(stream, n_objects, n_dims);
end

function S = similarity_by_definition(embedding)
% Slow loop over all triplets, directly from the definition (same as
% embedding2sim_stable.m). The maximum of the three dot products is
% subtracted before exp(), which does not change the ratio but prevents
% overflow.
n_objects = size(embedding, 1);
sim = embedding * embedding';
S = eye(n_objects);
for i = 1:n_objects
    for j = [1:i-1, i+1:n_objects]
        p = zeros(1, n_objects - 2);
        k_list = setdiff(1:n_objects, [i j]);
        for i_k = 1:numel(k_list)
            k = k_list(i_k);
            three = [sim(i, j), sim(i, k), sim(j, k)];
            three = exp(three - max(three));
            p(i_k) = three(1) / sum(three);
        end
        S(i, j) = mean(p);
    end
end
end
