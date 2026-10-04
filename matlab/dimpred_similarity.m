% function S = dimpred_similarity(embedding, method, features, model)
%
% Predicted similarity between all pairs of objects (e.g. images) from
% their SPoSE dimensions, e.g. the output of dimpred_predict.
%
% With features (the network features of the same objects, as passed to
% dimpred_predict), the network's own similarity is added for pairs that
% are very close in the network (close pairs). The dot products of the
% embedding are then replaced by
%
%   D(i,j) = e_i*e_j' + close_pairs_weight * max(0, cos(f_i, f_j) - close_pairs_threshold)
%
% where cos is the cosine of the features. The weight and the threshold
% come from the model (default: the default model; the threshold is the
% 90th percentile of the cosines between its 1854 training images, so only
% the closest pairs change). This is the default method of dimpred since
% version 1.2.0. Without features, S comes from the embedding alone.
%
% With method 'spose' (default), the similarity of objects i and j is the
% probability that i and j are picked as the most similar pair in an
% odd-one-out triplet with a random third object k, as in the SPoSE model
% (Hebart et al., 2020):
%
%   S(i,j) = mean over all k other than i and j of
%            exp(e_i*e_j') / (exp(e_i*e_j') + exp(e_i*e_k') + exp(e_j*e_k'))
%
% where e_i*e_j' is the dot product of the embeddings of i and j. The
% diagonal is set to 1. The similarity of two objects therefore depends on
% all other objects in the set. In each triplet exactly one pair is picked,
% so the mean of all values off the diagonal is always 1/3. This is the
% same definition as in embedding2sim_stable.m. Before exp, the largest dot
% product is subtracted (the largest of all pairs, or, if the dot products
% span 700 or more, the largest of the three in each triplet). This does
% not change the probabilities, but exp cannot overflow for large dot
% products.
%
% With method 'dot', S is the matrix of dot products, embedding * embedding'
% (with features: D above).
%
% Computing time: the spose similarity loops over the objects i and
% computes all triplets (i,j,k) with j > i at once, with a matrix of up to
% n x n values in each step. The time grows with n^3, i.e. twice as many
% objects take about 8 times as long (a few seconds for 1000 objects).
%
% Input:
%   embedding: n_objects x n_dims (one row per object), all values finite
%   method:    'spose' (default) or 'dot'; [] also gives 'spose'
%   features:  optional, the network features of the objects,
%              n_objects x n_features (the input of dimpred_predict), for
%              the close-pair term ([] or omitted: none)
%   model:     the model whose close-pair settings are used (default: [],
%              the default model); only used with features
%
% Output:
%   S: n_objects x n_objects similarity matrix (double, symmetric)
%
% Example:
%   features = dimpred_extract_features(dimpred_find_images('my_images'));
%   embedding = dimpred_predict(features);
%   S = dimpred_similarity(embedding, [], features);  % with the close pairs (recommended)
%   S_dims = dimpred_similarity(embedding);           % from the dimensions alone
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_PREDICT

% History:
% 2026/10/04: features and model: the close-pair term of the default model
% 2026/09/30: NaN and Inf give an error, as in Python
% 2026/09/30: written for the first release of the package, following
%   embedding2sim_stable_fast2.m, but vectorized over two objects

function S = dimpred_similarity(embedding, method, features, model)

% Check input
if ~exist('method', 'var') || isempty(method), method = 'spose'; end
if ~ismatrix(embedding)
    error('dimpred:notMatrix', ['The embedding has to be a matrix, n_objects x n_dims (one row per object), ' ...
        'but it has size %s.'], mat2str(size(embedding)))
end
not_finite = find(~all(isfinite(embedding), 2));
if ~isempty(not_finite)
    % a single such object makes all values of the spose similarity NaN,
    % because it is the third object of every pair
    error('dimpred:notFinite', ['The embedding contains NaN or Inf in %i of %i rows (the first is row %i). ' ...
        'Please remove these objects or check their features.'], numel(not_finite), size(embedding, 1), not_finite(1))
end
embedding = double(embedding);
if ~any(strcmpi(method, {'spose', 'dot'}))
    error('dimpred:unknownMethod', 'Unknown method ''%s''. Use ''spose'' or ''dot''.', method)
end

% The dot products of the embedding, with features plus the close-pair term
dots = embedding * embedding';
if exist('features', 'var') && ~isempty(features)
    if ~exist('model', 'var'), model = []; end
    dots = dots + close_pairs(features, size(embedding, 1), model);
end

switch lower(method)
    case 'dot'
        S = dots;
    case 'spose'
        S = spose_similarity(dots);
end


%% Subfunctions

function term = close_pairs(features, n_objects, model)
% The close-pair term: close_pairs_weight * max(0, cos(f_i, f_j) - close_pairs_threshold) for all pairs

model = dimpred_load_model(model);
name = model.file;
if isfield(model, 'info') && isfield(model.info, 'name'), name = model.info.name; end
if ~isfield(model, 'close_pairs_weight') || isempty(model.close_pairs_weight)
    error('dimpred:noClosePairs', ['The model %s has no close-pair settings, so dimpred_similarity cannot ' ...
        'use the features. Use a model with close_pairs_weight and close_pairs_threshold (e.g. the default ' ...
        'model alignet_siglip2b_66d_kernel), or leave out the features.'], name)
end
if ~ismatrix(features) || size(features, 1) ~= n_objects
    error('dimpred:wrongFeatureCount', ['The features have to be n_objects x n_features with one row per ' ...
        'object of the embedding (%i rows), but they have size %s.'], n_objects, mat2str(size(features)))
end
if size(features, 2) ~= size(model.weights, 1)
    error('dimpred:wrongFeatureCount', ['The model %s needs %i features per object, but the features have ' ...
        '%i. The features have to come from the network of the model.'], name, size(model.weights, 1), ...
        size(features, 2))
end
features = double(features);
norms = sqrt(sum(features .^ 2, 2));
if ~all(isfinite(features(:))) || any(norms == 0)
    error('dimpred:notFinite', ['The features contain NaN or Inf, or rows that are all 0, so their cosines ' ...
        'are not defined.'])
end
unit = features ./ norms;
term = model.close_pairs_weight * max(0, unit * unit' - model.close_pairs_threshold);


function S = spose_similarity(dots)

n_objects = size(dots, 1);
if n_objects < 3
    error('dimpred:tooFewObjects', ['The SPoSE similarity (method ''spose'') needs at least 3 objects (it is ' ...
        'defined by triplets of objects), but the embedding has %i.'], n_objects)
end

% Most of the time goes into exp. If the dot products lie within a range
% of 700, we compute exp once for all pairs, after subtracting the largest
% dot product, which does not change the probabilities but prevents
% overflow (exp(-700) is still an ordinary number, so nothing becomes 0).
% Otherwise, e.g. for very large embedding values, we subtract the largest
% of the three dot products in each triplet, as in embedding2sim_stable.m.
% Both give the same values, but the second is about 10 times slower.
exp_once = max(dots(:)) - min(dots(:)) < 700;
if exp_once
    exp_dots = exp(dots - max(dots(:)));
end

% For each object i, we compute the probabilities of all triplets (i,j,k)
% with j > i at once: j in the rows, k in the columns. The lower half of S
% follows from the upper half, since S(i,j) = S(j,i).
S = zeros(n_objects);
for i = 1:n_objects-1
    j = (i+1:n_objects)';
    if exp_once
        exp_ij = exp_dots(j, i); % column, expanded along k
        exp_ik = exp_dots(i, :); % row, expanded along j
        exp_jk = exp_dots(j, :);
    else
        largest = max(max(dots(j, i), dots(i, :)), dots(j, :)); % prevents overflow of exp
        exp_ij = exp(dots(j, i) - largest);
        exp_ik = exp(dots(i, :) - largest);
        exp_jk = exp(dots(j, :) - largest);
    end
    p = exp_ij ./ (exp_ij + exp_ik + exp_jk);

    % k has to be different from i and j
    p(:, i) = 0;
    p(sub2ind(size(p), 1:numel(j), j')) = 0;

    % mean across the n - 2 third objects
    S(i, j) = sum(p, 2)' / (n_objects - 2);
end
S = S + S';
S(1:n_objects+1:end) = 1; % diagonal
