% function S = dimpred_similarity(embedding, method)
%
% Predicted similarity between all pairs of objects (e.g. images) from
% their SPoSE dimensions, e.g. the output of dimpred_predict.
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
% With method 'dot', S is the matrix of dot products, embedding * embedding'.
%
% Computing time: the spose similarity loops over the objects i and
% computes all triplets (i,j,k) with j > i at once, with a matrix of up to
% n x n values in each step. The time grows with n^3, i.e. twice as many
% objects take about 8 times as long (a few seconds for 1000 objects).
%
% Input:
%   embedding: n_objects x n_dims (one row per object), all values finite
%   method:    'spose' (default) or 'dot'; [] also gives 'spose'
%
% Output:
%   S: n_objects x n_objects similarity matrix (double, symmetric)
%
% Example:
%   embedding = dimpred_predict(dimpred_extract_features(dimpred_find_images('my_images')));
%   S = dimpred_similarity(embedding);
%
% Martin Hebart, 2026/09/30
%
% See also DIMPRED_PREDICT

% History:
% 2026/09/30: NaN and Inf give an error, as in Python
% 2026/09/30: written for the first release of the package, following
%   embedding2sim_stable_fast2.m, but vectorized over two objects

function S = dimpred_similarity(embedding, method)

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

switch lower(method)
    case 'dot'
        S = embedding * embedding';
    case 'spose'
        S = spose_similarity(embedding);
    otherwise
        error('dimpred:unknownMethod', 'Unknown method ''%s''. Use ''spose'' or ''dot''.', method)
end


%% Subfunctions

function S = spose_similarity(embedding)

n_objects = size(embedding, 1);
if n_objects < 3
    error('dimpred:tooFewObjects', ['The SPoSE similarity (method ''spose'') needs at least 3 objects (it is ' ...
        'defined by triplets of objects), but the embedding has %i.'], n_objects)
end
dots = embedding * embedding';

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
