% function embedding = dimpred_predict(features, model)
%
% Predict the values of the SPoSE dimensions of images from their network
% features. The features are z-scored with the mean and std of the
% features of the 1854 training images, multiplied with the regression
% weights, and the mean of each dimension in the training images is added.
% Negative values are set to 0, because SPoSE dimensions are non-negative:
%
%   embedding = max(((features - feature_mean) ./ feature_scale) * weights + target_mean, 0)
%
% target_mean has to be added because the regressions were fit on centered
% dimension values. Some earlier, unofficial wrappers left it out, which
% makes the predictions too small and sets about 60% of them to 0.
%
% Each image is predicted on its own, i.e. the result for an image does
% not depend on which other images are predicted at the same time.
%
% Input:
%   features: network features, n_images x n_features (one row per image),
%             e.g. from dimpred_extract_features. The features have to come
%             from the network of the model (model.info.network). One image
%             is one row (1 x n_features); a column gives an error, because
%             it would be n_features images with one feature each.
%             Features of class single, as returned by
%             dimpred_extract_features, are converted to double. NaN or
%             Inf give an error.
%   model:    model name, path of a model file, or a model from
%             dimpred_load_model (default: [], the default model
%             vitb32_66d_elastic)
%
% Output:
%   embedding: predicted dimension values, double, n_images x n_dims (one
%              column per dimension, labels in model.labels)
%
% Example:
%   features = dimpred_extract_features(dimpred_find_images('my_images'));
%   embedding = dimpred_predict(features);
%
% Martin Hebart, 2026/09/30
%
% See also DIMPRED_LOAD_MODEL, DIMPRED_EXTRACT_FEATURES, DIMPRED_SIMILARITY

% History:
% 2026/09/30: NaN and Inf give an error, as in Python
% 2026/09/30: written for the first release of the package

function embedding = dimpred_predict(features, model)

if ~exist('model', 'var'), model = []; end
model = dimpred_load_model(model);

% Check input. We never transpose the features ourselves: for a square
% matrix we could not tell which way round it is meant.
n_features = size(model.weights, 1);
if ~ismatrix(features) || size(features, 2) ~= n_features
    % Name and network of the model for the message. A model file does not
    % have to include them in info, and a model made by hand may have no file.
    name = ''; network = 'unknown';
    if isfield(model, 'file'), name = model.file; end
    if isfield(model, 'info') && isfield(model.info, 'name'), name = model.info.name; end
    if isfield(model, 'info') && isfield(model.info, 'network'), network = model.info.network; end
    message = sprintf(['The model %s (network %s) needs %i features per image, one row per image, ' ...
        'but the features have size %s.'], name, network, n_features, mat2str(size(features)));
    if ismatrix(features) && size(features, 1) == n_features
        message = [message ' The images seem to be in the columns, please transpose the features.'];
    else
        message = [message ' Please use the features of this network, or a model for the network of your ' ...
            'features (see dimpred_list_models).'];
    end
    error('dimpred:wrongFeatureCount', '%s', message)
end
not_finite = find(~all(isfinite(features), 2));
if ~isempty(not_finite)
    % NaN or Inf would give NaN, Inf or 0 as predictions, and the zeros look
    % like real values
    error('dimpred:notFinite', ['The features contain NaN or Inf in %i of %i rows (the first is row %i). ' ...
        'Please check these images or remove them.'], numel(not_finite), size(features, 1), not_finite(1))
end

% z-score the features with the mean and std of the training images (not
% with those of the new images), apply the regression and add the mean of
% each dimension
z = (double(features) - model.feature_mean) ./ model.feature_scale;
embedding = z * model.weights + model.target_mean;

% SPoSE dimensions are non-negative
embedding = max(embedding, 0);
