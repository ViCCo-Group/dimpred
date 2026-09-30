import numpy as np

from .load_model import load_model

# History:
# 2026/09/30: written for the first release of the package


def predict(features, model=None):
    """
    embedding = predict(features, model=None)

    Predict the values of the SPoSE dimensions of images from their network
    features. The features are z-scored with the mean and std of the features
    of the 1854 training images, multiplied with the regression weights, and
    the mean of each dimension in the training images is added. Negative
    values are set to 0, because SPoSE dimensions are non-negative:

        embedding = max(((features - feature_mean) / feature_scale) @ weights + target_mean, 0)

    target_mean has to be added because the regressions were fit on
    centered dimension values. Some earlier, unofficial wrappers left it
    out, which makes the predictions too small and sets about 60% of them to 0.

    Each image is predicted on its own, i.e. the result for an image does not
    depend on which other images are predicted at the same time.

    Input:
        features: network features, n_images x n_features (one row per
                  image), e.g. from extract_features. The features have to
                  come from the network of the model (model["info"]["network"]).
                  A single vector (n_features,) is one image. Any float type
                  or a list is accepted and computed in float64. The input
                  is not changed. nan or inf give an error.
        model:    model name, path of a model file, or a model from
                  load_model (default: dimpred.DEFAULT_MODEL)

    Output:
        embedding: predicted dimension values, float64, n_images x n_dims
                   (one column per dimension, labels in model["labels"])

    Example:
        features = dimpred.extract_features(dimpred.find_images("my_images"))
        embedding = dimpred.predict(features)

    Martin Hebart, 2026/09/30

    See also: load_model, extract_features, similarity
    """

    model = load_model(model)

    # Check input
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features[np.newaxis, :]  # one image
    n_features = model["weights"].shape[0]
    if features.ndim != 2 or features.shape[1] != n_features:
        info = model.get("info", {})
        message = (f"The model {info.get('name', model.get('file'))} (network {info.get('network')}) needs "
                   f"{n_features} features per image, one row per image, but the features have shape "
                   f"{features.shape}.")
        if features.ndim == 2 and features.shape[0] == n_features:
            message += " The images seem to be in the columns, please transpose the features."
        else:
            message += (" Please use the features of this network, or a model for the network of your "
                        "features (see dimpred.list_models).")
        raise ValueError(message)
    not_finite = np.flatnonzero(~np.all(np.isfinite(features), axis=1))
    if not_finite.size > 0:
        # nan or inf would give nan, inf or 0 as predictions, and the zeros
        # look like real values
        raise ValueError(f"The features contain nan or inf in {not_finite.size} of {features.shape[0]} rows (the "
                         f"first is row {not_finite[0]}, counting from 0). Please check these images or remove them.")

    # z-score the features with the statistics of the training images and
    # apply the regression. Nothing is computed in place, so the input
    # features and the model stay unchanged.
    z = (features - model["feature_mean"]) / model["feature_scale"]
    embedding = z @ model["weights"] + model["target_mean"]

    # SPoSE dimensions are non-negative
    return np.maximum(embedding, 0)
