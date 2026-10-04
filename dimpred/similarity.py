import numpy as np

from .load_model import load_model

# History:
# 2026/10/04: features and model: the close-pair term of the default model
# 2026/09/30: written for the first release of the package, following
#   embedding2sim_stable_fast2.m, but vectorized over two objects


def similarity(embedding, method="spose", features=None, model=None):
    """
    S = similarity(embedding, method="spose", features=None, model=None)

    Predicted similarity between all pairs of objects (e.g. images) from their
    SPoSE dimensions, e.g. the output of predict.

    With features (the network features of the same objects, as passed to
    predict), the network's own similarity is added for pairs that are very
    close in the network (close pairs). The dimensions describe the
    differences between kinds of objects well, but less the fine
    differences within a kind, and the network knows these. The dot products
    of the embedding are then replaced by

        D[i, j] = e_i.e_j + close_pairs_weight * max(0, cos(f_i, f_j) - close_pairs_threshold)

    where cos is the cosine of the features. The weight and the threshold
    come from the model (default: dimpred.DEFAULT_MODEL; the threshold is the
    90th percentile of the cosines between its 1854 training images, so only
    the closest pairs change). This is the default method of dimpred since
    version 1.2.0: on the human similarity benchmark it predicts human
    similarity much better than the dimensions alone, above all within a
    category (see docs/details.md). Without features, S comes from the
    embedding alone, as before.

    With method "spose" (default), the similarity of objects i and j is the
    probability that i and j are picked as the most similar pair in an
    odd-one-out triplet with a random third object k, as in the SPoSE model
    (Hebart et al., 2020):

        S[i, j] = mean over all k other than i and j of
                  exp(e_i.e_j) / (exp(e_i.e_j) + exp(e_i.e_k) + exp(e_j.e_k))

    where e_i.e_j is the dot product of the embeddings of i and j. The
    diagonal is set to 1. The similarity of two objects therefore depends on
    all other objects in the set. In each triplet exactly one pair is picked,
    so the mean of all values off the diagonal is always 1/3. This is the
    same definition as in embedding2sim_stable.m. Before exp, the largest dot
    product is subtracted (the largest of all pairs, or, if the dot products
    span 700 or more, the largest of the three in each triplet). This does
    not change the probabilities, but exp cannot overflow for large dot
    products.

    With method "dot", S is the matrix of dot products, embedding @ embedding.T
    (with features: D above).

    Computing time: the spose similarity loops over the objects i and
    computes all triplets (i, j, k) with j > i at once, with a matrix of up
    to n x n values in each step. The time grows with n^3, i.e. twice as many
    objects take about 8 times as long (a few seconds for 1000 objects).

    Input:
        embedding: n_objects x n_dims (one row per object), all values finite
        method:    "spose" (default) or "dot"; None also gives "spose"
        features:  optional, the network features of the objects,
                   n_objects x n_features (the input of predict), for the
                   close-pair term
        model:     the model whose close-pair settings are used (name, path
                   or dict, default: dimpred.DEFAULT_MODEL); only used with
                   features

    Output:
        S: n_objects x n_objects similarity matrix (float64, symmetric)

    Example:
        features = dimpred.extract_features(dimpred.find_images("my_images"))
        embedding = dimpred.predict(features)
        S = dimpred.similarity(embedding, features=features)   # with the close pairs (recommended)
        S_dims = dimpred.similarity(embedding)                 # from the dimensions alone

    Hebartlab, 2026/09/30

    See also: predict
    """

    # Check input
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise ValueError(f"The embedding has to be a matrix, n_objects x n_dims (one row per object), but it has "
                         f"shape {embedding.shape}.")
    if method is None:
        method = "spose"  # as [] in MATLAB
    if method.lower() not in ["spose", "dot"]:
        raise ValueError(f"Unknown method '{method}'. Use 'spose' or 'dot'.")
    method = method.lower()
    not_finite = np.flatnonzero(~np.all(np.isfinite(embedding), axis=1))
    if not_finite.size > 0:
        # a single such object makes all values of the spose similarity nan,
        # because it is the third object of every pair
        raise ValueError(f"The embedding contains nan or inf in {not_finite.size} of {embedding.shape[0]} rows (the "
                         f"first is row {not_finite[0]}, counting from 0). Please remove these objects or check "
                         f"their features.")

    dots = embedding @ embedding.T
    if features is not None:
        dots = dots + close_pairs(features, embedding.shape[0], model)
    if method == "dot":
        return dots

    n = embedding.shape[0]
    if n < 3:
        raise ValueError(f"The SPoSE similarity (method 'spose') needs at least 3 objects (it is defined by "
                         f"triplets of objects), but the embedding has {n}.")

    # Most of the time goes into exp. If the dot products lie within a range
    # of 700, we compute exp once for all pairs, after subtracting the largest
    # dot product, which does not change the probabilities but prevents
    # overflow (exp(-700) is still an ordinary number, so nothing becomes 0).
    # Otherwise, e.g. for very large embedding values, we subtract the largest
    # of the three dot products in each triplet, as in embedding2sim_stable.m.
    # Both give the same values, but the second is about 10 times slower.
    exp_once = dots.max() - dots.min() < 700
    if exp_once:
        exp_dots = np.exp(dots - dots.max())

    # For each object i, we compute the probabilities of all triplets (i, j, k)
    # with j > i at once: j in the rows, k in the columns. The lower half of S
    # follows from the upper half, since S[i, j] = S[j, i].
    S = np.zeros((n, n))
    for i in range(n - 1):
        j = np.arange(i + 1, n)
        if exp_once:
            exp_ij = exp_dots[j, i][:, np.newaxis]  # column, expanded along k
            exp_ik = exp_dots[i, :]  # row, expanded along j
            exp_jk = exp_dots[j, :]
        else:
            dots_ij = dots[j, i][:, np.newaxis]
            largest = np.maximum(np.maximum(dots_ij, dots[i, :]), dots[j, :])  # prevents overflow of exp
            exp_ij = np.exp(dots_ij - largest)
            exp_ik = np.exp(dots[i, :] - largest)
            exp_jk = np.exp(dots[j, :] - largest)
        p = exp_ij / (exp_ij + exp_ik + exp_jk)

        # k has to be different from i and j
        p[:, i] = 0
        p[np.arange(len(j)), j] = 0

        # mean across the n - 2 third objects
        S[i, j] = p.sum(axis=1) / (n - 2)
    S = S + S.T
    np.fill_diagonal(S, 1)
    return S


def close_pairs(features, n_objects, model=None):
    """The close-pair term: close_pairs_weight * max(0, cos(f_i, f_j) - close_pairs_threshold) for all pairs."""

    model = load_model(model)
    name = model.get("info", {}).get("name", model.get("file"))
    if model.get("close_pairs_weight") is None:
        raise ValueError(f"The model {name} has no close-pair settings, so similarity cannot use the features. "
                         f"Use a model with close_pairs_weight and close_pairs_threshold (e.g. the default model "
                         f"alignet_siglip2b_66d_kernel), or leave out the features.")
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] != n_objects:
        raise ValueError(f"The features have to be n_objects x n_features with one row per object of the "
                         f"embedding ({n_objects} rows), but they have shape {features.shape}.")
    n_features = model["weights"].shape[0]
    if features.shape[1] != n_features:
        raise ValueError(f"The model {name} (network {model.get('info', {}).get('network')}) needs {n_features} "
                         f"features per object, but the features have {features.shape[1]}. The features have to "
                         f"come from the network of the model.")
    norms = np.linalg.norm(features, axis=1)
    if not np.all(np.isfinite(features)) or np.any(norms == 0):
        raise ValueError("The features contain nan or inf, or rows that are all 0, so their cosines are not defined.")
    unit = features / norms[:, np.newaxis]
    cosines = unit @ unit.T
    return model["close_pairs_weight"] * np.maximum(0, cosines - model["close_pairs_threshold"])
