import os

import numpy as np
import scipy.io

from .list_models import list_models

# History:
# 2026/10/04: optional kernel part (kernel_features, kernel_coefficients,
#   kernel_tau) and close-pair settings (close_pairs_weight,
#   close_pairs_threshold); new default model alignet_siglip2b_66d_kernel
# 2026/10/02: new default model alignet_siglip2b_66d_ridge
# 2026/09/30: written for the first release of the package

VARIABLES = ["weights", "feature_mean", "feature_scale", "target_mean", "labels", "info"]
# optional, each group all or none
KERNEL_VARIABLES = ["kernel_features", "kernel_coefficients", "kernel_tau"]
CLOSE_PAIRS_VARIABLES = ["close_pairs_weight", "close_pairs_threshold"]


def load_model(model=None):
    """
    model = load_model(model=None)

    Load a model, i.e. everything that is needed to predict the SPoSE
    dimensions from network features: the regression weights, the mean and
    std of the features in the 1854 training images (for z-scoring new
    features), the mean of each dimension in the training images, the labels
    of the dimensions and a description of the model (info).

    The mean of each dimension (target_mean) is needed for every prediction:
    the regressions were fit on centered dimension values, so it has to be
    added back (see predict). Without it, the predictions are too small and
    most of them are 0. For this reason, a model file that lacks one of the
    variables gives an error and is never filled with defaults.

    Model files are MATLAB .mat files (the same files are used by the MATLAB
    version of dimpred) with the variables
        weights        n_features x n_dims
        feature_mean   1 x n_features
        feature_scale  1 x n_features (std of the features, all > 0)
        target_mean    1 x n_dims (mean of each dimension)
        labels         n_dims x 1 cell array of char
        info           struct with the fields name, network, pretrained,
                       layer, preprocessing, embedding, regression,
                       training_images, source, note, created (text) and
                       n_features, n_dims (numbers)
    A model with a local kernel (e.g. the default model) also has
        kernel_features      n_train x n_features, the features of the
                             training images, scaled to length 1
        kernel_coefficients  n_train x n_dims
        kernel_tau           width of the kernel (see predict)
    and the settings of the close-pair term of similarity:
        close_pairs_weight, close_pairs_threshold
    You can build your own model files in the same format and pass their path.

    Input:
        model: one of
               None:  the default model (dimpred.DEFAULT_MODEL, "alignet_siglip2b_66d_kernel")
               name:  a model that comes with dimpred (see list_models)
               path:  the path of a model file (.mat)
               dict:  a model that was already loaded, returned unchanged

    Output:
        model: dict with the keys
               weights        float64 array, n_features x n_dims
               feature_mean   float64 array, (n_features,)
               feature_scale  float64 array, (n_features,)
               target_mean    float64 array, (n_dims,)
               labels         list of n_dims str
               info           dict (text fields as str, n_features and n_dims as int)
               file           absolute path of the model file
               kernel_features, kernel_coefficients (float64 arrays),
               kernel_tau, close_pairs_weight, close_pairs_threshold
                              (float): the kernel part and the close-pair
                              settings, None if the model has none

    Example:
        model = dimpred.load_model("rn50x64_49d_ridge")
        print(model["info"]["network"], model["weights"].shape)  # RN50x64 (1024, 49)

    Hebartlab, 2026/09/30

    See also: list_models, predict
    """

    # A model that was already loaded
    if isinstance(model, dict):
        return model

    # Find the file: first among the shipped models, then as a path
    if model is None:
        from . import DEFAULT_MODEL  # read at each call, so that a changed dimpred.DEFAULT_MODEL is used
        model = DEFAULT_MODEL
    model = os.fspath(model)
    if model in list_models():
        fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model + ".mat")
    elif os.path.isfile(model):
        fname = os.path.abspath(model)
    else:
        raise ValueError(f"Unknown model '{model}': this is neither the name of a model that comes with dimpred "
                         f"nor an existing model file. Available models: {', '.join(list_models())}")

    # Load the file. simplify_cells turns the cell array of labels into
    # strings and the struct info into a dict, but it also removes singleton
    # dimensions, so 1 x n row vectors come back with shape (n,).
    data = scipy.io.loadmat(fname, simplify_cells=True)
    missing = [variable for variable in VARIABLES if variable not in data]
    if missing:
        raise ValueError(f"The model file {fname} has no variable {', '.join(missing)}. A model file needs the "
                         f"variables {', '.join(VARIABLES)} (see help(dimpred.load_model)).")

    weights = np.asarray(data["weights"], dtype=float)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)  # a model with one dimension, loadmat removed the second axis
    info = dict(data["info"])
    for field in ["n_features", "n_dims"]:
        if field in info:
            info[field] = int(info[field])  # saved as double in the file

    for variable in KERNEL_VARIABLES + CLOSE_PAIRS_VARIABLES:
        if variable in data and np.size(data[variable]) == 0:
            del data[variable]  # an empty variable (e.g. [] from a MATLAB struct) means: not there
    for group in [KERNEL_VARIABLES, CLOSE_PAIRS_VARIABLES]:
        present = [variable for variable in group if variable in data]
        if present and len(present) < len(group):
            raise ValueError(f"The model file {fname} has {', '.join(present)} but not "
                             f"{', '.join(v for v in group if v not in present)}. These variables go together.")

    model = {
        "weights": weights,
        "feature_mean": np.asarray(data["feature_mean"], dtype=float).ravel(),
        "feature_scale": np.asarray(data["feature_scale"], dtype=float).ravel(),
        "target_mean": np.asarray(data["target_mean"], dtype=float).ravel(),
        "labels": [str(label) for label in np.atleast_1d(data["labels"])],  # one label comes back as plain str
        "info": info,
        "file": fname,
        "kernel_features": None,
        "kernel_coefficients": None,
        "kernel_tau": None,
        "close_pairs_weight": None,
        "close_pairs_threshold": None,
    }
    if "kernel_features" in data:
        model["kernel_features"] = np.atleast_2d(np.asarray(data["kernel_features"], dtype=float))
        coefficients = np.asarray(data["kernel_coefficients"], dtype=float)
        model["kernel_coefficients"] = coefficients.reshape(-1, 1) if coefficients.ndim == 1 else coefficients
        model["kernel_tau"] = float(data["kernel_tau"])
    if "close_pairs_weight" in data:
        model["close_pairs_weight"] = float(data["close_pairs_weight"])
        model["close_pairs_threshold"] = float(data["close_pairs_threshold"])

    # Check that the sizes fit together
    n_features, n_dims = weights.shape
    problems = []
    if model["feature_mean"].size != n_features:
        problems.append(f"feature_mean has {model['feature_mean'].size} values")
    if model["feature_scale"].size != n_features:
        problems.append(f"feature_scale has {model['feature_scale'].size} values")
    if model["target_mean"].size != n_dims:
        problems.append(f"target_mean has {model['target_mean'].size} values")
    if len(model["labels"]) != n_dims:
        problems.append(f"there are {len(model['labels'])} labels")
    if info.get("n_features", n_features) != n_features:
        problems.append(f"info.n_features is {info['n_features']}")
    if info.get("n_dims", n_dims) != n_dims:
        problems.append(f"info.n_dims is {info['n_dims']}")
    if model["kernel_features"] is not None:
        if model["kernel_features"].shape[1] != n_features:
            problems.append(f"kernel_features has {model['kernel_features'].shape[1]} columns")
        if model["kernel_coefficients"].shape != (model["kernel_features"].shape[0], n_dims):
            problems.append(f"kernel_coefficients is {model['kernel_coefficients'].shape[0]} x "
                            f"{model['kernel_coefficients'].shape[1]} (needed: {model['kernel_features'].shape[0]} "
                            f"training images x {n_dims})")
        if not model["kernel_tau"] > 0:
            problems.append(f"kernel_tau is {model['kernel_tau']} (it has to be > 0)")
    if problems:
        raise ValueError(f"The model file {fname} is inconsistent: weights are {n_features} x {n_dims} "
                         f"(n_features x n_dims), but {', '.join(problems)}.")

    return model
