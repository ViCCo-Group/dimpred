#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Service module used to create single predicted embeddings for others.

For a specified use-case this module creates a predicted embedding. This
module provides a wrapper of functions from `fit.py` in order to
more conveniently create embeddings.

The functions in this module can be imported into other Python scripts.
Alternatively `get_predicted_spose_for()` can be used from the CLI.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import pickle
import sys

import joblib
import numpy as np
from utils.utils import determine_base_path

from dimpred.fit import (
    load_data_from,
    predict_spose_for_new_imgset_with,
    train_model_with,
)


def get_trained_model_for(model, module, n_dim, regularization):
    """Trains a dimpred model for a specific use-case.

    For a specified use-case, a dimpred model is created. This function
    is intended to more conveniently create a model that is not used further
    in the dimpred project (i.e., when other people want an embedding). The
    model can ater be used to predict SPoSE dimension values for new image sets.

    Parameters
    ----------
    model : str
        Name of the model which should serve as a basis for the predicted embedding.
    module : str
        Name of the specific module of the `model` which should serve as a basis for the
        predicted embedding.
    n_dim : int
        Indication of the dimensionality the embedding should have. Allowed are
        either 49 or 66.
    regularization : str
        Indication of the regularization paradigm that should be used when
        selecting the pre-created statistical model.

    Returns
    -------
    model : object
        One statistical model fitted separately for each of multiple targets
        with target-specific optimal hyperparameters.
    """
    base_path = determine_base_path()
    random_state = 0

    try:
        X = load_data_from(
            f"{base_path}/dimpred/data/raw/dnns/{model}/{module}/1854ref/features.txt"
        )
    except (FileNotFoundError, IOError, OSError):
        X = load_data_from(
            f"{base_path}/dimpred/data/raw/dnns/{model}/{module}/1854ref/features-srp.txt"
        )

    y = load_data_from(
        f"{base_path}/dimpred/data/raw/original_spose/embedding_{n_dim}d.txt"
    )

    fitted_model, alphas, l1_ratios = train_model_with(
        X, y, regularization, k_in=3, n_in=3, random_state=random_state
    )
    joblib.dump(
        value=fitted_model,
        filename=f"{base_path}/dimpred/data/interim/dimpred/model_{n_dim}d_{regularization}_{model}_{module}.joblib",
        compress=False,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    print("...Success! Model saved...")
    return fitted_model


def get_predicted_spose_for(model, module, imagesets, n_dim, regularization):
    """Create a predicted embedding for a specific use-case.

    For a specified use-case, a predicted embedding is created. This function
    is intended to more conveniently create embeddings that are not used further
    in the dimpred project (i.e., when other people want an embedding). Note
    that this function assumes that a trained statistical model for the requested
    model-module-n_dim-regularization combination has been created beforehand
    via `run_train.py`.

    Parameters
    ----------
    model : str
        Name of the model which should serve as a basis for the predicted embedding.
    module : str
        Name of the specific module of the `model` which should serve as a basis for the
        predicted embedding.
    imagesets : list
        A list of strings with names of image sets for which a predicted
        embedding should be created.
    n_dim : int
        Indication of the dimensionality the embedding should have. Allowed are
        either 49 or 66.
    regularization : str
        Indication of the regularization paradigm that should be used when
        selecting the pre-created statistical model. Can be either 'ridge' or
        'elastic'.
    """
    base_path = determine_base_path()

    X = {}
    for imageset in imagesets + ["1854ref"]:
        try:
            X[imageset] = load_data_from(
                f"{base_path}/dimpred/data/raw/dnns/{model}/{module}/{imageset}/features.txt"
            )
        except (FileNotFoundError, IOError, OSError):
            X[imageset] = load_data_from(
                f"{base_path}/dimpred/data/raw/dnns/{model}/{module}/{imageset}/features-srp.txt"
            )

    y = load_data_from(
        f"{base_path}/dimpred/data/raw/original_spose/embedding_{n_dim}d.txt"
    )

    try:
        fitted_model = joblib.load(
            f"{base_path}/dimpred/data/interim/dimpred/model_{n_dim}d_{regularization}_{model}_{module}.joblib"
        )
    except (FileNotFoundError, IOError, OSError):
        print("...no model found, fitting one...")
        fitted_model = get_trained_model_for(model, module, n_dim, regularization)
    print("...model loaded...")

    y_predicted = predict_spose_for_new_imgset_with(fitted_model, X, y)

    for imageset in imagesets:
        np.savetxt(
            fname=f"{base_path}/dimpred/data/interim/dimpred/predictions_{n_dim}d_{regularization}_{model}_{module}_{imageset}.txt",
            X=y_predicted[imageset],
        )
        print(f"...predictions for {imageset} saved...")

    print("!!! done !!!")


if __name__ == "__main__":
    print("START")
    model, module, imagesets, n_dim, regularization = sys.argv[1:6]
    get_predicted_spose_for(model, module, [imagesets], n_dim, regularization)
    print("END")
