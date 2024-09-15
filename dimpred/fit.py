#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core module for `dimpred`.

In this module, all the functions doing the hard work live.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import numpy as np
from fracridge import FracRidgeRegressorCV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


def load_data_from(path):
    """Load data from disk and returns it in Fortran order.

    Parameters
    ----------
    path : str
        Path to the file that shall be loaded.

    Returns
    -------
    Data : nd.array
        Loaded data in Fortran order.
    """
    data = np.asfortranarray(np.loadtxt(path))
    return data


def preprocess_data(X_train, y_train):
    """Compute column-wise transformed versions of variables.

    Parameters
    ----------
    X_train : ndarray
        Predictor matrix of the training fold.
    y_train : ndarray
        Target of the training fold.

    Returns
    -------
    X_train_z : ndarray
        Column-wise z-transformed version of `X_train`.
    y_train_c : ndarray
        Column-wise centered version of `y_train`.
    X_standardizer : ndarray
        Standardizer that can be used later for transforming test predictor
        matrices with X_train's statistics.
    y_train_mean : ndarray
        Original mean of each column of `y_train`.
    """
    X_standardizer = StandardScaler(copy=False, with_mean=True, with_std=True)
    X_train_z = X_standardizer.fit_transform(X_train)
    center = StandardScaler(copy=False, with_mean=True, with_std=False)
    y_train_c = center.fit_transform(y_train)
    y_train_mean = center.mean_
    return X_train_z, y_train_c, X_standardizer, y_train_mean


def fit_model_with(
    X_train_z, y_train_c, regularization, n_splits=5, n_repeats=5, random_state=None
):
    """Find best set of hyperparameters for each target.

    Parameters
    ----------
    X_train_z : ndarray
        Column-wise z-transformed training predictor matrix.
    y_train_c : ndarray
        Column-wise centered targets.
    n_splits : int
        Determines the number of folds for the inner cross-validation, defaults
        to 5.
    n_repeats : int
        Determines how often the inner k-fold cross-validation shall be
        repeated, defaults to 5.
    random_state : int
        Sets the random_state. The default is None.

    Returns
    -------
    model : object
        One statistical model fitted separately for each of multiple targets
        with target-specific optimal hyperparameters.
    """
    cv = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    n_alphas = 10
    l1_ratio = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

    if regularization == "elastic":
        alphas = None
        base = ElasticNetCV(
            l1_ratio=l1_ratio,
            n_alphas=n_alphas,
            alphas=alphas,
            cv=cv,
            random_state=random_state,
            fit_intercept=False,
            max_iter=1000,
            n_jobs=-1,
        )
        model = MultiOutputRegressor(base, n_jobs=1)
        model.fit(X_train_z, y_train_c)
        alphas = [model.estimators_[i].alpha_ for i in range(len(model.estimators_))]
        l1_ratios = [
            model.estimators_[i].l1_ratio_ for i in range(len(model.estimators_))
        ]
    elif regularization == "ridge":
        base = FracRidgeRegressorCV(
            fit_intercept=False,
            normalize=False,
            copy_X=True,
            tol=1e-10,
            jit=True,
            cv=cv,
            scoring=None,
        )
        model = MultiOutputRegressor(base, n_jobs=-1)
        model.fit(
            X_train_z,
            y_train_c,
            frac_grid=np.linspace(0.1, 1, (n_alphas * len(l1_ratio))),
        )
        alphas, l1_ratios = None, None
    print("...fitted model...")
    return model, alphas, l1_ratios


def get_predictions_for(model, X_test_z, y_train_mean):
    """Compute predictions for each target.

    Applies a fitted statistical models to `X_test` to receive predictions for
    each of multiple targets. Negative predictions are set to 0.

    Parameters
    ----------
    model : object
        One statistical model fitted separately for each of multiple targets
        with target-specific optimal hyperparameters.
    X_test_z : nd.array
        Test predictor matrix that had been z-transformed column-wise with
        X_train's standardizer.
    y_train_mean : nd.array
        Original mean of each column of `y_train`.

    Returns
    -------
    y_predicted : nd.array
        Predictions for each target.
    """
    y_predicted = model.predict(X_test_z) + y_train_mean
    y_predicted[y_predicted < 0] = 0
    return y_predicted


def predict_spose_for_1854ref_with(
    X, y, regularization, k_out=2, k_in=2, n_in=1, random_state=None
):
    """Predict dimension values for the 1854 reference image set.

    For a given module of a specified deep neural network architecture, a
    regularized regression model is fitted to iteratively predict the values
    of the 1854 THINGS reference images on the SPoSE dimensions.

    X and y are itereatively split in train and test splits in an outer cross-
    validation. In each outer CV, the best hyperparamter set is determined
    in an inner CV on the outer training data. Using these best hyperparameters
    the statistical model is fitted on the outer training data.
    Finally, predicted dimension values are derived for the outer test images.

    Parameters
    ----------
    X : ndarray
        Image activations of a specific deep neural network model's module.
        Expected shape is (n_images, n_units).
    y : ndarray
        Ground truth SPoSE embedding of the 1854 reference images.
        Expected shape is (n_images, n_dims).
    regularization : {"ridge", "elastic"}
        Denotes which regularization scheme shall be used.
    k_out : int, optional
        Determines the number of folds for the outer cross-validation. The
        default is 2.
    k_in : int, optional
        Determines the number of folds for the inner cross-validation.
        The default is 2.
    n_in : int, optional
        Determines how often the inner k-fold cross-validation shall be
        repeated. The default is 1.
    random_state : int, optional
        Sets the random_state. The default is None.

    Returns
    -------
    y_predicted : ndarray
        Predicted values on all SPoSE dimensions for all 1854 reference images.
        Shape is (n_images, n_dims).
    """
    n_out = 1
    cv = RepeatedKFold(n_splits=k_out, n_repeats=n_out, random_state=random_state)
    n_objects = y.shape[0]
    n_dim = y.shape[1]
    y_predicted = np.zeros((n_objects, n_dim))
    alphas = np.zeros((k_out * n_out, n_dim))
    l1_ratios = np.zeros((k_out * n_out, n_dim))
    i = -1
    for train, test in cv.split(range(n_objects)):
        i += 1
        X_train_z, y_train_c, X_standardizer, y_train_mean = preprocess_data(
            X[train, :], y[train, :]
        )
        X_test_z = X_standardizer.transform(X[test, :])
        fitted_model, alphas[i, :], l1_ratios[i, :] = fit_model_with(
            X_train_z,
            y_train_c,
            regularization,
            n_splits=k_in,
            n_repeats=n_in,
            random_state=random_state,
        )
        y_predicted[test, :] = get_predictions_for(fitted_model, X_test_z, y_train_mean)
        print(f"...received predictions for fold {i} of {k_out}...")
    return y_predicted, alphas, l1_ratios


def train_model_with(X, y, regularization, k_in=2, n_in=1, random_state=None):
    """Trains a statistical model on the 1854 reference image set.

    For a given module of a specified deep neural network architecture,
    a regularized regression model is fitted to the data of the 1854
    THINGS reference images. This model can later be used to predict
    SPoSE dimension values for new image sets.

    Parameters
    ----------
    X : ndarray
        Image activations of a specific deep neural network model's module
        for the 1854 THINGS reference images.
        Expected shape is (n_images, n_units).
    y : ndarray
        Ground truth SPoSE embedding of the 1854 reference images.
        Expected shape is (n_images, n_dims).
    regularization : {"ridge", "elastic"}
        Denotes which regularization scheme shall be used.
    k_in : int
        Determines the number of folds for the inner cross-validation in
        which the best hyperparameter is determined.
        The default is 2.
    n_in : int, optional
        Determines how often the inner k-fold cross-validation shall be
        repeated. The default is 1.
    random_state : int, optional
        Sets the random_state. The default is None.

    Returns
    -------
    fitted_model : object
        Fitted statistical model. Can be saved for later use.
    """
    X_train_z, y_train_c, X_standardizer, y_train_mean = preprocess_data(X, y)
    fitted_model, alphas, l1_ratios = fit_model_with(
        X_train_z,
        y_train_c,
        regularization,
        n_splits=k_in,
        n_repeats=n_in,
        random_state=random_state,
    )
    print("...received model...")
    return fitted_model, alphas, l1_ratios


def predict_spose_for_new_imgset_with(fitted_model, X, y):
    """Predict dimension values for new image sets.

    Based on a fitted statistical model, SPoSE dimension values for
    new image sets for a given module of a specific deep neural network
    architecture are predicted.

    Parameters
    ----------
    fitted_model : object
        Fitted statistical model.
    X : dict
        Each key holds image activations of a specific deep neural
        network model's module for a different image set. Each value is
        an ndarray with expected shape of (n_images, n_units).
    y : ndarray
        Ground truth SPoSE embedding of the 1854 reference images.
        Expected shape is (n_images, n_dims).

    Returns
    -------
    y_predicted : dict
        Each key denotes one imageset. Each value is an ndarray holding
        the predicted values on all SPoSE dimensions for the imageset,
        with shape (n_images, n_dims).
    """

    y_predicted = {}
    _, _, X_standardizer, y_train_mean = preprocess_data(X["1854ref"], y)
    for imageset in X.keys():
        if imageset == "1854ref":
            continue
        X_test_z = X_standardizer.transform(X[imageset])
        y_predicted[imageset] = get_predictions_for(
            fitted_model, X_test_z, y_train_mean
        )
        print(f"...received predictions for {imageset}...")
    return y_predicted
