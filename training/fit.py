#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core module for `dimpred`.

In this module, all the functions doing the hard work live.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import re

import numpy as np
import sklearn
from fracridge import fracridge
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# History:
# 2026/10/02: ridge_cv, a ridge regression that keeps the penalty chosen by
#   cross-validation, is now "ridge"; the fractional ridge of the DimPred
#   paper is now "fracridge"
# 2026/09/30: moved from dimpred/fit.py to training/; fracridge_cv with one
#   SVD per fold (the same weights, much faster); scikit-learn 1.7 and newer
# Philipp Kaniuth's code of the DimPred paper otherwise

# scikit-learn 1.7 replaced ElasticNetCV(n_alphas=...) by ElasticNetCV(alphas=<int>)
# and 1.9 removed n_alphas. Both give the same grid of alphas. Only the first
# two numbers of the version are used, so that e.g. "1.7rc1" works as well.
SKLEARN_NEW_ALPHAS = tuple(int(v) for v in re.findall(r"\d+", sklearn.__version__)[:2]) >= (1, 7)


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
    regularization : {"ridge", "fracridge", "elastic"}
        "ridge": ridge regression with the penalty chosen directly (ridge_cv,
        lambda from 1e-6 to 1e3, 8 values per decade). "fracridge": the
        fractional ridge regression of the DimPred paper (fracridge_cv, 70
        fractions from 0.1 to 1); until 2026/10/02 this was called "ridge".
        "elastic": elastic net (ElasticNetCV).
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
    model : object or ndarray
        One statistical model fitted separately for each of multiple targets
        with target-specific optimal hyperparameters. For "elastic" this is a
        fitted MultiOutputRegressor, for "ridge" and "fracridge" the weight
        matrix of shape (n_units, n_targets) (see ridge_cv and fracridge_cv).
    alphas : list, ndarray or None
        Selected alpha of each target: for "elastic" the alpha of
        ElasticNetCV, for "ridge" the penalty of the refit on the
        sum-of-squares scale (see ridge_cv), for "fracridge" None.
    l1_ratios : list or None
        Selected l1 ratio of each target for "elastic", else None.
    """
    cv = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    n_alphas = 10
    l1_ratio = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

    if regularization == "elastic":
        alphas = None
        if SKLEARN_NEW_ALPHAS:
            alpha_grid = dict(alphas=n_alphas)
        else:
            alpha_grid = dict(n_alphas=n_alphas, alphas=alphas)
        base = ElasticNetCV(
            l1_ratio=l1_ratio,
            **alpha_grid,
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
        # model is the weight matrix (n_units, n_targets), see ridge_cv.
        # lambda is the penalty per image, 8 values per decade from 1e-6 to
        # 1e3 (for the shipped models the selected values are 0.18 to 1.8,
        # far from both ends).
        model, _, alphas = ridge_cv(
            X_train_z,
            y_train_c,
            lambda_grid=np.logspace(-6, 3, 73),
            cv=cv,
        )
        l1_ratios = None
    elif regularization == "fracridge":
        # model is the weight matrix (n_units, n_targets), see fracridge_cv
        model, _ = fracridge_cv(
            X_train_z,
            y_train_c,
            frac_grid=np.linspace(0.1, 1, (n_alphas * len(l1_ratio))),
            cv=cv,
        )
        alphas, l1_ratios = None, None
    else:
        raise ValueError(
            f"regularization has to be 'ridge', 'fracridge' or 'elastic', not {regularization!r}"
        )
    print("...fitted model...")
    return model, alphas, l1_ratios


def ridge_cv(X_train_z, y_train_c, lambda_grid, cv):
    """Ridge regression with a cross-validated penalty per target.

    This is the ridge regression of "ridge" since 2026/10/02. It replaces the
    fractional ridge regression (fracridge_cv) for new models, because
    fracridge_cv does not refit with the penalty it selected: it selects a
    fraction in the inner folds and refits on all data with the same
    fraction. But the penalty (alpha) that gives a fraction depends on the
    data, and for all data it is very different from the one in the inner
    folds. With the 70 fractions of fit_model_with, the refit of RN50x64 for
    the 66d embedding used a median of 9.4 times the alpha of the inner
    folds (4.1 to 110 times across dimensions), the refit of CLIP ViT-B/32 a
    median of 0.02 times. So the refit shrinks much more or much less than
    the inner cross-validation chose. Here the penalty itself is selected
    and kept.

    The penalty is on a per-image scale: for a lambda in lambda_grid, ridge
    minimizes the mean squared error + lambda * ||beta||^2, i.e. the sum of
    squares + alpha * ||beta||^2 with alpha = n * lambda, where n is the
    number of training images. In each inner fold, alpha = n_train * lambda
    with the training images of that fold. As in GridSearchCV, the lambda
    with the highest mean R^2 across folds is selected for each target (the
    first one in case of ties). The model is then refit on all data with
    the alpha of the inner folds, alpha = mean n_train * lambda (e.g. 1236 *
    lambda for 1854 images and 3 folds), so that the refit uses the penalty
    that was chosen. Refitting with alpha = 1854 * lambda instead (the
    convention of ElasticNetCV) shrank more and predicted worse, both per
    dimension and on new images.

    As in fracridge_cv, all targets and penalties are solved from one SVD
    per fold. With X = U S V', the weights are
    beta = V diag(s / (s^2 + alpha)) U' y, and the held-out predictions of
    all penalties and targets come from one matrix product. There is no
    intercept, because X_train_z and y_train_c are z-scored and centered
    (as for fracridge_cv). The weights are those of scikit-learn's
    Ridge(alpha=alpha, fit_intercept=False) on the same data.

    Parameters
    ----------
    X_train_z : ndarray
        Column-wise z-transformed training predictor matrix.
    y_train_c : ndarray
        Column-wise centered targets, shape (n_images, n_targets).
    lambda_grid : array-like
        Penalties per image to test (lambda, see above).
    cv : cross-validation generator
        E.g. RepeatedKFold. Only its split() method is used.

    Returns
    -------
    coef : ndarray
        Regression weights, shape (n_units, n_targets).
    best_lambda : ndarray
        Selected lambda for each target, shape (n_targets,).
    alpha : ndarray
        Penalty of the refit for each target (mean n_train * best_lambda),
        shape (n_targets,).
    """
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    n_targets, n_lambdas = y_train_c.shape[1], len(lambda_grid)
    scores = np.zeros((n_lambdas, n_targets))
    n_train = []
    for train, test in cv.split(X_train_z):
        U, s, Vt = np.linalg.svd(X_train_z[train], full_matrices=False)
        UTy = U.T @ y_train_c[train]
        # shrinkage of each singular value for all lambdas: (n_lambdas, n_singular_values)
        d = s[None, :] / (s[None, :] ** 2 + len(train) * lambda_grid[:, None])
        # weights of all lambdas and targets in the basis V (shrunk U' y),
        # then the predictions of all of them at once: (n_test, n_lambdas, n_targets)
        UTy_shrunk = (d.T[:, :, None] * UTy[:, None, :]).reshape(len(s), n_lambdas * n_targets)
        y_hat = ((X_train_z[test] @ Vt.T) @ UTy_shrunk).reshape(len(test), n_lambdas, n_targets)
        y_true = y_train_c[test]
        ss_res = ((y_true[:, None, :] - y_hat) ** 2).sum(axis=0)
        ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
        scores += 1 - ss_res / ss_tot  # R^2, as sklearn's default score
        n_train.append(len(train))
    best = np.argmax(scores / len(n_train), axis=0)
    best_lambda = lambda_grid[best]
    # refit on all data with the alpha of the inner folds
    alpha = np.mean(n_train) * best_lambda
    U, s, Vt = np.linalg.svd(X_train_z, full_matrices=False)
    coef = Vt.T @ (s[:, None] / (s[:, None] ** 2 + alpha[None, :]) * (U.T @ y_train_c))
    return coef, best_lambda, alpha


def fracridge_cv(X_train_z, y_train_c, frac_grid, cv):
    """Fractional ridge regression with a cross-validated fraction per target.

    This is the ridge regression of the DimPred paper ("fracridge", until
    2026/10/02 called "ridge"). For new models use ridge_cv, which keeps the
    selected penalty at the refit (see there).

    This gives the same result as
    ``MultiOutputRegressor(FracRidgeRegressorCV()).fit(X, y, frac_grid=...)``,
    which is what this module used before, if cv has an integer random_state
    (with random_state=None the old code drew different folds for each
    target, here all targets use the same folds). It is much faster: the old
    version passed each target separately to sklearn's GridSearchCV, which
    fits the model once per fraction, fold and target, and every fit computes
    a new SVD of the same X. `fracridge` itself can solve all targets and all
    fractions from a single SVD, so here we only need one SVD per fold (e.g.
    10 instead of about 42,000 for 66 targets, 70 fractions and 3x3 folds).
    The held-out predictions for all fractions and targets then come from one
    matrix product.

    As in GridSearchCV, the fraction with the highest mean R^2 across folds is
    selected for each target (the first one in case of ties), and the model is
    then refit on all data with this fraction.

    Parameters
    ----------
    X_train_z : ndarray
        Column-wise z-transformed training predictor matrix.
    y_train_c : ndarray
        Column-wise centered targets, shape (n_images, n_targets).
    frac_grid : array-like
        Fractions to test.
    cv : cross-validation generator
        E.g. RepeatedKFold. Only its split() method is used.

    Returns
    -------
    coef : ndarray
        Regression weights, shape (n_units, n_targets).
    best_frac : ndarray
        Selected fraction for each target, shape (n_targets,).
    """
    frac_grid = np.asarray(frac_grid, dtype=float)
    n_units, n_targets, n_fracs = X_train_z.shape[1], y_train_c.shape[1], len(frac_grid)
    scores = np.zeros((n_fracs, n_targets))
    n_folds = 0
    for train, test in cv.split(X_train_z):
        coef, _ = fracridge(X_train_z[train], y_train_c[train], fracs=frac_grid, tol=1e-10, jit=True)
        coef = coef.reshape(n_units, n_fracs, n_targets)
        # predictions of all fractions and targets at once: (n_test, n_fracs, n_targets)
        y_hat = np.einsum("ip,pft->ift", X_train_z[test], coef)
        y_true = y_train_c[test]
        ss_res = ((y_true[:, None, :] - y_hat) ** 2).sum(axis=0)
        ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
        scores += 1 - ss_res / ss_tot  # R^2, as sklearn's default score
        n_folds += 1
    best = np.argmax(scores / n_folds, axis=0)
    coef, _ = fracridge(X_train_z, y_train_c, fracs=frac_grid, tol=1e-10, jit=True)
    coef = coef.reshape(n_units, n_fracs, n_targets)[:, best, np.arange(n_targets)]
    return coef, frac_grid[best]


def get_predictions_for(model, X_test_z, y_train_mean):
    """Compute predictions for each target.

    Applies a fitted statistical models to `X_test` to receive predictions for
    each of multiple targets. Negative predictions are set to 0.

    Parameters
    ----------
    model : object or ndarray
        A fitted MultiOutputRegressor for "elastic", the weight matrix
        (n_units, n_targets) from ridge_cv for "ridge" or from fracridge_cv
        for "fracridge". Ridge models saved by the earlier version of this
        module (e.g. the models on OSF) are MultiOutputRegressor objects as
        well and work too.
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
    if isinstance(model, np.ndarray):  # ridge, fracridge: weight matrix
        y_predicted = X_test_z @ model + y_train_mean
    else:
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
    regularization : {"ridge", "fracridge", "elastic"}
        Denotes which regularization scheme shall be used (see
        fit_model_with).
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
    regularization : {"ridge", "fracridge", "elastic"}
        Denotes which regularization scheme shall be used (see
        fit_model_with).
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
    fitted_model : object or ndarray
        A fitted MultiOutputRegressor for "elastic", the weight matrix
        (n_units, n_targets) for "ridge" and "fracridge".
        Can be saved for later use.
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
    fitted_model : object or ndarray
        A fitted MultiOutputRegressor for "elastic", the weight matrix
        (n_units, n_targets) for "ridge" and "fracridge" (see
        get_predictions_for).
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
