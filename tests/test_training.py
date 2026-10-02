"""
Tests of Philipp Kaniuth's training code (training/fit.py) and of the part of
training/build_models.py that turns a fit into a model file.

fit.py has two ridge regressions. "ridge" (ridge_cv) chooses the penalty
itself in the inner cross-validation and refits with the same penalty. We
check it against scikit-learn's Ridge, fitted in loops over targets, penalties
and folds: the closed form has to give the weights of Ridge at the chosen
penalty, the penalty has to be chosen from the held-out scores of the inner
folds, and the refit on all data has to use the penalty of the inner folds
(alpha = mean training size of the inner folds x lambda), not alpha = n x
lambda with all n images.

"fracridge" (fracridge_cv) is Philipp's fractional ridge regression of the
DimPred paper. His original code fitted it with
MultiOutputRegressor(FracRidgeRegressorCV()), which fits fracridge once per
target, fraction and fold. fit.py now does the same with one fit per fold
(fracridge_cv). We check that this gives the same result as the slow way:
    - against a brute-force version written here (loops over targets,
      fractions and folds) on small random problems
    - against Philipp's original code itself on a small problem, where it
      still runs (scikit-learn up to about 1.3; skipped otherwise)
    - against the weights of Philipp's original code for RN50x64 and the
      66d embedding (stored in the fixtures; until 2026/10/02 this was the
      shipped rn50x64_66d_ridge model). These last tests need the RN50x64
      features of the 1854 reference images, which are not part of the
      repository; set DIMPRED_REFERENCE_FEATURES to the file
      (features_RN50x64.npy, as written by training/build_models.py
      --features) or to the folder that contains it, otherwise they are
      skipped. With the folder, the shipped models rn50x64_66d_ridge and
      alignet_siglip2b_66d_ridge are also fitted again and compared with the
      model files (features_AligNet-SigLIP2-B.npy for the second).
The elastic net branch has to run on the installed scikit-learn (1.7 renamed
n_alphas, 1.9 removed it) and give the coefficients of scikit-learn 1.3.
build_models.py has to save the statistics of the raw features, although
preprocess_data z-scores its input in place, it has to fit with the settings
of call.py (3-fold cross-validation repeated 3 times, random_state 0), it
has to pass the images in the order of data/reference_image_names.txt, which
is the order of the rows of the embedding, it has to fit each model with its
regression (ridge, fracridge or elastic), and it has to extract the features
on the cpu unless another device is given. call.py has to say which ridge it
fits with "ridge".

The whole file is skipped if scikit-learn or fracridge are not installed
(they are only needed for training, see training/environment.yml). The
training data themselves are tested in test_training_data.py, which always runs.

Hebartlab, 2026/09/30

See also: ../training/fit.py, ../training/build_models.py, test_training_data.py
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="the training tests need scikit-learn (training/environment.yml)")
pytest.importorskip("fracridge", reason="the training tests need fracridge (training/environment.yml)")

import sklearn  # noqa: E402
from fracridge import fracridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import KFold, PredefinedSplit, RepeatedKFold  # noqa: E402

from helpers import (MODELS_DIR, ORIGINAL_RIDGE_FILE, TOL_BEST_FRAC, TOL_ORIGINAL_WEIGHTS, TRAINING,  # noqa: E402
                     TRAINING_DATA, assert_close, load_mat, read_lines)

# The training code is not part of the dimpred package. build_models.py
# imports fit.py by its name, so the training folder has to be on the path.
sys.path.insert(0, TRAINING)
import build_models  # noqa: E402
import fit  # noqa: E402

# History:
# 2026/10/02: build_models.py extracts on the cpu by default; call.py says
#   which ridge "ridge" is (after the review)
# 2026/10/02: tests of the new ridge (ridge_cv), which is now "ridge"; the
#   fractional ridge of the paper is now "fracridge"; refit of the shipped
#   rn50x64_66d_ridge and alignet_siglip2b_66d_ridge from cached features
# 2026/09/30: tests of the choice between equally good fractions and of the
#   order in which build_models.py pairs the images with the embedding rows
#   (mistakes in both were not found by the other tests)
# 2026/09/30: comparison with Philipp's original FracRidgeRegressorCV where it
#   runs; the elastic net selects different l1 ratios and is compared with
#   the coefficients of scikit-learn 1.3; tests of build_models.fit_model,
#   of preprocess_data changing its input, and of the two prediction
#   functions of fit.py
# 2026/09/30: written together with the tests, before the package code

# Philipp's settings (fit.py, call.py): 70 fractions from 0.1 to 1,
# 7 l1 ratios with 10 alphas each for the elastic net
FRACTIONS = np.linspace(0.1, 1, 70)
L1_RATIOS = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

# Penalties of the new ridge (fit.py): lambda on a per-image scale, 8 values
# per decade from 1e-6 to 1e3. The tests against scikit-learn use a shorter
# grid, because they fit Ridge once per lambda, fold and target.
LAMBDAS = np.logspace(-6, 3, 73)
SHORT_LAMBDAS = np.logspace(-4, 2, 13)


def fracridge_cv_brute_force(X, y, frac_grid, cv):
    """The slow way, as in Philipp's original code.

    MultiOutputRegressor(FracRidgeRegressorCV()) runs a grid search for each
    target: fracridge is fitted for each fraction on each training fold and
    scored with R^2 on the held-out fold. The fraction with the highest mean
    score is selected (the first one if several are equally good), and the
    model is fitted again on all data with this fraction.
    """

    folds = list(cv.split(X))
    n_units, n_targets = X.shape[1], y.shape[1]
    coef = np.zeros((n_units, n_targets))
    best_frac = np.zeros(n_targets)
    for target in range(n_targets):
        mean_scores = []
        for frac in frac_grid:
            scores = []
            for train, test in folds:
                w, _ = fracridge(X[train], y[train, target], fracs=frac, tol=1e-10, jit=False)
                scores.append(r2_score(y[test, target], X[test] @ w))
            mean_scores.append(np.mean(scores))
        best_frac[target] = frac_grid[int(np.argmax(mean_scores))]
        w, _ = fracridge(X, y[:, target], fracs=best_frac[target], tol=1e-10, jit=False)
        coef[:, target] = w
    return coef, best_frac


def small_problem(seed=0, n=60, n_units=8, noise=(0.2, 1.0, 3.0, 10.0)):
    """Random z-scored predictors and centered targets with different amounts of noise.

    With little noise the best fraction is close to 1, with a lot of noise it
    is small, so the targets need different fractions.
    """

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_units))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = np.stack([X @ rng.standard_normal(n_units) + s * rng.standard_normal(n) for s in noise], axis=1)
    return X, y - y.mean(axis=0)


def raw_problem(seed=0, n=60, n_units=8):
    """As small_problem, but with a different mean and scale for each predictor and target."""

    X, y = small_problem(seed, n, n_units)
    rng = np.random.default_rng(seed + 100)
    X = X * rng.uniform(0.5, 3, n_units) + rng.uniform(-2, 5, n_units)
    return X, y + rng.uniform(0.5, 2, y.shape[1])


def z_score(X, X_train=None):
    """z-score X with the mean and std (ddof=0) of X_train (default: X itself)."""

    X_train = X if X_train is None else X_train
    return (X - X_train.mean(axis=0)) / X_train.std(axis=0)


# --- ridge_cv (the new ridge, regularization "ridge")

def ridge_cv_brute_force(X, y, lambda_grid, cv):
    """The new ridge the slow way, with scikit-learn's Ridge.

    For each target and lambda, Ridge (no intercept) is fitted on each inner
    training fold with alpha = n_train x lambda (n_train: the number of
    training images of this fold) and scored with R^2 on the held-out fold.
    The lambda with the highest mean score is selected (the first one if
    several are equally good), and Ridge is fitted again on all data with
    the alpha of the inner folds, alpha = mean n_train x lambda.
    """

    from sklearn.linear_model import Ridge

    folds = list(cv.split(X))
    n_inner = np.mean([len(train) for train, _ in folds])
    n_units, n_targets = X.shape[1], y.shape[1]
    coef = np.zeros((n_units, n_targets))
    best_lambda = np.zeros(n_targets)
    for target in range(n_targets):
        mean_scores = []
        for lam in lambda_grid:
            scores = []
            for train, test in folds:
                ridge = Ridge(alpha=len(train) * lam, fit_intercept=False).fit(X[train], y[train, target])
                scores.append(r2_score(y[test, target], ridge.predict(X[test])))
            mean_scores.append(np.mean(scores))
        best_lambda[target] = lambda_grid[int(np.argmax(mean_scores))]
        coef[:, target] = Ridge(alpha=n_inner * best_lambda[target], fit_intercept=False).fit(X, y[:, target]).coef_
    return coef, best_lambda, n_inner * best_lambda


def assert_close_relative(actual, expected, rtol, what):
    """As assert_close, with the tolerance relative to the largest absolute value of expected."""

    assert_close(actual, expected, rtol * np.abs(expected).max(), what)


@pytest.fixture(scope="module")
def ridge_and_brute_force():
    """ridge_cv and the brute force version on the same small problem.

    59 images, so the training folds have 39 and 40 images; 3 folds
    repeated 2 times.
    """

    X, y = small_problem()
    X, y = X[:59], y[:59]  # 59 images: training folds of 39 and 40 images
    cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    return X, y, cv, fit.ridge_cv(X, y, SHORT_LAMBDAS, cv), ridge_cv_brute_force(X, y, SHORT_LAMBDAS, cv)


def test_ridge_cv_selects_the_lambdas_of_the_brute_force_version(ridge_and_brute_force):
    # the selection uses the held-out R^2 of the inner folds, each fold with
    # its own alpha = n_train x lambda
    _, _, _, (_, best_lambda, _), (_, best_lambda_slow, _) = ridge_and_brute_force
    assert len(np.unique(best_lambda_slow)) > 1, "test problem too easy: all targets select the same lambda"
    np.testing.assert_array_equal(best_lambda, best_lambda_slow, err_msg="selected lambdas differ")


def test_ridge_cv_gives_the_weights_of_the_brute_force_version(ridge_and_brute_force):
    _, _, _, (coef, _, _), (coef_slow, _, _) = ridge_and_brute_force
    assert_close_relative(coef, coef_slow, 1e-10, "ridge_cv vs brute force weights")


def test_ridge_cv_weights_are_scikit_learn_ridge_at_the_chosen_alpha(ridge_and_brute_force):
    # the closed form from one SVD gives the weights of Ridge with the alpha
    # that ridge_cv returns, for every target
    from sklearn.linear_model import Ridge

    X, y, _, (coef, _, alpha), _ = ridge_and_brute_force
    for target in range(y.shape[1]):
        expected = Ridge(alpha=alpha[target], fit_intercept=False, solver="svd").fit(X, y[:, target]).coef_
        assert_close_relative(coef[:, target], expected, 1e-10, f"ridge_cv vs Ridge(alpha={alpha[target]:.4g})")


def test_ridge_cv_refit_uses_the_alpha_of_the_inner_folds(ridge_and_brute_force):
    # The refit penalizes as in the inner folds: alpha = mean training size
    # of the inner folds x lambda (here 39.33 x lambda), not n x lambda with
    # all 59 images, which would shrink more than the cross-validation chose.
    from sklearn.linear_model import Ridge

    X, y, cv, (coef, best_lambda, alpha), _ = ridge_and_brute_force
    n_train = [len(train) for train, _ in cv.split(X)]
    assert len(set(n_train)) > 1, "test problem should have training folds of different sizes"
    assert_close(alpha, np.mean(n_train) * best_lambda, 1e-12, "refit alpha vs mean inner training size x lambda")
    all_images = Ridge(alpha=len(X) * best_lambda[0], fit_intercept=False).fit(X, y[:, 0]).coef_
    assert np.abs(coef[:, 0] - all_images).max() > 1e-6, "the refit used alpha = n x lambda with all images"


def test_ridge_cv_chooses_lambda_on_held_out_images_only():
    # Without noise, a fit on the training images is always best with the
    # smallest penalty. With a lot of noise, the held-out images need a large
    # penalty, which only a selection on held-out images finds.
    rng = np.random.default_rng(11)
    X = z_score(rng.standard_normal((60, 20)))
    y = np.stack([X @ rng.standard_normal(20), 0.05 * X[:, 0] + 5 * rng.standard_normal(60)], axis=1)
    _, best_lambda, _ = fit.ridge_cv(X, y - y.mean(axis=0), SHORT_LAMBDAS, KFold(n_splits=3, shuffle=True,
                                                                                    random_state=0))
    assert best_lambda[0] == SHORT_LAMBDAS[0], f"target without noise selected lambda {best_lambda[0]:g}"
    assert best_lambda[1] >= 1, f"pure noise selected lambda {best_lambda[1]:g}, expected a large penalty"


def test_ridge_cv_selects_the_first_lambda_if_several_are_equally_good():
    # as in test_fracridge_cv_selects_the_first_fraction_if_several_are_equally_good:
    # held-out images with all features 0 give the same R^2 for every lambda
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 8))
    y = X @ rng.standard_normal((8, 3)) + rng.standard_normal((60, 3))
    X[40:] = 0
    cv = PredefinedSplit(np.r_[np.full(40, -1), np.zeros(20, dtype=int)])  # train on 0-39, test on 40-59
    _, best_lambda, _ = fit.ridge_cv(X, y, SHORT_LAMBDAS, cv)
    np.testing.assert_array_equal(best_lambda, np.full(3, SHORT_LAMBDAS[0]),
                                  err_msg="with equal scores, the first lambda of the grid should be selected")


def test_ridge_cv_output_shapes_and_lambdas_from_the_grid():
    X, y = small_problem(seed=2)
    coef, best_lambda, alpha = fit.ridge_cv(X, y, SHORT_LAMBDAS, KFold(n_splits=3))
    assert coef.shape == (8, 4), f"weights have shape {coef.shape}, expected (8, 4)"
    assert best_lambda.shape == (4,) and alpha.shape == (4,), f"shapes {best_lambda.shape}, {alpha.shape}"
    assert all(lam in SHORT_LAMBDAS for lam in best_lambda), f"best_lambda {best_lambda} contains values not in the grid"


def test_ridge_branch_uses_ridge_cv_with_the_lambda_grid():
    # fit_model_with(..., "ridge"): lambdas from 1e-6 to 1e3 (8 per decade),
    # repeated k-fold; the returned alphas are those of the refit
    X, y = small_problem(seed=3)
    weights, alphas, l1_ratios = fit.fit_model_with(X, y, "ridge", n_splits=3, n_repeats=2, random_state=0)
    coef, _, alpha = fit.ridge_cv(X, y, LAMBDAS, RepeatedKFold(n_splits=3, n_repeats=2, random_state=0))
    assert_close(weights, coef, 1e-12, "ridge weights of fit_model_with vs ridge_cv")
    assert_close(alphas, alpha, 1e-12, "alphas returned by fit_model_with vs ridge_cv")
    assert l1_ratios is None, f"l1_ratios should be None for ridge, not {l1_ratios}"


def test_train_model_with_ridge_equals_brute_force_on_raw_data():
    # train_model_with z-scores the features and centers the targets first
    X, y = raw_problem(seed=7)
    weights, _, _ = fit.train_model_with(X.copy(), y.copy(), "ridge", k_in=3, n_in=2, random_state=0)
    coef_slow, _, _ = ridge_cv_brute_force(z_score(X), y - y.mean(axis=0), LAMBDAS,
                                           RepeatedKFold(n_splits=3, n_repeats=2, random_state=0))
    assert_close_relative(weights, coef_slow, 1e-9, "ridge weights of train_model_with vs brute force")


def test_unknown_regularization_gives_error():
    X, y = small_problem(seed=3)
    with pytest.raises(ValueError, match="fracridge"):
        fit.fit_model_with(X, y, "lasso", n_splits=3, n_repeats=1, random_state=0)


# --- fracridge_cv (Philipp's fractional ridge, regularization "fracridge")

def test_fracridge_cv_equals_brute_force():
    X, y = small_problem()
    frac_grid = np.linspace(0.1, 1, 10)
    cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    coef, best_frac = fit.fracridge_cv(X, y, frac_grid, cv)
    coef_slow, best_frac_slow = fracridge_cv_brute_force(X, y, frac_grid, cv)
    assert len(np.unique(best_frac_slow)) > 1, "test problem too easy: all targets select the same fraction"
    np.testing.assert_array_equal(best_frac, best_frac_slow, err_msg="selected fractions differ")
    assert_close(coef, coef_slow, 1e-10, "fracridge_cv vs brute force weights")


def test_fracridge_cv_equals_brute_force_with_plain_kfold():
    # fracridge_cv only uses cv.split, so any sklearn splitter works
    X, y = small_problem(seed=1)
    frac_grid = np.linspace(0.1, 1, 10)
    cv = KFold(n_splits=4, shuffle=True, random_state=1)
    coef, best_frac = fit.fracridge_cv(X, y, frac_grid, cv)
    coef_slow, best_frac_slow = fracridge_cv_brute_force(X, y, frac_grid, cv)
    np.testing.assert_array_equal(best_frac, best_frac_slow, err_msg="selected fractions differ")
    assert_close(coef, coef_slow, 1e-10, "fracridge_cv vs brute force weights (KFold)")


def test_fracridge_cv_output_shapes():
    X, y = small_problem(seed=2)
    coef, best_frac = fit.fracridge_cv(X, y, np.linspace(0.1, 1, 10), KFold(n_splits=3))
    assert coef.shape == (8, 4), f"weights have shape {coef.shape}, expected (8, 4)"
    assert best_frac.shape == (4,), f"best_frac has shape {best_frac.shape}, expected (4,)"


def test_fracridge_cv_selects_fractions_from_the_grid():
    X, y = small_problem(seed=2)
    frac_grid = np.linspace(0.1, 1, 10)
    _, best_frac = fit.fracridge_cv(X, y, frac_grid, KFold(n_splits=3))
    assert all(f in frac_grid for f in best_frac), f"best_frac {best_frac} contains values not in the grid"


def test_fracridge_cv_selects_the_first_fraction_if_several_are_equally_good():
    # GridSearchCV, which Philipp's original code used, takes the first of
    # several equally good fractions. Random data never give exactly equal
    # scores, so we force them: one split whose held-out images have all
    # features 0, so that every fraction predicts exactly 0 there and gets
    # the same R^2.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 8))
    y = X @ rng.standard_normal((8, 3)) + rng.standard_normal((60, 3))
    X[40:] = 0
    cv = PredefinedSplit(np.r_[np.full(40, -1), np.zeros(20, dtype=int)])  # train on 0-39, test on 40-59
    frac_grid = np.linspace(0.1, 1, 10)
    _, best_frac = fit.fracridge_cv(X, y, frac_grid, cv)
    np.testing.assert_array_equal(best_frac, np.full(3, frac_grid[0]),
                                  err_msg="with equal scores, the first fraction of the grid should be selected")


def test_fracridge_branch_equals_brute_force_with_philipps_settings():
    # fit_model_with(..., "fracridge") with 70 fractions and repeated k-fold
    X, y = small_problem(seed=3)
    weights, alphas, l1_ratios = fit.fit_model_with(X, y, "fracridge", n_splits=3, n_repeats=2, random_state=0)
    coef_slow, _ = fracridge_cv_brute_force(X, y, FRACTIONS, RepeatedKFold(n_splits=3, n_repeats=2, random_state=0))
    assert_close(weights, coef_slow, 1e-10, "fracridge weights of fit_model_with vs brute force")
    assert alphas is None and l1_ratios is None, "fracridge should return None for alphas and l1_ratios, as before"


def test_train_model_with_fracridge_equals_brute_force_on_raw_data():
    # train_model_with z-scores the features and centers the targets first,
    # then fits with Philipp's settings
    X, y = raw_problem(seed=7)
    weights, _, _ = fit.train_model_with(X.copy(), y.copy(), "fracridge", k_in=3, n_in=2, random_state=0)
    coef_slow, _ = fracridge_cv_brute_force(z_score(X), y - y.mean(axis=0), FRACTIONS,
                                            RepeatedKFold(n_splits=3, n_repeats=2, random_state=0))
    assert_close(weights, coef_slow, 1e-10, "fracridge weights of train_model_with vs brute force")


# --- Philipp's original code, where it still runs

def fit_with_philipps_original_code(X, y, frac_grid, cv):
    """MultiOutputRegressor(FracRidgeRegressorCV(...)) with the settings of the original fit.py.

    The original used n_jobs=-1 for MultiOutputRegressor, which only makes it
    run in parallel. We use one job, because the computer may be shared.
    """

    from fracridge import FracRidgeRegressorCV
    from sklearn.multioutput import MultiOutputRegressor

    base = FracRidgeRegressorCV(fit_intercept=False, normalize=False, copy_X=True, tol=1e-10, jit=True, cv=cv,
                                scoring=None)
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(X, y, frac_grid=frac_grid)
    coef = np.stack([estimator.coef_ for estimator in model.estimators_], axis=1)
    best_frac = np.array([estimator.best_frac_ for estimator in model.estimators_])
    return coef, best_frac


@pytest.fixture(scope="module")
def original_and_sped_up():
    """Weights and fractions of the original code and of fracridge_cv on the same small problem."""

    X, y = small_problem()
    frac_grid = np.linspace(0.1, 1, 10)
    cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    try:
        original = fit_with_philipps_original_code(X, y, frac_grid, cv)
    except TypeError as error:
        # e.g. "_preprocess_data() got an unexpected keyword argument 'normalize'"
        pytest.skip(f"Philipp's original FracRidgeRegressorCV does not run on scikit-learn "
                    f"{sklearn.__version__} ({error})")
    return original, fit.fracridge_cv(X, y, frac_grid, cv)


def test_sped_up_ridge_selects_the_fractions_of_the_original_code(original_and_sped_up):
    (_, best_frac_original), (_, best_frac) = original_and_sped_up
    np.testing.assert_array_equal(best_frac, best_frac_original, err_msg="selected fractions differ")


def test_sped_up_ridge_gives_the_weights_of_the_original_code(original_and_sped_up):
    (coef_original, _), (coef, _) = original_and_sped_up
    assert_close(coef, coef_original, 1e-10, "fracridge_cv vs MultiOutputRegressor(FracRidgeRegressorCV())")


# --- preprocessing and prediction

def test_preprocess_data_z_scores_features_and_centers_targets():
    # these statistics end up in the model files (feature_mean,
    # feature_scale, target_mean), so they have to be the plain mean and the
    # std with ddof=0 of the training data
    X, y = raw_problem(seed=4, n=50, n_units=6)
    X_z, y_c, standardizer, y_mean = fit.preprocess_data(X.copy(), y.copy())
    assert_close(standardizer.mean_, X.mean(axis=0), 1e-12, "feature mean")
    assert_close(standardizer.scale_, X.std(axis=0), 1e-12, "feature scale (std with ddof=0)")
    assert_close(y_mean, y.mean(axis=0), 1e-12, "target mean")
    assert_close(X_z, z_score(X), 1e-12, "z-scored features")
    assert_close(y_c, y - y.mean(axis=0), 1e-12, "centered targets")


def test_preprocess_data_changes_its_input_in_place():
    # Philipp's preprocess_data uses StandardScaler(copy=False), so the
    # arrays given to it hold the z-scored features and centered targets
    # afterwards. Code that needs the raw features after the call, e.g. to
    # save their mean and std with a model, has to pass a copy (as
    # build_models.py does). Statistics computed from the changed arrays are
    # about 0 and 1, and a model file with them does not z-score new features.
    X, y = raw_problem(seed=4, n=50, n_units=6)
    X_given, y_given = X.copy(), y.copy()
    fit.preprocess_data(X_given, y_given)
    message = ("preprocess_data does not change its input anymore. That is fine, but then update this test "
               "and the comments in build_models.py.")
    assert np.allclose(X_given, z_score(X), rtol=0, atol=1e-12), message
    assert np.allclose(y_given, y - y.mean(axis=0), rtol=0, atol=1e-12), message


def test_ridge_predictions_add_the_target_mean_and_clip_at_0():
    rng = np.random.default_rng(5)
    weights = rng.standard_normal((6, 3))
    X_test_z = rng.standard_normal((10, 6))
    y_mean = np.array([0.5, 1.0, 2.0])
    prediction = fit.get_predictions_for(weights, X_test_z, y_mean)
    assert_close(prediction, np.maximum(X_test_z @ weights + y_mean, 0), 1e-12, "predictions of the ridge model")


def test_predictions_for_new_images_use_the_statistics_of_the_reference_images():
    # predict_spose_for_new_imgset_with, as used by call.py, with the weight
    # matrix that the ridge branch returns now
    X, y = raw_problem(seed=10, n=67)
    X_ref, X_new = X[:60], X[60:]
    weights, _, _ = fit.train_model_with(X_ref.copy(), y[:60].copy(), "ridge", k_in=3, n_in=1, random_state=0)
    predicted = fit.predict_spose_for_new_imgset_with(weights, {"1854ref": X_ref.copy(), "new": X_new.copy()},
                                                      y[:60].copy())
    assert list(predicted) == ["new"], f"predictions for {list(predicted)}, expected only 'new'"
    expected = np.maximum(z_score(X_new, X_ref) @ weights + y[:60].mean(axis=0), 0)
    assert_close(predicted["new"], expected, 1e-12, "predictions for new images")


def test_cross_validated_predictions_come_from_the_training_folds_only():
    # predict_spose_for_1854ref_with: each image is predicted by a model
    # fitted on the other fold, with the statistics of that fold. We compute
    # the same with the brute-force fractional ridge regression.
    X, y = raw_problem(seed=9, n=48)
    predicted, _, _ = fit.predict_spose_for_1854ref_with(X.copy(), y.copy(), "fracridge", k_out=2, k_in=2, n_in=1,
                                                         random_state=0)
    expected = np.zeros_like(y)
    for train, test in RepeatedKFold(n_splits=2, n_repeats=1, random_state=0).split(X):
        y_mean = y[train].mean(axis=0)
        coef, _ = fracridge_cv_brute_force(z_score(X[train]), y[train] - y_mean, FRACTIONS,
                                           RepeatedKFold(n_splits=2, n_repeats=1, random_state=0))
        expected[test] = np.maximum(z_score(X[test], X[train]) @ coef + y_mean, 0)
    assert_close(predicted, expected, 1e-10, "cross-validated predictions vs brute force")


# --- build_models.py: from the fit to the model file

@pytest.fixture(scope="module")
def built_ridge_model():
    """build_models.fit_model on a small raw problem, and the arrays given to it."""

    X, y = raw_problem(seed=8)
    X_given, y_given = X.copy(), y.copy()
    weights, feature_mean, feature_scale, target_mean = build_models.fit_model(X_given, y_given, "ridge")
    return dict(X=X, y=y, X_given=X_given, y_given=y_given, weights=weights, feature_mean=feature_mean,
                feature_scale=feature_scale, target_mean=target_mean)


def test_build_models_saves_the_mean_of_the_raw_features(built_ridge_model):
    m = built_ridge_model
    assert_close(m["feature_mean"], m["X"].mean(axis=0), 1e-12, "feature_mean vs mean of the raw features")


def test_build_models_saves_the_std_of_the_raw_features(built_ridge_model):
    # ddof=0, as sklearn's StandardScaler used in training
    m = built_ridge_model
    assert_close(m["feature_scale"], m["X"].std(axis=0), 1e-12, "feature_scale vs std of the raw features")


def test_build_models_saves_the_mean_of_the_targets(built_ridge_model):
    m = built_ridge_model
    assert_close(m["target_mean"], m["y"].mean(axis=0), 1e-12, "target_mean vs mean of the targets")


def test_build_models_does_not_change_its_input(built_ridge_model):
    m = built_ridge_model
    np.testing.assert_array_equal(m["X_given"], m["X"], err_msg="build_models.fit_model changed the features")
    np.testing.assert_array_equal(m["y_given"], m["y"], err_msg="build_models.fit_model changed the targets")


def test_build_models_ridge_uses_the_settings_of_call_py(built_ridge_model):
    # the new ridge with its lambda grid, 3-fold cross-validation repeated 3
    # times, random_state 0 (the settings of call.py). The slow tests below
    # refit the shipped models with build_models.fit_model.
    m = built_ridge_model
    coef, _, _ = fit.ridge_cv(z_score(m["X"]), m["y"] - m["y"].mean(axis=0), LAMBDAS,
                              RepeatedKFold(n_splits=3, n_repeats=3, random_state=0))
    assert_close(m["weights"], coef, 1e-10, "weights of build_models.fit_model vs ridge_cv with 3 x 3 folds")


def test_build_models_fracridge_uses_the_settings_of_call_py():
    # 70 fractions, 3-fold cross-validation repeated 3 times, random_state 0.
    # The slow tests below use fracridge_cv with these settings.
    X, y = raw_problem(seed=8)
    weights, _, _, _ = build_models.fit_model(X.copy(), y.copy(), "fracridge")
    coef, _ = fit.fracridge_cv(z_score(X), y - y.mean(axis=0), FRACTIONS,
                               RepeatedKFold(n_splits=3, n_repeats=3, random_state=0))
    assert_close(weights, coef, 1e-10, "weights of build_models.fit_model vs fracridge_cv with 3 x 3 folds")


@pytest.mark.parametrize("name, regression", [("alignet_siglip2b_66d_ridge", "ridge"), ("rn50x64_66d_ridge", "ridge"),
                                              ("rn50x64_49d_ridge", "fracridge"),
                                              ("rn50x64_66d_elastic", "elastic"), ("vitb32_66d_elastic", "elastic")])
def test_build_models_fits_each_model_with_its_regression(name, regression):
    spec = [spec for spec in build_models.MODELS if spec["name"] == name]
    assert len(spec) == 1, f"build_models.MODELS has {len(spec)} entries for {name}"
    assert spec[0]["regression"] == regression, f"{name} is fitted with {spec[0]['regression']}, expected {regression}"


def test_build_models_pairs_images_and_embedding_rows_in_the_order_of_the_name_list(tmp_path, monkeypatch):
    # The images have to be passed in the order of reference_image_names.txt,
    # which is the order of the rows of the embedding. Sorting the names
    # instead pairs 21 concepts with the wrong dimension values. Feature
    # extraction and fit are replaced by stand-ins that only record what
    # they get, and the images by empty files.
    names = read_lines(os.path.join(TRAINING_DATA, "reference_image_names.txt"))
    for name in names:
        (tmp_path / name).write_bytes(b"")
    given = {}

    def record_files(network, image_files, features_dir=None, device="cpu"):
        given["files"] = list(image_files)
        return np.zeros((len(image_files), 4))

    def record_embedding(features, embedding, regression):
        given["embedding"] = embedding.copy()
        return np.zeros((4, embedding.shape[1])), np.zeros(4), np.ones(4), embedding.mean(axis=0)

    monkeypatch.setattr(build_models, "get_features", record_files)
    monkeypatch.setattr(build_models, "fit_model", record_embedding)
    build_models.main(["--images", str(tmp_path), "--out", str(tmp_path / "models"), "--only", "rn50x64_66d_ridge"])
    assert [os.path.basename(f) for f in given["files"]] == names, "images are not in the order of the name list"
    np.testing.assert_array_equal(given["embedding"],
                                  np.loadtxt(os.path.join(TRAINING_DATA, "spose_embedding_66d.txt")),
                                  err_msg="the rows of the embedding were reordered")


def test_build_models_extracts_the_features_on_the_cpu_by_default(monkeypatch):
    # The shipped models were fitted on features extracted on the cpu. On
    # an Apple GPU (mps), the AligNet features differ by about 1.5e-5, which
    # changes the weights, so a rebuild with the default settings would not
    # give the shipped files. dimpred.extract_features is replaced by a
    # stand-in that only records its arguments.
    import types

    given = {}

    def record_arguments(image_files, **kwargs):
        given.update(kwargs)
        return np.zeros((len(image_files), 4), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "dimpred", types.SimpleNamespace(extract_features=record_arguments))
    build_models.get_features("RN50x64", ["a.jpg", "b.jpg"])
    assert given.get("device") == "cpu", f"extract_features was called with {given}, expected device='cpu'"
    build_models.get_features("RN50x64", ["a.jpg", "b.jpg"], device="mps")
    assert given.get("device") == "mps", f"extract_features was called with {given}, expected device='mps'"


@pytest.mark.parametrize("arguments, device", [([], "cpu"), (["--device", "mps"], "mps")])
def test_build_models_passes_the_device_to_the_feature_extraction(arguments, device, tmp_path, monkeypatch):
    names = read_lines(os.path.join(TRAINING_DATA, "reference_image_names.txt"))
    for name in names:
        (tmp_path / name).write_bytes(b"")
    given = {}

    def record_device(network, image_files, features_dir=None, device="cpu"):
        given["device"] = device
        return np.zeros((len(image_files), 4))

    def fit_nothing(features, embedding, regression):
        return np.zeros((4, embedding.shape[1])), np.zeros(4), np.ones(4), embedding.mean(axis=0)

    monkeypatch.setattr(build_models, "get_features", record_device)
    monkeypatch.setattr(build_models, "fit_model", fit_nothing)
    build_models.main(["--images", str(tmp_path), "--out", str(tmp_path / "models"), "--only", "rn50x64_66d_ridge"]
                      + arguments)
    assert given["device"] == device, f"features extracted on {given['device']!r}, expected {device!r}"


# --- call.py

def test_call_py_says_that_ridge_is_not_the_regression_of_the_paper(tmp_path, monkeypatch, capsys):
    # The models of the paper on OSF are named ..._ridge_... but are
    # fractional ridge models. Without such a file, call.py with "ridge"
    # fits the new ridge and saves it under the same kind of name, so it has
    # to say that the paper used "fracridge".
    import call

    X, y = raw_problem()
    features = tmp_path / "dimpred" / "data" / "raw" / "dnns" / "net" / "layer" / "1854ref" / "features.txt"
    embedding = tmp_path / "dimpred" / "data" / "raw" / "original_spose" / "embedding_4d.txt"
    for folder in [features.parent, embedding.parent, tmp_path / "dimpred" / "data" / "interim" / "dimpred"]:
        folder.mkdir(parents=True)  # call.py does not create the folder for the model file
    np.savetxt(features, X)
    np.savetxt(embedding, y)
    monkeypatch.setenv("DIMPRED_BASE_PATH", str(tmp_path))
    call.get_trained_model_for("net", "layer", 4, "ridge")
    output = capsys.readouterr().out
    assert "fracridge" in output, f"call.py should say that the paper used 'fracridge':\n{output}"
    call.get_trained_model_for("net", "layer", 4, "fracridge")
    assert "fracridge" not in capsys.readouterr().out, "the note should only be printed for 'ridge'"


# --- elastic net on the installed scikit-learn

def elastic_problem(seed=0, n=45, n_units=6):
    """Two targets: one depends on a single predictor, one a little on all of them.

    The first favors a high l1 ratio (sparse weights), the second a low one,
    so the selection of the l1 ratio is tested as well.
    """

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_units))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    w_sparse = np.zeros(n_units)
    w_sparse[0] = 2.0
    w_dense = np.full(n_units, 0.4)
    y = np.stack([X @ w_sparse + rng.standard_normal(n), X @ w_dense + rng.standard_normal(n)], axis=1)
    return X, y - y.mean(axis=0)


# Coefficients of the fit in elastic_fit on scikit-learn 1.3.2, the last
# version we tested on which Philipp's original training code runs
# (fit.fit_model_with, as below). scikit-learn 1.9.1 gives the same numbers
# (largest difference 4e-16).
ELASTIC_COEF_SKLEARN_1_3 = np.array([
    [1.9796861146200566, 0.2246536468376601],
    [-0.14162590367906744, 0.25608295080097865],
    [0.10260966566098115, 0.48228529108724627],
    [0.03536933012195729, 0.26758613493072686],
    [0.0, 0.20348706115411844],
    [-0.19939091115516905, 0.07407635301259428],
])


@pytest.fixture(scope="module")
def elastic_fit():
    X, y = elastic_problem()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LOKY_MAX_CPU_COUNT", "2")  # ElasticNetCV uses n_jobs=-1 (all cores)
        model, alphas, l1_ratios = fit.fit_model_with(X, y, "elastic", n_splits=3, n_repeats=1, random_state=0)
    return X, y, model, alphas, l1_ratios


def test_elastic_branch_gives_one_model_per_target(elastic_fit):
    _, _, model, alphas, l1_ratios = elastic_fit
    assert len(model.estimators_) == 2, f"{len(model.estimators_)} models for 2 targets"
    assert len(alphas) == 2 and len(l1_ratios) == 2, f"alphas {alphas}, l1 ratios {l1_ratios}, expected 2 each"


def test_elastic_net_selects_a_high_l1_ratio_for_the_sparse_target_and_a_low_one_for_the_dense(elastic_fit):
    _, _, _, _, l1_ratios = elastic_fit
    assert [float(r) for r in l1_ratios] == [0.99, 0.01], f"selected l1 ratios {l1_ratios}, expected [0.99, 0.01]"


def test_elastic_coefficients_are_those_of_scikit_learn_1_3(elastic_fit):
    _, _, model, _, _ = elastic_fit
    coef = np.stack([estimator.coef_ for estimator in model.estimators_], axis=1)
    assert_close(coef, ELASTIC_COEF_SKLEARN_1_3, 1e-8,
                 f"elastic net coefficients on scikit-learn {sklearn.__version__} vs 1.3.2")


def test_elastic_model_is_described_by_its_coefficients(elastic_fit):
    # build_models.py saves only the coefficients (np.stack of coef_). This
    # is enough because the models are fitted without intercept.
    X, _, model, _, _ = elastic_fit
    weights = np.stack([estimator.coef_ for estimator in model.estimators_], axis=1)
    assert_close(X @ weights, model.predict(X), 1e-12, "X @ coefficients vs predictions of the fitted model")


def test_elastic_predictions_add_the_target_mean_and_clip_at_0(elastic_fit):
    X, _, model, _, _ = elastic_fit
    y_mean = np.array([0.5, 0.1])
    expected = np.maximum(model.predict(X) + y_mean, 0)
    assert (expected == 0).any(), "test data should need clipping"
    assert_close(fit.get_predictions_for(model, X, y_mean), expected, 1e-12, "predictions of the elastic net")


def test_elastic_alpha_grid_is_philipps_grid(elastic_fit):
    # 10 alphas per l1 ratio, log-spaced from alpha_max down to alpha_max / 1000,
    # alpha_max = max |X' y| / (n * l1_ratio). scikit-learn computes the grid
    # like this with n_alphas=10 (before 1.7) and with alphas=10 (1.7 and later).
    X, y, model, _, _ = elastic_fit
    for target, estimator in enumerate(model.estimators_):
        assert estimator.alphas_.shape == (7, 10), f"alpha grid has shape {estimator.alphas_.shape}, expected (7, 10)"
        for i, l1_ratio in enumerate(L1_RATIOS):
            alpha_max = np.abs(X.T @ y[:, target]).max() / (len(X) * l1_ratio)
            expected = np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-3), 10)
            np.testing.assert_allclose(estimator.alphas_[i], expected, rtol=1e-8,
                                       err_msg=f"alphas for target {target}, l1_ratio {l1_ratio}")


# --- the real models: sped-up code vs Philipp's original code, and refits of the shipped models

def reference_features_file(network="RN50x64"):
    """features_<network>.npy in DIMPRED_REFERENCE_FEATURES (a file of RN50x64 features or the folder), or None.

    build_models.py --features writes these files, with spaces in the
    network name replaced by "-" (features_AligNet-SigLIP2-B.npy).
    """

    path = os.environ.get("DIMPRED_REFERENCE_FEATURES", "")
    if os.path.isdir(path):
        path = os.path.join(path, f"features_{network.replace(' ', '-')}.npy")
    elif network != "RN50x64":
        return None  # a single file holds the RN50x64 features
    return path if path and os.path.isfile(path) else None


@pytest.fixture(scope="module")
def sped_up_fit():
    """Fit RN50x64 66d with the sped-up fractional ridge on the 1854 reference images (takes about a minute).

    These are the steps of build_models.fit_model with the settings of
    call.py (see test_build_models_fracridge_uses_the_settings_of_call_py),
    but we also want the selected fractions.
    """

    fname = reference_features_file()
    if fname is None:
        pytest.skip("RN50x64 features of the 1854 reference images not found "
                    "(set DIMPRED_REFERENCE_FEATURES to features_RN50x64.npy or its folder)")
    X = np.load(fname).astype(np.float64)  # Philipp's features were read from text files (float64)
    y = np.loadtxt(os.path.join(TRAINING_DATA, "spose_embedding_66d.txt"))
    assert X.shape == (1854, 1024), f"reference features have shape {X.shape}, expected (1854, 1024)"
    X_z, y_c, _, _ = fit.preprocess_data(X, y)
    return fit.fracridge_cv(X_z, y_c, FRACTIONS, RepeatedKFold(n_splits=3, n_repeats=3, random_state=0))


@pytest.fixture(scope="module")
def original():
    assert os.path.exists(ORIGINAL_RIDGE_FILE), f"Missing test fixture {ORIGINAL_RIDGE_FILE}"
    return load_mat(ORIGINAL_RIDGE_FILE)


@pytest.mark.slow
def test_sped_up_ridge_selects_the_fractions_of_philipps_original_code(sped_up_fit, original):
    _, best_frac = sped_up_fit
    different = np.flatnonzero(np.abs(best_frac - original["best_frac"]) > TOL_BEST_FRAC)
    assert different.size == 0, (f"selected fractions differ for dimensions {different.tolist()}: "
                                 f"{best_frac[different]} vs {original['best_frac'][different]}")


@pytest.mark.slow
def test_sped_up_ridge_gives_the_weights_of_philipps_original_code(sped_up_fit, original):
    weights, _ = sped_up_fit
    assert_close(weights, original["weights"], TOL_ORIGINAL_WEIGHTS,
                 "fracridge_cv weights (RN50x64, 66d) vs Philipp's original code")


@pytest.mark.slow
@pytest.mark.parametrize("name, network", [("rn50x64_66d_ridge", "RN50x64"),
                                           ("alignet_siglip2b_66d_ridge", "AligNet SigLIP2-B")])
def test_shipped_ridge_model_is_the_refit_with_build_models(name, network):
    # The shipped model files are what build_models.fit_model gives on the
    # cached features of the 1854 reference images (a few seconds per model)
    fname = reference_features_file(network)
    if fname is None:
        pytest.skip(f"{network} features of the 1854 reference images not found (set DIMPRED_REFERENCE_FEATURES "
                    f"to the folder with features_{network.replace(' ', '-')}.npy)")
    X = np.load(fname)
    y = np.loadtxt(os.path.join(TRAINING_DATA, "spose_embedding_66d.txt"))
    weights, feature_mean, feature_scale, target_mean = build_models.fit_model(X, y, "ridge")
    shipped = load_mat(os.path.join(MODELS_DIR, name + ".mat"))
    assert_close(weights, shipped["weights"], TOL_ORIGINAL_WEIGHTS, f"{name}: refit vs shipped weights")
    assert_close(feature_mean, shipped["feature_mean"], 1e-12, f"{name}: feature_mean")
    assert_close(feature_scale, shipped["feature_scale"], 1e-12, f"{name}: feature_scale")
    assert_close(target_mean, shipped["target_mean"], 1e-12, f"{name}: target_mean")
