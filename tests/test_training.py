"""
Tests of Philipp Kaniuth's training code (training/fit.py) and of the part of
training/build_models.py that turns a fit into a model file.

Philipp's original code fitted the fractional ridge regression with
MultiOutputRegressor(FracRidgeRegressorCV()), which fits fracridge once per
target, fraction and fold. fit.py now does the same with one fit per fold
(fracridge_cv). We check that this gives the same result as the slow way:
    - against a brute-force version written here (loops over targets,
      fractions and folds) on small random problems
    - against Philipp's original code itself on a small problem, where it
      still runs (scikit-learn up to about 1.3; skipped otherwise)
    - against the weights of Philipp's original code for the
      rn50x64_66d_ridge model (stored in the fixtures). These last tests need
      the RN50x64 features of the 1854 reference images, which are not part
      of the repository; set DIMPRED_REFERENCE_FEATURES to the file
      (features_RN50x64.npy, as written by training/build_models.py
      --features) or to the folder that contains it, otherwise they are
      skipped.
The elastic net branch has to run on the installed scikit-learn (1.7 renamed
n_alphas, 1.9 removed it) and give the coefficients of scikit-learn 1.3.
build_models.py has to save the statistics of the raw features, although
preprocess_data z-scores its input in place, it has to fit with the settings
of call.py (3-fold cross-validation repeated 3 times, random_state 0), and it
has to pass the images in the order of data/reference_image_names.txt, which
is the order of the rows of the embedding.

The whole file is skipped if scikit-learn or fracridge are not installed
(they are only needed for training, see training/environment.yml). The
training data themselves are tested in test_training_data.py, which always runs.

Martin Hebart, 2026/09/30

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

from helpers import (ORIGINAL_RIDGE_FILE, TOL_BEST_FRAC, TOL_ORIGINAL_WEIGHTS, TRAINING, TRAINING_DATA,  # noqa: E402
                     assert_close, load_mat, read_lines)

# The training code is not part of the dimpred package. build_models.py
# imports fit.py by its name, so the training folder has to be on the path.
sys.path.insert(0, TRAINING)
import build_models  # noqa: E402
import fit  # noqa: E402

# History:
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


# --- fracridge_cv

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


def test_ridge_branch_equals_brute_force_with_philipps_settings():
    # fit_model_with(..., "ridge") with 70 fractions and repeated k-fold
    X, y = small_problem(seed=3)
    weights, _, _ = fit.fit_model_with(X, y, "ridge", n_splits=3, n_repeats=2, random_state=0)
    coef_slow, _ = fracridge_cv_brute_force(X, y, FRACTIONS, RepeatedKFold(n_splits=3, n_repeats=2, random_state=0))
    assert_close(weights, coef_slow, 1e-10, "ridge weights of fit_model_with vs brute force")


def test_train_model_with_ridge_equals_brute_force_on_raw_data():
    # train_model_with z-scores the features and centers the targets first,
    # then fits with Philipp's settings
    X, y = raw_problem(seed=7)
    weights, _, _ = fit.train_model_with(X.copy(), y.copy(), "ridge", k_in=3, n_in=2, random_state=0)
    coef_slow, _ = fracridge_cv_brute_force(z_score(X), y - y.mean(axis=0), FRACTIONS,
                                            RepeatedKFold(n_splits=3, n_repeats=2, random_state=0))
    assert_close(weights, coef_slow, 1e-10, "ridge weights of train_model_with vs brute force")


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
    # the same with the brute-force ridge regression.
    X, y = raw_problem(seed=9, n=48)
    predicted, _, _ = fit.predict_spose_for_1854ref_with(X.copy(), y.copy(), "ridge", k_out=2, k_in=2, n_in=1,
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
    # 70 fractions, 3-fold cross-validation repeated 3 times, random_state 0.
    # The slow tests below use fracridge_cv with these settings, so this
    # test ties them to the code that built the shipped model.
    m = built_ridge_model
    coef, _ = fit.fracridge_cv(z_score(m["X"]), m["y"] - m["y"].mean(axis=0), FRACTIONS,
                               RepeatedKFold(n_splits=3, n_repeats=3, random_state=0))
    assert_close(m["weights"], coef, 1e-10, "weights of build_models.fit_model vs fracridge_cv with 3 x 3 folds")


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

    def record_files(network, image_files, features_dir=None):
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


# --- the real model: sped-up code vs Philipp's original code

def reference_features_file():
    path = os.environ.get("DIMPRED_REFERENCE_FEATURES", "")
    if os.path.isdir(path):
        path = os.path.join(path, "features_RN50x64.npy")
    return path if path and os.path.isfile(path) else None


@pytest.fixture(scope="module")
def sped_up_fit():
    """Fit rn50x64_66d_ridge with the sped-up code on the 1854 reference images (takes about a minute).

    These are the steps of build_models.fit_model with the settings of
    call.py (see test_build_models_ridge_uses_the_settings_of_call_py), but
    we also want the selected fractions.
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
                 "rn50x64_66d_ridge weights vs Philipp's original code")
