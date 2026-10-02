"""
Tests of the shipped model files themselves (dimpred/models/*.mat).

The expected predictions in the fixtures (expected_*, human_r_48nonref) were
computed from these same files, so they only show that dimpred applies the
files correctly, not that the files are right. Here we check the numbers in
the files against sources that do not depend on them:
    - target_mean has to be the mean of each dimension of the SPoSE
      embedding in training/data (the regression was fit on centered values)
    - feature_mean and feature_scale of the rebuilt RN50x64 models have to
      be those of Philipp Kaniuth's published model (same network, same 1854
      training images)
    - the z-scored features of the 168 test images have to have a std of
      about 1, which is not the case when the statistics were computed from
      features that were already z-scored
    - the labels have to be those of the embedding in training/data
    - the weights of rn50x64_66d_ridge and alignet_siglip2b_66d_ridge have to
      be those of the same ridge fitted by the DimPred benchmark, a separate
      implementation (fixtures/benchmark_ridge_fits.mat)
These tests read the files directly and do not use dimpred. That the
shipped ridge models are what build_models.py fits, and that fracridge_cv
still gives the weights of Philipp's original code, is tested in
test_training.py (with the features of the 1854 reference images).

Hebartlab, 2026/09/30

See also: test_load_model.py, ../training/build_models.py
"""

import os

import numpy as np
import pytest

from helpers import (BENCHMARK_RIDGE_FILE, MODEL_NAMES, MODELS, MODELS_DIR, TOL_STANDARDIZER, TRAINING_DATA,
                     assert_close, features_for, load_mat, read_lines)

# History:
# 2026/10/02: the ridge models are compared with the fits of the DimPred
#   benchmark (before, only that rn50x64_66d_ridge is not the fractional
#   ridge anymore, which a wrong refit would also pass)
# 2026/10/02: rn50x64_66d_ridge is the new ridge, so it is not compared with
#   Philipp's original code anymore (fracridge_cv is, in test_training.py)
# 2026/09/30: written after the review of the first tests, which could not
#   find model files with wrong statistics

PHILIPPS_MODEL = "rn50x64_49d_ridge"  # his published weights and feature scaling
REBUILT_RN50X64_MODELS = ["rn50x64_66d_elastic", "rn50x64_66d_ridge"]

# The ridge models were also fitted in the DimPred benchmark (2026/10/01), with
# its own implementation of the same ridge. For RN50x64, it used the same
# features, so the weights are identical. For AligNet, it used the TensorFlow
# features, which differ by up to 3e-4 from those of the PyTorch port; the
# weights then differ by up to 2.5e-7, their sums by up to 1.7e-6.
TOL_BENCHMARK_RIDGE = {"rn50x64_66d_ridge": 1e-10, "alignet_siglip2b_66d_ridge": 1e-5}


def read_model_file(name):
    return load_mat(os.path.join(MODELS_DIR, name + ".mat"))


@pytest.fixture(scope="module")
def spose_embedding():
    """The SPoSE embeddings the models were trained on, by number of dimensions."""

    return {n_dims: np.loadtxt(os.path.join(TRAINING_DATA, f"spose_embedding_{n_dims}d.txt")) for n_dims in (49, 66)}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_target_mean_is_the_mean_of_the_spose_embedding(name, spose_embedding):
    model = read_model_file(name)
    embedding = spose_embedding[MODELS[name]["n_dims"]]
    assert_close(model["target_mean"], embedding.mean(axis=0), 1e-12,
                 f"{name}: target_mean vs mean of spose_embedding_{MODELS[name]['n_dims']}d.txt")


@pytest.mark.parametrize("name", REBUILT_RN50X64_MODELS)
def test_rn50x64_feature_mean_is_that_of_philipps_model(name):
    # difference relative to the feature scale, because the features have
    # very different ranges
    model, philipp = read_model_file(name), read_model_file(PHILIPPS_MODEL)
    difference = np.abs(model["feature_mean"] - philipp["feature_mean"]) / philipp["feature_scale"]
    assert difference.max() < TOL_STANDARDIZER, (
        f"{name}: feature_mean differs from that of {PHILIPPS_MODEL} by up to {difference.max():.3g} "
        f"feature scales (allowed: {TOL_STANDARDIZER:g})")


@pytest.mark.parametrize("name", REBUILT_RN50X64_MODELS)
def test_rn50x64_feature_scale_is_that_of_philipps_model(name):
    # the std with ddof=0, as sklearn's StandardScaler; ddof=1 would give
    # values that are 2.7e-4 too large
    model, philipp = read_model_file(name), read_model_file(PHILIPPS_MODEL)
    difference = np.abs(model["feature_scale"] / philipp["feature_scale"] - 1)
    assert difference.max() < TOL_STANDARDIZER, (
        f"{name}: feature_scale differs from that of {PHILIPPS_MODEL} by up to {difference.max():.3g} "
        f"(relative; allowed: {TOL_STANDARDIZER:g})")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_z_scored_test_features_have_a_std_close_to_1(name, ref):
    # The 168 test images were not used for training, so their z-scored
    # features do not have a std of exactly 1, but close to it (1.03 to
    # 1.04, 1.09 for AligNet). If feature_mean and feature_scale were
    # computed from features that were already z-scored, they are about 0
    # and 1, and the std stays that of the raw features (0.18 for RN50x64,
    # 0.47 for ViT-B-32, 0.68 for AligNet).
    model = read_model_file(name)
    z = (features_for(ref, name) - model["feature_mean"]) / model["feature_scale"]
    assert 0.8 < z.std() < 1.25, f"{name}: std of the z-scored test features is {z.std():.3f}, expected about 1"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_labels_are_the_labels_of_the_spose_embedding(name):
    n_dims = MODELS[name]["n_dims"]
    label_file = os.path.join(TRAINING_DATA, f"labels_{n_dims}d.txt")
    assert read_model_file(name)["labels"] == read_lines(label_file), f"{name}: labels differ from {label_file}"


@pytest.mark.parametrize("name", sorted(TOL_BENCHMARK_RIDGE))
def test_ridge_weights_are_those_of_the_benchmark_fit(name):
    # The fixture holds the norm and the sum of the weights of each dimension
    # of the benchmark fit. The norm of the weights of a ridge regression
    # shrinks with the penalty, so it shows whether each dimension was refit
    # with the right penalty, and the sum shows the sign. A refit with
    # alpha = 1854 x lambda instead of the alpha of the inner folds changes
    # the norm of every dimension by 0.006 to 0.017. The fractional ridge
    # of version 1.0.0 is also clearly different.
    assert os.path.exists(BENCHMARK_RIDGE_FILE), f"Missing test fixture {BENCHMARK_RIDGE_FILE}"
    benchmark = load_mat(BENCHMARK_RIDGE_FILE)
    weights = read_model_file(name)["weights"]
    tolerance = TOL_BENCHMARK_RIDGE[name]
    assert_close(np.linalg.norm(weights, axis=0), benchmark[f"norm_{name}"], tolerance,
                 f"{name}: norm of the weights of each dimension vs the benchmark fit")
    assert_close(weights.sum(axis=0), benchmark[f"sum_{name}"], tolerance,
                 f"{name}: sum of the weights of each dimension vs the benchmark fit")
