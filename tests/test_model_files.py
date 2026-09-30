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
    - rn50x64_66d_ridge has to be the result of Philipp's original training
      code
These tests read the files directly and do not use dimpred.

Martin Hebart, 2026/09/30

See also: test_load_model.py, ../training/build_models.py
"""

import os

import numpy as np
import pytest

from helpers import (MODEL_NAMES, MODELS, MODELS_DIR, ORIGINAL_RIDGE_FILE, TOL_ORIGINAL_WEIGHTS, TOL_STANDARDIZER,
                     TRAINING_DATA, assert_close, features_for, load_mat, read_lines)

# History:
# 2026/09/30: written after the review of the first tests, which could not
#   find model files with wrong statistics

PHILIPPS_MODEL = "rn50x64_49d_ridge"  # his published weights and feature scaling
REBUILT_RN50X64_MODELS = ["rn50x64_66d_elastic", "rn50x64_66d_ridge"]


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
    # 1.04). If feature_mean and feature_scale were computed from features
    # that were already z-scored, they are about 0 and 1, and the std stays
    # that of the raw features (0.18 for RN50x64, 0.47 for ViT-B-32).
    model = read_model_file(name)
    z = (features_for(ref, name) - model["feature_mean"]) / model["feature_scale"]
    assert 0.8 < z.std() < 1.25, f"{name}: std of the z-scored test features is {z.std():.3f}, expected about 1"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_labels_are_the_labels_of_the_spose_embedding(name):
    n_dims = MODELS[name]["n_dims"]
    label_file = os.path.join(TRAINING_DATA, f"labels_{n_dims}d.txt")
    assert read_model_file(name)["labels"] == read_lines(label_file), f"{name}: labels differ from {label_file}"


def test_shipped_ridge_model_equals_the_result_of_philipps_original_training():
    # rn50x64_66d_ridge was fitted with the sped-up training code. Philipp's
    # original (slow) code gave the weights stored in the fixture.
    assert os.path.exists(ORIGINAL_RIDGE_FILE), f"Missing test fixture {ORIGINAL_RIDGE_FILE}"
    original = load_mat(ORIGINAL_RIDGE_FILE)
    assert_close(read_model_file("rn50x64_66d_ridge")["weights"], original["weights"], TOL_ORIGINAL_WEIGHTS,
                 "rn50x64_66d_ridge weights vs Philipp's original training code")
