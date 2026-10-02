"""
Tests of dimpred.predict.

predict computes
    embedding = max(((features - feature_mean) / feature_scale) @ weights + target_mean, 0)

The most important tests compare the predictions with numbers that do not
come from dimpred: Philipp Kaniuth's published predictions of the DimPred
paper model, and the expected predictions of each shipped model stored in the
fixtures. The other tests check single parts of the formula, so that a
failing test says which part is wrong. Several tests guard against mistakes
that happened in earlier versions of this code:
    - forgetting to add target_mean (values too low, about 70% zeros)
    - z-scoring with wrong statistics, e.g. subtracting the mean twice or
      using the mean of the given images instead of the training images

Hebartlab, 2026/09/30

See also: test_load_model.py, test_validation_human.py, test_package.py
"""

import os

import numpy as np
import pytest

import dimpred
from helpers import (DEFAULT_MODEL, MODEL_NAMES, MODELS, MODELS_DIR, TOL_FRESH_PREDICTION, TOL_MODEL, assert_close,
                     features_for)

# History:
# 2026/10/02: the default model is alignet_siglip2b_66d_ridge (768 AligNet features)
# 2026/09/30: nan and inf give an error, as in MATLAB
# 2026/09/30: predictions of the CC0 reference features, transposed features,
#   one behavior per test; the test without torch moved to test_package.py
# 2026/09/30: written together with the tests, before the package code


def linear_part(features, model):
    """The formula of predict without the clipping at 0."""

    z = (np.asarray(features, dtype=float) - model["feature_mean"]) / model["feature_scale"]
    return z @ model["weights"] + model["target_mean"]


# --- comparison with known predictions

def test_reproduces_philipps_published_predictions(ref):
    prediction = dimpred.predict(ref["features_rn50x64"], "rn50x64_49d_ridge")
    assert_close(prediction, ref["published_rn50x64_49d_ridge"], TOL_MODEL,
                 "rn50x64_49d_ridge vs Philipp's published predictions")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_reproduces_expected_predictions_of_each_model(ref, name):
    prediction = dimpred.predict(features_for(ref, name), name)
    assert_close(prediction, ref["expected_" + name], TOL_MODEL, f"{name} vs expected_{name}")


def test_cc0_reference_features_give_philipps_published_predictions(ref):
    # Philipp's published predictions of the CC0 images come from his own
    # feature extraction, the reference features from a separate extraction
    # of the same images, so small differences are expected (up to 1e-3).
    # Without target_mean, the difference is 0.64.
    prediction = dimpred.predict(ref["cc0_features_rn50x64"], "rn50x64_49d_ridge")
    assert_close(prediction, ref["cc0_published_rn50x64_49d_ridge"], TOL_FRESH_PREDICTION,
                 "rn50x64_49d_ridge on the CC0 reference features vs Philipp's published predictions")


def test_default_model_is_used_if_no_model_is_given(ref):
    prediction = dimpred.predict(features_for(ref, DEFAULT_MODEL))
    assert_close(prediction, ref["expected_" + DEFAULT_MODEL], TOL_MODEL, "predict without model vs default model")


# --- target_mean

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_features_at_the_training_mean_give_target_mean(name):
    # z-scored features are then 0, so only target_mean is left (all values
    # are > 0, so nothing is clipped). If target_mean is forgotten, this gives 0.
    model = dimpred.load_model(name)
    features = np.tile(model["feature_mean"], (3, 1))
    prediction = dimpred.predict(features, model)
    assert_close(prediction, np.tile(model["target_mean"], (3, 1)), 1e-12,
                 f"{name}: prediction for features equal to feature_mean vs target_mean")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_for_real_images_are_not_mostly_zero(ref, name):
    # For the 168 test images, about 15-26% of the predicted values are 0 and
    # the mean prediction is about 90% of the mean of target_mean. Without
    # target_mean, about 70% are 0 and the mean drops below 45%.
    model = dimpred.load_model(name)
    prediction = dimpred.predict(features_for(ref, name), model)
    fraction_zero = np.mean(prediction == 0)
    ratio = prediction.mean() / model["target_mean"].mean()
    assert fraction_zero < 0.4, (
        f"{name}: {fraction_zero:.0%} of the predicted values are 0 (probably target_mean was not added)")
    assert ratio > 0.7, (
        f"{name}: mean prediction is only {ratio:.0%} of the mean of target_mean (probably target_mean was not added)")


# --- z-scoring and weights

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_one_standard_deviation_in_one_feature_adds_its_weights(name):
    # features = feature_mean, plus one feature_scale in feature k: the
    # z-scored features are 0 except a 1 at k, so the prediction is
    # target_mean + weights[k] (clipped at 0). This finds wrong scaling, e.g.
    # dividing by the variance or multiplying by the scale.
    model = dimpred.load_model(name)
    n_features = MODELS[name]["n_features"]
    ks = [0, 1, n_features // 2, n_features - 1]
    features = np.tile(model["feature_mean"], (len(ks), 1))
    for row, k in enumerate(ks):
        features[row, k] += model["feature_scale"][k]
    expected = np.maximum(model["target_mean"] + model["weights"][ks], 0)
    assert_close(dimpred.predict(features, model), expected, 1e-12,
                 f"{name}: prediction for one standard deviation in features {ks}")


def test_prediction_equals_the_formula_when_nothing_is_clipped():
    # features close to the training mean give only positive values, so the
    # prediction is the plain linear formula
    model = dimpred.load_model("rn50x64_49d_ridge")
    rng = np.random.default_rng(1)
    features = model["feature_mean"] + 0.05 * model["feature_scale"] * rng.standard_normal((20, 1024))
    expected = linear_part(features, model)
    assert expected.min() > 0, "test data should not need clipping, use smaller noise"
    assert_close(dimpred.predict(features, model), expected, 1e-12, "prediction vs linear formula")


def test_only_negative_values_are_set_to_0():
    model = dimpred.load_model("rn50x64_49d_ridge")
    rng = np.random.default_rng(2)
    features = model["feature_mean"] + 2 * model["feature_scale"] * rng.standard_normal((20, 1024))
    linear = linear_part(features, model)
    assert (linear < 0).any() and (linear > 0).any(), "test data should give both positive and negative values"
    prediction = dimpred.predict(features, model)
    assert np.all(prediction[linear < 0] == 0), "negative values of the linear formula should be set to 0"
    assert_close(prediction[linear >= 0], linear[linear >= 0], 1e-12, "positive values should not be changed")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_are_a_float64_array(ref, name):
    prediction = dimpred.predict(features_for(ref, name), name)
    assert isinstance(prediction, np.ndarray), f"{name}: predict returned a {type(prediction)}, not a numpy array"
    assert prediction.dtype == np.float64, f"{name}: dtype is {prediction.dtype}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_one_row_per_image_and_one_column_per_dimension(ref, name):
    prediction = dimpred.predict(features_for(ref, name), name)
    assert prediction.shape == (168, MODELS[name]["n_dims"]), f"{name}: shape is {prediction.shape}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_are_finite(ref, name):
    prediction = dimpred.predict(features_for(ref, name), name)
    assert np.all(np.isfinite(prediction)), f"{name}: predictions contain nan or inf"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_are_not_negative(ref, name):
    prediction = dimpred.predict(features_for(ref, name), name)
    assert prediction.min() >= 0, f"{name}: predictions have negative values (min {prediction.min():g})"


# --- each image is predicted on its own

@pytest.mark.parametrize("rows", [slice(20, 30), slice(7, 8), slice(0, 2)], ids=["10 rows", "1 row", "2 rows"])
def test_predicting_some_images_gives_the_same_rows(ref, rows):
    # Predictions must not depend on which other images are predicted at the
    # same time. This fails if the features are z-scored with the mean and std
    # of the given images instead of those of the training images.
    features = features_for(ref, DEFAULT_MODEL)
    all_rows = dimpred.predict(features)
    assert_close(dimpred.predict(features[rows]), all_rows[rows], 1e-12, "prediction of a subset of images")


def test_permuting_images_permutes_predictions(ref):
    features = ref["features_rn50x64"]
    order = np.random.default_rng(3).permutation(len(features))
    all_rows = dimpred.predict(features, "rn50x64_49d_ridge")
    assert_close(dimpred.predict(features[order], "rn50x64_49d_ridge"), all_rows[order], 1e-12,
                 "prediction of permuted images")


# --- form of the input

def test_one_feature_vector_gives_one_row(ref):
    features = features_for(ref, DEFAULT_MODEL)
    prediction = dimpred.predict(features[5])
    assert prediction.shape == (1, 66), f"shape is {prediction.shape}, expected (1, 66)"
    assert_close(prediction, dimpred.predict(features[5:6]), 0, "1-D input vs one row")


def test_list_of_rows_gives_the_same_result_as_an_array(ref):
    features = features_for(ref, DEFAULT_MODEL)[:4]
    assert_close(dimpred.predict(features.tolist()), dimpred.predict(features), 0, "list input vs array input")


def test_list_of_numbers_gives_one_row(ref):
    features = features_for(ref, DEFAULT_MODEL)[0]
    assert_close(dimpred.predict(list(features)), dimpred.predict(features[None, :]), 0, "list of numbers vs one row")


def test_float32_input_is_computed_in_float64(ref):
    # extract_features returns float32. These have to be converted to float64
    # before z-scoring, then the result equals that for the same numbers in float64.
    features32 = features_for(ref, DEFAULT_MODEL)[:10].astype(np.float32)
    prediction = dimpred.predict(features32)
    assert prediction.dtype == np.float64, f"dtype is {prediction.dtype}"
    assert_close(prediction, dimpred.predict(features32.astype(np.float64)), 1e-12, "float32 vs float64 input")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_input_features_are_not_changed(ref, dtype):
    features = features_for(ref, DEFAULT_MODEL)[:10].astype(dtype)
    before = features.copy()
    dimpred.predict(features)
    np.testing.assert_array_equal(features, before, err_msg="predict changed its input")


def test_model_is_not_changed(ref):
    model = dimpred.load_model("rn50x64_49d_ridge")
    before = {k: model[k].copy() for k in ["weights", "feature_mean", "feature_scale", "target_mean"]}
    dimpred.predict(ref["features_rn50x64"], model)
    for key, value in before.items():
        np.testing.assert_array_equal(model[key], value, err_msg=f"predict changed model['{key}']")


def test_model_can_be_given_by_name_path_or_dict(ref):
    features = ref["features_rn50x64"][:10]
    by_name = dimpred.predict(features, "rn50x64_66d_ridge")
    by_path = dimpred.predict(features, os.path.join(MODELS_DIR, "rn50x64_66d_ridge.mat"))
    by_dict = dimpred.predict(features, dimpred.load_model("rn50x64_66d_ridge"))
    assert_close(by_path, by_name, 0, "model given by path vs by name")
    assert_close(by_dict, by_name, 0, "model given as dict vs by name")


# --- errors

@pytest.mark.parametrize("name, features_key, network, n_expected", [
    ("vitb32_66d_elastic", "features_rn50x64", "ViT-B-32-quickgelu", 512),
    ("rn50x64_49d_ridge", "features_vitb32", "RN50x64", 1024),
    ("alignet_siglip2b_66d_ridge", "features_vitb32", "AligNet SigLIP2-B", 768),
])
def test_wrong_number_of_features_gives_error_with_model_network_and_count(ref, name, features_key, network,
                                                                          n_expected):
    with pytest.raises(ValueError) as error:
        dimpred.predict(ref[features_key][:3], name)
    message = str(error.value)
    for part in [name, network, str(n_expected)]:
        assert part in message, f"the error message should contain {part!r}: {message!r}"


@pytest.mark.parametrize("shape", ["transposed", "column"])
def test_features_with_images_in_columns_give_error(ref, shape):
    # predict never transposes: one image is one row. A (1024, 5) array
    # holds 5 images in columns, a (1024, 1) array one image as a column.
    # Both have to give an error and must not be turned around silently.
    features = ref["features_rn50x64"][:5].T if shape == "transposed" else ref["features_rn50x64"][:1].T
    with pytest.raises(ValueError):
        dimpred.predict(features, "rn50x64_49d_ridge")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nan_or_inf_in_features_gives_error(ref, value):
    # nan would give nan predictions, and -inf would give predictions of 0
    # that look like real values, so both have to give an error
    features = features_for(ref, DEFAULT_MODEL)[:4].copy()
    features[2, 10] = value
    with pytest.raises(ValueError, match="nan or inf"):
        dimpred.predict(features)
