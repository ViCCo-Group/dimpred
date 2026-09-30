"""
Tests of the test fixtures themselves (tests/fixtures).

All other tests compare dimpred against the numbers in these files, so we
first check that the files contain what make_fixtures.py writes: the right
variables and shapes, one entry per shipped model, image files that exist,
features of the two networks that belong to the same images in the same row
order, and expected predictions and human similarity that fit together.
These tests do not use dimpred at all.

Martin Hebart, 2026/09/30

See also: fixtures/make_fixtures.py
"""

import os

import numpy as np
import pytest

from helpers import (CC0_REORDERED, IMAGES, MODEL_NAMES, MODELS, MODELS_DIR, ORIGINAL_RIDGE_FILE, TOL_HUMAN_R,
                     TOL_MODEL, assert_close, load_mat, lower_triangle, row_correlations,
                     spose_similarity_by_definition)

# History:
# 2026/09/30: checks that the MATLAB tests of the fixtures already had
#   (unique names, different CC0 features, published = expected predictions,
#   human r of at least 0.80, not mostly zeros, one entry per model file);
#   second order of the CC0 images; the test with shuffled images moved here
#   from test_validation_human.py, since it checks the data and not dimpred
# 2026/09/30: written together with the tests, before the package code

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

FIXED_SHAPES = {
    "features_rn50x64": (168, 1024),
    "features_vitb32": (168, 512),
    "published_rn50x64_49d_ridge": (168, 49),
    "human_similarity_48nonref": (48, 48),
    "cc0_features_rn50x64": (3, 1024),
    "cc0_features_vitb32": (3, 512),
    "cc0_published_rn50x64_49d_ridge": (3, 49),
}


@pytest.mark.parametrize("variable", sorted(FIXED_SHAPES))
def test_reference_data_variable_has_right_shape(ref, variable):
    assert variable in ref, f"reference_data.mat has no variable {variable}"
    assert ref[variable].shape == FIXED_SHAPES[variable], (
        f"{variable} has shape {ref[variable].shape}, expected {FIXED_SHAPES[variable]}")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_reference_data_has_expected_predictions_of_each_model(ref, name):
    key = "expected_" + name
    assert key in ref, f"reference_data.mat has no variable {key}"
    assert ref[key].shape == (168, MODELS[name]["n_dims"]), f"{key} has shape {ref[key].shape}"


@pytest.mark.parametrize("variable", sorted(FIXED_SHAPES) + ["expected_" + n for n in MODEL_NAMES])
def test_reference_numbers_are_finite(ref, variable):
    assert np.all(np.isfinite(ref[variable])), f"{variable} contains nan or inf"


@pytest.mark.parametrize("variable", ["published_rn50x64_49d_ridge", "cc0_published_rn50x64_49d_ridge"]
                         + ["expected_" + n for n in MODEL_NAMES])
def test_reference_predictions_are_not_negative(ref, variable):
    # predictions are clipped at 0, because SPoSE dimensions are non-negative
    assert ref[variable].min() >= 0, f"{variable} has negative values (min {ref[variable].min():g})"


def test_one_file_name_and_image_set_per_row(ref):
    assert len(ref["files"]) == 168, f"files has {len(ref['files'])} entries, expected 168"
    assert len(ref["image_set"]) == 168, f"image_set has {len(ref['image_set'])} entries, expected 168"


def test_first_48_rows_are_48nonref_then_peterson_animals(ref):
    # the human similarity matrix belongs to the 48nonref images, in this order
    assert ref["image_set"][:48] == ["48nonref"] * 48, "rows 1-48 should be the 48nonref images"
    assert ref["image_set"][48:] == ["peterson-animals"] * 120, "rows 49-168 should be the Peterson animals"


def test_file_names_are_unique(ref):
    # two rows of the same image would point to a mistake in collecting the features
    rows = list(zip(ref["image_set"], ref["files"]))
    assert len(set(rows)) == len(rows), "some images appear in more than one row"


def test_human_similarity_is_symmetric(ref):
    human = ref["human_similarity_48nonref"]
    off_diagonal = ~np.eye(48, dtype=bool)
    np.testing.assert_allclose(human[off_diagonal], human.T[off_diagonal], rtol=0, atol=1e-12,
                               err_msg="human_similarity_48nonref is not symmetric")


def test_human_similarity_is_between_0_and_1(ref):
    # probabilities from the odd-one-out task
    values = lower_triangle(ref["human_similarity_48nonref"])
    assert values.min() >= 0, f"smallest human similarity is {values.min():g}"
    assert values.max() <= 1, f"largest human similarity is {values.max():g}"


def test_human_correlation_is_given_for_each_model(ref):
    human_r = ref["human_r_48nonref"]
    assert isinstance(human_r, dict), "human_r_48nonref should be a struct"
    missing = [n for n in MODEL_NAMES if n not in human_r]
    assert not missing, f"human_r_48nonref has no value for {missing}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_human_correlation_of_each_model_is_at_least_0_80(ref, name):
    # Each shipped model predicts human similarity at least about as well as
    # the model of the DimPred paper (r = 0.810; the rebuilt models reach
    # 0.82 to 0.83). The human data do not depend on the model files, so this
    # finds model files that are wrong but consistent with the expected
    # predictions: without target_mean, r drops to 0.76-0.78, with wrong
    # feature statistics to about 0.65-0.75.
    r = float(ref["human_r_48nonref"][name])
    assert 0.80 <= r < 1, f"human_r_48nonref.{name} = {r:.3f}, expected at least 0.80"


def test_paper_model_human_correlation_is_the_published_value(ref):
    # the DimPred paper model gives r = 0.81 on the 48nonref images
    r = float(ref["human_r_48nonref"]["rn50x64_49d_ridge"])
    assert abs(r - 0.810) <= TOL_HUMAN_R, f"human_r_48nonref.rn50x64_49d_ridge = {r:.4f}, expected 0.810"


def test_published_predictions_equal_expected_predictions_of_the_paper_model(ref):
    # rn50x64_49d_ridge holds Philipp's published weights, so its expected
    # predictions are his published predictions. If this fails, the model
    # file or the fixture is wrong, not the dimpred code.
    assert_close(ref["expected_rn50x64_49d_ridge"], ref["published_rn50x64_49d_ridge"], TOL_MODEL,
                 "expected_rn50x64_49d_ridge vs published_rn50x64_49d_ridge")


@pytest.mark.parametrize("variable", ["published_rn50x64_49d_ridge"] + ["expected_" + n for n in MODEL_NAMES])
def test_reference_predictions_are_not_mostly_zero(ref, variable):
    # SPoSE dimensions are sparse: 26% of Philipp's published predictions are
    # 0, and 15-26% of the expected predictions. Without target_mean, about
    # 70% would be 0.
    fraction_zero = np.mean(ref[variable] == 0)
    assert fraction_zero < 0.4, f"{fraction_zero:.0%} of {variable} are 0 (probably target_mean was not added)"


def test_every_model_file_has_fixture_numbers(ref):
    # when a model is added to dimpred/models, the fixtures have to be updated
    names = sorted(f[:-4] for f in os.listdir(MODELS_DIR) if f.endswith(".mat"))
    missing = [n for n in names if "expected_" + n not in ref or n not in ref["human_r_48nonref"]]
    assert not missing, f"reference_data.mat has no expected_* or human_r_48nonref for the model files {missing}"


def test_cc0_images_are_in_the_images_folder(ref):
    assert len(ref["cc0_files"]) == 3, f"cc0_files has {len(ref['cc0_files'])} entries, expected 3"
    missing = [f for f in ref["cc0_files"] if not os.path.isfile(os.path.join(IMAGES, f))]
    assert not missing, f"CC0 images not found in {IMAGES}: {missing}"


def test_images_folder_contains_only_the_cc0_images(ref):
    # tests of find_images and of the command line tool rely on this
    images = sorted(f for f in os.listdir(IMAGES)
                    if not f.startswith(".") and f.lower().endswith(IMAGE_EXTENSIONS))
    assert images == sorted(ref["cc0_files"]), f"images in {IMAGES}: {images}, cc0_files: {ref['cc0_files']}"


def test_cc0_files_are_unique(ref):
    assert len(set(ref["cc0_files"])) == len(ref["cc0_files"]), f"cc0_files lists an image twice: {ref['cc0_files']}"


def test_cc0_files_are_not_sorted(ref):
    # the extraction tests need a file order that differs from the sorted
    # order, to find code that sorts file names on the way
    assert ref["cc0_files"] != sorted(ref["cc0_files"]), "cc0_files should not be in sorted order"


def test_second_cc0_order_is_neither_sorted_nor_the_order_of_cc0_files(ref):
    # used by the tests of the row order (helpers.CC0_REORDERED)
    reordered = [ref["cc0_files"][i] for i in CC0_REORDERED]
    assert reordered != sorted(reordered), f"{reordered} is sorted"
    assert reordered != ref["cc0_files"], f"{reordered} is the order of cc0_files"


@pytest.mark.parametrize("variable", ["cc0_features_rn50x64", "cc0_features_vitb32"])
def test_cc0_features_differ_clearly_between_images(ref, variable):
    # The order tests can only tell the rows apart if the images have clearly
    # different features. They require r > 0.99999 with the right row;
    # different images here correlate about 0.4 to 0.5.
    r = np.corrcoef(ref[variable])[np.triu_indices(3, k=1)]
    assert r.max() < 0.9, f"two CC0 images have almost the same {variable} (r = {r.max():.4f})"


def test_rows_of_both_networks_belong_to_the_same_images(ref):
    # The RN50x64 and ViT features come from different sources. If both sets
    # are in the same image order, the 66d predictions of the two networks
    # should be most similar for the same image. For the 48nonref images
    # (48 different objects) this is true for every image when the order is
    # right, and for about 1 in 48 when it is not.
    rn = ref["expected_rn50x64_66d_elastic"][:48]
    vit = ref["expected_vitb32_66d_elastic"][:48]
    r = np.corrcoef(rn, vit)[:48, 48:]  # r[i, j]: RN50x64 image i vs ViT image j
    same_row_best = np.mean(np.argmax(r, axis=1) == np.arange(48))
    assert same_row_best >= 0.9, (
        f"only {same_row_best:.0%} of the 48nonref images are predicted most similarly by both networks "
        f"at the same row, so features_rn50x64 and features_vitb32 are probably not in the same image order")


def test_rows_of_both_networks_agree_for_all_images(ref):
    # the 120 Peterson animals are all similar to each other, so here we only
    # check that the same image agrees much better than two different images
    rn = ref["expected_rn50x64_66d_elastic"]
    vit = ref["expected_vitb32_66d_elastic"]
    same = row_correlations(rn, vit).mean()
    other = row_correlations(rn, np.roll(vit, 1, axis=0)).mean()
    assert same > other + 0.2, (
        f"mean r same image {same:.3f}, neighboring rows {other:.3f}; expected a clear difference")


# --- expected predictions and human similarity

@pytest.fixture(scope="module")
def predicted_similarity(ref_session):
    """SPoSE similarity of the expected predictions for the 48nonref images, per model."""

    return {name: spose_similarity_by_definition(ref_session["expected_" + name][:48]) for name in MODEL_NAMES}


def r_with_humans(similarity, ref):
    return np.corrcoef(lower_triangle(similarity), lower_triangle(ref["human_similarity_48nonref"]))[0, 1]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_human_correlation_fits_the_expected_predictions(ref, predicted_similarity, name):
    # human_r_48nonref has to be the correlation of the expected predictions
    # (first 48 rows) with the human similarity
    r = r_with_humans(predicted_similarity[name], ref)
    expected = float(ref["human_r_48nonref"][name])
    assert abs(r - expected) < 1e-6, f"{name}: r from expected_{name} is {r:.6f}, human_r_48nonref says {expected:.6f}"


def test_shuffled_images_do_not_correlate_with_human_similarity(ref, predicted_similarity):
    # This checks the data, not dimpred: it shows that the correlation with
    # human similarity (test_validation_human.py) would reveal images that
    # are paired with the wrong rows, as happens when file names are sorted
    # on the way. Here we shuffle the rows ourselves.
    order = np.random.default_rng(0).permutation(48)
    shuffled = predicted_similarity["rn50x64_49d_ridge"][np.ix_(order, order)]
    r = r_with_humans(shuffled, ref)
    assert abs(r) < 0.3, f"r with human similarity is {r:.3f} for shuffled images, expected about 0"


# --- Philipp's original training

def test_original_ridge_fixture():
    assert os.path.exists(ORIGINAL_RIDGE_FILE), f"Missing test fixture {ORIGINAL_RIDGE_FILE}"
    original = load_mat(ORIGINAL_RIDGE_FILE)
    assert original["weights"].shape == (1024, 66), f"weights have shape {original['weights'].shape}"
    assert original["best_frac"].shape == (66,), f"best_frac has shape {original['best_frac'].shape}"
    assert np.all(np.isfinite(original["weights"])), "weights contain nan or inf"
    # Philipp's grid: 70 fractions from 0.1 to 1
    grid = np.linspace(0.1, 1, 70)
    distance = np.abs(original["best_frac"][:, None] - grid[None, :]).min(axis=1)
    assert distance.max() < 1e-12, "best_frac contains values that are not in the grid linspace(0.1, 1, 70)"
