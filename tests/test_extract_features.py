"""
Tests of dimpred.extract_features (slow, need torch and open_clip).

We extract the features of the three CC0 test images and compare them with
reference features that were extracted independently of dimpred (and that
reproduce Philipp Kaniuth's features). Two mistakes of earlier versions are
tested explicitly:
    - using open_clip's plain "ViT-B-32" (GELU) instead of
      "ViT-B-32-quickgelu" for OpenAI's ViT. The features then correlate only
      about 0.98 with the reference, and the predictions change.
    - returning the rows in sorted file order instead of the given order,
      which pairs images with the wrong predictions
Different devices (cpu, mps, cuda) give slightly different numbers, so we
allow a correlation of 0.99999 and a difference of 2e-3. Wrong input (missing
files, folders) is tested in test_extract_features_errors.py.

Martin Hebart, 2026/09/30

See also: test_extract_features_errors.py, test_cli.py
"""

import numpy as np
import pytest

import dimpred
from helpers import TOL_FRESH_PREDICTION, assert_close, assert_features_match, row_correlations

# History:
# 2026/09/30: the order test uses a second unsorted order; a grayscale image
#   is compared with the same image in RGB; the tests of wrong input moved
#   to test_extract_features_errors.py (fast, without torch)
# 2026/09/30: written together with the tests, before the package code

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("open_clip_available")]


@pytest.fixture(scope="module")
def vit_features(open_clip_available, cc0_paths):
    """Features of the CC0 images with the default model (ViT-B-32-quickgelu), in cc0_files order."""
    return dimpred.extract_features(cc0_paths)


@pytest.fixture(scope="module")
def rn_features(open_clip_available, cc0_paths):
    """Features of the CC0 images with RN50x64, in cc0_files order."""
    return dimpred.extract_features(cc0_paths, model="rn50x64_49d_ridge")


# --- the features

def test_default_model_gives_the_reference_vit_features(ref, vit_features):
    assert_features_match(vit_features, ref["cc0_features_vitb32"], "ViT-B-32-quickgelu features of the CC0 images")


def test_rn50x64_model_gives_the_reference_rn50x64_features(ref, rn_features):
    assert_features_match(rn_features, ref["cc0_features_rn50x64"], "RN50x64 features of the CC0 images")


def test_plain_vit_b_32_does_not_give_the_reference_features(ref, cc0_paths):
    # open_clip's "ViT-B-32" loads the same OpenAI weights, but with GELU
    # instead of QuickGELU. This test shows that the comparison above would
    # find this mistake.
    features = dimpred.extract_features(cc0_paths, network="ViT-B-32")
    r = row_correlations(features, ref["cc0_features_vitb32"])
    assert r.max() < 0.9999, (
        f"plain ViT-B-32 gives r = {np.round(r, 6).tolist()} with the QuickGELU reference features. "
        f"If open_clip now uses QuickGELU for ViT-B-32 with OpenAI weights, this test (not dimpred) needs updating.")


def test_network_argument_overrides_the_model(ref, cc0_paths):
    features = dimpred.extract_features(cc0_paths, model="rn50x64_49d_ridge", network="ViT-B-32-quickgelu")
    assert_features_match(features, ref["cc0_features_vitb32"], "features with network='ViT-B-32-quickgelu'")


def test_model_can_be_given_as_dict(ref, cc0_paths):
    features = dimpred.extract_features(cc0_paths[:1], model=dimpred.load_model("vitb32_66d_elastic"))
    assert_features_match(features, ref["cc0_features_vitb32"][:1], "features with a model dict")


def test_features_are_a_float32_array(vit_features):
    assert isinstance(vit_features, np.ndarray), f"extract_features returned a {type(vit_features)}"
    assert vit_features.dtype == np.float32, f"dtype is {vit_features.dtype}"


def test_one_row_per_image_and_one_column_per_feature(vit_features):
    assert vit_features.shape == (3, 512), f"shape is {vit_features.shape}, expected (3, 512)"


def test_predictions_from_extracted_features_match_philipps_published_predictions(ref, rn_features):
    prediction = dimpred.predict(rn_features, "rn50x64_49d_ridge")
    assert_close(prediction, ref["cc0_published_rn50x64_49d_ridge"], TOL_FRESH_PREDICTION,
                 "predictions from extracted features vs Philipp's published predictions")


# --- order of the rows

def test_rows_are_in_the_given_order_not_sorted(ref, cc0_paths_reordered):
    # a second unsorted order (the other tests use the order of cc0_files)
    paths, rows = cc0_paths_reordered
    features = dimpred.extract_features(paths)
    assert_features_match(features, ref["cc0_features_vitb32"][rows], "features of images given in a second order")


def test_repeated_image_gives_repeated_row(ref, cc0_paths):
    paths = [cc0_paths[1], cc0_paths[0], cc0_paths[1]]
    features = dimpred.extract_features(paths)
    assert features.shape[0] == 3, f"{features.shape[0]} rows for 3 images"
    assert_features_match(features, ref["cc0_features_vitb32"][[1, 0, 1]], "features with a repeated image")


@pytest.mark.parametrize("batch_size", [1, 2])
def test_batch_size_does_not_change_the_features(ref, cc0_paths, batch_size):
    # with batch_size 2, the second batch has only one image
    features = dimpred.extract_features(cc0_paths, batch_size=batch_size)
    assert_features_match(features, ref["cc0_features_vitb32"], f"features with batch_size={batch_size}")


def test_single_path_gives_one_row(ref, cc0_paths):
    features = dimpred.extract_features(cc0_paths[2])
    assert_features_match(features, ref["cc0_features_vitb32"][2:3], "features of a single path")


def test_cpu_device(ref, cc0_paths):
    features = dimpred.extract_features(cc0_paths, device="cpu")
    assert_features_match(features, ref["cc0_features_vitb32"], "features on the cpu")


# --- other image types

def test_png_with_alpha_channel_gives_the_features_of_the_jpg(ref, cc0_paths, tmp_path):
    # the same pixels saved as RGBA png; the alpha channel has to be dropped
    from PIL import Image

    fname = str(tmp_path / "rgba.png")
    Image.open(cc0_paths[0]).convert("RGBA").save(fname)
    features = dimpred.extract_features([fname])
    assert_features_match(features, ref["cc0_features_vitb32"][:1], "features of an RGBA png")


def test_grayscale_image_gives_the_features_of_the_same_image_in_rgb(cc0_paths, tmp_path):
    # a grayscale image has one channel and has to be converted to RGB with
    # three equal channels, as if it had been saved in RGB
    from PIL import Image

    gray = Image.open(cc0_paths[0]).convert("L")
    gray.save(tmp_path / "gray.png")
    gray.convert("RGB").save(tmp_path / "gray_rgb.png")
    features = dimpred.extract_features([str(tmp_path / "gray.png"), str(tmp_path / "gray_rgb.png")])
    assert_features_match(features[:1], features[1:], "features of a grayscale png vs the same image in RGB")
