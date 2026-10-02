"""
Tests of dimpred.rise, the RISE heatmaps (Petsiuk et al., 2018) of the
predicted dimensions.

RISE masks the network input image with many random masks, predicts the
dimensions of each masked image, and the map of a dimension is the average
of its predictions, weighted by the masks. The parts are tested on their own
and then together:
    - the masks: shape, values 0 to 1, mean close to p, the same masks for
      the same seed, and the same masks as generate_masks of the original
      RISE code (github.com/eclique/RISE, explanations.py), transcribed
      below with the bilinear resize of skimage written out by its formula
      (and compared with skimage itself if it is installed)
    - the maps from the predictions of the masked images, on synthetic data:
      the two normalizations by their formulas, and if each prediction is
      the uncovered fraction of a known region, the map peaks in this region
    - the relevance map: the dimension maps averaged with the predicted
      dimension values of the image as weights (compute_aggregate_saliency
      of the DimPred paper code)
    - rise with a stand-in network (fast, needs torch): the network gets the
      masked 0 to 1 image with its own normalization, the maps are the
      formula applied to these masks and predictions, the same masks for
      every image and batch size, and the view of the image
    - wrong input, checked before the network is loaded (fast, no torch)
    - rise with the real networks (slow): the prediction of the unmasked
      image is the prediction of extract_features, and the output on the
      three CC0 images
    - the command line tool (python -m dimpred --rise): the variables of
      the .mat and .npz files and the PNG files, with the stand-in network
      (fast) and with a real network (slow); wrong options (fast)

Hebartlab, 2026/10/02

See also: ../dimpred/rise.py, test_cli.py
"""

import importlib
import json
import os
import sys
import types

import numpy as np
import pytest
import scipy.io

import dimpred
import dimpred.__main__
from helpers import assert_close, output_of, run_dimpred

# the module dimpred/rise.py (dimpred.rise is the function rise)
rise_module = importlib.import_module("dimpred.rise")

# History:
# 2026/10/02: written before dimpred/rise.py (test-driven)


# --- the original RISE code, transcribed

def resize_bilinear_reflect(image, output_shape):
    """skimage.transform.resize(image, output_shape, order=1, mode="reflect", anti_aliasing=False) by its formula.

    Along each axis, output pixel o lies at the input position
    (o + 0.5) * n_in / n_out - 0.5 (pixel centers). Its value is the linear
    interpolation between the two input pixels around this position. Pixels
    outside the image are mirrored at the border pixels (skimage's mode
    "reflect": c b | a b c d | c b), rows first, then columns.
    """

    def one_axis(n_in, n_out):
        weights = np.zeros((n_out, n_in))
        for o in range(n_out):
            position = (o + 0.5) * n_in / n_out - 0.5
            first = int(np.floor(position))
            for k, weight in [(first, 1 - (position - first)), (first + 1, position - first)]:
                while k < 0 or k > n_in - 1:  # mirror at the border pixels
                    k = -k if k < 0 else 2 * (n_in - 1) - k
                weights[o, k] += weight
        return weights

    image = np.asarray(image, dtype=float)
    rows = one_axis(image.shape[0], int(output_shape[0]))
    columns = one_axis(image.shape[1], int(output_shape[1]))
    return rows @ image @ columns.T


def original_generate_masks(N, s, p1, input_size, seed, resize=None):
    """generate_masks of the original RISE code (explanations.py, class RISE), line by line.

    The original uses np.random without a seed; here the global generator is
    seeded first (and restored afterwards), and skimage's
    resize(grid, up_size, order=1, mode="reflect", anti_aliasing=False) is
    resize_bilinear_reflect, unless another resize is given.
    """

    if resize is None:
        resize = resize_bilinear_reflect
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        masks = np.empty((N, *input_size))

        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size)[x:x + input_size[0], y:y + input_size[1]]
    finally:
        np.random.set_state(state)
    return masks


def compute_aggregate_saliency(saliency_maps, dimension_values):
    """The relevance map of the DimPred paper code (scripts/visualize_heatmaps.py), copied."""

    aggregate_saliency = np.zeros_like(saliency_maps[0])
    for smap, dim_val in zip(saliency_maps, dimension_values):
        aggregate_saliency += smap * dim_val
    aggregate_saliency /= np.sum(dimension_values)
    return aggregate_saliency


# --- the masks

def test_masks_have_the_input_size_and_are_float32():
    masks = rise_module.generate_masks(10, (13, 18), grid=4, p=0.3, seed=0)
    assert masks.shape == (10, 13, 18), f"shape is {masks.shape}, expected (10, 13, 18) (n_masks x height x width)"
    assert masks.dtype == np.float32, f"dtype is {masks.dtype}"


def test_mask_values_are_between_0_and_1():
    masks = rise_module.generate_masks(200, (24, 24), grid=6, p=0.2, seed=3)
    assert masks.min() >= 0 and masks.max() <= 1, f"mask values from {masks.min()} to {masks.max()}"
    # soft masks: the bilinear upsampling gives values between 0 and 1
    assert np.mean((masks > 0.01) & (masks < 0.99)) > 0.05, "the masks should not be binary"


def test_mean_of_the_masks_is_close_to_p():
    # each cell is kept with probability p, and the bilinear upsampling
    # keeps the mean, so the mean of many masks is close to p
    for p in [0.1, 0.5]:
        masks = rise_module.generate_masks(1000, (24, 24), grid=6, p=p, seed=0)
        assert abs(masks.mean() - p) < 0.01, f"mean of the masks is {masks.mean():.4f} for p = {p}"


def test_masks_are_the_same_for_the_same_seed():
    first = rise_module.generate_masks(30, (20, 20), grid=5, p=0.3, seed=7)
    second = rise_module.generate_masks(30, (20, 20), grid=5, p=0.3, seed=7)
    np.testing.assert_array_equal(first, second, err_msg="the same seed gave different masks")
    other = rise_module.generate_masks(30, (20, 20), grid=5, p=0.3, seed=8)
    assert not np.array_equal(first, other), "another seed gave the same masks"


def test_masks_do_not_change_the_global_random_generator():
    state = np.random.get_state()[1].copy()
    rise_module.generate_masks(5, (10, 10), grid=3, p=0.5, seed=1)
    assert np.array_equal(np.random.get_state()[1], state), "generate_masks changed the state of np.random"


@pytest.mark.parametrize("input_size, grid, p, seed", [((13, 18), 4, 0.3, 0), ((28, 28), 8, 0.1, 5),
                                                         ((17, 9), 3, 0.5, 2)])
def test_masks_are_those_of_the_original_rise_code(input_size, grid, p, seed):
    # small, non-square cases, so that rows and columns cannot be swapped
    expected = original_generate_masks(25, grid, p, input_size, seed)
    masks = rise_module.generate_masks(25, input_size, grid=grid, p=p, seed=seed)
    assert_close(masks, expected, 1e-6, f"masks vs the original generate_masks ({input_size}, grid {grid})")


def test_masks_are_those_of_the_original_rise_code_with_skimage():
    # the original code itself, with skimage's resize (skimage is not a
    # dependency of dimpred, so this test is skipped without it)
    skimage_transform = pytest.importorskip("skimage.transform", reason="needs scikit-image")

    def resize(grid, up_size):
        return skimage_transform.resize(grid, up_size, order=1, mode="reflect", anti_aliasing=False)

    expected = original_generate_masks(25, 4, 0.3, (13, 18), 0, resize=resize)
    masks = rise_module.generate_masks(25, (13, 18), grid=4, p=0.3, seed=0)
    assert_close(masks, expected, 1e-6, "masks vs the original generate_masks with skimage")


def test_the_formula_of_the_resize_is_that_of_skimage():
    # the check of the transcription above (skipped without skimage)
    skimage_transform = pytest.importorskip("skimage.transform", reason="needs scikit-image")
    grid = np.random.default_rng(0).random((5, 4))
    expected = skimage_transform.resize(grid, (24, 15), order=1, mode="reflect", anti_aliasing=False)
    assert_close(resize_bilinear_reflect(grid, (24, 15)), expected, 1e-12, "bilinear resize vs skimage")


def test_masks_in_parts_are_the_masks_of_all_at_once():
    # rise makes the masks batch by batch from random_grids and upsample_grids
    grids, shifts = rise_module.random_grids(40, (21, 30), grid=5, p=0.4, seed=2)
    assert grids.shape == (40, 5, 5) and shifts.shape == (40, 2), f"{grids.shape}, {shifts.shape}"
    parts = [rise_module.upsample_grids(grids[i:i + 7], shifts[i:i + 7], (21, 30)) for i in range(0, 40, 7)]
    expected = rise_module.generate_masks(40, (21, 30), grid=5, p=0.4, seed=2)
    np.testing.assert_array_equal(np.concatenate(parts), expected, err_msg="masks made in parts differ")


# --- the maps from the predictions of the masked images (synthetic data)

@pytest.fixture
def predictions_and_masks():
    rng = np.random.default_rng(0)
    masks = rng.random((30, 5, 7)).astype(np.float32)
    predictions = rng.random((30, 3))
    return predictions, masks


def test_pixel_normalization_divides_by_the_sum_of_the_masks_at_each_pixel(predictions_and_masks):
    predictions, masks = predictions_and_masks
    sums, coverage = rise_module.mask_sums(predictions, masks)
    maps = rise_module.saliency_maps(sums, coverage, 30, 0.1, "pixel")
    masks = masks.astype(float)  # the sums below in float64
    expected = np.zeros((3, 5, 7))
    for d in range(3):
        for row in range(5):
            for column in range(7):
                weighted = sum(predictions[i, d] * masks[i, row, column] for i in range(30))
                expected[d, row, column] = weighted / sum(masks[i, row, column] for i in range(30))
    assert_close(maps, expected, 1e-12, "maps with normalization 'pixel'")


def test_original_normalization_divides_by_n_masks_times_p(predictions_and_masks):
    # sal = p.T @ masks / N / p1 in the original RISE code
    predictions, masks = predictions_and_masks
    sums, coverage = rise_module.mask_sums(predictions, masks)
    maps = rise_module.saliency_maps(sums, coverage, 30, 0.1, "original")
    expected = (predictions.T @ masks.reshape(30, -1).astype(float)).reshape(3, 5, 7) / 30 / 0.1
    assert_close(maps, expected, 1e-12, "maps with normalization 'original'")


def test_mask_sums_are_the_weighted_sum_and_the_sum_of_the_masks(predictions_and_masks):
    predictions, masks = predictions_and_masks
    sums, coverage = rise_module.mask_sums(predictions, masks)
    assert_close(sums, np.einsum("nd,nhw->dhw", predictions, masks.astype(float)), 1e-12, "sums")
    assert_close(coverage, masks.astype(float).sum(axis=0), 1e-12, "coverage")


def test_pixels_that_no_mask_kept_are_nan_with_pixel_normalization():
    masks = np.ones((4, 3, 3), dtype=np.float32)
    masks[:, 1, 2] = 0
    sums, coverage = rise_module.mask_sums(np.ones((4, 2)), masks)
    maps = rise_module.saliency_maps(sums, coverage, 4, 0.5, "pixel")
    assert np.all(np.isnan(maps[:, 1, 2])), "a pixel without any mask should be nan"
    assert np.all(np.isfinite(np.delete(maps.reshape(2, -1), 5, axis=1))), "the other pixels should be finite"


REGION_A = (slice(2, 8), slice(14, 21))   # rows, columns of a 24 x 24 image
REGION_B = (slice(15, 22), slice(3, 9))


def outside(region, size=24, margin=3):
    """True for the pixels more than margin pixels away from region."""

    keep = np.ones((size, size), dtype=bool)
    keep[max(region[0].start - margin, 0):region[0].stop + margin,
         max(region[1].start - margin, 0):region[1].stop + margin] = False
    return keep


@pytest.mark.parametrize("normalization", ["pixel", "original"])
def test_map_peaks_in_the_region_that_drives_the_prediction(normalization):
    # dimension 1 is the uncovered fraction of region A, dimension 2 that of
    # region B: masks that keep the region give high values, so the map of
    # each dimension has to be high in its region
    masks = rise_module.generate_masks(600, (24, 24), grid=6, p=0.5, seed=1)
    predictions = np.stack([masks[:, REGION_A[0], REGION_A[1]].mean(axis=(1, 2)),
                            masks[:, REGION_B[0], REGION_B[1]].mean(axis=(1, 2))], axis=1)
    maps = rise_module.saliency_maps(*rise_module.mask_sums(predictions, masks), 600, 0.5, normalization)
    for d, (region, other) in enumerate([(REGION_A, REGION_B), (REGION_B, REGION_A)]):
        peak = np.unravel_index(np.argmax(maps[d]), maps[d].shape)
        assert region[0].start <= peak[0] < region[0].stop and region[1].start <= peak[1] < region[1].stop, (
            f"map {d + 1} peaks at {peak}, outside its region {region}")
        inside, far = maps[d][region].mean(), maps[d][outside(region)].mean()
        assert inside > far + 0.02, f"map {d + 1}: mean {inside:.3f} in its region, {far:.3f} far from it"
        assert maps[d][region].mean() > maps[d][other].mean(), f"map {d + 1} is higher in the other region"


# --- the relevance map

def test_relevance_is_the_average_of_the_dimension_maps_weighted_by_the_embedding():
    rng = np.random.default_rng(4)
    maps = rng.random((6, 9, 11))
    embedding = rng.random(6) * 2
    expected = compute_aggregate_saliency(maps, embedding)
    assert_close(rise_module.relevance_map(maps, embedding), expected, 1e-12, "relevance map")


def test_relevance_does_not_change_its_input():
    maps = np.ones((3, 4, 4))
    embedding = np.array([1.0, 2.0, 3.0])
    rise_module.relevance_map(maps, embedding)
    assert np.all(maps == 1) and np.all(embedding == [1, 2, 3]), "relevance_map changed its input"


def test_relevance_of_an_embedding_of_zeros_is_nan():
    relevance = rise_module.relevance_map(np.ones((3, 4, 4)), np.zeros(3))
    assert relevance.shape == (4, 4) and np.all(np.isnan(relevance)), "an embedding of zeros should give nan"


# --- the overlay for the PNG files

def test_overlay_is_40_percent_jet_and_60_percent_image():
    # jet as in Figure 7 of the DimPred paper: the minimum of the map dark
    # blue (0, 0, 0.5), the maximum dark red (0.5, 0, 0)
    view = np.full((4, 5, 3), 100, dtype=np.uint8)
    saliency = np.zeros((4, 5))
    saliency[1, 2] = 3.0   # maximum
    saliency[3, 4] = -1.0  # minimum
    image = rise_module.overlay(view, saliency)
    assert image.dtype == np.uint8 and image.shape == (4, 5, 3), f"{image.dtype}, {image.shape}"
    assert_close(image[1, 2], np.round(0.4 * np.array([127.5, 0, 0]) + 0.6 * 100), 1, "color of the maximum")
    assert_close(image[3, 4], np.round(0.4 * np.array([0, 0, 127.5]) + 0.6 * 100), 1, "color of the minimum")


def test_overlay_of_a_constant_map_is_the_image():
    view = np.random.default_rng(0).integers(0, 256, (6, 6, 3), dtype=np.uint8)
    np.testing.assert_array_equal(rise_module.overlay(view, np.full((6, 6), 0.3)), view)


# --- wrong input (fast: checked before the network is loaded)

@pytest.fixture
def without_torch(monkeypatch):
    """torch and open_clip cannot be imported during the test (as if they were not installed)."""

    for name in ("torch", "torchvision", "open_clip"):
        monkeypatch.setitem(sys.modules, name, None)  # restored after the test


def test_unknown_normalization_gives_error(cc0_paths, without_torch):
    with pytest.raises(ValueError) as error:
        dimpred.rise(cc0_paths[:1], normalization="per pixel")
    for part in ["per pixel", "'pixel'", "'original'"]:
        assert part in str(error.value), f"the message should mention {part}: {str(error.value)!r}"


@pytest.mark.parametrize("n_masks", [0, -10, 2.5, "100"])
def test_n_masks_below_1_or_not_a_whole_number_gives_error(cc0_paths, without_torch, n_masks):
    with pytest.raises(ValueError, match="n_masks"):
        dimpred.rise(cc0_paths[:1], n_masks=n_masks)


@pytest.mark.parametrize("setting, value", [("grid", 0), ("grid", 2.5), ("p", 0), ("p", 1.5), ("p", -0.1),
                                            ("seed", -1), ("seed", 0.5), ("map_size", 0), ("map_size", 22.5),
                                            ("batch_size", 0)])
def test_wrong_settings_give_error_that_names_them(cc0_paths, without_torch, setting, value):
    with pytest.raises(ValueError, match=setting):
        dimpred.rise(cc0_paths[:1], **{setting: value})


@pytest.mark.parametrize("given_as", ["path", "list"])
def test_folder_gives_error_that_points_to_find_images(cc0_paths, without_torch, given_as):
    from helpers import IMAGES

    images = IMAGES if given_as == "path" else [cc0_paths[0], IMAGES]
    with pytest.raises(ValueError) as error:
        dimpred.rise(images)
    assert "find_images" in str(error.value), f"the message should point to find_images: {str(error.value)!r}"


def test_missing_image_gives_error_that_names_it(cc0_paths, tmp_path, without_torch):
    with pytest.raises(FileNotFoundError, match="no_such_image.jpg"):
        dimpred.rise([cc0_paths[0], str(tmp_path / "no_such_image.jpg")])


def test_no_images_give_error(without_torch):
    with pytest.raises(ValueError):
        dimpred.rise([])


def test_without_torch_existing_images_give_import_error(cc0_paths, without_torch):
    with pytest.raises(ImportError):
        dimpred.rise(cc0_paths[:1], n_masks=2)


# --- rise with a stand-in network (fast, needs torch)

HEIGHT, WIDTH = 24, 40  # input size of the stand-in network: not square, so that rows and columns cannot be swapped
STAND_IN_REGIONS = [(slice(3, 9), slice(4, 12)), (slice(14, 21), slice(26, 36)), (slice(0, 24), slice(0, 40))]
MEAN, STD = 0.5, 0.25  # normalization of the stand-in network (all channels)

# For the stand-in network, feature k is the mean of the normalized image
# (4 x pixel value - 2) in region k. With this model, the prediction of
# dimension k is the mean pixel value (0 to 1) in region k, so for a white
# image the uncovered fraction of the region. Region 3 is the whole image.
STAND_IN_MODEL = dict(weights=np.eye(3), feature_mean=np.full(3, -2.0), feature_scale=np.full(3, 4.0),
                      target_mean=np.zeros(3), labels=["region 1", "region 2", "whole image"],
                      info=dict(name="stand_in_3d", network="stand-in network", pretrained="none"))


@pytest.fixture
def stand_in_network(monkeypatch):
    """Replace open_clip by a stand-in network. Returns the list of the batches the network got.

    The preprocessing is that of open_clip: a torchvision Compose whose last
    step is Normalize. The first step resizes the image to WIDTH x HEIGHT
    and gives values 0 to 1.
    """

    torch = pytest.importorskip("torch", reason="the stand-in network needs torch")
    transforms = pytest.importorskip("torchvision.transforms", reason="the stand-in preprocessing needs torchvision")
    batches = []

    def to_unit(image):
        pixels = np.asarray(image.resize((WIDTH, HEIGHT)), dtype=np.float32) / 255
        return torch.from_numpy(pixels).permute(2, 0, 1)

    def encode_image(batch):
        batches.append(batch.clone())
        return torch.stack([batch[:, :, rows, columns].mean(dim=(1, 2, 3)) for rows, columns in STAND_IN_REGIONS],
                           dim=1)

    def create_model_and_transforms(network, pretrained=None, device=None):
        assert network == "stand-in network", f"open_clip was asked for {network}"
        net = types.SimpleNamespace(eval=lambda: None, encode_image=encode_image)
        return net, None, transforms.Compose([to_unit, transforms.Normalize((MEAN,) * 3, (STD,) * 3)])

    open_clip = types.ModuleType("open_clip")
    open_clip.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)  # restored after the test
    return batches


@pytest.fixture
def white_image(tmp_path):
    from PIL import Image

    fname = str(tmp_path / "white.png")
    Image.new("RGB", (80, 50), (255, 255, 255)).save(fname)
    return fname


@pytest.fixture
def pattern_image(tmp_path):
    """An image with different values in each pixel and channel (not square)."""

    from PIL import Image

    rng = np.random.default_rng(0)
    fname = str(tmp_path / "pattern.png")
    Image.fromarray(rng.integers(0, 256, (37, 61, 3), dtype=np.uint8)).save(fname)
    return fname


@pytest.fixture
def broken_image(tmp_path, cc0_paths):
    """The first half of a CC0 image (a truncated JPEG file, which PIL cannot read)."""

    fname = str(tmp_path / "broken.jpg")
    with open(cc0_paths[0], "rb") as source, open(fname, "wb") as target:
        data = source.read()
        target.write(data[:len(data) // 2])
    return fname


def test_stand_in_network_gets_the_masked_image_with_its_normalization(stand_in_network, pattern_image):
    # the mask multiplies the 0 to 1 image, before the normalization of the
    # network, so masked pixels are black. The first batch is the image
    # without a mask.
    from PIL import Image

    dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=20, grid=4, p=0.3, batch_size=8, device="cpu")
    unit = np.asarray(Image.open(pattern_image).convert("RGB").resize((WIDTH, HEIGHT)), dtype=np.float32) / 255
    unit = unit.transpose(2, 0, 1)
    masks = rise_module.generate_masks(20, (HEIGHT, WIDTH), grid=4, p=0.3, seed=0)
    batches = [b.numpy() for b in stand_in_network]
    assert [b.shape[0] for b in batches] == [1, 8, 8, 4], f"batch sizes {[b.shape[0] for b in batches]}"
    assert_close(batches[0][0], (unit - MEAN) / STD, 1e-6, "the unmasked image")
    expected = (unit[None] * masks[:, None] - MEAN) / STD
    assert_close(np.concatenate(batches[1:]), expected, 1e-6, "the masked images")


@pytest.mark.parametrize("normalization", ["pixel", "original"])
def test_stand_in_maps_are_the_formula_applied_to_the_masks(stand_in_network, white_image, normalization):
    # with the white image, the prediction of each dimension is the uncovered
    # fraction of its region, which we compute from the masks here
    result = dimpred.rise([white_image], STAND_IN_MODEL, n_masks=150, grid=5, p=0.4, seed=3,
                          normalization=normalization, map_size=None, batch_size=64, device="cpu")
    masks = rise_module.generate_masks(150, (HEIGHT, WIDTH), grid=5, p=0.4, seed=3).astype(float)
    predictions = np.stack([masks[:, rows, columns].mean(axis=(1, 2)) for rows, columns in STAND_IN_REGIONS], axis=1)
    weighted = np.einsum("nd,nhw->dhw", predictions, masks)
    if normalization == "pixel":
        expected = weighted / masks.sum(axis=0)
    else:
        expected = weighted / 150 / 0.4
    assert_close(result["dimension_maps"][0], expected, 1e-5, f"dimension maps ({normalization})")
    assert_close(result["embedding"], [[1, 1, 1]], 1e-6, "prediction of the unmasked white image")


def test_stand_in_maps_peak_in_the_regions(stand_in_network, white_image):
    result = dimpred.rise([white_image], STAND_IN_MODEL, n_masks=400, grid=5, p=0.5, map_size=None, device="cpu")
    for d, (rows, columns) in enumerate(STAND_IN_REGIONS[:2]):
        peak = np.unravel_index(np.argmax(result["dimension_maps"][0, d]), (HEIGHT, WIDTH))
        assert rows.start <= peak[0] < rows.stop and columns.start <= peak[1] < columns.stop, (
            f"map of dimension {d + 1} peaks at {peak}, outside its region {(rows, columns)}")


def test_stand_in_output(stand_in_network, white_image, pattern_image):
    result = dimpred.rise([white_image, pattern_image], STAND_IN_MODEL, n_masks=30, map_size=50, device="cpu")
    shapes = {key: np.shape(result[key]) for key in ["relevance", "dimension_maps", "embedding", "view"]}
    assert shapes == {"relevance": (2, 50, 50), "dimension_maps": (2, 3, 50, 50), "embedding": (2, 3),
                      "view": (2, 50, 50, 3)}, shapes
    assert result["view"].dtype == np.uint8, f"view has dtype {result['view'].dtype}"
    assert result["relevance"].dtype == np.float32 and result["dimension_maps"].dtype == np.float32
    assert result["labels"] == STAND_IN_MODEL["labels"], f"labels: {result['labels']}"
    assert result["files"] == [white_image, pattern_image], f"files: {result['files']}"
    assert result["model"] == "stand_in_3d", f"model: {result['model']!r}"
    expected_settings = dict(n_masks=30, grid=8, p=0.1, seed=0, normalization="pixel", input_size=[HEIGHT, WIDTH],
                             map_size=50)
    assert result["settings"] == expected_settings, f"settings: {result['settings']}"


def test_stand_in_relevance_is_the_weighted_average_of_the_dimension_maps(stand_in_network, pattern_image,
                                                                         white_image):
    result = dimpred.rise([pattern_image, white_image], STAND_IN_MODEL, n_masks=40, device="cpu")
    for i in range(2):
        expected = compute_aggregate_saliency(result["dimension_maps"][i].astype(float), result["embedding"][i])
        assert_close(result["relevance"][i], expected, 1e-5, f"relevance map of image {i + 1}")


def test_stand_in_view_is_the_image_as_the_network_sees_it(stand_in_network, pattern_image):
    from PIL import Image

    result = dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=5, map_size=None, device="cpu")
    expected = np.asarray(Image.open(pattern_image).convert("RGB").resize((WIDTH, HEIGHT)))
    np.testing.assert_array_equal(result["view"][0], expected, err_msg="view at the input size of the network")


def test_stand_in_maps_are_resized_to_map_size(stand_in_network, pattern_image):
    # the maps are computed at the input size of the network (24 x 40) and
    # then resized: they are close to the maps at the input size, resized
    # with PIL
    from PIL import Image

    small = dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=50, map_size=None, device="cpu")
    large = dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=50, map_size=80, device="cpu")
    assert small["dimension_maps"].shape == (1, 3, HEIGHT, WIDTH), f"{small['dimension_maps'].shape}"
    assert large["dimension_maps"].shape == (1, 3, 80, 80) and large["view"].shape == (1, 80, 80, 3)
    assert_close(large["embedding"], small["embedding"], 0, "embedding with and without resizing")
    for d in range(3):
        resized = np.asarray(Image.fromarray(small["dimension_maps"][0, d]).resize((80, 80), Image.BILINEAR))
        r = np.corrcoef(resized.ravel(), large["dimension_maps"][0, d].ravel())[0, 1]
        assert r > 0.99, f"map of dimension {d + 1}: r = {r:.4f} with the map at the input size, resized"


def test_stand_in_same_masks_for_every_image(stand_in_network, pattern_image, white_image):
    # the same image twice gives the same maps, also after another image
    result = dimpred.rise([pattern_image, white_image, pattern_image], STAND_IN_MODEL, n_masks=40, device="cpu")
    np.testing.assert_array_equal(result["dimension_maps"][0], result["dimension_maps"][2],
                                  err_msg="the first and the third image (the same file) have different maps")


def test_stand_in_batch_size_does_not_change_the_maps(stand_in_network, pattern_image):
    first = dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=45, batch_size=7, device="cpu")
    second = dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=45, batch_size=100, device="cpu")
    assert_close(first["dimension_maps"], second["dimension_maps"], 1e-6, "maps with batch size 7 and 100")


def test_stand_in_seed_sets_the_masks(stand_in_network, pattern_image):
    first = dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=30, seed=1, device="cpu")
    again = dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=30, seed=1, device="cpu")
    other = dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=30, seed=2, device="cpu")
    np.testing.assert_array_equal(first["relevance"], again["relevance"], err_msg="the same seed gave other maps")
    assert not np.allclose(first["relevance"], other["relevance"]), "another seed gave the same maps"


def test_preprocessing_without_values_from_0_to_1_gives_error(monkeypatch, pattern_image):
    # an open_clip network whose preprocessing does not end with Normalize:
    # rise cannot find the 0 to 1 image, which it needs for the masks
    torch = pytest.importorskip("torch", reason="the stand-in network needs torch")

    def preprocess(image):  # values 0 to 255, no Normalize step
        return torch.from_numpy(np.asarray(image.resize((WIDTH, HEIGHT)), dtype=np.float32)).permute(2, 0, 1)

    def create_model_and_transforms(network, pretrained=None, device=None):
        net = types.SimpleNamespace(eval=lambda: None, encode_image=lambda batch: batch.mean(dim=(2, 3)))
        return net, None, preprocess

    open_clip = types.ModuleType("open_clip")
    open_clip.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    with pytest.raises(ValueError, match="0 to 1"):
        dimpred.rise([pattern_image], STAND_IN_MODEL, n_masks=2, device="cpu")


def test_stand_in_broken_image_gives_error_before_any_masks(stand_in_network, pattern_image, broken_image):
    # a broken image anywhere in the list has to stop rise before the maps
    # of the images before it are computed (minutes each), with its name
    with pytest.raises(OSError, match="broken.jpg"):
        dimpred.rise([pattern_image, broken_image], STAND_IN_MODEL, n_masks=10, device="cpu")
    assert stand_in_network == [], f"the network got {len(stand_in_network)} batches before the error"


# --- rise with the real networks (slow)

VIT = "vitb32_66d_elastic"


@pytest.mark.slow
@pytest.mark.parametrize("model", [VIT, "alignet_siglip2b_66d_ridge"])
def test_unmasked_prediction_is_the_prediction_of_extract_features(request, cc0_paths, open_clip_available, model):
    if model == "alignet_siglip2b_66d_ridge":
        request.getfixturevalue("alignet_available")
    result = dimpred.rise(cc0_paths[1:2], model, n_masks=2, device="cpu")
    expected = dimpred.predict(dimpred.extract_features(cc0_paths[1:2], model, device="cpu"), model)
    assert_close(result["embedding"], expected, 1e-10, f"{model}: rise embedding vs predict(extract_features)")


@pytest.fixture(scope="module")
def vit_rise(open_clip_available, cc0_paths):
    """rise on the three CC0 images with vitb32_66d_elastic and 40 masks (default device)."""

    return dimpred.rise(cc0_paths, VIT, n_masks=40)


@pytest.mark.slow
def test_cc0_output_shapes(vit_rise):
    shapes = {key: np.shape(vit_rise[key]) for key in ["relevance", "dimension_maps", "embedding", "view"]}
    assert shapes == {"relevance": (3, 224, 224), "dimension_maps": (3, 66, 224, 224), "embedding": (3, 66),
                      "view": (3, 224, 224, 3)}, shapes


@pytest.mark.slow
def test_cc0_maps_are_finite_and_not_constant(vit_rise):
    for key in ["relevance", "dimension_maps", "embedding"]:
        assert np.all(np.isfinite(vit_rise[key])), f"{key} is not finite everywhere"
    for i in range(3):
        assert vit_rise["relevance"][i].std() > 0, f"the relevance map of image {i + 1} is constant"


@pytest.mark.slow
def test_cc0_labels_files_model_and_settings(vit_rise, cc0_paths):
    assert vit_rise["labels"] == dimpred.load_model(VIT)["labels"], "labels are not those of the model"
    assert vit_rise["files"] == cc0_paths, f"files: {vit_rise['files']}"
    assert vit_rise["model"] == VIT, f"model: {vit_rise['model']!r}"
    assert vit_rise["settings"] == dict(n_masks=40, grid=8, p=0.1, seed=0, normalization="pixel",
                                        input_size=[224, 224], map_size=224), f"settings: {vit_rise['settings']}"


@pytest.mark.slow
def test_cc0_relevance_is_the_weighted_average_of_the_dimension_maps(vit_rise):
    for i in range(3):
        expected = compute_aggregate_saliency(vit_rise["dimension_maps"][i].astype(float), vit_rise["embedding"][i])
        assert_close(vit_rise["relevance"][i], expected, 1e-5, f"relevance map of CC0 image {i + 1}")


@pytest.mark.slow
def test_cc0_view_is_the_clip_preprocessing_without_its_normalization(vit_rise, cc0_paths):
    # open_clip's preprocessing of ViT-B-32-quickgelu (resize, center crop,
    # normalization), with the normalization undone
    import open_clip

    from dimpred.extract_features import read_image

    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained=None)
    normalize = preprocess.transforms[-1]
    for i, fname in enumerate(cc0_paths):
        pixels = preprocess(read_image(fname)).numpy()
        pixels = pixels * np.array(normalize.std)[:, None, None] + np.array(normalize.mean)[:, None, None]
        assert_close(vit_rise["view"][i], np.round(pixels.transpose(1, 2, 0) * 255), 1, f"view of {fname}")


@pytest.mark.slow
def test_alignet_view_is_the_whole_image_resized(cc0_paths, alignet_available):
    from dimpred import alignet
    from dimpred.extract_features import read_image

    result = dimpred.rise(cc0_paths[:1], "alignet_siglip2b_66d_ridge", n_masks=4)
    expected = alignet.resize_cubic(np.asarray(read_image(cc0_paths[0])))
    np.testing.assert_array_equal(result["view"][0], expected, err_msg="AligNet view vs resize_cubic of the image")
    assert result["settings"]["input_size"] == [224, 224], f"input size {result['settings']['input_size']}"


@pytest.mark.slow
def test_map_size_argument(cc0_paths, open_clip_available):
    result = dimpred.rise(cc0_paths[:1], VIT, n_masks=10, map_size=100)
    assert result["relevance"].shape == (1, 100, 100), f"relevance has shape {result['relevance'].shape}"
    assert result["dimension_maps"].shape == (1, 66, 100, 100), f"{result['dimension_maps'].shape}"
    assert result["view"].shape == (1, 100, 100, 3), f"view has shape {result['view'].shape}"


@pytest.mark.slow
def test_rn50x64_maps_are_computed_at_448_and_returned_at_224(cc0_paths, open_clip_available):
    result = dimpred.rise(cc0_paths[2:3], "rn50x64_66d_ridge", n_masks=6, device="cpu")
    assert result["settings"]["input_size"] == [448, 448], f"input size {result['settings']['input_size']}"
    assert result["relevance"].shape == (1, 224, 224), f"relevance has shape {result['relevance'].shape}"
    assert result["view"].shape == (1, 224, 224, 3), f"view has shape {result['view'].shape}"
    expected = dimpred.predict(dimpred.extract_features(cc0_paths[2:3], "rn50x64_66d_ridge", device="cpu"),
                               "rn50x64_66d_ridge")
    assert_close(result["embedding"], expected, 1e-10, "RN50x64: rise embedding vs predict(extract_features)")


# --- the command line tool with the stand-in network (fast, needs torch)

@pytest.fixture
def stand_in_model_file(tmp_path):
    """STAND_IN_MODEL saved as a model file, for --model."""

    fname = str(tmp_path / "stand_in_3d.mat")
    scipy.io.savemat(fname, dict(weights=STAND_IN_MODEL["weights"], feature_mean=STAND_IN_MODEL["feature_mean"][None],
                                 feature_scale=STAND_IN_MODEL["feature_scale"][None],
                                 target_mean=STAND_IN_MODEL["target_mean"][None],
                                 labels=np.array(STAND_IN_MODEL["labels"], dtype=object).reshape(-1, 1),
                                 info=STAND_IN_MODEL["info"]))
    return fname


def run_main(args):
    """Run the command line tool in this process (with the stand-in network)."""

    dimpred.__main__.main([str(a) for a in args])


@pytest.mark.parametrize("extension", [".mat", ".npz"])
def test_command_line_saves_the_maps_of_rise(stand_in_network, stand_in_model_file, white_image, pattern_image,
                                             tmp_path, extension):
    out = str(tmp_path / ("heatmaps" + extension))
    run_main([pattern_image, white_image, "--rise", "--model", stand_in_model_file, "--n-masks", 30,
              "--device", "cpu", "--out", out])
    expected = dimpred.rise([pattern_image, white_image], dimpred.load_model(stand_in_model_file), n_masks=30,
                            device="cpu")
    if extension == ".mat":
        saved = scipy.io.loadmat(out, simplify_cells=True)
        settings = saved["settings"]
        settings["input_size"] = settings["input_size"].tolist()
    else:
        saved = dict(np.load(out))  # without allow_pickle: no object arrays
        settings = json.loads(str(saved["settings"]))
    for key in ["relevance", "dimension_maps", "embedding", "view"]:
        assert np.shape(saved[key]) == np.shape(expected[key]), f"{key}: {np.shape(saved[key])}"
        assert_close(saved[key], expected[key], 0, f"{key} in the {extension} file")
    assert saved["view"].dtype == np.uint8, f"view has dtype {saved['view'].dtype}"
    assert [str(f) for f in saved["labels"]] == expected["labels"], f"labels: {saved['labels']}"
    assert [str(f) for f in saved["files"]] == [pattern_image, white_image], f"files: {saved['files']}"
    assert str(saved["model"]) == "stand_in_3d", f"model: {saved['model']!r}"
    assert settings == expected["settings"], f"settings: {settings}"


def test_command_line_default_output_is_dimpred_heatmaps_mat(stand_in_network, stand_in_model_file, white_image,
                                                             tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_main([white_image, "--rise", "--model", stand_in_model_file, "--n-masks", 5, "--device", "cpu"])
    assert os.path.isfile(tmp_path / "dimpred_heatmaps.mat"), f"files: {os.listdir(tmp_path)}"


def test_command_line_writes_the_png_files(stand_in_network, stand_in_model_file, white_image, pattern_image,
                                           tmp_path):
    # per image: the relevance map and the maps of the 3 dimensions with the
    # largest predicted values, each on the view of the image
    from PIL import Image

    folder = tmp_path / "png"
    run_main([pattern_image, white_image, "--rise", "--model", stand_in_model_file, "--n-masks", 30,
              "--device", "cpu", "--out", tmp_path / "out.mat", "--png", folder])
    result = dimpred.rise([pattern_image, white_image], dimpred.load_model(stand_in_model_file), n_masks=30,
                          device="cpu")
    names = sorted(os.listdir(folder))
    assert len(names) == 8, f"PNG files: {names}"
    for i, stem in enumerate(["pattern", "white"]):
        top = np.argsort(-result["embedding"][i], kind="stable")[:3]
        expected = [f"{stem}_relevance.png"] + [f"{stem}_top{rank + 1}_dim{d + 1}.png" for rank, d in enumerate(top)]
        assert all(name in names for name in expected), f"expected {expected}, found {names}"
        relevance = np.asarray(Image.open(folder / f"{stem}_relevance.png"))
        assert relevance.shape == (224, 224, 3), f"PNG of the relevance map has shape {relevance.shape}"
        np.testing.assert_array_equal(relevance, rise_module.overlay(result["view"][i], result["relevance"][i]))
        first = np.asarray(Image.open(folder / expected[1]))
        np.testing.assert_array_equal(first,
                                      rise_module.overlay(result["view"][i], result["dimension_maps"][i, top[0]]))


def test_command_line_png_files_of_images_with_the_same_name_are_numbered(stand_in_network, stand_in_model_file,
                                                                         pattern_image, tmp_path):
    # the same file name in two folders must not overwrite the PNG files
    os.makedirs(tmp_path / "other")
    copy = str(tmp_path / "other" / "pattern.png")
    with open(pattern_image, "rb") as source, open(copy, "wb") as target:
        target.write(source.read())
    folder = tmp_path / "png"
    run_main([pattern_image, copy, "--rise", "--model", stand_in_model_file, "--n-masks", 5, "--device", "cpu",
              "--out", tmp_path / "out.mat", "--png", folder])
    names = sorted(os.listdir(folder))
    assert len(names) == 8, f"PNG files: {names}"
    assert "1_pattern_relevance.png" in names and "2_pattern_relevance.png" in names, f"PNG files: {names}"


@pytest.mark.parametrize("extension", [".NPZ", ".MAT"])
def test_command_line_keeps_an_extension_in_upper_case(stand_in_network, stand_in_model_file, white_image, tmp_path,
                                                       extension):
    # np.savez adds .npz to a name that does not end with .npz in lower case
    out = tmp_path / ("heatmaps" + extension)
    run_main([white_image, "--rise", "--model", stand_in_model_file, "--n-masks", 5, "--device", "cpu", "--out", out])
    names = [name for name in os.listdir(tmp_path) if name.startswith("heatmaps")]
    assert names == ["heatmaps" + extension], f"files written: {names}"


def test_command_line_broken_image_gives_error_before_any_masks(stand_in_network, stand_in_model_file,
                                                                pattern_image, broken_image, tmp_path):
    # one message that names the file, before the first image is computed
    with pytest.raises(SystemExit) as error:
        run_main([pattern_image, broken_image, "--rise", "--model", stand_in_model_file, "--n-masks", 10,
                  "--device", "cpu", "--out", tmp_path / "out.mat"])
    assert "broken.jpg" in str(error.value.code), f"the error should name the file: {error.value.code!r}"
    assert stand_in_network == [], f"the network got {len(stand_in_network)} batches before the error"


# --- wrong options of the command line tool (fast, no torch needed)

def assert_clear_error(result, *parts):
    assert result.returncode != 0, f"python -m dimpred should have failed\n{output_of(result)}"
    output = result.stdout + result.stderr
    for part in parts:
        assert part in output, f"the error message should mention {part!r}\n{output_of(result)}"
    assert "Traceback" not in output, f"error shown as a Python traceback\n{output_of(result)}"


@pytest.mark.parametrize("options, part", [(["--features", "f.mat"], "--features"),
                                           (["--features-only"], "--features-only"),
                                           (["--out", "out.csv"], ".npz"),
                                           (["--n-masks", "0"], "n_masks")])
def test_command_line_rise_with_wrong_options_gives_clear_error(tmp_path, cc0_paths, options, part):
    result = run_dimpred([cc0_paths[0], "--rise"] + options, cwd=tmp_path, without_torch=True)
    assert_clear_error(result, part)


@pytest.mark.parametrize("option", [["--n-masks", "100"], ["--png", "folder"]])
def test_command_line_rise_options_without_rise_give_clear_error(tmp_path, cc0_paths, option):
    result = run_dimpred([cc0_paths[0]] + option, cwd=tmp_path, without_torch=True)
    assert_clear_error(result, "--rise")


def test_command_line_rise_checks_the_output_folder_before_it_starts(tmp_path, cc0_paths):
    # a wrong output file would otherwise only show up after hours
    result = run_dimpred([cc0_paths[0], "--rise", "--out", str(tmp_path / "no_such_folder" / "out.mat")],
                         cwd=tmp_path, without_torch=True)
    assert_clear_error(result, "no_such_folder")


def test_command_line_help_explains_rise(tmp_path):
    result = run_dimpred(["--help"], cwd=tmp_path)
    assert result.returncode == 0, output_of(result)
    text = " ".join(result.stdout.split())
    for part in ["--rise", "--n-masks", "--png", "(default: 6000)", "dimpred_heatmaps.mat"]:
        assert part in text, f"the help should mention {part!r}\n{output_of(result)}"


# --- the command line tool with a real network (slow)

@pytest.fixture(scope="module")
def cli_rise(tmp_path_factory, cc0_paths, open_clip_available):
    """python -m dimpred --rise on one CC0 image with vitb32_66d_elastic and 8 masks, .mat and PNG files."""

    folder = tmp_path_factory.mktemp("cli_rise")
    result = run_dimpred([cc0_paths[0], "--rise", "--model", VIT, "--n-masks", "8", "--out", "out.mat",
                          "--png", "png"], cwd=folder)
    assert result.returncode == 0, f"python -m dimpred --rise failed\n{output_of(result)}"
    return folder, result


@pytest.mark.slow
def test_command_line_rise_mat_file_with_a_real_network(cli_rise, cc0_paths):
    folder, _ = cli_rise
    saved = scipy.io.loadmat(folder / "out.mat")  # without simplify_cells: the sizes as MATLAB sees them
    shapes = {key: saved[key].shape for key in ["relevance", "dimension_maps", "embedding", "view"]}
    assert shapes == {"relevance": (1, 224, 224), "dimension_maps": (1, 66, 224, 224), "embedding": (1, 66),
                      "view": (1, 224, 224, 3)}, shapes
    simple = scipy.io.loadmat(folder / "out.mat", simplify_cells=True)
    assert simple["model"] == VIT and simple["settings"]["n_masks"] == 8, f"{simple['model']}, {simple['settings']}"
    expected = dimpred.predict(dimpred.extract_features(cc0_paths[:1], VIT), VIT)
    assert_close(saved["embedding"], expected, 2e-3, "embedding in the .mat file vs predict(extract_features)")


@pytest.mark.slow
def test_command_line_rise_png_files_with_a_real_network(cli_rise, cc0_paths):
    folder, result = cli_rise
    names = sorted(os.listdir(folder / "png"))
    stem = os.path.splitext(os.path.basename(cc0_paths[0]))[0]
    assert len(names) == 4 and f"{stem}_relevance.png" in names, f"PNG files: {names}"
    assert "png" in result.stdout, f"the summary should name the PNG folder\n{output_of(result)}"
