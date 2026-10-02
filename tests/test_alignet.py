"""
Tests of dimpred/alignet.py, the PyTorch port of AligNet SigLIP2-B (the
network of the default model).

The most important test compares the features of the three CC0 images with
features of the same images from the released TensorFlow model (made with
TensorFlow, not with dimpred, see fixtures/make_fixtures.py). It needs the
weights (378 MB), which the tests never download: set DIMPRED_ALIGNET_WEIGHTS
to alignet_siglip2_b.safetensors, otherwise these tests are skipped.

The other tests run without the weights:
    - resize_cubic against a resize written here from the formula of the
      cubic kernel of OpenCV's INTER_CUBIC, with loops over the pixels (not
      against OpenCV itself, which is not installed)
    - preprocess: whole image to 224 x 224, values 0 to 1, no normalization
    - the activation functions: all GELUs have to be the tanh approximation
      (older timm versions use the exact GELU in the attention pooling head)
    - get_weights: DIMPRED_ALIGNET_WEIGHTS, the download into ~/.cache/dimpred
      (from a local file instead of GitHub), the sha256 check, clear errors
      when the download fails, and downloads that run at the same time
All tests need torch (alignet.py is only imported for feature extraction).

Hebartlab, 2026/10/02

See also: ../dimpred/alignet.py, test_extract_features.py
"""

import hashlib
import os

import numpy as np
import pytest

pytest.importorskip("torch", reason="dimpred.alignet needs torch")

import dimpred  # noqa: E402
from dimpred import alignet  # noqa: E402
from helpers import TOL_ALIGNET_DIFF, TOL_EXTRACT_R, assert_close, row_correlations  # noqa: E402

# History:
# 2026/10/02: tests of the activation functions (timm version) and of
#   downloads that run at the same time, after the review
# 2026/10/02: written together with dimpred/alignet.py


def cubic_kernel(x, a=-0.75):
    """Weight of the cubic convolution kernel (Keys) at distance x, as used by OpenCV with a = -0.75."""

    x = abs(x)
    if x <= 1:
        return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1
    if x < 2:
        return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a
    return 0.0


def resize_by_definition(pixels, size):
    """Slow reference of cv2.resize(pixels, (size, size), interpolation=cv2.INTER_CUBIC) for uint8.

    For each output pixel, the position in the input is (i + 0.5) * scale -
    0.5. The 4 input pixels around it are weighted with the cubic kernel,
    pixels outside the image are the border pixels, and the result is
    rounded and clipped to 0 to 255. Rows first, then columns.
    """

    def one_axis(n_in):
        weights = np.zeros((size, n_in))
        for i in range(size):
            position = (i + 0.5) * n_in / size - 0.5
            first = int(np.floor(position))
            for k in range(first - 1, first + 3):
                weights[i, min(max(k, 0), n_in - 1)] += cubic_kernel(position - k)
        return weights

    pixels = np.asarray(pixels, dtype=float)
    rows, columns = one_axis(pixels.shape[0]), one_axis(pixels.shape[1])
    resized = np.stack([rows @ pixels[:, :, c] @ columns.T for c in range(pixels.shape[2])], axis=2)
    return np.clip(np.floor(resized + 0.5), 0, 255).astype(np.uint8)


def sha256_of(fname):
    with open(fname, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --- resize_cubic and preprocess (no weights needed)

@pytest.mark.parametrize("shape", [(37, 53, 3), (300, 250, 3), (224, 300, 3)])
def test_resize_cubic_is_the_cubic_resize_by_its_formula(shape):
    # smaller and larger images, one with the right height already. Both are
    # computed in float64, so only values that are exactly x.5 before
    # rounding could come out differently; random images hardly have them.
    pixels = np.random.default_rng(sum(shape)).integers(0, 256, shape, dtype=np.uint8)
    resized = alignet.resize_cubic(pixels)
    expected = resize_by_definition(pixels, 224)
    assert resized.dtype == np.uint8 and resized.shape == (224, 224, 3), f"{resized.dtype}, {resized.shape}"
    n_different = int(np.sum(resized != expected))
    assert n_different <= 2, f"{n_different} of {expected.size} values differ from the cubic resize by definition"
    assert np.abs(resized.astype(int) - expected).max() <= 1, "values differ by more than 1"


def test_resize_cubic_of_a_224_image_changes_nothing():
    pixels = np.random.default_rng(1).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    np.testing.assert_array_equal(alignet.resize_cubic(pixels), pixels, err_msg="resize to the same size changed pixels")


def test_resize_cubic_keeps_a_constant_image():
    pixels = np.full((100, 61, 3), 77, dtype=np.uint8)
    np.testing.assert_array_equal(alignet.resize_cubic(pixels), np.full((224, 224, 3), 77, dtype=np.uint8))


def test_preprocess_gives_3_x_224_x_224_values_from_0_to_1_without_normalization():
    # white is 1 and black is 0 (no mean or std of a dataset is subtracted),
    # and the whole image is resized, not cropped (the right half stays black)
    from PIL import Image

    pixels = np.zeros((90, 400, 3), dtype=np.uint8)
    pixels[:, :200] = 255
    tensor = alignet.preprocess(Image.fromarray(pixels))
    assert tuple(tensor.shape) == (3, 224, 224), f"shape is {tuple(tensor.shape)}"
    assert str(tensor.dtype) == "torch.float32", f"dtype is {tensor.dtype}"
    values = tensor.numpy()
    assert_close(values[:, :, :100], np.ones((3, 224, 100)), 0, "left half (white)")
    assert_close(values[:, :, 124:], np.zeros((3, 224, 100)), 0, "right half (black)")


def test_preprocess_is_resize_cubic_divided_by_255(cc0_paths):
    from PIL import Image

    image = Image.open(cc0_paths[0]).convert("RGB")
    expected = alignet.resize_cubic(np.asarray(image)).transpose(2, 0, 1) / 255
    assert_close(alignet.preprocess(image).numpy(), expected, 1e-7, "preprocess vs resize_cubic / 255")


# --- the network without its weights

def test_all_gelu_activations_are_the_tanh_approximation():
    # big_vision uses GELU with the tanh approximation everywhere: in the 12
    # blocks and in the MLP of the attention pooling head. timm before 1.0.15
    # does not pass act_layer on to the attention pooling head, which then
    # uses the exact GELU, with the same parameter names, so the weights load
    # without any error, but the features change by up to 3e-3.
    net = alignet.AligNet()
    gelus = {name: type(module).__name__ for name, module in net.named_modules() if "GELU" in type(module).__name__}
    assert len(gelus) == 13, f"{len(gelus)} GELU activations, expected 13 (12 blocks and the pooling head): {gelus}"
    assert set(gelus.values()) == {"GELUTanh"}, f"not all GELU activations are the tanh approximation: {gelus}"


def test_timm_with_the_exact_gelu_in_the_pooling_head_gives_an_error(monkeypatch):
    # as with timm before 1.0.15: the error has to say that timm is too old
    import timm
    import torch

    create_model = timm.create_model

    def create_model_of_old_timm(*args, **kwargs):
        model = create_model(*args, **kwargs)
        model.attn_pool.mlp.act = torch.nn.GELU()
        return model

    monkeypatch.setattr(timm, "create_model", create_model_of_old_timm)
    with pytest.raises(ImportError) as error:
        alignet.AligNet()
    message = str(error.value)
    for part in ["timm", "1.0.15", timm.__version__]:
        assert part in message, f"the error should mention {part!r}: {message!r}"


# --- get_weights: where the weights come from (no real weights needed)

@pytest.fixture
def fake_weights(tmp_path, monkeypatch):
    """A small file in place of the released weights, served from a file:// URL, and an empty home folder.

    The sha256 that dimpred expects is set to the sha256 of this file, so
    the download and the check work as with the real file.
    """

    source = tmp_path / "server" / "alignet_siglip2_b.safetensors"
    source.parent.mkdir()
    source.write_bytes(b"not really the weights, but some bytes" * 100)
    monkeypatch.setattr(alignet, "WEIGHTS_URL", source.as_uri())
    monkeypatch.setattr(alignet, "WEIGHTS_SHA256", sha256_of(source))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DIMPRED_ALIGNET_WEIGHTS", raising=False)
    return source


def test_get_weights_downloads_into_the_cache_folder(fake_weights, tmp_path):
    fname = alignet.get_weights()
    expected = tmp_path / "home" / ".cache" / "dimpred" / "alignet_siglip2_b.safetensors"
    assert os.path.samefile(fname, expected), f"weights file is {fname}, expected {expected}"
    assert sha256_of(fname) == sha256_of(fake_weights), "the downloaded file differs from the source"
    assert os.listdir(expected.parent) == ["alignet_siglip2_b.safetensors"], "files left in the cache folder"


def test_get_weights_downloads_only_once(fake_weights):
    first = alignet.get_weights()
    fake_weights.unlink()  # a second download would fail now
    assert alignet.get_weights() == first


def test_failed_download_gives_one_line_error_that_says_what_to_do(fake_weights, tmp_path, monkeypatch):
    monkeypatch.setattr(alignet, "WEIGHTS_URL", (tmp_path / "no_such_file.safetensors").as_uri())
    with pytest.raises(OSError) as error:
        alignet.get_weights()
    message = str(error.value)
    assert "\n" not in message, f"the error should be one line: {message!r}"
    for part in ["DIMPRED_ALIGNET_WEIGHTS", "no_such_file.safetensors"]:
        assert part in message, f"the error should mention {part!r}: {message!r}"
    cache = tmp_path / "home" / ".cache" / "dimpred"
    assert not cache.exists() or not os.listdir(cache), f"files left in the cache folder: {os.listdir(cache)}"


def test_download_with_the_wrong_sha256_is_not_kept(fake_weights, tmp_path, monkeypatch):
    monkeypatch.setattr(alignet, "WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(OSError, match="sha256"):
        alignet.get_weights()
    cache = tmp_path / "home" / ".cache" / "dimpred"
    assert not os.listdir(cache), f"a file with the wrong sha256 was kept: {os.listdir(cache)}"


def test_download_does_not_touch_the_file_of_another_download(fake_weights, tmp_path):
    # Several jobs can start at the same time with the same home folder (e.g.
    # an array job on a cluster). Each download has to go into its own
    # temporary file, so that they do not overwrite or delete each other's.
    # The file of the other download is given the name that dimpred used
    # for all downloads before.
    cache = tmp_path / "home" / ".cache" / "dimpred"
    cache.mkdir(parents=True)
    other = cache / "alignet_siglip2_b.safetensors.part"
    other.write_bytes(b"the first half of the download of another job")
    fname = alignet.get_weights()
    assert sha256_of(fname) == sha256_of(fake_weights), "the downloaded file differs from the source"
    assert other.exists(), "the file of the other download was deleted"
    assert other.read_bytes() == b"the first half of the download of another job", (
        "the file of the other download was changed")
    assert sorted(os.listdir(cache)) == sorted([other.name, "alignet_siglip2_b.safetensors"]), (
        f"files left in the cache folder: {os.listdir(cache)}")


def test_environment_variable_gives_the_weights_file(fake_weights, monkeypatch):
    monkeypatch.setenv("DIMPRED_ALIGNET_WEIGHTS", str(fake_weights))
    monkeypatch.setattr(alignet, "WEIGHTS_URL", "file:///no/download/allowed")
    assert alignet.get_weights() == str(fake_weights)


def test_environment_variable_with_a_missing_file_gives_clear_error(fake_weights, tmp_path, monkeypatch):
    monkeypatch.setenv("DIMPRED_ALIGNET_WEIGHTS", str(tmp_path / "missing.safetensors"))
    with pytest.raises(FileNotFoundError) as error:
        alignet.get_weights()
    assert "DIMPRED_ALIGNET_WEIGHTS" in str(error.value), f"message: {str(error.value)!r}"


def test_environment_variable_with_the_wrong_file_gives_clear_error(fake_weights, tmp_path, monkeypatch):
    wrong = tmp_path / "wrong.safetensors"
    wrong.write_bytes(b"something else")
    monkeypatch.setenv("DIMPRED_ALIGNET_WEIGHTS", str(wrong))
    with pytest.raises(OSError, match="sha256"):
        alignet.get_weights()


def test_weights_url_is_the_github_release():
    assert alignet.WEIGHTS_URL == ("https://github.com/ViCCo-Group/dimpred/releases/download/alignet-weights-v1/"
                                   "alignet_siglip2_b.safetensors")
    assert alignet.WEIGHTS_SHA256 == "2ce461e04ac11271c736d32477d873f7d14fe6932c73d8757fec2854c6482647"


# --- the network (needs the weights)

@pytest.fixture(scope="module")
def alignet_features(alignet_available, cc0_paths):
    """AligNet features of the CC0 images with the default model, in cc0_files order."""

    return dimpred.extract_features(cc0_paths)


@pytest.mark.slow
def test_default_model_gives_the_tensorflow_features(ref, alignet_features):
    expected = ref["cc0_features_alignet"]
    r = row_correlations(alignet_features, expected)
    assert r.min() > TOL_EXTRACT_R, f"correlation with the TensorFlow features is {np.round(r, 6).tolist()}"
    assert_close(alignet_features, expected, TOL_ALIGNET_DIFF, "AligNet features vs TensorFlow")


@pytest.mark.slow
def test_alignet_features_are_768_float32_per_image(alignet_features):
    assert alignet_features.dtype == np.float32, f"dtype is {alignet_features.dtype}"
    assert alignet_features.shape == (3, 768), f"shape is {alignet_features.shape}, expected (3, 768)"


@pytest.mark.slow
def test_alignet_rows_are_in_the_given_order(ref, alignet_available, cc0_paths_reordered):
    paths, rows = cc0_paths_reordered
    features = dimpred.extract_features(paths, batch_size=2, device="cpu")
    assert_close(features, ref["cc0_features_alignet"][rows], TOL_ALIGNET_DIFF,
                 "AligNet features of images given in a second order (cpu, batch size 2)")


@pytest.mark.slow
def test_network_argument_selects_alignet(ref, alignet_available, cc0_paths):
    features = dimpred.extract_features(cc0_paths[:1], model="rn50x64_49d_ridge", network="AligNet SigLIP2-B")
    assert_close(features, ref["cc0_features_alignet"][:1], TOL_ALIGNET_DIFF, "features with network='AligNet SigLIP2-B'")


@pytest.mark.slow
def test_load_alignet_gives_the_three_outputs(alignet_available, cc0_paths):
    import torch
    from PIL import Image

    net = alignet.load_alignet(device="cpu")
    images = torch.stack([alignet.preprocess(Image.open(f).convert("RGB")) for f in cc0_paths])
    with torch.no_grad():
        out = net(images)
    shapes = {key: tuple(value.shape) for key, value in out.items()}
    assert shapes == {"pre_logits": (3, 768), "triplet_logits": (3, 1024), "i1k_logits": (3, 1000)}, shapes
