"""
Tests of the warning for images that are not square (warn_non_square and
square_input_of in dimpred/extract_features.py). These run without torch and
without a network, except the last ones, which check with stand-in networks
(as in test_stand_in_network.py) that extract_features and rise give the
warning.

Hebartlab, 2026/10/26

See also: test_stand_in_network.py, test_rise.py
"""

import sys
import types
import warnings

import numpy as np
import pytest

import dimpred
import dimpred.alignet
from dimpred.extract_features import square_input_of, warn_non_square
from helpers import ALIGNET

pytest.importorskip("PIL", reason="the test images are written with pillow")


def write_image(folder, name, width, height):
    """A gray image of width x height pixels, returns its path."""

    from PIL import Image

    fname = str(folder / name)
    Image.new("RGB", (width, height), (128, 128, 128)).save(fname)
    return fname


def non_square_warnings(images, square_input, network="RN50x64"):
    """The warnings of warn_non_square, as a list of their messages."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_non_square(images, network, square_input)
    return [str(w.message) for w in caught]


# --- warn_non_square

@pytest.mark.parametrize("square_input, what", [("crop", "removes the ends of their longer edge"),
                                                ("squeeze", "squeezes their longer edge")])
def test_one_warning_with_the_number_of_images_and_what_the_network_does(tmp_path, square_input, what):
    images = [write_image(tmp_path, "square.png", 40, 40), write_image(tmp_path, "wide.png", 40, 30),
              write_image(tmp_path, "tall.png", 30, 40)]
    messages = non_square_warnings(images, square_input)
    assert len(messages) == 1, f"expected one warning for all images, got {messages}"
    assert messages[0].startswith("Not all images are square: 2 of 3 images"), messages[0]
    assert "wide.png, 40 x 30" in messages[0], f"the first non-square image is not named: {messages[0]}"
    assert what in messages[0], f"the warning for {square_input} does not say what happens: {messages[0]}"


def test_no_warning_for_square_images(tmp_path):
    images = [write_image(tmp_path, "a.png", 40, 40), write_image(tmp_path, "b.png", 500, 500)]
    assert non_square_warnings(images, "crop") == []


def test_no_warning_for_a_difference_within_one_percent(tmp_path):
    images = [write_image(tmp_path, "almost.png", 200, 199)]
    assert non_square_warnings(images, "squeeze") == []


def test_no_warning_if_the_network_does_not_squeeze_or_crop(tmp_path):
    images = [write_image(tmp_path, "wide.png", 40, 30)]
    assert non_square_warnings(images, None) == []


def test_file_that_is_not_an_image_is_skipped(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_text("not an image")
    images = [str(broken), write_image(tmp_path, "wide.png", 40, 30)]
    messages = non_square_warnings(images, "crop")
    assert len(messages) == 1 and "1 of 2 images" in messages[0], messages


def test_warning_can_be_switched_off_by_its_message(tmp_path):
    images = [write_image(tmp_path, "wide.png", 40, 30)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message="Not all images are square")
        warn_non_square(images, "RN50x64", "crop")
    assert caught == [], "the documented filter does not switch the warning off"


# --- square_input_of

class CenterCrop:
    pass


class CenterCropOrPad:
    pass


class ResizeKeepRatio:
    pass


class Resize:
    def __init__(self, size):
        self.size = size


@pytest.mark.parametrize("steps, expected", [
    ([Resize(448), CenterCrop()], "crop"),            # OpenAI's CLIP networks
    ([Resize((224, 224))], "squeeze"),                # e.g. SigLIP
    ([ResizeKeepRatio(), CenterCropOrPad()], None),   # open_clip's "longest": pads
    ([Resize(224)], None),
    ([], None),
])
def test_square_input_of_preprocessing_steps(steps, expected):
    assert square_input_of(steps) == expected


def test_square_input_of_open_clip_preprocessing():
    # the real torchvision steps of open_clip, without loading a network
    open_clip = pytest.importorskip("open_clip", reason="needs open_clip")
    crop = open_clip.image_transform(448, is_train=False)
    squeeze = open_clip.image_transform(224, is_train=False, resize_mode="squash")
    assert square_input_of(crop.transforms) == "crop"
    assert square_input_of(squeeze.transforms) == "squeeze"


# --- extract_features and rise give the warning

STAND_IN_MODEL = dict(weights=np.eye(1), feature_mean=np.zeros(1), feature_scale=np.ones(1), target_mean=np.zeros(1),
                      labels=["mean"], info=dict(name="stand_in_1d", network="stand-in network", pretrained="none"))


@pytest.fixture
def stand_in_networks(monkeypatch):
    """open_clip with a cropping stand-in network (torchvision steps) and AligNet by a stand-in; one feature, the mean."""

    torch = pytest.importorskip("torch", reason="the stand-in network needs torch")
    transforms = pytest.importorskip("torchvision.transforms", reason="the stand-in preprocessing needs torchvision")
    import dimpred.alignet

    def one_feature(batch):
        return batch.mean(dim=(1, 2, 3)).reshape(-1, 1)

    def create_model_and_transforms(network, pretrained=None, device=None):
        net = types.SimpleNamespace(eval=lambda: None, encode_image=one_feature)
        steps = [transforms.Resize(8), transforms.CenterCrop(8), transforms.ToTensor(),
                 transforms.Normalize((0.5,) * 3, (0.25,) * 3)]
        return net, None, transforms.Compose(steps)

    def preprocess(image):  # AligNet: resized to a square
        return torch.from_numpy(np.array(image.resize((8, 8)), dtype=np.float32) / 255).permute(2, 0, 1)

    open_clip = types.ModuleType("open_clip")
    open_clip.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)  # restored after the test
    monkeypatch.setattr(dimpred.alignet, "load_alignet", lambda weights_file=None, device="cpu":
                        lambda batch: {"pre_logits": one_feature(batch)})
    monkeypatch.setattr(dimpred.alignet, "preprocess", preprocess)


@pytest.mark.parametrize("network, what", [("stand-in network", "removes the ends of their longer edge"),
                                           (ALIGNET, "squeezes their longer edge")])
def test_extract_features_warns_for_non_square_images(stand_in_networks, tmp_path, network, what):
    images = [write_image(tmp_path, "square.png", 40, 40), write_image(tmp_path, "wide.png", 40, 30)]
    with pytest.warns(UserWarning, match="Not all images are square: 1 of 2") as caught:
        dimpred.extract_features(images, network=network, device="cpu")
    assert what in str(caught[0].message), str(caught[0].message)


def test_extract_features_does_not_warn_for_square_images(stand_in_networks, tmp_path):
    images = [write_image(tmp_path, "square.png", 40, 40)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dimpred.extract_features(images, network="stand-in network", device="cpu")


def test_rise_warns_for_non_square_images(stand_in_networks, tmp_path):
    images = [write_image(tmp_path, "wide.png", 40, 30)]
    with pytest.warns(UserWarning, match="Not all images are square: 1 of 1"):
        dimpred.rise(images, STAND_IN_MODEL, n_masks=4, map_size=8, device="cpu")
