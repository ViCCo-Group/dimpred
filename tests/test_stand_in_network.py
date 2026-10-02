"""
Fast tests of dimpred.extract_features and the command line tool with a
stand-in for the network (no open_clip, no network weights).

The slow tests (test_extract_features.py, test_cli.py) find these mistakes
with the real networks, but they do not run with -m "not slow":
    - the wrong network, e.g. open_clip's "ViT-B-32" instead of
      "ViT-B-32-quickgelu"
    - rows in sorted file order instead of the given order, in
      extract_features and in the command line tool
    - feature names in the csv header that count from 0
Here open_clip and the AligNet network (dimpred.alignet.load_alignet and
preprocess) are replaced by stand-ins that record which network was asked
for and return one feature per image, its mean pixel value. The stand-ins
use torch (torch.stack and torch.no_grad are called by extract_features), so
the tests are skipped without torch.

Hebartlab, 2026/09/30

See also: test_extract_features.py, test_cli.py
"""

import csv
import sys
import types

import numpy as np
import pytest

import dimpred
import dimpred.__main__
from helpers import ALIGNET, MODEL_NAMES, MODELS, assert_close

# History:
# 2026/10/02: stand-in for the AligNet network of the new default model
# 2026/09/30: new file (these mistakes were only found by the slow tests)

SIZE = 8  # the stand-in preprocessing makes every image SIZE x SIZE pixels


def mean_pixel_value(fname):
    """The feature that the stand-in network gives for an image."""

    from PIL import Image

    with Image.open(fname) as image:
        return np.array(image.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32).mean()


@pytest.fixture
def stand_in_network(monkeypatch):
    """Replace open_clip and AligNet by stand-ins. Returns a dict with the network and weights that were asked for.

    For AligNet, pretrained is recorded as "alignet" (load_alignet has no
    weights argument in extract_features, the weights come from get_weights).
    """

    torch = pytest.importorskip("torch", reason="the stand-in network needs torch")
    import dimpred.alignet

    asked_for = {}

    def preprocess(image):  # PIL image to a tensor of 3 x SIZE x SIZE
        pixels = np.array(image.resize((SIZE, SIZE)), dtype=np.float32)
        return torch.from_numpy(pixels).permute(2, 0, 1)

    def one_feature(batch):  # one feature per image
        return batch.mean(dim=(1, 2, 3)).reshape(-1, 1)

    def create_model_and_transforms(network, pretrained=None, device=None):
        asked_for["network"] = network
        asked_for["pretrained"] = pretrained
        net = types.SimpleNamespace(eval=lambda: None, encode_image=one_feature)
        return net, None, preprocess

    def load_alignet(weights_file=None, device="cpu"):
        asked_for["network"] = ALIGNET
        asked_for["pretrained"] = "alignet"
        return lambda batch: {"pre_logits": one_feature(batch), "triplet_logits": None, "i1k_logits": None}

    open_clip = types.ModuleType("open_clip")
    open_clip.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)  # restored after the test
    monkeypatch.setattr(dimpred.alignet, "load_alignet", load_alignet)
    monkeypatch.setattr(dimpred.alignet, "preprocess", preprocess)
    return asked_for


# --- network

def test_default_model_uses_alignet(stand_in_network, cc0_paths):
    dimpred.extract_features(cc0_paths[:1], device="cpu")
    asked_for = (stand_in_network["network"], stand_in_network["pretrained"])
    assert asked_for == (ALIGNET, "alignet"), f"the default model asked for {asked_for}"


def test_vit_model_asks_for_the_quickgelu_network(stand_in_network, cc0_paths):
    dimpred.extract_features(cc0_paths[:1], model="vitb32_66d_elastic", device="cpu")
    asked_for = (stand_in_network["network"], stand_in_network["pretrained"])
    assert asked_for == ("ViT-B-32-quickgelu", "openai"), f"vitb32_66d_elastic asked open_clip for {asked_for}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_sets_the_network(stand_in_network, cc0_paths, name):
    dimpred.extract_features(cc0_paths[:1], model=name, device="cpu")
    asked_for = (stand_in_network["network"], stand_in_network["pretrained"])
    expected = (MODELS[name]["network"], "alignet" if MODELS[name]["network"] == ALIGNET else "openai")
    assert asked_for == expected, f"{name} asked for {asked_for}, expected {expected}"


def test_network_argument_selects_alignet(stand_in_network, cc0_paths):
    # "alignet": load_alignet was called, not open_clip with the name AligNet SigLIP2-B
    dimpred.extract_features(cc0_paths[:1], model="vitb32_66d_elastic", network=ALIGNET, device="cpu")
    asked_for = (stand_in_network["network"], stand_in_network["pretrained"])
    assert asked_for == (ALIGNET, "alignet"), f"network {ALIGNET!r} asked for {asked_for}"


# --- order of the rows

@pytest.mark.parametrize("model", ["alignet_siglip2b_66d_ridge", "vitb32_66d_elastic"])
@pytest.mark.parametrize("batch_size", [1, 2, 32])
def test_rows_are_in_the_given_order(stand_in_network, cc0_paths_reordered, batch_size, model):
    paths, _ = cc0_paths_reordered
    expected = [mean_pixel_value(p) for p in paths]
    assert len(set(np.round(expected))) == len(paths), "the test images should differ in their mean pixel value"
    features = dimpred.extract_features(paths, model=model, device="cpu", batch_size=batch_size)
    assert_close(features[:, 0], expected, 1e-3, f"stand-in features of images given in a second order "
                                                 f"({model}, batch_size={batch_size})")


def test_command_line_tool_keeps_the_given_order_and_counts_features_from_1(stand_in_network,
                                                                            cc0_paths_reordered, tmp_path):
    paths, _ = cc0_paths_reordered
    dimpred.__main__.main(paths + ["--features-only", "--device", "cpu", "--out", str(tmp_path / "out.csv")])
    with open(tmp_path / "out.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["image", "feature_1"], f"csv header is {rows[0]}"
    assert [row[0] for row in rows[1:]] == paths, f"image column is {[row[0] for row in rows[1:]]}"
    assert_close([float(row[1]) for row in rows[1:]], [mean_pixel_value(p) for p in paths], 1e-3,
                 "stand-in features in the csv file")
