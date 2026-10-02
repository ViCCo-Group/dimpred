"""
AligNet SigLIP2-B in PyTorch, the network of the default model.

AligNet (Muttenthaler et al., 2025, Nature, github.com/google-deepmind/alignet)
is the SigLIP2 ViT-B/16 image encoder at 224 px, fine-tuned on odd-one-out
choices of a teacher network that was aligned with the human odd-one-out
judgments of THINGS (the data behind the SPoSE dimensions). It was released
as a TensorFlow SavedModel only (SigLIP2-B-alignet). We converted its
weights to PyTorch (training/alignet). For the same input arrays, the
outputs are those of the TensorFlow model up to float32 rounding (largest
difference 3e-5 on 2880 images). From the image file, the features differ
from those of TensorFlow by up to 3e-4 (for pixel values that are exactly
x.5 before rounding in the resize). Neither TensorFlow nor OpenCV is needed.

The image encoder is timm's vit_base_patch16_siglip_224 (the image tower of
open_clip's ViT-B-16-SigLIP2), except that GELU is the tanh approximation, as
in big_vision. This needs timm 1.0.15 or newer. dimpred uses pre_logits, the
output of the attention pooling head (768 features).

The weights (378 MB) are not part of the package. They are downloaded on
first use from the release alignet-weights-v1 of github.com/ViCCo-Group/dimpred
into ~/.cache/dimpred and checked with their sha256. If you cannot download
them there, download alignet_siglip2_b.safetensors yourself and set the
environment variable DIMPRED_ALIGNET_WEIGHTS to its path.

License: see training/alignet/ALIGNET_MODELS_LICENSE.md (Apache 2.0 for
software, CC-BY 4.0 for other materials; SigLIP 2 is Apache 2.0).
Changes: the weights were converted from the TensorFlow SavedModel to
PyTorch (safetensors), without changing their values. If you use the
default model, please cite Muttenthaler, L., Greff, K., Born, F., Spitzer,
B., Kornblith, S., Mozer, M. C., Mueller, K.-R., Unterthiner, T., &
Lampinen, A. K. (2025). Aligning machine and human visual representations
across abstraction levels. Nature 647, 349-355, in addition to the DimPred
paper.

Functions:
    load_alignet  the network with its weights
    get_weights   path of the weights file (downloads it the first time)
    preprocess    image to network input (resize to 224 x 224, values 0 to 1)
    resize_cubic  resize with the formula of OpenCV's INTER_CUBIC, in numpy

Hebartlab, 2026/10/02

See also: extract_features
"""

import hashlib
import os
import shutil
import urllib.request
import uuid

import numpy as np
import torch

# History:
# 2026/10/02: check of the activation functions (timm version); each
#   download goes to its own temporary file; resize_cubic converts only the
#   rows it needs to float64
# 2026/10/02: written for dimpred, from the port of 2026/10/01 (training/alignet)

# The name of the network in the model files (info["network"]). extract_features
# uses this module when a model has this network.
NETWORK = "AligNet SigLIP2-B"

# The weights file: where it is downloaded from, its name and its checksum
WEIGHTS_URL = ("https://github.com/ViCCo-Group/dimpred/releases/download/alignet-weights-v1/"
               "alignet_siglip2_b.safetensors")
WEIGHTS_NAME = "alignet_siglip2_b.safetensors"
WEIGHTS_SHA256 = "2ce461e04ac11271c736d32477d873f7d14fe6932c73d8757fec2854c6482647"

IMAGE_SIZE = 224


class AligNet(torch.nn.Module):
    """
    AligNet SigLIP2-B image encoder with its two heads. Use load_alignet to
    get it with the weights.

    forward(images) takes float32 images, n x 3 x 224 x 224, values 0 to 1
    (see preprocess), and returns a dict with
        pre_logits:     n x 768, the output of the attention pooling head
                        (the features dimpred uses)
        triplet_logits: n x 1024, pre_logits through the linear head that
                        AligNet was trained with
        i1k_logits:     n x 1000, ImageNet logits (LayerNorm and linear
                        layer on pre_logits)
    """

    def __init__(self):
        super().__init__()
        import timm  # comes with open_clip_torch

        # big_vision uses GELU with the tanh approximation, timm's default
        # is the exact GELU, which changes the features by up to 1.7e-2
        self.encoder = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=0,
                                         act_layer="gelu_tanh")
        # timm before 1.0.15 does not pass act_layer on to the MLP of the
        # attention pooling head, which then uses the exact GELU. The
        # parameter names are the same, so the weights load without any
        # error, but the features change by up to 3e-3. So we check all
        # activation functions.
        activations = [type(block.mlp.act).__name__ for block in self.encoder.blocks]
        activations.append(type(self.encoder.attn_pool.mlp.act).__name__)
        if set(activations) != {"GELUTanh"}:
            raise ImportError(f"AligNet needs timm 1.0.15 or newer (installed: {timm.__version__}). Older "
                              f"versions build the network with another activation function, which gives "
                              f"wrong features. Please update timm (pip install -U timm).")
        self.triplet_head = torch.nn.Linear(768, 1024)
        self.i1k_norm = torch.nn.LayerNorm(768, eps=1e-6)
        self.i1k_head = torch.nn.Linear(768, 1000)

    def forward(self, images):
        pre_logits = self.encoder(images)
        return {"pre_logits": pre_logits,
                "triplet_logits": self.triplet_head(pre_logits),
                "i1k_logits": self.i1k_head(self.i1k_norm(pre_logits))}


def load_alignet(weights_file=None, device="cpu"):
    """
    net = load_alignet(weights_file=None, device="cpu")

    The AligNet SigLIP2-B network with its weights, in eval mode. The
    outputs are those of the TensorFlow SavedModel (SigLIP2-B-alignet,
    signature serving_default), see AligNet.

    Input:
        weights_file: path of alignet_siglip2_b.safetensors (default: the
                      file from get_weights, downloaded the first time)
        device:       "cpu", "mps" or "cuda" (default: "cpu")

    Output:
        net: AligNet (torch module); call net(images) inside torch.no_grad()

    Example:
        net = load_alignet()
        images = torch.stack([preprocess(read_image(f)) for f in files])
        with torch.no_grad():
            features = net(images)["pre_logits"]

    Hebartlab, 2026/10/02

    See also: get_weights, preprocess
    """

    from safetensors.torch import load_file  # comes with open_clip_torch and timm

    if weights_file is None:
        weights_file = get_weights()
    net = AligNet()
    net.load_state_dict(load_file(weights_file))  # strict: all weights have to fit
    return net.eval().to(device)


def get_weights():
    """
    fname = get_weights()

    Path of the AligNet weights file. If the environment variable
    DIMPRED_ALIGNET_WEIGHTS is set, this is the file it points to. Otherwise
    it is ~/.cache/dimpred/alignet_siglip2_b.safetensors, which is
    downloaded the first time (378 MB, from WEIGHTS_URL). A file that does
    not have the sha256 of the released weights gives an error, also when it
    is given with DIMPRED_ALIGNET_WEIGHTS (e.g. after an incomplete
    download), and a downloaded file is only kept if it has the right sha256.
    Computing the sha256 takes 0.2 s (Apple M1 Max).

    Input:
        none

    Output:
        fname: path of alignet_siglip2_b.safetensors

    Hebartlab, 2026/10/02

    See also: load_alignet
    """

    fname = os.environ.get("DIMPRED_ALIGNET_WEIGHTS", "")
    if fname:
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"The environment variable DIMPRED_ALIGNET_WEIGHTS is {fname}, but this file "
                                    f"does not exist. It has to point to {WEIGHTS_NAME}.")
    else:
        fname = os.path.join(os.path.expanduser("~"), ".cache", "dimpred", WEIGHTS_NAME)
        if not os.path.isfile(fname):
            download_weights(fname)
    if file_sha256(fname) != WEIGHTS_SHA256:
        raise OSError(f"The AligNet weights {fname} are not the released file (wrong sha256), maybe the download "
                      f"was incomplete. Please delete it and try again, or download {WEIGHTS_URL} and set the "
                      f"environment variable DIMPRED_ALIGNET_WEIGHTS to its path.")
    return fname


def download_weights(fname):
    """
    download_weights(fname)

    Download the AligNet weights from WEIGHTS_URL to fname. The download goes
    to a temporary file in the same folder first and is only renamed to
    fname if it has the right sha256, so an interrupted download never looks
    like a complete file. Each download has its own temporary file, so
    downloads that run at the same time (e.g. several jobs on a cluster with
    the same home folder) do not overwrite each other. If the download
    fails, the error says what to do instead.
    """

    print(f"Downloading the AligNet SigLIP2-B weights (378 MB, only needed once) to {fname}", flush=True)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    partial = f"{fname}.{uuid.uuid4().hex[:12]}.part"  # a new name for each download
    try:
        try:
            with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as response, open(partial, "xb") as f:
                shutil.copyfileobj(response, f, 2 ** 20)
        except (OSError, ValueError) as error:  # no connection, file not found on the server, ...
            raise OSError(f"Could not download the AligNet weights from {WEIGHTS_URL} ({error}). Please "
                          f"download this file yourself and set the environment variable "
                          f"DIMPRED_ALIGNET_WEIGHTS to its path.") from error
        if file_sha256(partial) != WEIGHTS_SHA256:
            raise OSError(f"The AligNet weights downloaded from {WEIGHTS_URL} are not the released file (wrong "
                          f"sha256), maybe the download was incomplete. Please try again, or download the file "
                          f"yourself and set the environment variable DIMPRED_ALIGNET_WEIGHTS to its path.")
        # renaming is atomic: other processes see either no file or the
        # complete one. If another download finished first, its file is
        # replaced by an identical one.
        os.replace(partial, fname)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def file_sha256(fname):
    """sha256 of a file as hex text, read in parts of 16 MB."""

    sha256 = hashlib.sha256()
    with open(fname, "rb") as f:
        for part in iter(lambda: f.read(2 ** 24), b""):
            sha256.update(part)
    return sha256.hexdigest()


def preprocess(image):
    """
    pixels = preprocess(image)

    Preprocessing of an image as in the AligNet training code
    (configs/siglip.py: cv2.resize with INTER_CUBIC to 224 x 224, then values
    / 255): the whole image is resized to 224 x 224 (no crop, the aspect
    ratio is not kept), with the bicubic interpolation of OpenCV, written
    in numpy (see resize_cubic), and the values are scaled to 0 to 1,
    without mean or standard deviation normalization. OpenCV is not needed. PIL's bicubic
    resize is not a substitute: it smooths when it shrinks an image and
    changes the features (per-image r about 0.986).

    Input:
        image: PIL image in mode RGB (e.g. from read_image in
               extract_features.py), or a uint8 array, height x width x 3

    Output:
        pixels: float32 tensor, 3 x 224 x 224, values 0 to 1

    Hebartlab, 2026/10/02

    See also: resize_cubic, load_alignet
    """

    pixels = resize_cubic(np.asarray(image, dtype=np.uint8))
    return torch.from_numpy(pixels).permute(2, 0, 1).float() / 255


def resize_cubic(pixels, size=IMAGE_SIZE):
    """
    resized = resize_cubic(pixels, size=224)

    Resize an image to size x size with the formula of cv2.resize(pixels,
    (size, size), interpolation=cv2.INTER_CUBIC) for uint8 images: bicubic
    interpolation with a = -0.75, pixel centers at +0.5, border pixels
    repeated, no antialiasing, result rounded to uint8. We compute in
    float64. OpenCV uses fixed-point weights (11 bit), so single pixels can
    differ by 1 from OpenCV. This was not compared with OpenCV itself.

    Input:
        pixels: uint8 array, height x width x channels
        size:   output height and width (default: 224)

    Output:
        resized: uint8 array, size x size x channels

    Hebartlab, 2026/10/02

    See also: preprocess
    """

    def weights(n_in):
        # for each output pixel: the 4 input pixels and their weights (as in
        # cv2's interpolateCubic, the last weight is 1 minus the others)
        a = -0.75
        position = (np.arange(size) + 0.5) * (n_in / size) - 0.5
        start = np.floor(position).astype(int)
        x = position - start
        w0 = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a
        w1 = ((a + 2) * x - (a + 3)) * x * x + 1
        w2 = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1
        w3 = 1 - w0 - w1 - w2
        index = np.clip(start[:, None] + np.arange(-1, 3), 0, n_in - 1)  # border pixels repeated
        return index, np.stack([w0, w1, w2, w3], axis=1)

    pixels = np.asarray(pixels)
    rows, row_weights = weights(pixels.shape[0])
    columns, column_weights = weights(pixels.shape[1])
    # only the 4 input rows of each output row are converted to float64 (the
    # whole of a 24 megapixel photo would take 576 MB)
    resized = np.einsum("ok,okwc->owc", row_weights, pixels[rows].astype(np.float64))  # vertical, then horizontal
    resized = np.einsum("pk,opkc->opc", column_weights, resized[:, columns])
    return np.clip(np.floor(resized + 0.5), 0, 255).astype(np.uint8)
