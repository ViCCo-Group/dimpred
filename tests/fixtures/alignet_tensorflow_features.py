"""
AligNet SigLIP2-B features of images with the released TensorFlow model, for
the test fixtures.

Usage (in an environment with tensorflow, numpy and pillow):
    python alignet_tensorflow_features.py --model <SigLIP2-B-alignet folder> --out <file.npy> IMAGE [IMAGE ...]

The tests compare the features of dimpred's PyTorch port (dimpred/alignet.py)
with these features, so they are made here without dimpred and without
PyTorch: with the TensorFlow SavedModel released by the AligNet authors
(https://storage.googleapis.com/alignet/models/SigLIP2-B-alignet.tar.gz,
signature serving_default, output pre_logits) and with the preprocessing
used for the TensorFlow reference features of the DimPred benchmark: PIL
convert("RGB") (16-bit grayscale scaled by 1/257), cv2.resize(image,
(224, 224), interpolation=cv2.INTER_CUBIC) written in numpy with full
weight matrices (OpenCV is not needed), values / 255. The rows of the output
are in the order of the given images.

make_fixtures.py reads the output file (--alignet-cc0).

Hebartlab, 2026/10/02

See also: make_fixtures.py, ../../dimpred/alignet.py
"""

import argparse
import os

import numpy as np

# History:
# 2026/10/02: written for the fixtures of the AligNet model

SIZE = 224


def read_image(fname):
    """RGB uint8 array, height x width x 3 (as dimpred's read_image)."""

    from PIL import Image

    with Image.open(fname) as image:
        image.load()
        if image.mode in ["I;16", "I;16B", "I;16L", "I;16N"]:
            pixels = np.asarray(image, dtype=float) / 257
            image = Image.fromarray(pixels.round().astype(np.uint8))
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def cubic_matrix(n_in, n_out, a=-0.75):
    """n_out x n_in matrix of the weights of cv2.INTER_CUBIC along one axis (border pixels repeated)."""

    weights = np.zeros((n_out, n_in))
    scale = n_in / n_out  # computed first, as for the reference features (rounding decides values at x.5)
    for i in range(n_out):
        position = (i + 0.5) * scale - 0.5
        start = int(np.floor(position))
        x = position - start
        c0 = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a
        c1 = ((a + 2) * x - (a + 3)) * x * x + 1
        c2 = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1
        c3 = 1 - c0 - c1 - c2
        for k, c in zip(range(-1, 3), (c0, c1, c2, c3)):
            weights[i, min(max(start + k, 0), n_in - 1)] += c
    return weights


def preprocess(fname):
    """224 x 224 x 3 float32 image, values 0 to 1."""

    pixels = read_image(fname).astype(np.float64)
    rows, columns = cubic_matrix(pixels.shape[0], SIZE), cubic_matrix(pixels.shape[1], SIZE)
    resized = np.stack([rows @ pixels[:, :, c] @ columns.T for c in range(3)], axis=2)  # vertical, then horizontal
    resized = np.clip(np.floor(resized + 0.5), 0, 255).astype(np.uint8)
    return resized.astype(np.float32) / 255


def main(argv=None):
    parser = argparse.ArgumentParser(description="AligNet SigLIP2-B features with TensorFlow.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model", required=True, help="folder of the SavedModel SigLIP2-B-alignet")
    parser.add_argument("--out", required=True, help="output file (.npy), n_images x 768")
    args = parser.parse_args(argv)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # on the cpu, as the reference features
    forward = tf.saved_model.load(args.model).signatures["serving_default"]
    images = np.stack([preprocess(f) for f in args.images])  # n x 224 x 224 x 3
    features = forward(images=tf.constant(images))["pre_logits"].numpy()
    assert features.shape == (len(args.images), 768), features.shape
    np.save(args.out, features.astype(np.float32))
    print(f"Saved the features of {len(args.images)} images to {args.out}")


if __name__ == "__main__":
    main()
