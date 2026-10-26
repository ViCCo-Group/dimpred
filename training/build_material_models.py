"""
Build the shipped models of the 36 material dimensions (dimpred/models/*_36d_*.mat).

Usage:
    python build_material_models.py --images <folder> [--features <folder>] [--out <folder>]
                                    [--only NAME ...] [--device DEV]

The 36 material dimensions are the SPoSE embedding of the 600 images of the
STUFF dataset (200 materials x 3 images; Schmidt, Hebart, Schmid & Fleming,
2025, PNAS), learned from 1.87 million triplet 2AFC judgments: which of two
images is more similar to a reference image. The embedding (sorted as in the
paper), the labels and the image names are in data/. The images are on OSF
(https://osf.io/myutc, STUFF_dataset_600_images.zip, with a password for
agreeing to the license). Pass their folder with --images: the images are
found by their number (0001.jpg to 0600.jpg) anywhere below it, so the folder
of the zip file (one subfolder per material) works as it is. The order of the
rows of the embedding is the order of these numbers.

What happens for each model:
    1. Extract the features of the 600 images with dimpred.extract_features,
       or load them from the --features folder
       (features_stuff600_<network>.npy).
    2. alignet_siglip2b_36d_ridge and alignet_siglip2b_36d_kernel: one ridge
       regression per dimension on the z-scored features (std with ddof 0,
       divided by the square root of the number of features), with the
       penalty from 10 inner folds of whole materials (the 3 images of a
       material are always in the same fold; the penalty with the best R^2
       of the held-out predictions of all folds together, from 10^-4 to
       10^4, 4 per decade). For the kernel model, the ridge plus a local
       kernel (fit.kernel_ridge_fit, as alignet_siglip2b_66d_kernel), with
       the same penalties and tau and beta from 5 inner folds of whole
       materials (the best mean R^2 across dimensions). The kernel model has
       no close-pair settings: on the human data of STUFF, the close pairs
       did not improve the similarity.
    3. rn50x64_36d_ridge is the model of the paper for new images
       (predict_from_clip_ridge_newsample.m): kernel ridge regression on the
       z-scored features (std with ddof 1), lambda 1000 for all dimensions,
       negative values set to 0, then each dimension multiplied by a scale
       factor (the least squares fit of the embedding on the paper's
       leave-one-material-out predictions). The lambdas and the scale factors
       are in data/stuff_rn50x64_paper.txt. Since the scale factors are > 0,
       they are folded into the weights and target_mean. The shipped file
       was built from the features of the paper (equal to those of
       dimpred.extract_features within 1.4e-4).

The hyperparameters were chosen in the comparison of 2026/10/26 (RN50x64,
SigLIP2-B/16 and AligNet SigLIP2-B, ridge and ridge + kernel, leave one
material out; dimpred-materials, not part of this repository), where
AligNet SigLIP2-B with the local kernel predicted the dimensions best.

Input:
    --images:   folder with the 600 STUFF images (not needed if the features
                are cached in --features)
    --features: optional folder for cached features
                (features_stuff600_<network>.npy, spaces replaced by "-")
    --out:      output folder (default: ../dimpred/models, overwrites the
                shipped files)
    --only:     names of the models to build (default: all)
    --device:   device for the feature extraction (default: cpu, as for the
                shipped models)

Needs the training environment (environment.yml) plus dimpred with feature
extraction (pip install -e "..[extract]").

Hebartlab, 2026/10/26

See also: build_models.py, fit.py, README.md
"""


import argparse
import glob
import os
import sys

import numpy as np

from build_models import PREPROCESSING, read_lines, save_model
from fit import kernel_ridge_fit

# History:
# 2026/10/26: written for the models of the 36 material dimensions

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

ALIGNET_PRETRAINED = ("alignet_siglip2_b.safetensors (SigLIP2-B-alignet of Muttenthaler et al., 2025, converted "
                      "from TensorFlow to PyTorch)")
ALIGNET_LAYER = "pre_logits (output of the attention pooling head of the image encoder, not normalized)"
CLIP_LAYER = "image embedding (output of the image encoder, not normalized)"
EMBEDDING = ("SPoSE 36d embedding of the 600 STUFF images (200 materials; Schmidt, Hebart, Schmid & Fleming, 2025, "
             "PNAS), from triplet 2AFC judgments (which of two images is more similar to a reference)")
TRAINING_IMAGES = "600 STUFF images (200 materials x 3 images; Schmidt et al., 2025)"
SIMILARITY_NOTE = "Similarity: dimpred.similarity(embedding, method=\"2afc\")."

MODELS = [
    dict(name="alignet_siglip2b_36d_kernel", network="AligNet SigLIP2-B", dims=36, regression="ridge + kernel",
         pretrained=ALIGNET_PRETRAINED, layer=ALIGNET_LAYER, embedding=EMBEDDING, training_images=TRAINING_IMAGES,
         regression_text="ridge regression with the penalty chosen directly, one per dimension (as "
                         "alignet_siglip2b_36d_ridge), plus a local kernel fit together with it (training/fit.py "
                         "kernel_ridge_fit; a Gaussian process with the covariance beta exp((cos - 1) / tau) over "
                         "the cosine of the features, tau and beta chosen in inner folds of whole materials)",
         note="AligNet SigLIP2-B for the 36 material dimensions (STUFF), ridge plus a local kernel; the "
              "recommended model for materials. " + SIMILARITY_NOTE),
    dict(name="alignet_siglip2b_36d_ridge", network="AligNet SigLIP2-B", dims=36, regression="ridge",
         pretrained=ALIGNET_PRETRAINED, layer=ALIGNET_LAYER, embedding=EMBEDDING, training_images=TRAINING_IMAGES,
         regression_text="ridge regression with the penalty chosen directly, one per dimension (penalty from "
                         "10^-4 to 10^4, 4 per decade, on the z-scored features divided by the square root of "
                         "their number, selected by 10 inner folds of whole materials)",
         note="AligNet SigLIP2-B for the 36 material dimensions (STUFF), ridge with the penalty chosen directly. "
              + SIMILARITY_NOTE),
    dict(name="rn50x64_36d_ridge", network="RN50x64", dims=36, regression="paper", pretrained="openai",
         layer=CLIP_LAYER, embedding=EMBEDDING, training_images=TRAINING_IMAGES,
         regression_text="kernel ridge regression of Schmidt et al. (2025), one per dimension (lambda 1000, "
                         "negative values set to 0, then scaled by the least squares factor of the paper's "
                         "leave-one-material-out predictions)",
         note="Model of the STUFF paper (Schmidt et al., 2025) for new images, rebuilt from its MATLAB code. "
              + SIMILARITY_NOTE),
]

ALPHAS = 10.0 ** np.arange(-4, 4.01, 0.25)
TAUS = [0.1, 0.2, 0.5, 1.0]
BETAS = [1.0, 3.0, 10.0, 30.0, 100.0]


def find_images(folder):
    """The 600 STUFF images by number (0001.jpg to 0600.jpg), anywhere below folder."""

    found = {}
    for fname in glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True):
        stem = os.path.splitext(os.path.basename(fname))[0]
        if stem.isdigit() and 1 <= int(stem) <= 600:
            found.setdefault(int(stem), []).append(fname)
    missing = [i for i in range(1, 601) if i not in found]
    if missing:
        sys.exit(f"{len(missing)} of the 600 STUFF images were not found below {folder}, e.g. {missing[0]:04d}.jpg")
    double = [i for i in found if len(found[i]) > 1]
    if double:
        sys.exit(f"{len(double)} image numbers occur more than once below {folder}, e.g. {found[double[0]]}")
    return [found[i][0] for i in range(1, 601)]


def get_features(network, image_files, features_dir=None, device="cpu"):
    """Load cached features of the 600 images or extract them (on device)."""

    fname = f"features_stuff600_{network.replace(' ', '-')}.npy"
    if features_dir:
        fname = os.path.join(features_dir, fname)
        if os.path.exists(fname):
            print(f"Loading cached features {fname}")
            return np.load(fname)
    if image_files is None:
        sys.exit(f"No cached {network} features found, so the images are needed: pass --images <folder>")

    import dimpred  # only needed here, so the rest works without torch

    print(f"Extracting {network} features for {len(image_files)} images on the {device}")
    features = dimpred.extract_features(image_files, network=network, device=device)
    if features_dir:
        os.makedirs(features_dir, exist_ok=True)
        np.save(fname, features)
    return features


def inner_folds(groups, n_folds):
    """Folds of whole groups (materials), shuffled with seed 0."""

    order = np.random.default_rng(0).permutation(np.unique(groups))
    fold_of_group = {g: i % n_folds for i, g in enumerate(order)}
    fold = np.array([fold_of_group[g] for g in groups])
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_folds)]


def zscore(X_train, X_test):
    """z-score with the training mean and std (ddof 0), divided by sqrt(n_features)."""

    m, s = X_train.mean(0), X_train.std(0)
    assert np.all(s > 1e-10), "features with zero variance"
    root = np.sqrt(X_train.shape[1])
    return (X_train - m) / s / root, (X_test - m) / s / root


def r2(Y_hat, Y):
    """R^2 per dimension; Y_hat (..., n, d)."""

    return 1 - ((Y - Y_hat) ** 2).sum(-2) / ((Y - Y.mean(0)) ** 2).sum(0)


def select_alpha(X, Y, groups, n_folds=10):
    """Penalty per dimension: best R^2 of the held-out predictions of all inner folds together."""

    P = np.zeros((len(ALPHAS),) + Y.shape)
    for train, test in inner_folds(groups, n_folds):
        Z, Z_test = zscore(X[train], X[test])
        m = Y[train].mean(0)
        s, U = np.linalg.eigh(Z @ Z.T)
        s = np.maximum(s, 0)
        UY = U.T @ (Y[train] - m)
        ZU = (Z_test @ Z.T) @ U
        P[:, test] = np.stack([ZU @ (UY / (s + a)[:, None]) for a in ALPHAS]) + m
    return ALPHAS[np.argmax(r2(P, Y), axis=0)]


def select_kernel(X, Y, groups, alpha, n_folds=5):
    """tau and beta of the kernel: best mean R^2 across dimensions (held-out predictions of all folds)."""

    unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    P = np.zeros((len(TAUS), len(BETAS)) + Y.shape)
    for train, test in inner_folds(groups, n_folds):
        Z, Z_test = zscore(X[train], X[test])
        m = Y[train].mean(0)
        C, C_test = unit[train] @ unit[train].T, unit[test] @ unit[train].T
        for i, tau in enumerate(TAUS):
            K_test = np.exp((C_test - 1) / tau)
            for j, beta in enumerate(BETAS):
                weights, coefficients = kernel_ridge_fit(Z, Y[train] - m, X[train], alpha, beta, tau)
                P[i, j, test] = Z_test @ weights + K_test @ coefficients + m
    scores = r2(P, Y).mean(-1)
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    return TAUS[i], BETAS[j]


def fit_alignet(features, embedding, groups, kernel):
    """Ridge (and ridge + kernel) with the hyperparameters from inner folds of whole materials."""

    X = np.asarray(features, dtype=float)
    alpha = select_alpha(X, embedding, groups)
    feature_mean, feature_scale = X.mean(0), X.std(0)
    target_mean = embedding.mean(0)
    Z, _ = zscore(X, X[:1])
    root = np.sqrt(X.shape[1])
    extra = None
    if kernel:
        tau, beta = select_kernel(X, embedding, groups, alpha)
        print(f"tau {tau}, beta {beta}")
        weights, coefficients = kernel_ridge_fit(Z, embedding - target_mean, X, alpha, beta, tau)
        unit = X / np.linalg.norm(X, axis=1, keepdims=True)
        extra = dict(kernel_features=unit.astype(np.float32), kernel_coefficients=coefficients,
                     kernel_tau=float(tau))
    else:
        weights, _ = kernel_ridge_fit(Z, embedding - target_mean, X, alpha, 0.0, 1.0)  # beta 0: the ridge
    print(f"penalties: {np.unique(np.log10(alpha), return_counts=True)}")
    # the weights are for z / sqrt(n_features); the file has them for z
    return weights / root, feature_mean, feature_scale, target_mean, extra


def fit_paper(features, embedding):
    """The kernel ridge regression of the paper for new images, with its lambdas and scale factors."""

    settings = np.loadtxt(os.path.join(DATA, "stuff_rn50x64_paper.txt"))
    lambdas, scale = settings[:, 0], settings[:, 1]
    assert np.all(scale > 0), "the scale factors have to be > 0 to be folded into the weights"
    X = np.asarray(features, dtype=float)
    feature_mean, feature_scale = X.mean(0), X.std(0, ddof=1)  # MATLAB's std
    assert np.all(feature_scale > 1e-10), "features with zero variance"
    Z = (X - feature_mean) / feature_scale
    target_mean = embedding.mean(0)
    K = Z @ Z.T
    weights = np.zeros((X.shape[1], embedding.shape[1]))
    for d in range(embedding.shape[1]):
        a = np.linalg.solve(K + lambdas[d] * np.eye(len(K)), embedding[:, d] - target_mean[d])
        weights[:, d] = Z.T @ a
    # max(x, 0) * scale = max(x * scale, 0) for scale > 0
    return weights * scale, feature_mean, feature_scale, target_mean * scale


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the dimpred models of the 36 material dimensions.")
    parser.add_argument("--images", help="folder with the 600 STUFF images")
    parser.add_argument("--features", help="folder for cached features")
    parser.add_argument("--out", default=os.path.join(HERE, "..", "dimpred", "models"),
                        help="output folder (default: ../dimpred/models, overwrites the shipped models)")
    parser.add_argument("--only", nargs="*", metavar="NAME", help="build only these models (default: all)")
    parser.add_argument("--device", default="cpu", help="device for the feature extraction (default: cpu)")
    args = parser.parse_args(argv)

    image_files = find_images(args.images) if args.images else None
    embedding = np.loadtxt(os.path.join(DATA, "stuff_embedding_36d.txt"))
    labels = read_lines(os.path.join(DATA, "labels_36d.txt"))
    names = read_lines(os.path.join(DATA, "stuff_image_names.txt"))
    assert embedding.shape == (600, 36) and len(labels) == 36 and len(names) == 600
    groups = np.array([n.split("/")[0] for n in names])  # the material of each image
    os.makedirs(args.out, exist_ok=True)

    for spec in MODELS:
        if args.only and spec["name"] not in args.only:
            continue
        print(f"\n*** {spec['name']} ***")
        features = get_features(spec["network"], image_files, args.features, args.device)
        assert features.shape[0] == 600, "number of images and rows of the embedding differ"
        extra = None
        if spec["regression"] == "paper":
            weights, feature_mean, feature_scale, target_mean = fit_paper(features, embedding)
            source = ("rebuilt with training/build_material_models.py from the MATLAB code of Schmidt et al. "
                      "(2025), predict_from_clip_ridge_newsample.m, with its lambdas and scale factors "
                      "(training/data/stuff_rn50x64_paper.txt)")
        else:
            weights, feature_mean, feature_scale, target_mean, extra = fit_alignet(
                features, embedding, groups, kernel=spec["regression"] == "ridge + kernel")
            source = "fitted with training/build_material_models.py (inner folds of whole materials)"
        assert spec["network"] in PREPROCESSING
        save_model(os.path.join(args.out, spec["name"] + ".mat"), spec, weights, feature_mean, feature_scale,
                   target_mean, labels, source, extra)


if __name__ == "__main__":
    main()
