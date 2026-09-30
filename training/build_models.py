"""
Build the shipped model files of dimpred (dimpred/models/*.mat).

Usage:
    python build_models.py --images <folder> [--features <folder>] [--published <folder>]
                           [--out <folder>] [--only NAME ...]

This script recreates the models in dimpred/models from the 1854 THINGS
reference images, i.e. the images used in the odd-one-out experiments of
Hebart et al. (2020), one image per object concept. The images are not part
of this repository (if you don't have them, ask us, e.g. in an issue on
GitHub). They have to be named as in data/reference_image_names.txt, which
also sets their order. The order is important: it has to match the rows of
the SPoSE embeddings in data/. Never sort the file names instead, because
Python's sorting and the THINGS order differ for 21 concepts (e.g. camera1,
camera2, camera_lens), which silently pairs these images with the wrong
dimension values.

What happens for each model:
    1. Extract the features of the 1854 images with dimpred.extract_features
       (the output of the image encoder of the network, i.e. open_clip's
       image embedding; the same code users run on their images).
    2. Fit one regression per SPoSE dimension with Philipp Kaniuth's training
       code (fit.py, train_model_with), using the settings of call.py: 3-fold
       cross-validation repeated 3 times for picking the regularization,
       random_state 0.
    3. Save the weights together with everything that is needed to use them:
       the mean and std of the features (for z-scoring new features) and the
       mean of each dimension. The regression is fit on centered dimension
       values, so the mean has to be added back to every prediction.

The only exception is rn50x64_49d_ridge. This is the model of the DimPred
paper, and we don't refit it but convert Philipp's published weights (OSF,
https://osf.io/jtekq, data/interim.7z, folder interim/heatmaps_base). Pass
this folder with --published. A refit gives nearly the same weights
(per-dimension r = 1.0000, largest difference 2e-6, because the extracted
features differ slightly from Philipp's), but not the exact ones.

Input:
    --images:    folder with the 1854 reference images (not needed for
                 models whose features are cached in --features)
    --features:  optional folder for cached features (features_<network>.npy).
                 Features found there are used instead of extracting them.
    --published: folder with Philipp's published files for
                 rn50x64_49d_ridge (coefs_49d_ridge_OpenCLIP-RN50x64-openai_visual.npy
                 and Xstandardizer_49d_ridge_OpenCLIP-RN50x64-openai_visual.pkl)
    --out:       output folder (default: ../dimpred/models, so the shipped
                 model files are overwritten)
    --only:      names of the models to build (default: all)

Needs the training environment (environment.yml) plus dimpred with feature
extraction (pip install -e "..[extract]").

Martin Hebart, 2026/09/30

See also: fit.py, call.py, README.md
"""


import argparse
import os
import pickle
import sys

import numpy as np
import scipy.io

from fit import preprocess_data, train_model_with

# History:
# 2026/09/30: written for the first release of the dimpred model files

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# All shipped models. network is the open_clip model name, pretrained the
# weights. For OpenAI's ViT models the "-quickgelu" version has to be used: the
# original CLIP ViTs use QuickGELU, and open_clip's plain "ViT-B-32" silently
# uses GELU with the same weights, which gives different features (r ~ 0.98
# with the features used by Philipp and Oliver instead of 1.0).
MODELS = [
    dict(name="vitb32_66d_elastic", network="ViT-B-32-quickgelu", dims=66, regression="elastic",
         note="Default model. Rebuilt version of the model used by Contier et al. (2024)."),
    dict(name="rn50x64_49d_ridge", network="RN50x64", dims=49, regression="ridge",
         note="Model of the DimPred paper (Kaniuth et al., 2025), Philipp Kaniuth's published weights."),
    dict(name="rn50x64_66d_ridge", network="RN50x64", dims=66, regression="ridge",
         note="Same procedure as the DimPred paper, for the 66d embedding."),
    dict(name="rn50x64_66d_elastic", network="RN50x64", dims=66, regression="elastic",
         note="As rn50x64_66d_ridge, but with elastic net (generalizes slightly better)."),
]

PREPROCESSING = {
    "ViT-B-32-quickgelu": "resize shortest side to 224 px (bicubic), center crop 224 x 224, RGB, "
                          "normalize with the OpenAI CLIP mean and std",
    "RN50x64": "resize shortest side to 448 px (bicubic), center crop 448 x 448, RGB, "
               "normalize with the OpenAI CLIP mean and std",
}

REGRESSION = {
    "ridge": "fractional ridge regression, one per dimension (fracridge; 70 fractions from 0.1 to 1, "
             "selected by 3-fold cross-validation repeated 3 times)",
    "elastic": "elastic net, one per dimension (sklearn ElasticNetCV; l1_ratio 0.01, 0.1, 0.3, 0.5, 0.7, "
               "0.9, 0.99 and 10 alphas, selected by 3-fold cross-validation repeated 3 times)",
}

EMBEDDING = {
    49: "SPoSE 49d embedding of the 1854 THINGS concepts (Hebart et al., 2020, Nature Human Behaviour)",
    66: "SPoSE 66d embedding of the 1854 THINGS concepts (Hebart et al., 2023, eLife; corrected version of January 2023)",
}


def read_lines(fname):
    with open(fname) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def get_features(network, image_files, features_dir=None):
    """Load cached features of a network or extract them from the images."""

    if features_dir:
        fname = os.path.join(features_dir, f"features_{network}.npy")
        if os.path.exists(fname):
            print(f"Loading cached features {fname}")
            return np.load(fname)

    if image_files is None:
        sys.exit(f"No cached {network} features found, so the reference images are needed: "
                 f"pass --images <folder> (or --features <folder with features_{network}.npy>)")

    import dimpred  # only needed here, so the rest works without torch

    print(f"Extracting {network} features for {len(image_files)} images, this takes a few minutes")
    features = dimpred.extract_features(image_files, network=network)
    if features_dir:
        os.makedirs(features_dir, exist_ok=True)
        np.save(os.path.join(features_dir, f"features_{network}.npy"), features)
    return features


def fit_model(features, embedding, regression):
    """Fit one regression per dimension with Philipp's training code.

    Returns the weights (n_features x n_dims) and the numbers needed to apply
    them to new features: feature mean and std and the mean of each dimension.
    """

    # Philipp's inputs were text files (float64), the extracted features are
    # float32, so we convert them to get the same numerical precision
    features = np.asarray(features, dtype=float)

    # z-score features and center the dimensions exactly as in training
    _, _, standardizer, target_mean = preprocess_data(features.copy(), embedding.copy())

    # settings of call.py (get_trained_model_for)
    model, _, _ = train_model_with(features.copy(), embedding.copy(), regression,
                                   k_in=3, n_in=3, random_state=0)
    if regression == "ridge":
        weights = model  # fracridge_cv already returns the weight matrix
    else:
        weights = np.stack([est.coef_ for est in model.estimators_], axis=1)

    return weights, standardizer.mean_, standardizer.scale_, target_mean


def load_published_model(published_dir):
    """Philipp's published rn50x64_49d_ridge weights and feature scaling."""

    weights = np.load(os.path.join(published_dir, "coefs_49d_ridge_OpenCLIP-RN50x64-openai_visual.npy"))
    with open(os.path.join(published_dir, "Xstandardizer_49d_ridge_OpenCLIP-RN50x64-openai_visual.pkl"), "rb") as f:
        standardizer = pickle.load(f)  # sklearn StandardScaler, needs sklearn to unpickle
    return weights, standardizer.mean_, standardizer.scale_


def save_model(fname, spec, weights, feature_mean, feature_scale, target_mean, labels, source):
    """Write a model file that both dimpred_load_model.m and dimpred.load_model can read."""

    n_features, n_dims = weights.shape
    assert len(feature_mean) == n_features and len(feature_scale) == n_features, "feature statistics do not match weights"
    assert len(target_mean) == n_dims and len(labels) == n_dims, "dimension means or labels do not match weights"

    info = dict(
        name=spec["name"],
        network=spec["network"],
        pretrained="openai",
        layer="image embedding (output of the image encoder, not normalized)",
        preprocessing=PREPROCESSING[spec["network"]],
        n_features=float(n_features),
        n_dims=float(n_dims),
        embedding=EMBEDDING[spec["dims"]],
        regression=REGRESSION[spec["regression"]],
        training_images="1854 THINGS reference images (Hebart et al., 2020)",
        source=source,
        note=spec["note"],
        created="2026-09-30",
    )
    scipy.io.savemat(
        fname,
        dict(
            weights=np.asarray(weights, dtype=float),
            feature_mean=np.asarray(feature_mean, dtype=float).reshape(1, -1),
            feature_scale=np.asarray(feature_scale, dtype=float).reshape(1, -1),
            target_mean=np.asarray(target_mean, dtype=float).reshape(1, -1),
            labels=np.array(labels, dtype=object).reshape(-1, 1),  # becomes a cell array in MATLAB
            info=info,
        ),
        do_compression=True,
    )
    print(f"Saved {fname}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the dimpred model files.")
    parser.add_argument("--images", help="folder with the 1854 reference images")
    parser.add_argument("--features", help="folder for cached features")
    parser.add_argument("--published", help="folder with Philipp's published rn50x64_49d_ridge files")
    parser.add_argument("--out", default=os.path.join(HERE, "..", "dimpred", "models"),
                        help="output folder (default: ../dimpred/models, overwrites the shipped models)")
    parser.add_argument("--only", nargs="*", metavar="NAME",
                        help="build only these models, e.g. --only rn50x64_66d_ridge (default: all)")
    args = parser.parse_args(argv)

    names = read_lines(os.path.join(DATA, "reference_image_names.txt"))
    image_files = [os.path.join(args.images, n) for n in names] if args.images else None
    if image_files:
        missing = [f for f in image_files if not os.path.exists(f)]
        if missing:
            sys.exit(f"{len(missing)} of the 1854 reference images were not found, e.g. {missing[0]}")

    embedding = {49: np.loadtxt(os.path.join(DATA, "spose_embedding_49d.txt")),
                 66: np.loadtxt(os.path.join(DATA, "spose_embedding_66d.txt"))}
    labels = {49: read_lines(os.path.join(DATA, "labels_49d.txt")),
              66: read_lines(os.path.join(DATA, "labels_66d.txt"))}
    os.makedirs(args.out, exist_ok=True)

    for spec in MODELS:
        if args.only and spec["name"] not in args.only:
            continue
        print(f"\n*** {spec['name']} ***")
        fname = os.path.join(args.out, spec["name"] + ".mat")
        y = embedding[spec["dims"]]

        if spec["name"] == "rn50x64_49d_ridge":
            if not args.published:
                print("Skipping: needs --published (Philipp's published weights)")
                continue
            weights, feature_mean, feature_scale = load_published_model(args.published)
            target_mean = y.mean(axis=0)
            source = ("Philipp Kaniuth's published weights and feature scaling (https://osf.io/jtekq, "
                      "data/interim.7z, interim/heatmaps_base), identical to model_49d_ridge_OpenCLIP-RN50x64-openai_visual.joblib")
        else:
            features = get_features(spec["network"], image_files, args.features)
            assert features.shape[0] == y.shape[0], "number of images and rows of the embedding differ"
            weights, feature_mean, feature_scale, target_mean = fit_model(features, y, spec["regression"])
            source = "fitted with training/build_models.py (Philipp Kaniuth's training code in training/fit.py)"

        save_model(fname, spec, weights, feature_mean, feature_scale, target_mean, labels[spec["dims"]], source)


if __name__ == "__main__":
    main()
