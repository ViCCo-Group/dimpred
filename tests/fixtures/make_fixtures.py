"""
Create the test fixtures in this folder (reference_data.mat, images/).

Usage:
    python make_fixtures.py --export <folder> --osf <folder> --cc0 <folder> --vit <folder>

The tests compare dimpred against numbers that do not come from dimpred
itself, mostly Philipp Kaniuth's published results. This script collects
these numbers once. You only need to run it again if the shipped models
change, and you need data that are not part of the repository:

    --export: Florian Mahner's export of the DimPred paper model
              (dimpred-heatmaps-export), which contains Philipp's RN50x64
              features and published predictions for the 48nonref images and
              the 120 Peterson animals (_models/dimpred.mat)
    --osf:    extracted folders of the DimPred OSF project
              (https://osf.io/jtekq, data/interim.7z and data/raw.7z)
    --cc0:    the THINGSplus CC0 images of the 1854 concepts (1854ref-cc0)
    --vit:    folder with ViT-B-32-quickgelu features of the 48nonref and
              Peterson images (<set>.npy and <set>_files.txt)

The features of the CC0 images are extracted here with open_clip directly
and not with dimpred.extract_features, so that the tests check dimpred
against an independent implementation. The extraction was validated before:
it reproduces Philipp's RN50x64 features (r = 1.0, max difference 3e-5) and
the ViT-B/32 features used by Philipp and Oliver Contier (r = 1.0).

The third fixture, philipp_original_ridge_rn50x64_66d.mat, is not made by
this script, because it takes hours. It holds the weights (weights), the
selected fractions (best_frac) and the run time in seconds (seconds) of
Philipp's original, unchanged ridge code for the rn50x64_66d_ridge model,
for comparison with the faster fracridge_cv in training/fit.py. It was made
with the version of fit.py before the move to training/ (git show
42c3ab1:dimpred/fit.py), on scikit-learn 1.3.2 (FracRidgeRegressorCV does not
run on newer versions), with the settings of call.py:

    model, _, _ = train_model_with(X, y, "ridge", k_in=3, n_in=3, random_state=0)
    weights = np.stack([est.coef_ for est in model.estimators_], axis=1)
    best_frac = np.array([est.best_frac_ for est in model.estimators_])

where X are the RN50x64 features of the 1854 reference images as float64
(features_RN50x64.npy written by training/build_models.py --features) and y
is training/data/spose_embedding_66d.txt. It took 8810 s.

Martin Hebart, 2026/09/30

See also: ../../training/build_models.py
"""


import argparse
import glob
import os
import shutil

import numpy as np
import scipy.io

# History:
# 2026/09/30: written to create the test fixtures

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "..", "dimpred", "models")
CC0_IMAGES = ["burrito.jpg", "apron.jpg", "barbed_wire.jpg"]  # not sorted on purpose


def spose_similarity(embedding):
    """Reference SPoSE similarity, same as embedding2sim_stable.m (slow but simple)."""
    n = embedding.shape[0]
    sim = embedding @ embedding.T
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            k = np.array([k for k in range(n) if k not in (i, j)])
            three = np.stack([np.full(len(k), sim[i, j]), sim[i, k], sim[j, k]])
            three = np.exp(three - three.max(axis=0))  # stabilize
            out[i, j] = out[j, i] = np.mean(three[0] / three.sum(axis=0))
    return out


def lower_triangle_r(a, b):
    ind = np.tril_indices(a.shape[0], -1)
    return np.corrcoef(a[ind], b[ind])[0, 1]


def predict(features, model):
    z = (features - model["feature_mean"]) / model["feature_scale"]
    return np.maximum(z @ model["weights"] + model["target_mean"], 0)


def extract(files, network):
    import open_clip
    import torch
    from PIL import Image

    net, _, preprocess = open_clip.create_model_and_transforms(network, pretrained="openai")
    net = net.float().eval()  # on the cpu, so fixtures do not depend on the gpu
    with torch.no_grad():
        x = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in files])
        return net.encode_image(x).float().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create the dimpred test fixtures.")
    parser.add_argument("--export", required=True)
    parser.add_argument("--osf", required=True)
    parser.add_argument("--cc0", required=True)
    parser.add_argument("--vit", required=True)
    args = parser.parse_args(argv)

    # Philipp's features and published predictions for 48nonref + Peterson animals
    exp = scipy.io.loadmat(os.path.join(args.export, "_models", "dimpred.mat"), simplify_cells=True)["openclip_rn50x64_49d"]
    files = [str(f) for f in exp["filenames"]]
    image_set = ["48nonref"] * 48 + ["peterson-animals"] * 120
    fix = dict(features_rn50x64=exp["features"], files=files, image_set=image_set,
               published_rn50x64_49d_ridge=exp["expected_predictions"])

    # ViT-B-32-quickgelu features of the same images, in the same order
    vit = {}
    for s in ["48nonref", "peterson-animals"]:
        names = open(os.path.join(args.vit, f"{s}_files.txt")).read().split("\n")
        feats = np.load(os.path.join(args.vit, f"{s}.npy"))
        vit.update({(s, n): f for n, f in zip(names, feats)})
    fix["features_vitb32"] = np.stack([vit[(s, n)] for s, n in zip(image_set, files)]).astype(float)

    # Human similarity of the 48nonref images (odd-one-out data of the DimPred paper)
    human = np.loadtxt(os.path.join(args.osf, "raw", "ground_truth_representational_matrices", "similarity_49d_48nonref.txt"))
    fix["human_similarity_48nonref"] = human

    # Expected predictions and human correlations for every shipped model
    human_r = {}
    for fname in sorted(glob.glob(os.path.join(MODELS, "*.mat"))):
        name = os.path.splitext(os.path.basename(fname))[0]
        model = scipy.io.loadmat(fname, simplify_cells=True)
        features = fix["features_vitb32"] if model["info"]["network"].startswith("ViT") else fix["features_rn50x64"]
        fix["expected_" + name] = predict(features, model)
        human_r[name] = lower_triangle_r(spose_similarity(fix["expected_" + name][:48]), human)
        print(f"{name}: r with human similarity (48nonref) = {human_r[name]:.3f}")
    fix["human_r_48nonref"] = human_r

    # CC0 images: files, reference features, Philipp's published predictions
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    cc0_files = [os.path.join(HERE, "images", f) for f in CC0_IMAGES]
    for f in CC0_IMAGES:
        shutil.copy(os.path.join(args.cc0, f), os.path.join(HERE, "images", f))
    fix["cc0_files"] = CC0_IMAGES
    fix["cc0_features_rn50x64"] = extract(cc0_files, "RN50x64").astype(float)
    fix["cc0_features_vitb32"] = extract(cc0_files, "ViT-B-32-quickgelu").astype(float)
    cc0_names = [n.strip() for n in open(os.path.join(args.cc0, "..", "file_names_1854ref-cc0.txt")) if n.strip()]
    cc0_pred = np.loadtxt(os.path.join(args.osf, "interim", "dimpred", "predictions_49d_ridge_OpenCLIP-RN50x64-openai_visual_1854ref-cc0.txt"))
    fix["cc0_published_rn50x64_49d_ridge"] = cc0_pred[[cc0_names.index(f) for f in CC0_IMAGES]]

    # cell arrays for MATLAB
    for key in ["files", "image_set", "cc0_files"]:
        fix[key] = np.array(fix[key], dtype=object).reshape(-1, 1)
    scipy.io.savemat(os.path.join(HERE, "reference_data.mat"), fix, do_compression=True)
    print("Saved reference_data.mat")


if __name__ == "__main__":
    main()
