"""
Figures of the README: how well the default model predicts the 66
dimensions and the similarity of new images.

    dimension_accuracy.png: r between predicted and measured values for
        each dimension, 10-fold cross-validation on the 1854 THINGS
        reference images (KFold(10, shuffle=True, random_state=0), the
        default model refit in each fold as in training/build_models.py),
        and the consistency of each dimension
        (training/data/consistency_66d.txt, see consistency below)
    similarity_48nonref.png: human similarity of the 48nonref images (48 new
        images of THINGS objects, odd-one-out judgments of the DimPred
        paper) next to the similarity predicted by the default model (with
        the close pairs), objects in the order of a clustering of the human
        similarity
    rose_plots.png: the measured dimensions of four objects (the SPoSE
        embedding of the object) and the dimensions predicted for a CC0
        photo of the same object (THINGSplus, docs/images/examples and
        tests/fixtures/images), as rose plots

The colors of the dimensions are those of Hebart et al. (2023)
(training/data/colors_66d.txt). The 48nonref features come from
tests/fixtures/reference_data.mat. The 1854 reference images are not part
of the repository; pass the folder with them with --things, either
object_images_behavior of THINGS (aardvark.jpg, abacus.jpg, ...) or
object_images (aardvark/aardvark_01b.jpg, ...). Their features are saved
with --features and read from there the next time.

The consistency of the dimensions is read from
training/data/consistency_66d.txt. With --seed-models and --sortind it is
computed again from the 72 SPoSE models that Hebart et al. (2023) fitted
with different random seeds (folders seed00 ... seed71, the model of seed63
left out as in their code) and saved there.

Example:
    python docs/images/make_readme_figures.py --things /data/THINGS/object_images_behavior --features alignet_1854.npy

Hebartlab, 2026/11/09
"""

import argparse
import os
import sys

import numpy as np
import scipy.io
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "training"))
import dimpred
from build_models import CLOSE_PAIRS, KERNEL, fit_kernel, fit_model

DATA = os.path.join(REPO, "training", "data")
MODEL = "alignet_siglip2b_66d_kernel"
# the photos of the rose plots (CC0 photos of THINGSplus) and their objects
EXAMPLES = [("hot-air balloon", "hot-air_balloon", os.path.join(HERE, "examples", "hot-air_balloon.jpg")),
            ("phone booth", "phone_booth", os.path.join(HERE, "examples", "phone_booth.jpg")),
            ("barbed wire", "barbed_wire", os.path.join(REPO, "tests", "fixtures", "images", "barbed_wire.jpg")),
            ("jellyfish", "jellyfish", os.path.join(HERE, "examples", "jellyfish.jpg"))]


def read_lines(fname):
    with open(fname) as f:
        return [line.strip() for line in f if line.strip()]


def reference_features(things, cache):
    """AligNet features of the 1854 reference images (extracted once, then read from cache)."""
    if cache and os.path.exists(cache):
        return np.load(cache)
    if not things:
        raise SystemExit("Pass the THINGS image folder with --things (or a feature file with --features).")
    names = read_lines(os.path.join(DATA, "reference_image_names.txt"))
    concepts = [n.rsplit("_", 1)[0] for n in names]
    files = [os.path.join(things, c + ".jpg") for c in concepts]  # object_images_behavior
    if not all(os.path.exists(f) for f in files):
        files = [os.path.join(things, c, n) for c, n in zip(concepts, names)]  # object_images
    features = dimpred.extract_features(files, MODEL)
    if cache:
        np.save(cache, features)
    return features


def cross_validated(features, embedding):
    """Out-of-fold predictions of the default model for the 1854 images."""
    labels = read_lines(os.path.join(DATA, "labels_66d.txt"))
    predicted = np.zeros_like(embedding)
    for train, test in KFold(n_splits=10, shuffle=True, random_state=0).split(features):
        weights, feature_mean, feature_scale, target_mean, alphas = fit_model(
            features[train], embedding[train], "ridge", return_alphas=True)
        weights, extra = fit_kernel(features[train], embedding[train], feature_mean, feature_scale,
                                    target_mean, alphas, KERNEL[MODEL])
        model = dict(weights=weights, feature_mean=feature_mean, feature_scale=feature_scale,
                     target_mean=target_mean, labels=labels, **extra)
        predicted[test] = dimpred.predict(features[test], model)
    return predicted


def consistency(embedding, models_dir, sortind_file):
    """Consistency of each dimension as in Extended Data Fig. 1 of Hebart et al.
    (2023): for each of 71 models fitted with other random seeds, the largest
    correlation of the dimension with any dimension of that model (dimensions
    with a value above 0.1), averaged over the models after Fisher z."""
    import glob
    sortind = scipy.io.loadmat(sortind_file, squeeze_me=True)["sortind"].astype(int) - 1
    z = (embedding - embedding.mean(0)) / embedding.std(0)
    best = []
    for seed in range(72):
        if seed == 63:  # left out in the paper
            continue
        fname = sorted(glob.glob(os.path.join(models_dir, f"seed{seed:02d}", "0.00385", "*.txt")))[-1]
        other = np.loadtxt(fname).T
        other = other[:, (other > 0.1).any(axis=0)][sortind]
        other = (other - other.mean(0)) / other.std(0)
        best.append((z.T @ other / len(z)).max(axis=1))
    return np.tanh(np.arctanh(np.clip(np.array(best), -0.999999, 0.999999)).mean(axis=0))


def short_label(label):
    """First part of a label, e.g. "food-related", "transportation/movement" for
    "transportation-/movement-related"."""
    parts = [p.strip() for p in label.replace("  ", " / ").split("/")]
    if parts[0].endswith("-"):
        return parts[0][:-1] + "/" + parts[1].replace("-related", "")
    return parts[0]


def plot_accuracy(r, consistent, colors, labels, fname):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.4))
    x = np.arange(1, len(r) + 1)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.bar(x, r, color=colors, edgecolor="none", width=0.8)
    ax.hlines(consistent, x - 0.45, x + 0.45, color="black", linewidth=1.6)
    ax.legend([Patch(facecolor="0.6"), Line2D([], [], color="black", linewidth=1.6)],
              ["prediction", "consistency of the dimension"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(lab) for lab in labels], rotation=90, fontsize=7)
    ax.set_xlim(0.3, len(r) + 0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("r (predicted, measured)")
    ax.set_title("Prediction of each of the 66 dimensions for images left out of training "
                 "(10-fold cross-validation, 1854 THINGS images)", fontsize=10, pad=20)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_similarity(human, predicted, r, fname):
    import matplotlib.pyplot as plt
    off = ~np.eye(len(human), dtype=bool)
    order = leaves_list(linkage(squareform(1 - human, checks=False), "average"))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    for ax, S, title in [(axes[0], human, "Measured (human odd-one-out judgments)"),
                         (axes[1], predicted, "Predicted by DimPred")]:
        S = S[np.ix_(order, order)].astype(float)
        vmin, vmax = np.percentile(S[off], [1, 99])
        S[~off] = np.nan
        ax.imshow(S, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Similarity of 48 new images of THINGS objects (r = {r:.2f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def rose(ax, values, colors, vmax):
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.bar(angles, values, width=2 * np.pi / n * 0.9, color=colors, edgecolor="none")
    ax.set_ylim(0, vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_color("0.85")


def plot_roses(names, images, measured, predicted, colors, labels, fname):
    import matplotlib.pyplot as plt
    from PIL import Image
    n = len(names)
    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.87 * n), gridspec_kw=dict(wspace=0.05, hspace=0.25))
    for i, (name, image, m, p) in enumerate(zip(names, images, measured, predicted)):
        vmax = max(m.max(), p.max()) * 1.02
        ax = axes[i, 0]
        ax.imshow(Image.open(image).convert("RGB"))
        ax.axis("off")
        top = np.argsort(m)[::-1][:3]
        ax.set_title(name, fontsize=11)
        ax.text(0.5, -0.04, "largest: " + ", ".join(short_label(labels[k]) for k in top), transform=ax.transAxes,
                ha="center", va="top", fontsize=8)
        for j, (values, title) in enumerate([(m, "measured (the object)"),
                                             (p, "predicted (this photo)")]):
            axes[i, 1 + j].remove()
            ax = fig.add_subplot(n, 3, 3 * i + 2 + j, projection="polar")
            rose(ax, values, colors, vmax)
            ax.set_title(title, fontsize=9, pad=6)
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--things", help="folder with the 1854 reference images (object_images_behavior of THINGS)")
    parser.add_argument("--features", help="file for the features of the 1854 images (.npy)")
    parser.add_argument("--out", default=HERE)
    parser.add_argument("--seed-models", help="folder with the SPoSE models seed00 ... seed71 (to compute the consistency again)")
    parser.add_argument("--sortind", help="sortind.mat of these models (the order of the objects)")
    args = parser.parse_args()

    embedding = np.loadtxt(os.path.join(DATA, "spose_embedding_66d.txt"))
    labels = read_lines(os.path.join(DATA, "labels_66d.txt"))
    colors = np.loadtxt(os.path.join(DATA, "colors_66d.txt"), delimiter=",")
    model = dimpred.load_model(MODEL)

    # 1. each dimension, cross-validated on the 1854 images
    features = reference_features(args.things, args.features)
    oof = cross_validated(features, embedding)
    r = np.array([np.corrcoef(oof[:, d], embedding[:, d])[0, 1] for d in range(embedding.shape[1])])
    consistency_file = os.path.join(DATA, "consistency_66d.txt")
    if args.seed_models:
        np.savetxt(consistency_file, consistency(embedding, args.seed_models, args.sortind), fmt="%.6f")
    consistent = np.loadtxt(consistency_file)
    above = np.flatnonzero(r > consistent)
    print(f"per dimension: mean r {r.mean():.3f} (min {r.min():.3f}, max {r.max():.3f}); consistency mean "
          f"{consistent.mean():.3f}; prediction above the consistency for {len(above)} dimensions: "
          + ", ".join(f"{short_label(labels[d])} ({r[d]:.3f} vs {consistent[d]:.3f})" for d in above))
    plot_accuracy(r, consistent, colors, labels, os.path.join(args.out, "dimension_accuracy.png"))

    # 2. similarity of the 48nonref images
    fixtures = scipy.io.loadmat(os.path.join(REPO, "tests", "fixtures", "reference_data.mat"), squeeze_me=True)
    in_set = np.array([s == "48nonref" for s in fixtures["image_set"]])
    f48 = fixtures["features_alignet"][in_set]
    human = fixtures["human_similarity_48nonref"]
    predicted = dimpred.similarity(dimpred.predict(f48, model), features=f48, model=model)
    lower = np.tril_indices(48, -1)
    r48 = np.corrcoef(human[lower], predicted[lower])[0, 1]
    print(f"48nonref: r = {r48:.3f}")
    plot_similarity(human, predicted, r48, os.path.join(args.out, "similarity_48nonref.png"))

    # 3. rose plots of the CC0 photos
    names = read_lines(os.path.join(DATA, "reference_image_names.txt"))
    images = [f for _, _, f in EXAMPLES]
    measured = [embedding[names.index(f"{key}_01b.jpg")] for _, key, _ in EXAMPLES]
    predicted_examples = dimpred.predict(dimpred.extract_features(images, MODEL), model)
    plot_roses([n for n, _, _ in EXAMPLES], images, measured, predicted_examples, colors, labels,
               os.path.join(args.out, "rose_plots.png"))
    for (name, _, _), m, p in zip(EXAMPLES, measured, predicted_examples):
        print(f"{name}: r measured vs predicted {np.corrcoef(m, p)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
