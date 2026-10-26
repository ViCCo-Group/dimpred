"""
Tests of the models of the 36 material dimensions (STUFF, Schmidt et al., 2025)
and of the 2AFC similarity (dimpred.similarity, method "2afc").

The fixture materials_data.mat (made by fixtures/make_fixtures.py
--materials) has, for the 48 out-of-sample images of the STUFF paper, the
RN50x64 features from which the paper computed its predictions, the AligNet
SigLIP2-B features, the published predictions, the human similarity (choice
probabilities of the triplet 2AFC task) and, for each model, the expected
predictions and the correlation with the human similarity. Here we check:
    - rn50x64_36d_ridge, rebuilt from the paper's MATLAB code, gives the
      published predictions, and the paper's similarity of these predictions
      gives the published r = 0.79
    - each model gives the expected predictions and correlation with the
      human similarity (and at least 0.80)
    - the 2AFC similarity follows its definition (written out with loops)
    - the training data in training/data and the model files fit together
    - (slow, needs the cached features of the 600 STUFF images) the shipped
      files are what training/build_material_models.py builds

Hebartlab, 2026/10/26

See also: test_similarity.py, test_validation_human.py, ../training/build_material_models.py
"""

import os
import sys

import numpy as np
import pytest

import dimpred
from helpers import (MATERIAL_MODEL_NAMES, MATERIALS_FILE, MODELS, TOL_HUMAN_R, TOL_MODEL, TRAINING, TRAINING_DATA,
                     assert_close, load_mat, lower_triangle, read_lines, similarity_2afc_by_definition)

# History:
# 2026/10/26: written for the models of the 36 material dimensions

# The published predictions are float32 values in the paper's .mat file
TOL_PUBLISHED = 1e-5
KERNEL_MODEL = "alignet_siglip2b_36d_kernel"
RIDGE_MODEL = "alignet_siglip2b_36d_ridge"
PAPER_MODEL = "rn50x64_36d_ridge"


@pytest.fixture(scope="module")
def stuff():
    assert os.path.exists(MATERIALS_FILE), f"Missing test fixture {MATERIALS_FILE}"
    return load_mat(MATERIALS_FILE)


def features_of(stuff, name):
    return stuff["stuff48_features_alignet" if MODELS[name]["network"] == "AligNet SigLIP2-B"
                 else "stuff48_features_rn50x64"]


def r_with_humans(similarity, stuff):
    return np.corrcoef(lower_triangle(similarity), lower_triangle(stuff["stuff48_human_similarity"]))[0, 1]


def paper_similarity(embedding):
    """The similarity of the paper (embedding2sim_pair.m): only i is the reference for i < j."""

    dots = embedding @ embedding.T
    n = len(dots)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            k = np.array([k for k in range(n) if k not in (i, j)])
            S[i, j] = S[j, i] = np.mean(1 / (1 + np.exp(dots[i, k] - dots[i, j])))
    return S


# --- the fixture

def test_fixture_has_all_variables_with_their_sizes(stuff):
    sizes = {"stuff48_features_rn50x64": (48, 1024), "stuff48_features_alignet": (48, 768),
             "stuff48_published_rn50x64_36d_ridge": (48, 36), "stuff48_human_similarity": (48, 48)}
    sizes.update({"stuff48_expected_" + name: (48, 36) for name in MATERIAL_MODEL_NAMES})
    for key, size in sizes.items():
        assert key in stuff, f"materials_data.mat has no {key}"
        assert stuff[key].shape == size, f"{key} has shape {stuff[key].shape}, expected {size}"
        assert np.all(np.isfinite(stuff[key])), f"{key} has nan or inf"
    assert len(stuff["stuff48_files"]) == 48 and len(set(stuff["stuff48_files"])) == 48
    assert sorted(stuff["stuff48_human_r"]) == MATERIAL_MODEL_NAMES, f"human_r for {sorted(stuff['stuff48_human_r'])}"


def test_human_similarity_is_symmetric_and_between_0_and_1(stuff):
    human = stuff["stuff48_human_similarity"]
    assert_close(human, human.T, 1e-12, "human similarity vs its transpose")
    values = lower_triangle(human)
    assert values.min() >= 0 and values.max() <= 1, f"human similarity from {values.min()} to {values.max()}"


def test_paper_similarity_of_the_published_predictions_gives_the_published_r(stuff):
    # The paper reports r = 0.79 for the 48 images (with its similarity, embedding2sim_pair.m)
    r = r_with_humans(paper_similarity(stuff["stuff48_published_rn50x64_36d_ridge"]), stuff)
    assert abs(r - 0.788) <= TOL_HUMAN_R, f"r of the published predictions is {r:.4f}, published: 0.79"


# --- the models

def test_paper_model_gives_the_published_predictions(stuff):
    predicted = dimpred.predict(stuff["stuff48_features_rn50x64"], PAPER_MODEL)
    assert_close(predicted, stuff["stuff48_published_rn50x64_36d_ridge"], TOL_PUBLISHED,
                 f"{PAPER_MODEL} vs the published predictions of the 48 images")


@pytest.mark.parametrize("name", MATERIAL_MODEL_NAMES)
def test_model_gives_the_expected_predictions(stuff, name):
    assert_close(dimpred.predict(features_of(stuff, name), name), stuff["stuff48_expected_" + name], TOL_MODEL,
                 f"{name}: predictions of the 48 images vs stuff48_expected_{name}")


@pytest.mark.parametrize("name", MATERIAL_MODEL_NAMES)
def test_predicted_similarity_correlates_with_human_similarity_as_expected(stuff, name):
    similarity = dimpred.similarity(dimpred.predict(features_of(stuff, name), name), method="2afc")
    r, expected = r_with_humans(similarity, stuff), float(stuff["stuff48_human_r"][name])
    assert abs(r - expected) <= TOL_HUMAN_R, f"{name}: r is {r:.4f}, expected {expected:.4f} +- {TOL_HUMAN_R}"


@pytest.mark.parametrize("name", MATERIAL_MODEL_NAMES)
def test_predicted_similarity_correlates_at_least_0_80_with_human_similarity(stuff, name):
    # Independent of the stored values: the paper model gives 0.818 (0.788 with the similarity of the
    # paper), the AligNet models 0.835 (ridge) and 0.842 (kernel); the noise ceiling is 0.975
    similarity = dimpred.similarity(dimpred.predict(features_of(stuff, name), name), method="2afc")
    r = r_with_humans(similarity, stuff)
    assert r > 0.80, f"{name}: r with human similarity is only {r:.3f}, expected more than 0.80"


def test_stored_human_correlations_fit_the_expected_predictions(stuff):
    for name in MATERIAL_MODEL_NAMES:
        r = r_with_humans(similarity_2afc_by_definition(stuff["stuff48_expected_" + name]), stuff)
        expected = float(stuff["stuff48_human_r"][name])
        assert abs(r - expected) < 1e-6, f"{name}: r from the expected predictions {r:.6f}, stored {expected:.6f}"


@pytest.mark.parametrize("name", MATERIAL_MODEL_NAMES)
def test_z_scored_stuff_features_have_a_std_close_to_1(stuff, name):
    # the 48 images were not used for training (see test_model_files.py for the THINGS models)
    model = dimpred.load_model(name)
    z = (features_of(stuff, name) - model["feature_mean"]) / model["feature_scale"]
    assert 0.8 < z.std() < 1.25, f"{name}: std of the z-scored features is {z.std():.3f}, expected about 1"


def test_kernel_model_has_the_kernel_of_the_ridge_model_and_no_close_pairs():
    kernel, ridge = dimpred.load_model(KERNEL_MODEL), dimpred.load_model(RIDGE_MODEL)
    assert kernel["kernel_tau"] == 0.5, f"kernel_tau is {kernel['kernel_tau']}"
    assert kernel["close_pairs_weight"] is None and kernel["close_pairs_threshold"] is None
    for key in ["feature_mean", "feature_scale", "target_mean"]:
        assert_close(kernel[key], ridge[key], 1e-12, f"{key} of {KERNEL_MODEL} vs {RIDGE_MODEL}")
    norms = np.linalg.norm(kernel["kernel_features"], axis=1)
    assert_close(norms, np.ones(600), 1e-6, "length of the kernel features (single precision)")


def test_similarity_with_features_gives_error_for_the_material_kernel_model(stuff):
    features = stuff["stuff48_features_alignet"]
    embedding = dimpred.predict(features, KERNEL_MODEL)
    with pytest.raises(ValueError, match="no close-pair settings"):
        dimpred.similarity(embedding, method="2afc", features=features, model=KERNEL_MODEL)


@pytest.mark.parametrize("name", MATERIAL_MODEL_NAMES)
def test_info_names_the_stuff_data(name):
    info = dimpred.load_model(name)["info"]
    assert "STUFF" in info["embedding"] and "2AFC" in info["embedding"], f"info['embedding'] = {info['embedding']!r}"
    assert info["training_images"].startswith("600 STUFF images"), f"info['training_images'] = {info['training_images']!r}"
    assert 'method="2afc"' in info["note"], f"info['note'] = {info['note']!r}"


# --- the 2AFC similarity

def test_2afc_similarity_follows_its_definition():
    embedding = np.random.default_rng(0).random((7, 4)) * 2
    assert_close(dimpred.similarity(embedding, method="2afc"), similarity_2afc_by_definition(embedding), 1e-12,
                 "2afc similarity vs its definition")


def test_2afc_similarity_has_no_overflow_for_large_values():
    embedding = np.random.default_rng(1).random((6, 3)) * 100  # dot products up to 30000
    similarity = dimpred.similarity(embedding, method="2afc")
    assert np.all(np.isfinite(similarity)), "2afc similarity has nan or inf for large values"
    assert_close(similarity, similarity_2afc_by_definition(embedding), 1e-12, "2afc similarity for large values")


def test_2afc_similarity_is_symmetric_with_diagonal_1_and_mean_one_half():
    # for each reference, j is chosen over k or k over j, so the mean off the diagonal is exactly 1/2
    similarity = dimpred.similarity(np.random.default_rng(2).random((30, 36)), method="2afc")
    assert_close(similarity, similarity.T, 0, "2afc similarity vs its transpose")
    assert_close(np.diag(similarity), np.ones(30), 0, "diagonal")
    assert abs(similarity[~np.eye(30, dtype=bool)].mean() - 0.5) < 1e-12, "mean off the diagonal"


def test_2afc_similarity_follows_the_order_of_the_objects():
    embedding = np.random.default_rng(3).random((10, 5))
    order = np.random.default_rng(4).permutation(10)
    assert_close(dimpred.similarity(embedding[order], method="2afc"),
                 dimpred.similarity(embedding, method="2afc")[np.ix_(order, order)], 1e-12, "reordered objects")


def test_2afc_method_name_ignores_case():
    embedding = np.random.default_rng(5).random((5, 3))
    assert_close(dimpred.similarity(embedding, method="2AFC"), dimpred.similarity(embedding, method="2afc"), 0,
                 "method '2AFC' vs '2afc'")


def test_2afc_similarity_needs_three_objects():
    with pytest.raises(ValueError, match="at least 3 objects"):
        dimpred.similarity(np.ones((2, 3)), method="2afc")


# --- training data

def test_stuff_training_data_fit_together():
    embedding = np.loadtxt(os.path.join(TRAINING_DATA, "stuff_embedding_36d.txt"))
    labels = read_lines(os.path.join(TRAINING_DATA, "labels_36d.txt"))
    names = read_lines(os.path.join(TRAINING_DATA, "stuff_image_names.txt"))
    # (the embedding of the paper has a few values just below 0, down to -0.0013)
    assert embedding.shape == (600, 36) and embedding.min() > -0.01, f"embedding {embedding.shape}, min {embedding.min()}"
    # the dimensions are sorted by their sum, as in the paper
    assert np.all(np.diff(embedding.sum(axis=0)) <= 0), "dimensions are not sorted by their sum"
    assert len(labels) == 36 and len(set(labels)) == 36, "labels"
    assert [n.split("/")[1] for n in names] == [f"{i:04d}.jpg" for i in range(1, 601)], "image numbers"
    materials = [n.split("/")[0] for n in names]
    assert len(set(materials)) == 200 and all(materials.count(m) == 3 for m in set(materials)), "3 images per material"
    for name in MATERIAL_MODEL_NAMES:
        assert dimpred.load_model(name)["labels"] == labels, f"{name}: labels differ from labels_36d.txt"


def test_paper_settings():
    settings = np.loadtxt(os.path.join(TRAINING_DATA, "stuff_rn50x64_paper.txt"))
    assert settings.shape == (36, 2), f"settings have shape {settings.shape}"
    assert np.all(settings[:, 0] == 1000), "the paper chose lambda 1000 for all dimensions"
    assert np.all(settings[:, 1] > 1) and np.all(settings[:, 1] < 1.5), f"scale factors {settings[:, 1]}"


# --- rebuilding the shipped files (slow, needs the cached features of the 600 images)

@pytest.mark.slow
@pytest.mark.parametrize("name", MATERIAL_MODEL_NAMES)
def test_build_material_models_gives_the_shipped_files(tmp_path, name):
    # DIMPRED_STUFF_FEATURES: folder with features_stuff600_<network>.npy, the features the shipped files
    # were built from (not part of the repository); needs the training environment (fit.py)
    folder = os.environ.get("DIMPRED_STUFF_FEATURES")
    if not folder or not os.path.isdir(folder):
        pytest.skip("set DIMPRED_STUFF_FEATURES to the folder with the cached features of the 600 STUFF images")
    pytest.importorskip("fracridge")
    sys.path.insert(0, TRAINING)
    try:
        import build_material_models
    finally:
        sys.path.remove(TRAINING)
    build_material_models.main(["--features", folder, "--out", str(tmp_path), "--only", name])
    built = load_mat(os.path.join(tmp_path, name + ".mat"))
    shipped = load_mat(os.path.join(os.path.dirname(dimpred.__file__), "models", name + ".mat"))
    for key in ["weights", "feature_mean", "feature_scale", "target_mean", "kernel_features", "kernel_coefficients"]:
        if key in shipped:
            assert_close(built[key], shipped[key], 1e-10, f"{name}: {key} rebuilt vs shipped")
    assert built["labels"] == shipped["labels"]
