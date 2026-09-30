"""
Tests of dimpred.list_models, dimpred.load_model and the shipped model files.

We check that every shipped model can be loaded and is complete (shapes that
fit together, finite numbers, positive scales and dimension means, labels and
info), that a model can be given by name, by path or as an already loaded
dict with the same result, and that wrong input gives a clear error. The
numbers in the model files themselves are checked in test_model_files.py.

Martin Hebart, 2026/09/30

See also: test_predict.py, test_model_files.py
"""

import os
import shutil

import numpy as np
import pytest
import scipy.io

import dimpred
from helpers import DEFAULT_MODEL, MODEL_NAMES, MODELS, MODELS_DIR, load_mat

# History:
# 2026/09/30: a model file without one of its variables has to give an error;
#   the comparison with Philipp's original training moved to test_model_files.py
# 2026/09/30: written together with the tests, before the package code

INFO_TEXT_FIELDS = ["name", "network", "pretrained", "layer", "preprocessing", "embedding", "regression",
                    "training_images", "source", "note", "created"]
MODEL_VARIABLES = ["weights", "feature_mean", "feature_scale", "target_mean", "labels", "info"]


def save_model_file(fname, model, leave_out=()):
    """Write a model dict to a .mat file in the format of the shipped models.

    The variables named in leave_out are not written.
    """

    variables = dict(
        weights=np.asarray(model["weights"], dtype=float),
        feature_mean=np.asarray(model["feature_mean"], dtype=float).reshape(1, -1),
        feature_scale=np.asarray(model["feature_scale"], dtype=float).reshape(1, -1),
        target_mean=np.asarray(model["target_mean"], dtype=float).reshape(1, -1),
        labels=np.array(model["labels"], dtype=object).reshape(-1, 1),
        info={k: (float(v) if k in ("n_features", "n_dims") else v) for k, v in model["info"].items()},
    )
    scipy.io.savemat(fname, {k: v for k, v in variables.items() if k not in leave_out})


def assert_same_model(a, b):
    """Two loaded models have the same numbers, labels and info."""

    for key in ["weights", "feature_mean", "feature_scale", "target_mean"]:
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"'{key}' differs")
    assert a["labels"] == b["labels"], "labels differ"
    assert a["info"] == b["info"], "info differs"


# --- list_models and DEFAULT_MODEL

def test_list_models_gives_the_shipped_models():
    assert dimpred.list_models() == MODEL_NAMES


def test_list_models_gives_the_files_in_the_models_folder():
    names = sorted(f[:-4] for f in os.listdir(MODELS_DIR) if f.endswith(".mat"))
    assert dimpred.list_models() == names


def test_default_model_is_vitb32_66d_elastic():
    assert dimpred.DEFAULT_MODEL == "vitb32_66d_elastic"


def test_load_model_without_input_gives_the_default_model():
    assert dimpred.load_model()["info"]["name"] == DEFAULT_MODEL


def test_load_model_with_none_gives_the_default_model():
    assert dimpred.load_model(None)["info"]["name"] == DEFAULT_MODEL


# --- content of each shipped model

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_has_all_keys(name):
    model = dimpred.load_model(name)
    missing = {"weights", "feature_mean", "feature_scale", "target_mean", "labels", "info", "file"} - set(model)
    assert not missing, f"{name}: keys missing in the loaded model: {sorted(missing)}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_arrays_have_the_right_shapes(name):
    model = dimpred.load_model(name)
    n_features, n_dims = MODELS[name]["n_features"], MODELS[name]["n_dims"]
    assert model["weights"].shape == (n_features, n_dims), f"{name}: weights have shape {model['weights'].shape}"
    assert model["feature_mean"].shape == (n_features,), f"{name}: feature_mean has shape {model['feature_mean'].shape}"
    assert model["feature_scale"].shape == (n_features,), (
        f"{name}: feature_scale has shape {model['feature_scale'].shape}")
    assert model["target_mean"].shape == (n_dims,), f"{name}: target_mean has shape {model['target_mean'].shape}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_arrays_are_float64(name):
    model = dimpred.load_model(name)
    for key in ["weights", "feature_mean", "feature_scale", "target_mean"]:
        assert isinstance(model[key], np.ndarray), f"{name}: '{key}' is a {type(model[key])}, not a numpy array"
        assert model[key].dtype == np.float64, f"{name}: '{key}' has dtype {model[key].dtype}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_arrays_are_finite(name):
    model = dimpred.load_model(name)
    for key in ["weights", "feature_mean", "feature_scale", "target_mean"]:
        assert np.all(np.isfinite(model[key])), f"{name}: '{key}' contains nan or inf"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_feature_scale_is_positive(name):
    # the features are divided by it
    scale = dimpred.load_model(name)["feature_scale"]
    assert scale.min() > 0, f"{name}: feature_scale has values <= 0 (min {scale.min():g})"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_target_mean_is_positive(name):
    # SPoSE dimensions are non-negative, so their means across the 1854
    # training images are positive
    target_mean = dimpred.load_model(name)["target_mean"]
    assert target_mean.min() > 0, f"{name}: target_mean has values <= 0 (min {target_mean.min():g})"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_weights_are_not_all_zero(name):
    weights = dimpred.load_model(name)["weights"]
    assert np.all(np.abs(weights).max(axis=0) > 0), f"{name}: some dimensions have only zero weights"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_labels_are_one_string_per_dimension(name):
    labels = dimpred.load_model(name)["labels"]
    assert isinstance(labels, list), f"{name}: labels are a {type(labels)}, not a list"
    assert len(labels) == MODELS[name]["n_dims"], (
        f"{name}: {len(labels)} labels for {MODELS[name]['n_dims']} dimensions")
    assert all(isinstance(label, str) and label.strip() for label in labels), f"{name}: empty or non-text labels"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_info_text_fields(name):
    info = dimpred.load_model(name)["info"]
    for field in INFO_TEXT_FIELDS:
        assert field in info, f"{name}: info has no field '{field}'"
        assert isinstance(info[field], str) and info[field].strip(), f"{name}: info['{field}'] is empty or not text"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_info_number_fields_are_int_and_fit_the_arrays(name):
    model = dimpred.load_model(name)
    info = model["info"]
    for field in ["n_features", "n_dims"]:
        assert type(info[field]) is int, (
            f"{name}: info['{field}'] is {info[field]!r} ({type(info[field]).__name__}), expected int")
    numbers = (info["n_features"], info["n_dims"])
    assert numbers == model["weights"].shape, f"{name}: info gives {numbers}, the weights have {model['weights'].shape}"
    expected = (MODELS[name]["n_features"], MODELS[name]["n_dims"])
    assert numbers == expected, f"{name}: info gives {numbers}, expected {expected}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_info_describes_the_model(name):
    info = dimpred.load_model(name)["info"]
    assert info["name"] == name, f"info['name'] is {info['name']!r}, expected {name!r}"
    assert info["network"] == MODELS[name]["network"], f"{name}: info['network'] is {info['network']!r}"
    assert info["pretrained"] == "openai", f"{name}: info['pretrained'] is {info['pretrained']!r}"
    # "fractional ridge regression ..." or "elastic net ..."
    assert MODELS[name]["regression"] in info["regression"].lower(), (
        f"{name}: info['regression'] = {info['regression']!r}")


def test_vit_model_uses_the_quickgelu_network():
    # OpenAI's CLIP ViTs use QuickGELU. open_clip's plain "ViT-B-32" uses GELU
    # with the same weights and gives different features (r ~ 0.98), so the
    # network name has to be the -quickgelu version.
    assert dimpred.load_model("vitb32_66d_elastic")["info"]["network"] == "ViT-B-32-quickgelu"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_file_is_the_absolute_path_of_the_model_file(name):
    fname = dimpred.load_model(name)["file"]
    assert os.path.isabs(fname), f"{name}: file {fname!r} is not an absolute path"
    assert os.path.samefile(fname, os.path.join(MODELS_DIR, name + ".mat")), f"{name}: file is {fname!r}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_loaded_numbers_are_the_numbers_in_the_file(name):
    # read the file independently of dimpred: nothing may be transposed,
    # rescaled or reordered on the way
    raw = load_mat(os.path.join(MODELS_DIR, name + ".mat"))
    model = dimpred.load_model(name)
    for key in ["weights", "feature_mean", "feature_scale", "target_mean"]:
        np.testing.assert_array_equal(model[key], raw[key], err_msg=f"{name}: '{key}' differs from the file")
    assert model["labels"] == raw["labels"], f"{name}: labels differ from the file"


# --- different ways to give the model

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_load_by_name_and_by_path_give_the_same_model(name):
    by_name = dimpred.load_model(name)
    by_path = dimpred.load_model(os.path.join(MODELS_DIR, name + ".mat"))
    assert_same_model(by_name, by_path)


def test_passing_a_loaded_model_returns_it_unchanged():
    model = dimpred.load_model("rn50x64_49d_ridge")
    weights = model["weights"].copy()
    assert dimpred.load_model(model) is model, "load_model should return the given dict itself"
    np.testing.assert_array_equal(model["weights"], weights, err_msg="load_model changed the given model")


def test_load_model_from_a_copied_file(tmp_path):
    fname = tmp_path / "my_model.mat"
    shutil.copy(os.path.join(MODELS_DIR, DEFAULT_MODEL + ".mat"), fname)
    model = dimpred.load_model(str(fname))
    assert_same_model(model, dimpred.load_model(DEFAULT_MODEL))
    assert os.path.samefile(model["file"], fname), f"file is {model['file']!r}, expected {str(fname)!r}"


def test_load_model_from_a_relative_path(tmp_path, monkeypatch):
    os.makedirs(tmp_path / "copies")
    shutil.copy(os.path.join(MODELS_DIR, "rn50x64_49d_ridge.mat"), tmp_path / "copies" / "my_model.mat")
    monkeypatch.chdir(tmp_path)
    model = dimpred.load_model(os.path.join("copies", "my_model.mat"))
    assert os.path.isabs(model["file"]), f"file {model['file']!r} is not an absolute path"
    assert os.path.samefile(model["file"], tmp_path / "copies" / "my_model.mat"), f"file is {model['file']!r}"


def test_load_model_from_a_file_written_like_the_shipped_models(tmp_path):
    # users can build their own model files in the same format
    model = dimpred.load_model("rn50x64_49d_ridge")
    fname = tmp_path / "own_model.mat"
    save_model_file(fname, model)
    assert_same_model(dimpred.load_model(str(fname)), model)


# --- errors

def test_unknown_model_name_gives_error_that_lists_the_models():
    with pytest.raises(ValueError) as error:
        dimpred.load_model("no_such_model")
    message = str(error.value)
    missing = [name for name in MODEL_NAMES if name not in message]
    assert not missing, f"the error message should list the available models, but {missing} are missing: {message!r}"


def test_missing_model_file_gives_error(tmp_path):
    with pytest.raises((ValueError, OSError)):
        dimpred.load_model(str(tmp_path / "does_not_exist.mat"))


@pytest.mark.parametrize("variable", ["weights", "feature_mean", "feature_scale", "target_mean", "labels"])
def test_inconsistent_model_file_gives_error(tmp_path, variable):
    # remove the last entry of one variable, so that the shapes do not fit together anymore
    model = dimpred.load_model("rn50x64_49d_ridge")
    broken = dict(model)
    if variable == "weights":
        broken["weights"] = model["weights"][:, :-1]
    else:
        broken[variable] = model[variable][:-1]
    fname = tmp_path / "broken.mat"
    save_model_file(fname, broken)
    with pytest.raises(ValueError):
        dimpred.load_model(str(fname))


@pytest.mark.parametrize("variable", MODEL_VARIABLES)
def test_model_file_without_a_variable_gives_error(tmp_path, variable):
    # A file without target_mean would otherwise give predictions that are
    # far too small and mostly 0, so a missing variable must not be replaced
    # by a default. The message has to say which variable is missing.
    fname = tmp_path / f"no_{variable}.mat"
    save_model_file(fname, dimpred.load_model("rn50x64_49d_ridge"), leave_out=[variable])
    with pytest.raises((ValueError, KeyError)) as error:
        dimpred.load_model(str(fname))
    assert variable in str(error.value), f"the error message should name {variable!r}: {str(error.value)!r}"
