"""
Tests of the command line tool (python -m dimpred).

Each test runs the tool in a new Python process in a temporary folder. The
route with precomputed features (--features FILE.mat) is fast and must work
without torch, so these tests always run. The route with images needs torch
and open_clip and is marked slow. Where several things are checked on the
output of one run, the run is done once in a fixture and each test checks
one of them.

Hebartlab, 2026/09/30

See also: test_predict.py, test_extract_features.py, test_package.py
"""

import csv
import importlib.util
import os
import re
import shutil

import numpy as np
import pytest
import scipy.io

import dimpred
from helpers import (DEFAULT_MODEL, IMAGES, MODELS, TOL_ALIGNET_DIFF, TOL_FRESH_PREDICTION, TOL_MODEL, assert_close,
                     assert_features_match, features_for, imported_packages, load_mat, output_of, run_dimpred)

# History:
# 2026/10/02: an empty MKL_NUM_THREADS (as from MATLAB) gives no warning
# 2026/10/02: the default model is alignet_siglip2b_66d_ridge; the tests of
#   folders, order and options from images use vitb32_66d_elastic, so they
#   do not need the AligNet weights
# 2026/09/30: features files with a single image and in single precision; the
#   route with images without torch; the features route must not import
#   torch; clearer error and summary checks; one behavior per test; the
#   order test uses a second unsorted order
# 2026/09/30: written together with the tests, before the package code

# Numbers in a csv file are text, so they may be rounded a bit
TOL_CSV = 1e-5

TORCH_INSTALLED = importlib.util.find_spec("torch") is not None  # does not import torch


def save_features(fname, features, files=None, dtype=np.float64):
    """Write features (and optionally file names) as the command line tool expects them.

    dtype=np.float32 gives single precision, as dimpred_extract_features
    returns in MATLAB.
    """

    data = dict(features=np.asarray(features, dtype=dtype))
    if files is not None:
        data["files"] = np.array(files, dtype=object).reshape(-1, 1)  # cell array
    scipy.io.savemat(fname, data)


def read_csv(fname):
    """Header, image column and numbers of a csv file written by the tool."""

    with open(fname, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    images = [row[0] for row in body]
    values = np.array([[float(v) for v in row[1:]] for row in body])
    return header, images, values


def assert_success(result):
    assert result.returncode == 0, f"python -m dimpred failed with exit code {result.returncode}\n{output_of(result)}"


def assert_clear_error(result, *parts):
    """Non-zero exit code, the message mentions all parts, and no Python traceback."""

    assert result.returncode != 0, f"python -m dimpred should have failed\n{output_of(result)}"
    output = result.stdout + result.stderr
    for part in parts:
        assert part in output, f"the error message should mention {part!r}\n{output_of(result)}"
    # user errors should be explained in one message, not with a traceback
    assert "Traceback" not in output, f"error shown as a Python traceback\n{output_of(result)}"


# --- predictions from precomputed features (fast, no torch)

@pytest.fixture(scope="module")
def features_to_csv(tmp_path_factory, ref_session):
    """Run the tool once on 5 images with the default model, output as csv."""

    folder = tmp_path_factory.mktemp("features_to_csv")
    save_features(folder / "features.mat", features_for(ref_session, DEFAULT_MODEL)[:5], ref_session["files"][:5])
    result = run_dimpred(["--features", "features.mat", "--out", "out.csv"], cwd=folder)
    assert_success(result)
    return read_csv(folder / "out.csv")


def test_csv_header_is_image_and_the_labels(features_to_csv):
    header, _, _ = features_to_csv
    assert header == ["image"] + dimpred.load_model(DEFAULT_MODEL)["labels"], f"csv header is {header}"


def test_csv_image_column_has_the_file_names(features_to_csv, ref):
    _, images, _ = features_to_csv
    assert images == ref["files"][:5], f"image column is {images}"


def test_csv_values_are_the_predictions_of_the_default_model(features_to_csv, ref):
    _, _, values = features_to_csv
    assert_close(values, ref["expected_" + DEFAULT_MODEL][:5], TOL_CSV, "csv predictions")


@pytest.fixture(scope="module")
def features_to_mat(tmp_path_factory, ref_session):
    """Run the tool once on 6 images with the paper model, output as .mat."""

    folder = tmp_path_factory.mktemp("features_to_mat")
    save_features(folder / "features.mat", ref_session["features_rn50x64"][:6], ref_session["files"][:6])
    result = run_dimpred(["--features", "features.mat", "--model", "rn50x64_49d_ridge", "--out", "out.mat"],
                         cwd=folder)
    assert_success(result)
    return load_mat(folder / "out.mat")


def test_mat_embedding_is_philipps_published_prediction(features_to_mat, ref):
    assert_close(features_to_mat["embedding"], ref["published_rn50x64_49d_ridge"][:6], TOL_MODEL,
                 "embedding in the .mat file vs Philipp's published predictions")


def test_mat_files_are_the_file_names(features_to_mat, ref):
    assert features_to_mat["files"] == ref["files"][:6], f"files in the .mat file: {features_to_mat['files']}"


def test_mat_labels_are_the_labels_of_the_model(features_to_mat):
    assert features_to_mat["labels"] == dimpred.load_model("rn50x64_49d_ridge")["labels"], "labels differ"


def test_mat_model_is_the_model_name(features_to_mat):
    assert features_to_mat["model"] == "rn50x64_49d_ridge", f"model in the .mat file: {features_to_mat['model']!r}"


def test_default_output_file_is_dimpred_predictions_csv(tmp_path, ref):
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[:3], ref["files"][:3])
    assert_success(run_dimpred(["--features", "features.mat"], cwd=tmp_path))
    assert os.path.exists(tmp_path / "dimpred_predictions.csv"), "dimpred_predictions.csv was not written"


def test_rows_keep_the_order_of_the_features_file(tmp_path, ref):
    # file names in unsorted order must not be sorted on the way
    rows = [4, 0, 3, 1, 2]
    files = [ref["files"][i] for i in rows]
    assert files != sorted(files), "test data should not be sorted"
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[rows], files)
    assert_success(run_dimpred(["--features", "features.mat", "--out", "out.csv"], cwd=tmp_path))
    _, images, values = read_csv(tmp_path / "out.csv")
    assert images == files, f"image column is {images}, expected {files}"
    assert_close(values, ref["expected_" + DEFAULT_MODEL][rows], TOL_CSV, "predictions of unsorted rows")


def test_file_names_are_optional(tmp_path, ref):
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[:4])
    assert_success(run_dimpred(["--features", "features.mat", "--out", "out.csv"], cwd=tmp_path))
    _, images, values = read_csv(tmp_path / "out.csv")
    assert len(images) == 4, f"{len(images)} rows for 4 images"
    assert_close(values, ref["expected_" + DEFAULT_MODEL][:4], TOL_CSV, "predictions without file names")


def test_single_image_to_csv(tmp_path, ref):
    # scipy.io.loadmat turns a 1 x 1 cell into a plain str and 1 x p features
    # into a vector, which the tool has to undo: one row, with the whole file name
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[3:4], ref["files"][3:4])
    assert_success(run_dimpred(["--features", "features.mat", "--out", "out.csv"], cwd=tmp_path))
    _, images, values = read_csv(tmp_path / "out.csv")
    assert images == ref["files"][3:4], f"image column is {images}, expected {ref['files'][3:4]}"
    assert_close(values, ref["expected_" + DEFAULT_MODEL][3:4], TOL_CSV, "csv prediction of a single image")


def test_single_image_to_mat_gives_one_row(tmp_path, ref):
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[3:4], ref["files"][3:4])
    assert_success(run_dimpred(["--features", "features.mat", "--out", "out.mat"], cwd=tmp_path))
    raw = scipy.io.loadmat(tmp_path / "out.mat")  # without simplify_cells, which would drop the row dimension
    assert raw["embedding"].shape == (1, 66), f"embedding has shape {raw['embedding'].shape}, expected (1, 66)"
    assert_close(raw["embedding"], ref["expected_" + DEFAULT_MODEL][3:4], TOL_MODEL, "embedding of a single image")


def test_single_image_to_mat_keeps_the_file_name(tmp_path, ref):
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[3:4], ref["files"][3:4])
    assert_success(run_dimpred(["--features", "features.mat", "--out", "out.mat"], cwd=tmp_path))
    files = load_mat(tmp_path / "out.mat")["files"]
    assert files == ref["files"][3:4], f"files in the .mat file: {files}, expected {ref['files'][3:4]}"


def test_single_precision_features(tmp_path, ref):
    # dimpred_extract_features returns single in MATLAB, so users save
    # features in single precision. They are computed in float64, as in predict.
    features = features_for(ref, DEFAULT_MODEL)[:4].astype(np.float32)
    save_features(tmp_path / "features.mat", features, ref["files"][:4], dtype=np.float32)
    assert_success(run_dimpred(["--features", "features.mat", "--out", "out.csv"], cwd=tmp_path))
    _, _, values = read_csv(tmp_path / "out.csv")
    assert_close(values, dimpred.predict(features.astype(np.float64)), TOL_CSV, "predictions of single features")


def test_prints_a_summary_with_number_of_images_model_and_output_file(tmp_path, ref):
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[:7], ref["files"][:7])
    result = run_dimpred(["--features", "features.mat", "--out", "out.csv"], cwd=tmp_path)
    assert_success(result)
    for part, found in [("the number of images (7)", re.search(r"\b7\b", result.stdout)),
                        (f"the model ({DEFAULT_MODEL})", DEFAULT_MODEL in result.stdout),
                        ("the output file (out.csv)", "out.csv" in result.stdout)]:
        assert found, f"the summary should mention {part}\n{output_of(result)}"


def test_features_route_works_without_torch(tmp_path, ref):
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[:5], ref["files"][:5])
    result = run_dimpred(["--features", "features.mat", "--out", "out.csv"], cwd=tmp_path, without_torch=True)
    assert_success(result)
    _, _, values = read_csv(tmp_path / "out.csv")
    assert_close(values, ref["expected_" + DEFAULT_MODEL][:5], TOL_CSV, "csv predictions without torch")


@pytest.mark.skipif(not TORCH_INSTALLED, reason="torch is not installed, so it cannot be imported")
def test_features_route_does_not_import_torch(tmp_path, ref):
    # importing torch takes several seconds, which would make every call of
    # the tool slow, also when no images are processed
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[:3], ref["files"][:3])
    result = run_dimpred(["--features", "features.mat", "--out", "out.csv"], cwd=tmp_path,
                         python_options=["-X", "importtime"])
    assert_success(result)
    imported = sorted(imported_packages(result.stderr) & {"torch", "open_clip"})
    assert not imported, f"python -m dimpred --features imported {imported}"


def test_help_names_the_default_model(tmp_path):
    result = run_dimpred(["--help"], cwd=tmp_path)
    assert_success(result)
    assert f"(default: {DEFAULT_MODEL})" in " ".join(result.stdout.split()), (
        f"the help should say that the default model is {DEFAULT_MODEL}\n{output_of(result)}")


# --- errors (fast)

def test_missing_features_file_gives_clear_error(tmp_path):
    assert_clear_error(run_dimpred(["--features", "no_such_file.mat"], cwd=tmp_path), "no_such_file.mat")


def test_features_file_without_features_variable_gives_clear_error(tmp_path):
    # the message has to name the file and the missing variable
    scipy.io.savemat(tmp_path / "wrong.mat", dict(something_else=np.ones((3, 512))))
    assert_clear_error(run_dimpred(["--features", "wrong.mat"], cwd=tmp_path), "wrong.mat", "features")


def test_wrong_number_of_features_gives_clear_error(tmp_path, ref):
    # RN50x64 features (1024) for the default AligNet model (768)
    save_features(tmp_path / "features.mat", ref["features_rn50x64"][:3])
    n_features = str(MODELS[DEFAULT_MODEL]["n_features"])
    assert_clear_error(run_dimpred(["--features", "features.mat"], cwd=tmp_path), n_features)


def test_unknown_model_gives_clear_error(tmp_path, ref):
    save_features(tmp_path / "features.mat", features_for(ref, DEFAULT_MODEL)[:3])
    result = run_dimpred(["--features", "features.mat", "--model", "no_such_model"], cwd=tmp_path)
    assert_clear_error(result, "no_such_model", DEFAULT_MODEL)


def test_no_input_gives_error(tmp_path):
    result = run_dimpred([], cwd=tmp_path)
    assert result.returncode != 0, f"python -m dimpred without input should fail\n{output_of(result)}"


def test_missing_image_gives_clear_error(tmp_path, cc0_paths):
    # The paths are checked before the network is loaded. Without torch, a
    # tool that loads the network first fails with a message about torch.
    result = run_dimpred([cc0_paths[0], "no_such_image.jpg"], cwd=tmp_path, without_torch=True)
    assert_clear_error(result, "no_such_image.jpg")


def test_images_without_torch_give_clear_error(tmp_path, cc0_paths):
    # the message has to say that torch (open_clip_torch) is needed for images
    result = run_dimpred([cc0_paths[0], "--out", "out.csv"], cwd=tmp_path, without_torch=True)
    assert_clear_error(result, "torch")


# --- predictions from images (slow, needs torch)

@pytest.fixture(scope="module")
def images_to_csv(tmp_path_factory, cc0_paths_reordered, alignet_available):
    """Run the tool once on the CC0 images in the second unsorted order with the default model, output as csv."""

    folder = tmp_path_factory.mktemp("images_to_csv")
    paths, rows = cc0_paths_reordered
    result = run_dimpred(paths + ["--out", "out.csv"], cwd=folder)
    assert_success(result)
    return read_csv(folder / "out.csv"), paths, rows


@pytest.mark.slow
def test_images_to_csv_header(images_to_csv):
    (header, _, _), _, _ = images_to_csv
    assert header == ["image"] + dimpred.load_model(DEFAULT_MODEL)["labels"], f"csv header is {header}"


@pytest.mark.slow
def test_images_to_csv_keeps_the_given_order(images_to_csv):
    (_, images, _), paths, _ = images_to_csv
    names = [os.path.basename(f) for f in images]
    assert names == [os.path.basename(p) for p in paths], f"image column is {names}"


@pytest.mark.slow
def test_images_to_csv_values(images_to_csv, ref):
    (_, _, values), _, rows = images_to_csv
    expected = dimpred.predict(ref["cc0_features_alignet"][rows])
    assert_close(values, expected, TOL_FRESH_PREDICTION, "csv predictions from images")


@pytest.mark.slow
def test_paper_model_from_images_gives_philipps_published_predictions(tmp_path, ref, cc0_paths, open_clip_available):
    result = run_dimpred(cc0_paths + ["--model", "rn50x64_49d_ridge", "--out", "out.mat"], cwd=tmp_path)
    assert_success(result)
    out = load_mat(tmp_path / "out.mat")
    assert_close(out["embedding"], ref["cc0_published_rn50x64_49d_ridge"], TOL_FRESH_PREDICTION,
                 "predictions from images vs Philipp's published predictions")


@pytest.mark.slow
def test_folder_is_expanded_in_sorted_order(tmp_path, ref, open_clip_available):
    result = run_dimpred([IMAGES, "--model", "vitb32_66d_elastic", "--features-only", "--out", "out.mat"],
                         cwd=tmp_path)
    assert_success(result)
    out = load_mat(tmp_path / "out.mat")
    names = [os.path.basename(f) for f in out["files"]]
    assert names == sorted(ref["cc0_files"]), f"files in the .mat file: {names}"
    rows = [ref["cc0_files"].index(n) for n in names]
    assert_features_match(out["features"], ref["cc0_features_vitb32"][rows], "features of the images folder")


@pytest.mark.slow
def test_files_and_folders_keep_the_given_order(tmp_path, ref, cc0_paths, open_clip_available):
    # one file first, then a folder with two images: the file stays first,
    # the folder is expanded in sorted order
    os.makedirs(tmp_path / "folder")
    shutil.copy(cc0_paths[0], tmp_path / "folder" / "b.jpg")
    shutil.copy(cc0_paths[1], tmp_path / "folder" / "a.jpg")
    result = run_dimpred([cc0_paths[2], str(tmp_path / "folder"), "--model", "vitb32_66d_elastic", "--features-only",
                          "--out", "out.csv"], cwd=tmp_path)
    assert_success(result)
    header, images, values = read_csv(tmp_path / "out.csv")
    assert header == ["image"] + [f"feature_{i}" for i in range(1, 513)], f"csv header starts with {header[:3]}"
    names = [os.path.basename(f) for f in images]
    assert names == [ref["cc0_files"][2], "a.jpg", "b.jpg"], f"image column is {names}"
    assert_features_match(values, ref["cc0_features_vitb32"][[2, 1, 0]], "features of a file and a folder")


@pytest.mark.slow
def test_device_and_batch_size_options(tmp_path, ref, cc0_paths, open_clip_available):
    result = run_dimpred(cc0_paths + ["--model", "vitb32_66d_elastic", "--features-only", "--device", "cpu",
                                      "--batch-size", "2", "--out", "out.mat"], cwd=tmp_path)
    assert_success(result)
    assert_features_match(load_mat(tmp_path / "out.mat")["features"], ref["cc0_features_vitb32"],
                          "features with --device cpu --batch-size 2")


@pytest.mark.slow
def test_empty_mkl_num_threads_gives_no_warning(tmp_path, cc0_paths, open_clip_available, monkeypatch):
    # MATLAB starts Python with MKL_NUM_THREADS set to an empty text. torch
    # then warns on every run that the value is invalid, which looks like an
    # error in the output of dimpred_rise and in the errors of the MATLAB
    # functions, although torch still uses all threads
    monkeypatch.setenv("MKL_NUM_THREADS", "")  # run_dimpred passes os.environ on
    result = run_dimpred(cc0_paths[:1] + ["--model", "vitb32_66d_elastic", "--features-only", "--device", "cpu",
                                          "--out", "out.mat"], cwd=tmp_path)
    assert_success(result)
    assert "MKL_NUM_THREADS" not in result.stdout + result.stderr, (
        f"an empty MKL_NUM_THREADS should give no warning\n{output_of(result)}")


@pytest.mark.slow
def test_default_model_features_only_gives_the_alignet_features(tmp_path, ref, cc0_paths, alignet_available):
    result = run_dimpred(cc0_paths + ["--features-only", "--out", "out.mat"], cwd=tmp_path)
    assert_success(result)
    assert_close(load_mat(tmp_path / "out.mat")["features"], ref["cc0_features_alignet"], TOL_ALIGNET_DIFF,
                 "AligNet features from the command line tool vs TensorFlow")
