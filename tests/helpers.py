"""
Constants and small helper functions shared by the dimpred tests.

The test files import this module directly (conftest.py puts the tests folder
on the Python path). The pytest fixtures are in conftest.py.

The numbers here (model table, tolerances) are kept in one place, so that all
tests use the same values. The MATLAB tests use the same tolerances.

Martin Hebart, 2026/09/30

See also: conftest.py
"""

import math
import os
import subprocess
import sys

import numpy as np
import scipy.io

# History:
# 2026/09/30: tolerances of the training tests, second order of the CC0
#   images, reference SPoSE similarity (moved here from test_similarity.py),
#   helpers to check which modules a process imported
# 2026/09/30: written together with the tests, before the package code

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
FIXTURES = os.path.join(TESTS, "fixtures")
REFERENCE_FILE = os.path.join(FIXTURES, "reference_data.mat")
ORIGINAL_RIDGE_FILE = os.path.join(FIXTURES, "philipp_original_ridge_rn50x64_66d.mat")
IMAGES = os.path.join(FIXTURES, "images")
MODELS_DIR = os.path.join(REPO, "dimpred", "models")
MATLAB_CODE = os.path.join(REPO, "matlab")
MATLAB_TESTS = os.path.join(TESTS, "matlab")
TRAINING = os.path.join(REPO, "training")
TRAINING_DATA = os.path.join(TRAINING, "data")

DEFAULT_MODEL = "vitb32_66d_elastic"

# The shipped models (table in README.md, section Models)
MODELS = {
    "rn50x64_49d_ridge": dict(network="RN50x64", n_features=1024, n_dims=49, regression="ridge"),
    "rn50x64_66d_elastic": dict(network="RN50x64", n_features=1024, n_dims=66, regression="elastic"),
    "rn50x64_66d_ridge": dict(network="RN50x64", n_features=1024, n_dims=66, regression="ridge"),
    "vitb32_66d_elastic": dict(network="ViT-B-32-quickgelu", n_features=512, n_dims=66, regression="elastic"),
}
MODEL_NAMES = sorted(MODELS)

# Variables in reference_data.mat that hold the features of each network
FEATURES_KEY = {"RN50x64": "features_rn50x64", "ViT-B-32-quickgelu": "features_vitb32"}
CC0_FEATURES_KEY = {"RN50x64": "cc0_features_rn50x64", "ViT-B-32-quickgelu": "cc0_features_vitb32"}

# Most tests use the CC0 images in the order of cc0_files, which is not
# sorted. The tests of the row order use this second order (rows 3, 1, 2 of
# cc0_files), which is neither sorted nor the order of cc0_files, so that
# they find mistakes the other tests cannot find.
CC0_REORDERED = [2, 0, 1]

# Tolerances
TOL_MODEL = 1e-10          # shipped models vs expected_* and vs Philipp's published predictions
TOL_EXTRACT_R = 0.99999    # extracted vs reference features: minimum correlation per image
TOL_EXTRACT_DIFF = 2e-3    # extracted vs reference features: maximum abs difference (devices differ a bit)
TOL_FRESH_PREDICTION = 2e-3  # predictions from freshly extracted features vs published predictions
TOL_HUMAN_R = 0.002        # correlation between predicted and human similarity

# Tolerances of the training tests.
# When the sped-up and the original ridge code select the same fraction,
# their weights for the rn50x64_66d_ridge model differ by at most 4e-13, so
# 1e-8 leaves room for other computers but would show any real difference.
# The fractions come from the same grid, so they have to be identical up to
# rounding.
TOL_ORIGINAL_WEIGHTS = 1e-8
TOL_BEST_FRAC = 1e-12

# The rebuilt RN50x64 models were trained on features that we extracted
# again, Philipp's model on his own features of the same images. The feature
# means and scales differ by less than 5e-6 (relative to the scale). 1e-4
# leaves room for extraction on a gpu and is still small enough to show the
# std with ddof=1 (relative difference 2.7e-4 for 1854 images).
TOL_STANDARDIZER = 1e-4

# Variables that are cell arrays of char in our .mat files. scipy returns a
# cell with a single entry as a plain str, so we need the names to undo this.
CELL_VARIABLES = ("files", "image_set", "cc0_files", "labels", "names", "found_images")


def load_mat(fname):
    """Load a .mat file into a dict of numpy arrays, lists of str and dicts.

    scipy.io.loadmat with simplify_cells=True turns structs into dicts and cell
    arrays of char into arrays of str, and it removes all singleton
    dimensions (a 1 x n row vector comes back with shape (n,)). Cell arrays are
    returned as lists of str here, numbers as float64 arrays.
    """

    data = scipy.io.loadmat(fname, simplify_cells=True)
    out = {}
    for key, value in data.items():
        if key.startswith("__"):
            continue  # file header, version, globals
        if key in CELL_VARIABLES or (isinstance(value, np.ndarray) and value.dtype.kind in "OUS"):
            out[key] = [str(v) for v in np.atleast_1d(value)]
        elif isinstance(value, (dict, str)):
            out[key] = value
        else:
            out[key] = np.asarray(value, dtype=float)
    return out


def read_lines(fname):
    """Lines of a text file without line ends, empty lines left out (as in build_models.py)."""

    with open(fname) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def assert_close(actual, expected, atol, what):
    """Check shape and maximum absolute difference of two arrays.

    Gives a short message with the largest difference and where it is, which is
    easier to read than numpy's printout of two large matrices.
    """

    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{what}: shape is {actual.shape}, expected {expected.shape}"
    n_nan = int(np.sum(np.isnan(actual) != np.isnan(expected)))
    assert n_nan == 0, f"{what}: {n_nan} values are nan in one array but not in the other"
    if actual.size == 0:
        return
    diff = np.abs(actual - expected)
    diff[np.isnan(diff)] = 0  # nan at the same place in both arrays counts as equal
    worst = np.unravel_index(np.argmax(diff), diff.shape)
    assert diff[worst] <= atol, (
        f"{what}: max abs difference {diff[worst]:.3g} at index {tuple(int(i) for i in worst)} "
        f"(got {actual[worst]:.8g}, expected {expected[worst]:.8g}); allowed: {atol:g}"
    )


def row_correlations(a, b):
    """Pearson correlation between row i of a and row i of b, for all rows."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def assert_features_match(actual, expected, what):
    """Compare extracted features with reference features (TOL_EXTRACT_R and TOL_EXTRACT_DIFF)."""

    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{what}: shape is {actual.shape}, expected {expected.shape}"
    r = row_correlations(actual, expected)
    assert r.min() > TOL_EXTRACT_R, (
        f"{what}: correlation with the reference features is {np.round(r, 6).tolist()} per image, "
        f"should be > {TOL_EXTRACT_R}. This happens with the wrong network (e.g. ViT-B-32 instead of "
        f"ViT-B-32-quickgelu), the wrong preprocessing, or images in the wrong order."
    )
    assert_close(actual, expected, TOL_EXTRACT_DIFF, what)


def lower_triangle(matrix):
    """Values below the diagonal of a square matrix, as a vector."""

    matrix = np.asarray(matrix)
    return matrix[np.tril_indices(matrix.shape[0], k=-1)]


def spose_similarity_by_definition(embedding):
    """Slow reference SPoSE similarity: the definition written out with three loops.

    S[i, j] is the mean over all k not in {i, j} of
        exp(e_i.e_j) / (exp(e_i.e_j) + exp(e_i.e_k) + exp(e_j.e_k))
    and S[i, i] = 1. As in embedding2sim_stable.m, the largest of the three
    dot products is subtracted before exp. This does not change the
    probability, but exp cannot overflow. All pairs (i, j) are computed, not
    only i < j, so that the reference does not assume symmetry.
    """

    embedding = np.asarray(embedding, dtype=float)
    n = embedding.shape[0]
    dots = embedding @ embedding.T
    similarity = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            probabilities = []
            for k in range(n):
                if k == i or k == j:
                    continue
                three = [dots[i, j], dots[i, k], dots[j, k]]
                largest = max(three)
                e_ij, e_ik, e_jk = [math.exp(v - largest) for v in three]
                probabilities.append(e_ij / (e_ij + e_ik + e_jk))
            similarity[i, j] = sum(probabilities) / len(probabilities)
    return similarity


def features_for(ref, name):
    """Features in reference_data.mat that belong to the network of a shipped model."""

    return ref[FEATURES_KEY[MODELS[name]["network"]]]


def python_env():
    """Environment for new Python processes: use the dimpred package of this repository."""

    env = dict(os.environ)
    paths = [REPO] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


# Code for "python -c" that makes torch and open_clip not importable (as if
# they were not installed) and then runs the command line tool. Setting a
# module to None in sys.modules makes every later import of it fail.
_WITHOUT_TORCH = (
    "import runpy, sys\n"
    "for name in ('torch', 'torchvision', 'open_clip'):\n"
    "    sys.modules[name] = None\n"
    "sys.argv = ['dimpred'] + sys.argv[1:]\n"
    "runpy.run_module('dimpred', run_name='__main__', alter_sys=True)\n"
)


def run_dimpred(args, cwd, without_torch=False, python_options=(), timeout=900):
    """Run the command line tool (python -m dimpred ARGS) in a new process.

    Returns the subprocess.CompletedProcess (returncode, stdout, stderr).
    With without_torch=True, torch and open_clip cannot be imported.
    python_options are given to Python before -m, e.g. ["-X", "importtime"].
    """

    args = [str(a) for a in args]
    if without_torch:
        cmd = [sys.executable, *python_options, "-c", _WITHOUT_TORCH] + args
    else:
        cmd = [sys.executable, *python_options, "-m", "dimpred"] + args
    return subprocess.run(cmd, cwd=str(cwd), env=python_env(), capture_output=True, text=True, timeout=timeout)


def imported_packages(importtime_output):
    """Top-level names of all modules imported by a process started with python -X importtime.

    With -X importtime, Python writes one line per imported module to stderr,
    e.g. "import time:       345 |       1210 |   torch._C". The module name is
    the last field.
    """

    names = set()
    for line in importtime_output.splitlines():
        if line.startswith("import time:") and "|" in line:
            module = line.rsplit("|", 1)[1].strip()
            if module and module != "imported package":  # skip the header line
                names.add(module.split(".")[0])
    return names


def output_of(result):
    """stdout and stderr of a finished process, for error messages."""

    return f"--- stdout ---\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-3000:]}"
