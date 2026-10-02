"""
Tests of the MATLAB version (slow, need MATLAB).

The MATLAB functions (folder matlab/) have their own tests in tests/matlab,
which compare against the same fixture numbers as the Python tests. Here we
run these tests with "matlab -batch" and, in addition, run the MATLAB
functions on the fixtures and compare the results with the fixtures and with
the Python functions, so that both versions give the same numbers.

MATLAB is searched in the environment variable DIMPRED_MATLAB (full path of
the matlab program), then in /Applications/MATLAB_*/bin/matlab, then on the
system path. If it is not found, all tests here are skipped. Starting MATLAB
takes about 20 s.

Hebartlab, 2026/09/30

See also: tests/matlab
"""

import glob
import os
import shutil
import subprocess
import sys

import pytest

import dimpred
from helpers import (DEFAULT_MODEL, IMAGES, MATLAB_CODE, MATLAB_TESTS, MODEL_NAMES, REFERENCE_FILE, TOL_MODEL,
                     assert_close, assert_features_match, features_for, load_mat, output_of, python_env)

# History:
# 2026/10/02: AligNet features for the new default model alignet_siglip2b_66d_ridge
# 2026/09/30: the extraction order test uses a second unsorted order
# 2026/09/30: written together with the tests, before the package code

pytestmark = pytest.mark.slow


def find_matlab():
    if os.environ.get("DIMPRED_MATLAB"):
        return os.environ["DIMPRED_MATLAB"]  # if this is wrong, the tests fail and say so
    installed = sorted(glob.glob("/Applications/MATLAB_*/bin/matlab"))
    if installed:
        return installed[-1]  # newest version
    return shutil.which("matlab")


MATLAB = find_matlab()
if MATLAB is None:
    pytest.skip("MATLAB not found (set DIMPRED_MATLAB to the matlab program to run these tests)",
                allow_module_level=True)


def run_matlab(script, folder, timeout=1800):
    """Write a MATLAB script to folder and run it with matlab -batch."""

    fname = os.path.join(folder, "dimpred_test_script.m")
    with open(fname, "w") as f:
        f.write(script)
    env = python_env()  # for dimpred_extract_features, which calls python -m dimpred
    env["DIMPRED_PYTHON"] = sys.executable
    return subprocess.run([MATLAB, "-batch", f"run('{fname}')"], cwd=folder, env=env,
                          capture_output=True, text=True, timeout=timeout)


def assert_matlab_success(result):
    assert result.returncode == 0, f"MATLAB failed with exit code {result.returncode}\n{output_of(result)}"


# --- the MATLAB test suite

def test_matlab_test_suite_passes(tmp_path):
    # DIMPRED_PYTHON is set to this Python (see run_matlab), so the MATLAB
    # tests of the feature extraction run as well
    test_files = sorted(glob.glob(os.path.join(MATLAB_TESTS, "test*.m")))
    assert test_files, f"no MATLAB tests (test*.m) found in {MATLAB_TESTS}"
    if os.path.exists(os.path.join(MATLAB_TESTS, "run_dimpred_tests.m")):
        # the runner of the MATLAB tests, it gives an error if a test fails
        script = f"cd('{MATLAB_TESTS}');\nrun_dimpred_tests;\n"
    else:
        script = (
            f"addpath('{MATLAB_CODE}');\n"
            f"results = runtests('{MATLAB_TESTS}');\n"
            "disp(table(results));\n"
            "assert(~isempty(results), 'no MATLAB tests were run');\n"
            "n_failed = sum([results.Failed]);\n"
            "assert(n_failed == 0, '%d of %d MATLAB tests failed', n_failed, numel(results));\n"
        )
    assert_matlab_success(run_matlab(script, str(tmp_path)))


# --- MATLAB functions on the fixtures, compared with the fixtures and with Python

@pytest.fixture(scope="module")
def matlab_results(tmp_path_factory):
    """Run the MATLAB functions once on the fixtures and return what they computed."""

    folder = str(tmp_path_factory.mktemp("matlab"))
    out_file = os.path.join(folder, "matlab_results.mat")
    script = f"""
addpath('{MATLAB_CODE}');
ref = load('{REFERENCE_FILE}');
rows = strcmp(ref.image_set, '48nonref');
out = struct();
out.names = dimpred_list_models();
default_model = dimpred_load_model();
out.default_name = default_model.info.name;
empty_model = dimpred_load_model([]);
out.empty_name = empty_model.info.name;
for i = 1:numel(out.names)
    name = out.names{{i}};
    model = dimpred_load_model(name);
    if strcmp(model.info.network, 'RN50x64')
        features = ref.features_rn50x64;
    elseif strcmp(model.info.network, 'AligNet SigLIP2-B')
        features = ref.features_alignet;
    else
        features = ref.features_vitb32;
    end
    out.(['weights_' name]) = model.weights;
    out.(['embedding_' name]) = dimpred_predict(features, model);
    out.(['similarity_' name]) = dimpred_similarity(out.(['embedding_' name])(rows, :));
end
out.embedding_by_name = dimpred_predict(ref.features_rn50x64, 'rn50x64_49d_ridge');
out.embedding_default = dimpred_predict(ref.features_alignet);
out.similarity_dot = dimpred_similarity(ref.expected_rn50x64_49d_ridge(1:10, :), 'dot');
out.found_images = dimpred_find_images('{IMAGES}');
save('{out_file}', '-struct', 'out');
"""
    result = run_matlab(script, folder)
    assert_matlab_success(result)
    return load_mat(out_file)


def test_matlab_lists_the_same_models(matlab_results):
    assert matlab_results["names"] == MODEL_NAMES, f"dimpred_list_models gives {matlab_results['names']}"


def test_matlab_default_model(matlab_results):
    assert matlab_results["default_name"] == DEFAULT_MODEL, f"without input: {matlab_results['default_name']!r}"
    assert matlab_results["empty_name"] == DEFAULT_MODEL, f"with []: {matlab_results['empty_name']!r}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_matlab_loads_the_same_weights(matlab_results, name):
    assert_close(matlab_results["weights_" + name], dimpred.load_model(name)["weights"], 0,
                 f"{name}: weights loaded in MATLAB vs Python")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_matlab_predictions_equal_the_expected_predictions(matlab_results, ref, name):
    assert_close(matlab_results["embedding_" + name], ref["expected_" + name], TOL_MODEL,
                 f"{name}: MATLAB predictions vs expected_{name}")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_matlab_and_python_predictions_are_the_same(matlab_results, ref, name):
    assert_close(matlab_results["embedding_" + name], dimpred.predict(features_for(ref, name), name), 1e-12,
                 f"{name}: MATLAB vs Python predictions")


def test_matlab_model_given_by_name_gives_philipps_published_predictions(matlab_results, ref):
    assert_close(matlab_results["embedding_by_name"], ref["published_rn50x64_49d_ridge"], TOL_MODEL,
                 "MATLAB predictions (model by name) vs Philipp's published predictions")


def test_matlab_uses_the_default_model_without_model(matlab_results, ref):
    assert_close(matlab_results["embedding_default"], ref["expected_" + DEFAULT_MODEL], TOL_MODEL,
                 "MATLAB predictions without model vs default model")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_matlab_and_python_similarity_are_the_same(matlab_results, ref, name):
    rows = [i for i, s in enumerate(ref["image_set"]) if s == "48nonref"]
    python = dimpred.similarity(dimpred.predict(features_for(ref, name)[rows], name))
    assert_close(matlab_results["similarity_" + name], python, 1e-12, f"{name}: MATLAB vs Python similarity")


def test_matlab_dot_similarity(matlab_results, ref):
    embedding = ref["expected_rn50x64_49d_ridge"][:10]
    assert_close(matlab_results["similarity_dot"], embedding @ embedding.T, 1e-12, "MATLAB dot similarity")


def test_matlab_finds_the_same_images_in_the_same_order(matlab_results):
    matlab = matlab_results["found_images"]
    python = dimpred.find_images(IMAGES)
    names = [os.path.basename(f) for f in matlab]
    assert names == [os.path.basename(f) for f in python], f"dimpred_find_images gives {names}"
    assert all(os.path.isabs(f) for f in matlab), f"MATLAB should return full paths: {matlab}"


# --- feature extraction from MATLAB (calls the Python command line tool)

def test_matlab_extract_features_keeps_the_given_order(tmp_path, ref, cc0_paths_reordered, open_clip_available):
    paths, rows = cc0_paths_reordered  # neither sorted nor the order of cc0_files
    out_file = str(tmp_path / "matlab_features.mat")
    images = ", ".join(f"'{p}'" for p in paths)
    script = f"""
addpath('{MATLAB_CODE}');
cfg = struct();
cfg.python = '{sys.executable}';
[features, files] = dimpred_extract_features({{{images}}}, 'vitb32_66d_elastic', cfg);
save('{out_file}', 'features', 'files');
"""
    assert_matlab_success(run_matlab(script, str(tmp_path)))
    out = load_mat(out_file)
    names = [os.path.basename(f) for f in out["files"]]
    assert names == [os.path.basename(p) for p in paths], f"files returned by MATLAB: {names}"
    assert_features_match(out["features"], ref["cc0_features_vitb32"][rows], "features extracted from MATLAB")
