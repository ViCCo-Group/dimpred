"""
pytest configuration and shared fixtures for the dimpred tests.

Run all tests from the repository folder with
    python -m pytest tests
and leave out the slow ones (feature extraction with torch, MATLAB, training
on the 1854 reference images, speed of the similarity) with
    python -m pytest tests -m "not slow"
The marker "slow" is registered in pyproject.toml.

The fixtures (tests/fixtures/reference_data.mat, tests/fixtures/images/) are
part of the repository. If they are missing, the tests that need them fail and
are not skipped, because without them we cannot check any numbers. Tests are
only skipped if an optional program or package is missing (torch, open_clip,
MATLAB, scikit-learn) or an optional large file that is not in the repository
(the features of the 1854 reference images for the training test, the
weights of the AligNet network: set DIMPRED_ALIGNET_WEIGHTS to
alignet_siglip2_b.safetensors, the tests never download it).

Hebartlab, 2026/09/30

See also: helpers.py
"""

import copy
import os
import sys

import pytest

# History:
# 2026/10/02: fixture alignet_available (AligNet weights and timm)
# 2026/09/30: the marker "slow" is only registered in pyproject.toml; second
#   order of the CC0 images (cc0_paths_reordered)
# 2026/09/30: written together with the tests, before the package code

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)

# Test the package in this repository, also if another version of dimpred is
# installed, and make helpers.py importable from the test files
sys.path.insert(0, TESTS)
sys.path.insert(0, REPO)

import helpers  # noqa: E402


@pytest.fixture(scope="session")
def ref_session():
    """Contents of tests/fixtures/reference_data.mat, loaded once."""

    if not os.path.exists(helpers.REFERENCE_FILE):
        pytest.fail(f"Missing test fixture {helpers.REFERENCE_FILE}. It belongs to the repository "
                    f"(created with tests/fixtures/make_fixtures.py).")
    return helpers.load_mat(helpers.REFERENCE_FILE)


@pytest.fixture
def ref(ref_session):
    """Contents of reference_data.mat, a fresh copy for each test.

    With a copy, a function that changes its input by mistake only breaks the
    test that checks this, and not all tests that come after it.
    """

    return copy.deepcopy(ref_session)


@pytest.fixture(scope="session")
def cc0_paths(ref_session):
    """Full paths of the three CC0 test images, in the order of cc0_files."""

    paths = [os.path.join(helpers.IMAGES, f) for f in ref_session["cc0_files"]]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        pytest.fail(f"Missing test images (they belong to the repository): {missing}")
    return paths


@pytest.fixture(scope="session")
def cc0_paths_reordered(cc0_paths):
    """The CC0 test images in the second unsorted order (helpers.CC0_REORDERED).

    Returns the paths and the rows of these images in the cc0 reference
    features. test_fixtures.py checks that this order is neither sorted nor
    the order of cc0_files.
    """

    rows = list(helpers.CC0_REORDERED)
    return [cc0_paths[i] for i in rows], rows


@pytest.fixture(scope="session")
def open_clip_available():
    """Skip the test if torch or open_clip are not installed (needed for feature extraction)."""

    pytest.importorskip("torch", reason="feature extraction needs torch")
    pytest.importorskip("open_clip", reason="feature extraction needs open_clip_torch")


@pytest.fixture(scope="session")
def alignet_available(open_clip_available):
    """Skip the test if the AligNet weights are not available without a download (see helpers.alignet_weights)."""

    pytest.importorskip("timm", reason="AligNet needs timm (comes with open_clip_torch)")
    pytest.importorskip("safetensors", reason="AligNet needs safetensors (comes with open_clip_torch)")
    if helpers.alignet_weights() is None:
        pytest.skip("AligNet weights not found (set DIMPRED_ALIGNET_WEIGHTS to alignet_siglip2_b.safetensors)")
