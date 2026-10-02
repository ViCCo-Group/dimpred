"""
Tests of wrong input to dimpred.extract_features (fast, no torch needed).

The image files are checked before the network is loaded, so a wrong path
gives an error at once and not after loading the network, which takes
several seconds (or a download the first time). To show this, torch and
open_clip cannot be imported in these tests: an implementation that loads
the network first fails with ImportError instead of the expected error.
The same checks are in the MATLAB tests (test_dimpred_extract_features_errors.m),
where they happen before Python is started.

Hebartlab, 2026/09/30

See also: test_extract_features.py, test_find_images.py
"""

import sys

import pytest

import dimpred
from helpers import IMAGES

# History:
# 2026/09/30: new file; the tests of a missing file and of a folder moved here
#   from test_extract_features.py, so that they run without torch


@pytest.fixture
def without_torch(monkeypatch):
    """torch and open_clip cannot be imported during the test (as if they were not installed)."""

    for name in ("torch", "torchvision", "open_clip"):
        monkeypatch.setitem(sys.modules, name, None)  # restored after the test


def test_missing_file_gives_file_not_found_error(cc0_paths, tmp_path, without_torch):
    with pytest.raises(FileNotFoundError):
        dimpred.extract_features([cc0_paths[0], str(tmp_path / "no_such_image.jpg")])


def test_error_message_names_the_missing_file(cc0_paths, tmp_path, without_torch):
    with pytest.raises(FileNotFoundError) as error:
        dimpred.extract_features([cc0_paths[0], str(tmp_path / "no_such_image.jpg")])
    assert "no_such_image.jpg" in str(error.value), f"the message should name the file: {str(error.value)!r}"


def test_missing_single_path_gives_file_not_found_error(tmp_path, without_torch):
    with pytest.raises(FileNotFoundError):
        dimpred.extract_features(str(tmp_path / "no_such_image.jpg"))


@pytest.mark.parametrize("given_as", ["path", "list"])
def test_folder_gives_error_that_points_to_find_images(without_torch, given_as):
    # Folders have to be listed with find_images first, so that the order of
    # the rows is always the order of the given files. Letting PIL fail on
    # the folder (IsADirectoryError) is not enough, the message has to say
    # what to do.
    images = IMAGES if given_as == "path" else [IMAGES]
    with pytest.raises((ValueError, FileNotFoundError)) as error:
        dimpred.extract_features(images)
    assert "find_images" in str(error.value), f"the message should point to find_images: {str(error.value)!r}"


def test_without_torch_existing_images_give_import_error(cc0_paths, without_torch):
    # the command line tool turns this into a message on how to install torch
    with pytest.raises(ImportError):
        dimpred.extract_features(cc0_paths)
