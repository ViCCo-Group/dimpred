"""
Tests of dimpred.find_images.

find_images returns the image files directly in a folder (not in subfolders),
with the extensions .jpg .jpeg .png .bmp .tif .tiff .webp in any upper or
lower case, without hidden files, as full paths, sorted by file name with a
plain string sort (the same order as sort() in MATLAB). The tests create
empty files in a temporary folder, because only the names matter here.

Martin Hebart, 2026/09/30

See also: test_extract_features.py
"""

import os

import pytest

import dimpred
from helpers import IMAGES

# History:
# 2026/09/30: a file instead of a folder has to give an error
# 2026/09/30: written together with the tests, before the package code


def make_files(folder, names):
    for name in names:
        (folder / name).write_bytes(b"")


def found_names(folder):
    return [os.path.basename(f) for f in dimpred.find_images(str(folder))]


def test_finds_all_image_extensions(tmp_path):
    names = ["a.jpg", "b.jpeg", "c.png", "d.bmp", "e.tif", "f.tiff", "g.webp"]
    make_files(tmp_path, names)
    assert found_names(tmp_path) == names


def test_extensions_can_be_upper_or_lower_case(tmp_path):
    names = ["a.JPG", "b.Jpeg", "c.PNG", "d.BMP", "e.Tif", "f.TIFF", "g.WebP"]
    make_files(tmp_path, names)
    assert found_names(tmp_path) == names


def test_other_files_are_left_out(tmp_path):
    make_files(tmp_path, ["image.jpg", "notes.txt", "features.mat", "data.npy", "README", "jpg", "image.jpg.txt"])
    assert found_names(tmp_path) == ["image.jpg"]


def test_hidden_files_are_left_out(tmp_path):
    # e.g. the "._" files that macOS writes on external drives
    make_files(tmp_path, ["photo.jpg", ".hidden.jpg", "._photo.jpg"])
    assert found_names(tmp_path) == ["photo.jpg"]


def test_subfolders_are_not_searched(tmp_path):
    make_files(tmp_path, ["top.jpg"])
    os.makedirs(tmp_path / "sub")
    make_files(tmp_path / "sub", ["inside.jpg"])
    os.makedirs(tmp_path / "album.jpg")  # a folder, even if its name looks like an image
    assert found_names(tmp_path) == ["top.jpg"]


def test_files_are_sorted_by_plain_string_sort(tmp_path):
    # upper case before lower case, "1" < "10" < "2" < "_", no natural sorting
    make_files(tmp_path, ["camera2.jpg", "camera_lens.png", "apple.jpg", "camera10.jpg", "Zebra.jpg", "camera1.jpg"])
    assert found_names(tmp_path) == ["Zebra.jpg", "apple.jpg", "camera1.jpg", "camera10.jpg", "camera2.jpg",
                                     "camera_lens.png"]


def test_returns_a_list_of_full_paths(tmp_path):
    make_files(tmp_path, ["b.png", "a.jpg"])
    files = dimpred.find_images(str(tmp_path))
    assert isinstance(files, list), f"find_images returned a {type(files)}, not a list"
    assert all(isinstance(f, str) for f in files), "all entries should be str"
    assert all(os.path.isabs(f) for f in files), f"not all paths are absolute: {files}"
    assert os.path.samefile(files[0], tmp_path / "a.jpg"), f"first file is {files[0]!r}"
    assert os.path.samefile(files[1], tmp_path / "b.png"), f"second file is {files[1]!r}"


def test_relative_folder_gives_full_paths(tmp_path, monkeypatch):
    os.makedirs(tmp_path / "images")
    make_files(tmp_path / "images", ["a.jpg"])
    monkeypatch.chdir(tmp_path)
    files = dimpred.find_images("images")
    assert len(files) == 1, f"expected one file, got {files}"
    assert os.path.isabs(files[0]), f"{files[0]!r} is not an absolute path"
    assert os.path.samefile(files[0], tmp_path / "images" / "a.jpg"), f"{files[0]!r} is not images/a.jpg"


def test_finds_the_test_images(ref):
    assert found_names(IMAGES) == sorted(ref["cc0_files"])


def test_missing_folder_gives_error(tmp_path):
    with pytest.raises(ValueError):
        dimpred.find_images(str(tmp_path / "does_not_exist"))


def test_file_instead_of_folder_gives_error(tmp_path):
    make_files(tmp_path, ["a.jpg"])
    with pytest.raises(ValueError):
        dimpred.find_images(str(tmp_path / "a.jpg"))


def test_empty_folder_gives_error(tmp_path):
    with pytest.raises(ValueError):
        dimpred.find_images(str(tmp_path))


def test_folder_without_images_gives_error(tmp_path):
    make_files(tmp_path, ["notes.txt", ".hidden.jpg"])
    with pytest.raises(ValueError):
        dimpred.find_images(str(tmp_path))
