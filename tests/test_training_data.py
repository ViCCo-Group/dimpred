"""
Tests of the training data in training/data.

build_models.py pairs the 1854 reference images with the rows of the SPoSE
embeddings by the order of reference_image_names.txt. This order is the
order of the THINGS concepts, which differs from a plain sort of the file
names for 21 concepts (e.g. camera_lens comes before camera1 and camera2).
If the list is sorted, or built from a folder listing, these images are
silently paired with the wrong dimension values. We pin the order here, and
check that the embeddings and labels fit the list.

These tests need no extra packages and always run.

Hebartlab, 2026/09/30

See also: ../training/build_models.py, test_model_files.py
"""

import os

import numpy as np
import pytest

from helpers import TRAINING_DATA, read_lines

# History:
# 2026/09/30: written after the review of the first tests (order of the
#   reference images was not checked)


@pytest.fixture(scope="module")
def names():
    return read_lines(os.path.join(TRAINING_DATA, "reference_image_names.txt"))


def test_there_are_1854_reference_images(names):
    assert len(names) == 1854, f"reference_image_names.txt has {len(names)} names, expected 1854"


def test_reference_image_names_are_unique(names):
    assert len(set(names)) == len(names), "reference_image_names.txt lists some images more than once"


def test_reference_images_are_in_things_order_not_sorted(names):
    # a plain sort moves 21 names; with 0 the list has been sorted
    n_moved = sum(a != b for a, b in zip(names, sorted(names)))
    assert n_moved == 21, (
        f"{n_moved} names are at a different place after sorting, expected 21. The list has to keep the "
        f"order of the THINGS concepts (the rows of the embeddings), it must not be sorted.")


def test_camera_lens_comes_before_camera1_and_camera2(names):
    # the first place where the THINGS order and a plain sort differ
    assert names[246:249] == ["camera_lens_01b.jpg", "camera1_01b.jpg", "camera2_01b.jpg"], (
        f"names 247-249 are {names[246:249]}, expected camera_lens, camera1, camera2 (THINGS order)")


def test_first_and_last_reference_images(names):
    assert names[:2] == ["aardvark_01b.jpg", "abacus_01b.jpg"], f"first names are {names[:2]}"
    assert names[-2:] == ["zipper_01b.jpg", "zucchini_01b.jpg"], f"last names are {names[-2:]}"


@pytest.mark.parametrize("n_dims", [49, 66])
def test_embedding_has_one_row_per_reference_image(n_dims):
    embedding = np.loadtxt(os.path.join(TRAINING_DATA, f"spose_embedding_{n_dims}d.txt"))
    assert embedding.shape == (1854, n_dims), (
        f"spose_embedding_{n_dims}d.txt has shape {embedding.shape}, expected (1854, {n_dims})")


@pytest.mark.parametrize("n_dims", [49, 66])
def test_one_label_per_dimension(n_dims):
    labels = read_lines(os.path.join(TRAINING_DATA, f"labels_{n_dims}d.txt"))
    assert len(labels) == n_dims, f"labels_{n_dims}d.txt has {len(labels)} labels, expected {n_dims}"
    assert len(set(labels)) == n_dims, f"labels_{n_dims}d.txt has labels that appear more than once"
