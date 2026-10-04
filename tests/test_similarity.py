"""
Tests of dimpred.similarity.

SPoSE similarity of objects i and j is the probability that i and j are
picked as the most similar pair in an odd-one-out triplet with a random third
object k:
    S[i, j] = mean over all k not in {i, j} of
              exp(e_i.e_j) / (exp(e_i.e_j) + exp(e_i.e_k) + exp(e_j.e_k))
with S[i, i] = 1. We check the values against small examples computed by
hand, against a slow reference that follows this definition with three loops
(as embedding2sim_stable.m), and against properties that follow from the
definition (symmetry, mean of 1/3, permutations). Large embedding values must
not give nan or inf. Users compute the similarity of hundreds or thousands
of images, so it also has to be fast (slow test).

Hebartlab, 2026/09/30

See also: test_validation_human.py
"""

import math
import time

import numpy as np
import pytest

import dimpred
from helpers import assert_close, spose_similarity_by_definition

# History:
# 2026/10/04: tests of the close-pair term (features, model)
# 2026/09/30: nan and inf give an error, as in MATLAB
# 2026/09/30: test of large dot products within a small range, for the
#   version of similarity.py that computes exp once
# 2026/09/30: speed compared with plain Python loops; the slow reference
#   moved to helpers.py (test_fixtures.py uses it as well)
# 2026/09/30: written together with the tests, before the package code


def random_embedding(n, d, seed, scale=1.0):
    """Non-negative random embedding, like SPoSE dimensions."""

    return scale * np.random.default_rng(seed).random((n, d))


# --- values

def test_three_objects_computed_by_hand():
    # dot products: e0.e1 = 0, e0.e2 = 1, e1.e2 = 2. With three objects, the
    # third object of each pair is the remaining one.
    embedding = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    e = math.e
    s01 = 1 / (1 + e + e ** 2)       # exp(0) / (exp(0) + exp(1) + exp(2))
    s02 = e / (e + 1 + e ** 2)       # exp(1) / (exp(1) + exp(0) + exp(2))
    s12 = e ** 2 / (e ** 2 + 1 + e)  # exp(2) / (exp(2) + exp(0) + exp(1))
    expected = np.array([[1, s01, s02],
                         [s01, 1, s12],
                         [s02, s12, 1]])
    assert_close(dimpred.similarity(embedding), expected, 1e-14, "3 objects")


def test_four_objects_computed_by_hand():
    # with four objects, each pair has two possible third objects and the
    # similarity is the mean of the two probabilities (divided by n - 2 = 2)
    embedding = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    # dot products: e0.e2 = e1.e2 = 1, all others 0
    e = math.e
    s01 = (1 / (1 + e + e) + 1 / (1 + 1 + 1)) / 2  # third object 2, third object 3
    s02 = (e / (e + 1 + e) + e / (e + 1 + 1)) / 2  # third object 1, third object 3
    s12 = s02                                      # the same dot products as for 0 and 2
    s03 = (1 / (1 + 1 + 1) + 1 / (1 + e + 1)) / 2  # third object 1, third object 2
    s13 = s03                                      # the same dot products as for 0 and 3
    s23 = (1 / (1 + e + 1) + 1 / (1 + e + 1)) / 2  # third object 0, third object 1
    expected = np.array([[1, s01, s02, s03],
                         [s01, 1, s12, s13],
                         [s02, s12, 1, s23],
                         [s03, s13, s23, 1]])
    assert_close(dimpred.similarity(embedding), expected, 1e-14, "4 objects")


def test_equals_slow_reference_for_random_embeddings():
    embedding = random_embedding(15, 6, seed=1)
    assert_close(dimpred.similarity(embedding), spose_similarity_by_definition(embedding), 1e-12,
                 "similarity vs slow reference")


def test_equals_slow_reference_for_predicted_embeddings(ref):
    embedding = ref["expected_rn50x64_49d_ridge"][:20]
    assert_close(dimpred.similarity(embedding), spose_similarity_by_definition(embedding), 1e-12,
                 "similarity of predicted embeddings vs slow reference")


@pytest.mark.parametrize("sign", ["positive", "mixed"])
def test_large_values_give_no_nan_or_inf(sign):
    # dot products of several thousand: exp() alone would overflow to inf and
    # give inf / inf = nan
    rng = np.random.default_rng(4)
    embedding = 30 * (rng.random((10, 5)) if sign == "positive" else rng.standard_normal((10, 5)))
    assert (embedding @ embedding.T).max() > 1000, "test data should have large dot products"
    similarity = dimpred.similarity(embedding)
    assert np.all(np.isfinite(similarity)), "similarity contains nan or inf for large values"
    assert_close(similarity, spose_similarity_by_definition(embedding), 1e-12, "similarity for large values")


def test_large_dot_products_within_a_small_range_give_no_nan_or_inf():
    # all dot products are about 1000, but they differ by much less than 700.
    # similarity.py then computes exp once for all pairs, so this case needs
    # its own protection against overflow (the test above covers the other).
    embedding = 14 + np.random.default_rng(19).random((10, 5))
    dots = embedding @ embedding.T
    assert dots.min() > 710 and dots.max() - dots.min() < 700, "test data should have large, similar dot products"
    similarity = dimpred.similarity(embedding)
    assert np.all(np.isfinite(similarity)), "similarity contains nan or inf for large, similar dot products"
    assert_close(similarity, spose_similarity_by_definition(embedding), 1e-12, "similarity for large, similar values")


def test_identical_objects_all_have_similarity_one_third():
    # all dot products are equal, so each of the three pairs of a triplet is
    # picked with probability 1/3
    similarity = dimpred.similarity(np.ones((5, 3)))
    off_diagonal = ~np.eye(5, dtype=bool)
    assert_close(similarity[off_diagonal], np.full(20, 1 / 3), 1e-15, "identical objects")


# --- properties

def test_similarity_is_symmetric():
    similarity = dimpred.similarity(random_embedding(12, 4, seed=5))
    assert_close(similarity, similarity.T, 1e-14, "similarity vs its transpose")


def test_diagonal_is_one():
    similarity = dimpred.similarity(random_embedding(12, 4, seed=6))
    np.testing.assert_array_equal(np.diag(similarity), np.ones(12), err_msg="diagonal should be 1")


def test_mean_of_off_diagonal_values_is_one_third():
    # In each triplet exactly one of the three pairs is picked, so the three
    # probabilities of a triplet add up to 1. Summing over all pairs and
    # third objects gives a mean of exactly 1/3 for any embedding.
    n = 20
    similarity = dimpred.similarity(random_embedding(n, 8, seed=7, scale=2))
    mean = similarity[~np.eye(n, dtype=bool)].mean()
    assert abs(mean - 1 / 3) < 1e-12, f"mean of the off-diagonal values is {mean!r}, expected 1/3"


def test_values_are_between_0_and_1():
    similarity = dimpred.similarity(random_embedding(12, 4, seed=8, scale=3))
    assert similarity.min() > 0, f"smallest value is {similarity.min()}"
    assert similarity.max() <= 1, f"largest value is {similarity.max()}"


def test_permuting_objects_permutes_the_similarity_matrix():
    embedding = random_embedding(12, 4, seed=9)
    order = np.random.default_rng(10).permutation(12)
    expected = dimpred.similarity(embedding)[np.ix_(order, order)]
    assert_close(dimpred.similarity(embedding[order]), expected, 1e-14, "similarity of permuted objects")


def test_output_is_a_float_matrix_with_one_row_and_column_per_object():
    similarity = dimpred.similarity(random_embedding(7, 3, seed=11))
    assert isinstance(similarity, np.ndarray), f"similarity returned a {type(similarity)}, not a numpy array"
    assert similarity.shape == (7, 7), f"shape is {similarity.shape}"
    assert similarity.dtype == np.float64, f"dtype is {similarity.dtype}"


def test_input_is_not_changed():
    embedding = random_embedding(8, 3, seed=12)
    before = embedding.copy()
    dimpred.similarity(embedding)
    np.testing.assert_array_equal(embedding, before, err_msg="similarity changed its input")


# --- methods

def test_default_method_is_spose():
    embedding = random_embedding(8, 3, seed=13)
    assert_close(dimpred.similarity(embedding), dimpred.similarity(embedding, "spose"), 0, "default vs spose")


def test_dot_method_gives_dot_products():
    embedding = random_embedding(8, 3, seed=14)
    assert_close(dimpred.similarity(embedding, "dot"), embedding @ embedding.T, 1e-14, "dot method")


def test_dot_method_can_be_given_as_keyword():
    embedding = random_embedding(8, 3, seed=15)
    assert_close(dimpred.similarity(embedding, method="dot"), embedding @ embedding.T, 1e-14, "method='dot'")


# --- errors

@pytest.mark.parametrize("n", [1, 2])
def test_spose_needs_at_least_three_objects(n):
    with pytest.raises(ValueError):
        dimpred.similarity(random_embedding(n, 3, seed=16))


def test_unknown_method_gives_error():
    with pytest.raises(ValueError):
        dimpred.similarity(random_embedding(5, 3, seed=17), method="cosine")


@pytest.mark.parametrize("method", ["spose", "dot"])
def test_one_dimensional_input_gives_error(method):
    with pytest.raises(ValueError):
        dimpred.similarity(np.ones(5), method=method)


@pytest.mark.parametrize("method", ["spose", "dot"])
def test_three_dimensional_input_gives_error(method):
    with pytest.raises(ValueError):
        dimpred.similarity(np.ones((4, 3, 2)), method=method)


@pytest.mark.parametrize("method", ["spose", "dot"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nan_or_inf_in_embedding_gives_error(method, value):
    # with spose, one such object would make every value nan, because it is
    # the third object of every pair
    embedding = random_embedding(6, 3, seed=19)
    embedding[4, 1] = value
    with pytest.raises(ValueError, match="nan or inf"):
        dimpred.similarity(embedding, method=method)


# --- speed

def spose_similarity_python_loops(embedding):
    """SPoSE similarity with plain Python loops over the pairs i < j and all k.

    This is the most direct way to write the definition in Python. It serves
    as the yardstick for the speed of dimpred.similarity.
    """

    n = len(embedding)
    dots = (np.asarray(embedding) @ np.asarray(embedding).T).tolist()
    similarity = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            total = 0.0
            for k in range(n):
                if k == i or k == j:
                    continue
                largest = max(dots[i][j], dots[i][k], dots[j][k])
                e_ij = math.exp(dots[i][j] - largest)
                total += e_ij / (e_ij + math.exp(dots[i][k] - largest) + math.exp(dots[j][k] - largest))
            similarity[i, j] = similarity[j, i] = total / (n - 2)
    return similarity


def seconds(function, *args):
    start = time.perf_counter()
    function(*args)
    return time.perf_counter() - start


@pytest.mark.slow
def test_similarity_is_much_faster_than_plain_python_loops():
    # We compare with plain Python loops on the same computer instead of
    # using a fixed time limit, because the speed of the computer (and how
    # busy it is) should not decide the test. similarity.py, which loops over
    # one object and computes the other two with numpy, is about 100 times
    # faster than the plain loops for 200 objects.
    embedding = random_embedding(200, 66, seed=18)
    loops = seconds(spose_similarity_python_loops, embedding)
    fastest = min(seconds(dimpred.similarity, embedding) for _ in range(3))
    assert 3 * fastest < loops, (
        f"similarity of 200 objects took {fastest:.2f} s, plain Python loops {loops:.2f} s. It should be at "
        f"least 3 times faster than the loops.")


# --- the close-pair term (features)

def spose_from_dots(dots):
    """Slow reference: the SPoSE similarity written out for a matrix of dot products."""

    n = len(dots)
    S = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                p = [np.exp(dots[i, j]) / (np.exp(dots[i, j]) + np.exp(dots[i, k]) + np.exp(dots[j, k]))
                     for k in range(n) if k not in (i, j)]
                S[i, j] = np.mean(p)
    return S


def features_with_close_pairs(n, n_features, seed):
    """Random features where objects 0 and 1, and 2 and 3, are very close (cosine about 0.99)."""

    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, n_features))
    features[1] = features[0] + 0.1 * rng.standard_normal(n_features)
    features[3] = features[2] + 0.1 * rng.standard_normal(n_features)
    return features


def close_pair_term(features, weight, threshold):
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    return weight * np.maximum(0, unit @ unit.T - threshold)


def test_close_pairs_add_the_network_similarity_of_the_close_pairs():
    model = dimpred.load_model()  # the default model has the close-pair settings
    embedding = random_embedding(6, 66, 11, scale=0.5)
    features = features_with_close_pairs(6, model["weights"].shape[0], 12)
    D = embedding @ embedding.T + close_pair_term(features, model["close_pairs_weight"], model["close_pairs_threshold"])
    assert_close(dimpred.similarity(embedding, "dot", features=features), D, 1e-12, "dot with the close pairs")
    assert_close(dimpred.similarity(embedding, features=features), spose_from_dots(D), 1e-12, "spose with the close pairs")
    changed = ~np.isclose(D, embedding @ embedding.T)
    np.fill_diagonal(changed, False)
    assert changed[0, 1] and changed[2, 3] and changed.sum() == 4, "only the two close pairs should change"


def test_features_without_close_pairs_change_nothing():
    # random features in 768 dimensions have cosines near 0, far below the threshold
    embedding = random_embedding(8, 66, 13)
    features = np.random.default_rng(14).standard_normal((8, 768))
    S = dimpred.similarity(embedding, features=features)
    off = ~np.eye(8, dtype=bool)
    assert_close(S[off], dimpred.similarity(embedding)[off], 1e-12, "similarity with features without close pairs")


def test_close_pairs_use_the_settings_of_the_given_model():
    model = dict(dimpred.load_model(), close_pairs_weight=3.0, close_pairs_threshold=0.2)
    embedding = random_embedding(5, 66, 15)
    features = features_with_close_pairs(5, 768, 16)
    expected = embedding @ embedding.T + close_pair_term(features, 3.0, 0.2)
    assert_close(dimpred.similarity(embedding, "dot", features=features, model=model), expected, 1e-12,
                 "dot with the close-pair settings of the given model")


def test_input_is_not_changed_with_features():
    embedding = random_embedding(5, 66, 17)
    features = features_with_close_pairs(5, 768, 18)
    copies = embedding.copy(), features.copy()
    dimpred.similarity(embedding, features=features)
    assert np.array_equal(embedding, copies[0]) and np.array_equal(features, copies[1]), "input changed"


def test_model_without_close_pair_settings_gives_error():
    with pytest.raises(ValueError, match="no close-pair settings"):
        dimpred.similarity(random_embedding(4, 66, 19), features=np.ones((4, 1024)), model="rn50x64_66d_ridge")


@pytest.mark.parametrize("shape, message", [((4, 768), "one row per object"), ((5, 512), "needs 768 features")])
def test_features_of_the_wrong_shape_give_error(shape, message):
    with pytest.raises(ValueError, match=message):
        dimpred.similarity(random_embedding(5, 66, 20), features=np.ones(shape))


@pytest.mark.parametrize("value", [0.0, np.nan])
def test_features_with_zero_rows_or_nan_give_error(value):
    features = features_with_close_pairs(5, 768, 21)
    features[2] = value
    with pytest.raises(ValueError, match="cosines are not defined"):
        dimpred.similarity(random_embedding(5, 66, 22), features=features)
