"""
Validation against human behavior: predicted vs human similarity.

For the 48nonref images we have human similarity from the odd-one-out task.
For each shipped model, we predict the dimensions of these images, compute the
SPoSE similarity, and correlate it with the human similarity (Pearson r of
the values below the diagonal). The expected r of each model is stored in the
fixtures, e.g. 0.81 for the model of the DimPred paper. This tests predict and
similarity together on the task that DimPred is used for. That the
comparison with humans would find images in the wrong order is shown in
test_fixtures.py.

Hebartlab, 2026/09/30

See also: test_predict.py, test_similarity.py, test_fixtures.py
"""

import numpy as np
import pytest

import dimpred
from helpers import MODEL_NAMES, TOL_HUMAN_R, features_for, lower_triangle

# History:
# 2026/09/30: the threshold of the test that does not use the fixture values
#   is now 0.80 (0.6 also passed without target_mean); the test with shuffled
#   images moved to test_fixtures.py, since it does not test dimpred
# 2026/09/30: written together with the tests, before the package code


def rows_48nonref(ref):
    rows = [i for i, image_set in enumerate(ref["image_set"]) if image_set == "48nonref"]
    assert len(rows) == 48, f"found {len(rows)} rows of the 48nonref images in reference_data.mat, expected 48"
    return rows


def r_with_humans(similarity, ref):
    return np.corrcoef(lower_triangle(similarity), lower_triangle(ref["human_similarity_48nonref"]))[0, 1]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predicted_similarity_correlates_with_human_similarity_as_expected(ref, name):
    features = features_for(ref, name)[rows_48nonref(ref)]
    similarity = dimpred.similarity(dimpred.predict(features, name))
    r = r_with_humans(similarity, ref)
    expected = float(ref["human_r_48nonref"][name])
    assert abs(r - expected) <= TOL_HUMAN_R, (
        f"{name}: r with human similarity is {r:.4f}, expected {expected:.4f} +- {TOL_HUMAN_R}")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predicted_similarity_correlates_at_least_0_80_with_human_similarity(ref, name):
    # Independent of the fixture values: all models predict human similarity
    # at least about as well as the paper model (published: r = 0.81). This
    # fails without target_mean (r = 0.76 to 0.78), when predictions are
    # clipped before adding target_mean (0.74 to 0.76), and with wrong
    # feature statistics (0.65 to 0.75).
    features = features_for(ref, name)[rows_48nonref(ref)]
    r = r_with_humans(dimpred.similarity(dimpred.predict(features, name)), ref)
    assert r > 0.80, f"{name}: r with human similarity is only {r:.3f}, expected more than 0.80"
