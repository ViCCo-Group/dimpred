"""
dimpred: predict the SPoSE dimensions of human object representations for any image

DimPred predicts, for any image, its values on the SPoSE dimensions of human
mental object representations (49 dimensions in Hebart et al., 2020, or 66 in
Hebart et al., 2023). A network (the image encoder of CLIP) turns each image
into a feature vector, and one linear regression per dimension, trained on
the 1854 THINGS reference images, maps these features to dimension values.
From the predicted dimensions, we can then compute the predicted perceived
similarity between images.

Functions:
    list_models       names of the models that come with dimpred
    load_model        load a model (weights, feature mean and scale, dimension means, labels, info)
    find_images       image files in a folder, sorted by name
    extract_features  network features of images (needs torch and open_clip)
    predict           predicted dimension values from network features
    similarity        predicted similarity between images from their predicted dimensions

Example:
    import dimpred
    files = dimpred.find_images("my_images")
    features = dimpred.extract_features(files)  # default model: vitb32_66d_elastic
    embedding = dimpred.predict(features)       # n_images x 66
    S = dimpred.similarity(embedding)           # n_images x n_images

From the command line: python -m dimpred my_images --out predictions.csv
(python -m dimpred --help lists all options)

Reference: Kaniuth, P., Mahner, F. P., Perkuhn, J., & Hebart, M. N. (2025). A
high-throughput approach for the efficient prediction of perceived similarity
of natural objects. eLife 14:RP105394.

Martin Hebart, 2026/09/30
"""

from .list_models import list_models
from .load_model import load_model
from .find_images import find_images
from .extract_features import extract_features
from .predict import predict
from .similarity import similarity

# History:
# 2026/09/30: written for the first release of the package

# The model that is used when no model is given. load_model reads it each
# time it is called, so if you set dimpred.DEFAULT_MODEL to another model,
# all functions use the new default.
DEFAULT_MODEL = "vitb32_66d_elastic"
