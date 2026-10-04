"""
dimpred: predict the SPoSE dimensions of human object representations for any image

DimPred predicts, for any image, its values on the SPoSE dimensions of human
mental object representations (49 dimensions in Hebart et al., 2020, or 66 in
Hebart et al., 2023). A network (by default the image encoder of AligNet
SigLIP2-B, Muttenthaler et al., 2025) turns each image into a feature vector,
and one regression per dimension, trained on the 1854 THINGS reference
images, maps these features to dimension values: a ridge regression plus a
local kernel, i.e. a correction from the training images that are similar
to the image in the network (default model since 1.2.0). From the predicted
dimensions, we can then compute the predicted perceived similarity between
images. With the features, similarity also uses the network's own similarity
for pairs that are very close in the network (close pairs).

Functions:
    list_models       names of the models that come with dimpred
    load_model        load a model (weights, feature mean and scale, dimension means, labels, info)
    find_images       image files in a folder, sorted by name
    extract_features  network features of images (needs torch and open_clip)
    predict           predicted dimension values from network features
    similarity        predicted similarity between images from their predicted dimensions
                      (and, for the close pairs, their features)
    rise              heatmaps: which parts of an image drive its predicted dimensions (RISE, needs torch)

The network of the default model is in dimpred.alignet (import dimpred.alignet, needs torch).
The parts of rise (masks, maps, overlay for PNG files) are in dimpred/rise.py
(from dimpred.rise import overlay).

Example:
    import dimpred
    files = dimpred.find_images("my_images")
    features = dimpred.extract_features(files)  # default model: alignet_siglip2b_66d_kernel
    embedding = dimpred.predict(features)       # n_images x 66
    S = dimpred.similarity(embedding, features=features)  # n_images x n_images, with the close pairs

From the command line: python -m dimpred my_images --out predictions.csv
(python -m dimpred --help lists all options)

Reference: Kaniuth, P., Mahner, F. P., Perkuhn, J., & Hebart, M. N. (2025). A
high-throughput approach for the efficient prediction of perceived similarity
of natural objects. eLife 14:RP105394.
For the default model, please also cite AligNet: Muttenthaler, L., et al.
(2025). Aligning machine and human visual representations across abstraction
levels. Nature 647, 349-355.

Hebartlab, 2026/09/30
"""

from .list_models import list_models
from .load_model import load_model
from .find_images import find_images
from .extract_features import extract_features
from .predict import predict
from .similarity import similarity
from .rise import rise

# History:
# 2026/10/04: new default model alignet_siglip2b_66d_kernel (ridge + local
#   kernel); similarity with the close-pair term (features)
# 2026/10/02: rise, the RISE heatmaps of the predicted dimensions
# 2026/10/02: new default model alignet_siglip2b_66d_ridge
# 2026/09/30: written for the first release of the package

# The model that is used when no model is given. load_model reads it each
# time it is called, so if you set dimpred.DEFAULT_MODEL to another model,
# all functions use the new default.
DEFAULT_MODEL = "alignet_siglip2b_66d_kernel"
