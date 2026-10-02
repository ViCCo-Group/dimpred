# Training

This folder holds the training code of the DimPred paper (`fit.py`, `call.py`,
`environment.yml`) and `build_models.py`, which makes the shipped models in
`dimpred/models` with it. You only need it to reproduce the paper or to train
models, not to use the models (see the [main README](../README.md)).
`alignet/` has the conversion of the AligNet network to PyTorch
([alignet/README.md](alignet/README.md)).

Contents:
[Reproducing the DimPred paper](#reproducing-the-dimpred-paper) |
[The ridge regression of 1.1.0](#the-ridge-regression-of-110) |
[Rebuilding the shipped models](#rebuilding-the-shipped-models) |
[Feature extraction](#feature-extraction) |
[training/data](#trainingdata) |
[The 1854 reference images](#the-1854-reference-images) |
[Using call.py](#using-callpy) |
[What changed in the code of the paper](#what-changed-in-the-code-of-the-paper) |
[Tests of the training code](#tests-of-the-training-code)

## Reproducing the DimPred paper

- The code for the analyses and figures of the paper is in
  https://github.com/ViCCo-Group/dimpred_paper.
- The data and all 53 fitted models of the paper (49d, one per network and
  layer, as pickled scikit-learn objects) are on OSF: https://osf.io/jtekq
  (`data/interim.7z`).
- The paper used the fractional ridge regression, which is `"fracridge"`
  (`fracridge_cv`) in `fit.py`. Until dimpred 1.1.0 it was called `"ridge"`,
  and the models on OSF are named `..._ridge_...`. `"ridge"` is now another
  ridge regression ([below](#the-ridge-regression-of-110)).
- The RN50x64 49d model of the paper (used for the heatmaps of Figure 7 and
  the fMRI encoding) is the shipped model `rn50x64_49d_ridge`, identical to
  the published `model_49d_ridge_OpenCLIP-RN50x64-openai_visual.joblib`.
  With the features of the paper it gives the published predictions
  (tolerance 1e-10), with features from `dimpred.extract_features` nearly the
  same (differences up to 2e-3). So for this model you do not need the
  training code:
  `dimpred.predict(dimpred.extract_features(files, "rn50x64_49d_ridge"), "rn50x64_49d_ridge")`.
- The other 52 models are only on OSF. They need scikit-learn and fracridge
  to load (the environment of this folder), and `call.py` can use them. They
  were saved with scikit-learn 1.0.2. Newer versions load them with a version
  warning, and the predictions work (tested with 1.3.2 and 1.9.1).

To predict an image set with a model of the paper, or to fit it again:

```
conda env create -f training/environment.yml
conda activate dimpred-training
cd training
DIMPRED_BASE_PATH=<folder> python call.py <model> <module> <imageset> 49 ridge       # the model on OSF, if it is in the folder
DIMPRED_BASE_PATH=<folder> python call.py <model> <module> <imageset> 49 fracridge   # fit the regression of the paper again
```

with the features and models in the folders described in
[Using call.py](#using-callpy). With `ridge`, `call.py` loads the model of the
paper if its file is there, and otherwise fits the new ridge. With
`fracridge`, it fits the fractional ridge with the settings of the paper (70
fractions, 3-fold cross-validation repeated 3 times, `random_state` 0). This
code is faster than the original code and gives the same weights (differences
of at most 4e-13, see
[What changed in the code of the paper](#what-changed-in-the-code-of-the-paper)).

On the 1854 reference images (cross-validated, 49d, fractional ridge, from
the published predictions of the paper), the correlation between predicted
and human similarity is r = 0.888 for OpenCLIP ViT-g-14, 0.887 for OpenCLIP
RN50x64 and 0.866 for CLIP ViT-B/32.

## The ridge regression of 1.1.0

`fit.py` has two ridge regressions (since 2026/10/02):

- `"ridge"` (`ridge_cv`): ridge with the penalty chosen directly. Used for
  `rn50x64_66d_ridge` and `alignet_siglip2b_66d_ridge`, and for new models.
- `"fracridge"` (`fracridge_cv`): the fractional ridge regression of the
  DimPred paper. Use it to reproduce the paper. `rn50x64_49d_ridge` is the
  published model of the paper with it.

**Why.** `fracridge_cv` chooses a fraction in the inner cross-validation and
refits on all 1854 images with the same fraction. But the same fraction
means a different penalty on different data. With the 70 fractions of the
paper, the refit of RN50x64 for the 66d embedding used a median of 9.4 times
the penalty (alpha) of the inner folds (4.1 to 110 times across
dimensions), the refit of CLIP ViT-B/32 a median of 0.02 times. So the final
model was much more or much less regularized than the cross-validation
chose.

**What `ridge_cv` does.** For each dimension, it chooses lambda from
`np.logspace(-6, 3, 73)`, a penalty per image (mean squared error + lambda
||beta||^2, i.e. alpha = n_train x lambda in each inner fold), by the mean
R^2 over the inner folds (3-fold cross-validation repeated 3 times,
`random_state` 0, the first maximum on ties, as before). The refit on all
images uses the alpha of the inner folds (mean n_train x lambda, 1236 x
lambda for 1854 images). The weights come in closed form from one SVD per
fold and equal those of scikit-learn's `Ridge` at the chosen alpha
(`tests/test_training.py`). A fit takes 2 to 4 s.

**What it gives** (our comparison of 15 networks on the 10 test sets of
[docs/models.md](../docs/models.md), 2026/10/01):

- RN50x64, 66d: per dimension (10-fold cross-validation on the 1854 images,
  mean r) 0.779 instead of 0.758 with `fracridge` (better for 63 of 66
  dimensions). On the 10 test sets 0.583 instead of 0.595: better on
  Peterson various (+0.042), Kriegeskorte-92 (+0.025), 48new (+0.020) and
  Peterson animals (+0.011), worse on Peterson fruits (-0.083), vegetables
  (-0.059), furniture (-0.052) and automobiles (-0.019). The test sets
  prefer less shrinkage than the cross-validation chooses, so the
  per-dimension and the test-set results disagree here.
- For the 6 networks that also have an elastic net, it is within -0.003 to
  +0.007 of the elastic net on the test sets.
- Refitting with alpha = n x lambda with all 1854 images (the convention of
  ElasticNetCV) shrank more and was worse for all 15 networks on the test
  sets and for 14 of 15 per dimension.
- The chosen lambda is 0.18 to 1.8 for RN50x64 and 0.24 to 1.3 for AligNet
  SigLIP2-B, far from both ends of the grid.

## Rebuilding the shipped models

```
conda env create -f training/environment.yml
conda activate dimpred-training
pip install -e ".[extract]"
cd training
python build_models.py --images <folder with the 1854 reference images> --features <cache folder> --published <folder>
```

For each model, `build_models.py` extracts the features of the 1854 reference
images with `dimpred.extract_features` (the same code users run on their
images), fits one regression per dimension with `train_model_with` from
`fit.py` with the settings of `call.py` (3-fold cross-validation repeated 3
times for choosing the regularization, `random_state` 0), and saves the weights
with the mean and standard deviation of the features and the mean of each
dimension.

- `--features` is an optional folder where the features are cached
  (`features_<network>.npy`, spaces replaced by `-`, e.g.
  `features_AligNet-SigLIP2-B.npy`). With cached features, `--images` is not
  needed. Training and extraction can then run in different environments:
  extract in one with torch, fit in one with scikit-learn and fracridge.
- `--only NAME ...` builds only some of the models.
- `--device` is the device for the feature extraction (default: cpu). The
  shipped models were fitted on features extracted on the cpu. Other
  devices give slightly different features (AligNet on an Apple GPU:
  differences of about 1.5e-5) and so slightly different weights.
- `--out <folder>` writes the files elsewhere. By default the files in
  `dimpred/models` are overwritten.
- `--published` is the folder with the published files of the paper for
  `rn50x64_49d_ridge`, which is not refit but converted from the published
  weights and feature scaling (OSF, `data/interim.7z`, folder
  `interim/heatmaps_base`, `coefs_49d_ridge_OpenCLIP-RN50x64-openai_visual.npy`
  and `Xstandardizer_49d_ridge_OpenCLIP-RN50x64-openai_visual.pkl`). A refit
  gives nearly the same weights (per-dimension r = 1.0000, largest difference
  2e-6, because the extracted features differ slightly from those of the
  paper), but not the exact ones.

`rn50x64_66d_ridge` and `alignet_siglip2b_66d_ridge` were built on
2026/10/02 (AligNet features extracted on the cpu; later on the same day,
only the `info.note` of the AligNet model was changed). `vitb32_66d_elastic`
and `rn50x64_66d_elastic` are the models of 2026/09/30 (only their
`info.note` was updated on 2026/10/02). The RN50x64 refit equals the fit of
our comparison of 15 networks exactly. The AligNet model, fitted on the
features of the PyTorch port, differs from the fit on the TensorFlow features
by at most 2.5e-7 in the weights, with the same lambda for all 66 dimensions.
`tests/test_model_files.py` compares both models with the fits of that
comparison (`tests/fixtures/benchmark_ridge_fits.mat`).

Features extracted on another device differ slightly, and the elastic net can
differ slightly between scikit-learn versions (change 2 in
[What changed in the code of the paper](#what-changed-in-the-code-of-the-paper)),
so a rebuild on another computer gives nearly but not exactly the shipped
weights. Some tests then fail: the comparison of the shipped models with the
expected predictions (tolerance 1e-10), until the fixtures are made again with
`tests/fixtures/make_fixtures.py` (it needs data that are not part of the
repository), and the comparison of the shipped ridge models with a refit from
the cached features (tolerance 1e-8).

## Feature extraction

`dimpred.extract_features` gives the features used for training. open_clip
`RN50x64` and `ViT-B-32-quickgelu` reproduce the features of the DimPred paper
(r = 1.000000), so thingsvision is not needed; the plain `ViT-B-32`
configuration gives different features. Details:
[Feature extraction](../docs/details.md#feature-extraction). For AligNet
SigLIP2-B, see [alignet/README.md](alignet/README.md).

## training/data

- `reference_image_names.txt`: the file names of the 1854 reference images, in
  the order of the rows of the embeddings
- `spose_embedding_49d.txt`, `spose_embedding_66d.txt`: the SPoSE embeddings
  (1854 x 49 and 1854 x 66). The 66d embedding is the corrected version of
  January 2023.
- `labels_49d.txt`, `labels_66d.txt`: the labels of the dimensions

## The 1854 reference images

The models are trained on the 1854 THINGS reference images, the images of the
odd-one-out experiments, one per object concept. They are not part of this
repository (if you don't have them, ask us, e.g. in an issue on GitHub). They
have to be named as in
[data/reference_image_names.txt](data/reference_image_names.txt) and be in
this order, which is the order of the rows of the embeddings. Do not sort the
file names instead: sorting in Python puts 21 concepts in a different order
(e.g. camera_lens comes before camera1 and camera2 in the embedding; the
others are concepts starting with chicken, crystal, hot, ice and pepper),
which silently pairs these images with the wrong dimension values.

The reference images are a separate set from the images of the THINGS
database: of the 1663 reference concepts that also have an `_01b` image in the
THINGS database, 115 have a different photo there, and 191 of the 1854
reference images are not in THINGS at all.

## Using call.py

`call.py` is the script of the paper code to train a model for one network and
layer and to predict the dimensions of an image set with it:

```
cd training
DIMPRED_BASE_PATH=<folder> python call.py <model> <module> <imageset> <n_dim> <regularization>
```

It expects the features as text files (images x features) in this layout below
`DIMPRED_BASE_PATH`:

```
dimpred/data/raw/dnns/<model>/<module>/<imageset>/features.txt   (or features-srp.txt)
dimpred/data/raw/dnns/<model>/<module>/1854ref/features.txt      (the 1854 reference images)
dimpred/data/raw/original_spose/embedding_<n_dim>d.txt
```

and writes the fitted model and the predictions to

```
dimpred/data/interim/dimpred/model_<n_dim>d_<regularization>_<model>_<module>.joblib
dimpred/data/interim/dimpred/predictions_<n_dim>d_<regularization>_<model>_<module>_<imageset>.txt
```

Create the folder `dimpred/data/interim/dimpred` first, `call.py` does not
create it. If the model file already exists, it is loaded instead of fitted
again. `<regularization>` is `ridge`, `fracridge` or `elastic`. The models of
the paper on OSF are named `..._ridge_...` but are fractional ridge models:
with `ridge`, `call.py` loads them if they are there and only fits a new
model (the new ridge) if not. To fit the regression of the paper again, use
`fracridge`. For new work, the shipped models and the functions of the
package are easier to use.

## What changed in the code of the paper

The code used to be in `dimpred/` (`fit.py`, `call.py`), with `environment.yml`
and `setup.py` at the top level. `setup.py` was replaced by `pyproject.toml`,
and the rest moved to `training/` unchanged, except for these small changes:

1. **Speed-up of the fractional ridge** (`fracridge_cv` in `fit.py`). The
   original code used `MultiOutputRegressor(FracRidgeRegressorCV())`, which
   hands every dimension separately to scikit-learn's `GridSearchCV`. This
   refits the model for every fraction and fold, and every fit computes a new
   SVD of the same feature matrix: about 42,000 SVDs for 66 dimensions, 70
   fractions and 3 x 3 folds. The core function `fracridge()` of the fracridge
   package solves all dimensions and all fractions from one SVD, so one SVD
   per fold is enough (10 in total, with the final fit on all data), and the
   held-out predictions for all fractions and dimensions come from one matrix
   product. The fraction is selected with the same rule as in `GridSearchCV`
   (highest mean R^2 across folds, the first one in case of ties), and the
   model is then refit on all data. For the 66d RN50x64 model, the original
   code took 2 h 27 min (8810 s) on 10 cores (one dimension alone: 557 s of
   CPU time, about 10 CPU hours in total), the new code about 25 s. The result
   is the same: the weights differ by at most 4e-13 (floating point
   precision), the per-dimension r is 1.0, and the same fraction is selected
   for all 66 dimensions. The weights of the original code are stored in the
   test fixtures (see `tests/fixtures/make_fixtures.py`) and compared in
   `tests/test_training.py`. fracridge itself is unchanged, only its core
   function is used. For the fractional ridge (and the new ridge),
   `train_model_with` now returns the weight matrix (features x dimensions)
   instead of a scikit-learn object, and `call.py` saves this matrix in the
   .joblib file. `get_predictions_for` works with both, so the models on OSF
   still work.
2. **scikit-learn compatibility.** `FracRidgeRegressorCV` no longer works with
   newer scikit-learn (from about 1.4 on; it fails on 1.9.1 with
   "_preprocess_data() got an unexpected keyword argument 'normalize'" and
   worked on 1.3.2). After change 1 it is not needed anymore.
   `ElasticNetCV(n_alphas=...)` was replaced by `alphas=<int>` in scikit-learn
   1.7 and removed in 1.9. `fit.py` picks the right argument for the installed
   version (same alphas, max difference 4e-8).
3. **call.py**: the import after the move (`from fit import ...`), and
   `determine_base_path`, which came from a `utils` module that was never in
   this repository, now reads the environment variable `DIMPRED_BASE_PATH`
   (default: the current folder).
4. **build_models.py** (new) recreates the shipped model files, and
   `training/data` holds the data it needs.
5. **The new ridge** (2026/10/02, see
   [The ridge regression of 1.1.0](#the-ridge-regression-of-110)): `ridge_cv`,
   and the regularization `"fracridge"` for the old one. `fit_model_with` now
   gives an error for an unknown regularization. When `call.py` fits a model
   with `"ridge"`, it prints that this is the new ridge and that the paper
   used `"fracridge"`.

`environment.yml` now creates the environment `dimpred-training` (before:
`dimpred`), with scipy for `build_models.py` and without ipykernel. It was
tested with Python 3.11, scikit-learn 1.9.1 and fracridge 2.0, and with
scikit-learn 1.3.2.

## Tests of the training code

`tests/test_training.py` needs scikit-learn and fracridge, so run it in this
environment (install pytest there first, `pip install pytest`). It checks the
new ridge against scikit-learn's `Ridge`, the speed-up of the fractional ridge
against a brute-force version and against the original code, and
`tests/test_training_data.py` checks the order of the reference images in
`data/`. From the repository folder:

```
python -m pytest tests/test_training.py tests/test_training_data.py
```

- The two comparisons with the original `FracRidgeRegressorCV` on a small
  problem only run with scikit-learn up to about 1.3 (e.g. 1.3.2) and are
  skipped on newer versions.
- The tests against the weights of the original code in addition need the
  RN50x64 features of the 1854 reference images. Set
  `DIMPRED_REFERENCE_FEATURES` to `features_RN50x64.npy` (as written by
  `build_models.py --features`) or to the folder that contains it. With the
  folder, the shipped `rn50x64_66d_ridge` and `alignet_siglip2b_66d_ridge`
  are also fitted again and compared with the model files (the second needs
  `features_AligNet-SigLIP2-B.npy` there).
