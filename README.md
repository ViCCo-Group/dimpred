# DimPred

DimPred predicts, for any image, its values on the SPoSE dimensions of human
mental object representations (the 49 dimensions of Hebart et al., 2020, or
the 66 dimensions of Hebart et al., 2023), and from these dimensions the
perceived similarity between images. A deep neural network (the image encoder
of CLIP) turns each image into a feature vector. One linear regression per
dimension, trained on the 1854 THINGS reference images, maps the features to
the dimension values. The method and its evaluation are described in

Kaniuth, P., Mahner, F. P., Perkuhn, J., & Hebart, M. N. (2025). A high-throughput
approach for the efficient prediction of perceived similarity of natural objects.
eLife 14:RP105394. https://doi.org/10.7554/eLife.105394

This repository contains a small Python package and a parallel MATLAB version
that apply the trained models, the trained models themselves, Philipp
Kaniuth's training code, and the tests. The Python and MATLAB functions have
the same names (`dimpred.predict` in Python, `dimpred_predict` in MATLAB), the
same arguments (except that `dimpred_extract_features` takes its options in
cfg, see [All functions](#all-functions)) and give the same numbers.

If you know the earlier version of this repository: it only contained
Philipp's training code (`dimpred/fit.py`, `dimpred/call.py`), and the fitted
models had to be downloaded from OSF. This code now lives in `training/` (see
[Training](#training)), and `dimpred/` is the package that applies the
models, with the models included. Scripts that imported `dimpred.fit` or
`dimpred.call` have to import `fit` from `training/` instead (run them there
or add `training/` to the Python path).

Contents:
[Installation](#installation) |
[Quick start](#quick-start) |
[Images and limitations](#images-and-limitations) |
[Models](#models) |
[How the predictions are computed](#how-the-predictions-are-computed) |
[Similarity](#similarity) |
[Feature extraction](#feature-extraction) |
[Which model was used where](#which-model-was-used-where) |
[Training](#training) |
[Tests](#tests) |
[Reproducing the paper](#reproducing-the-paper) |
[Repository layout](#repository-layout) |
[Citation](#citation) |
[Credits](#credits) |
[License](#license)


## Installation

Python 3.9 or newer. From a clone of this repository:

```
git clone https://github.com/ViCCo-Group/dimpred.git
cd dimpred
pip install -e .              # predictions and similarity from features (installs numpy and scipy)
pip install -e ".[extract]"   # in addition feature extraction from images (torch, open_clip_torch, pillow)
pip install -e ".[test]"      # pytest, for running the tests
```

Without the `extract` part you can do everything except extracting features
from images, e.g. predict dimensions from features you already have. torch and
open_clip are only imported inside `extract_features`, so the rest of the
package does not load them.

MATLAB (R2016b or newer): add the folder `matlab/` to the path.

```matlab
addpath('/path/to/dimpred/matlab')
```

The MATLAB functions find the model files in `../dimpred/models` relative to
their own folder, so keep `matlab/` inside the repository. Everything runs in
MATLAB alone except the feature extraction: `dimpred_extract_features` calls
the Python command line tool, so it needs a Python with numpy, scipy, torch,
open_clip_torch and pillow (e.g. after `pip install -e ".[extract]"`, or
`pip install numpy scipy torch open_clip_torch pillow`). dimpred itself does not
have to be installed in this Python: `dimpred_extract_features` puts the
repository on the Python path, so Python and MATLAB use the same code and models.


## Quick start

### Python

```python
import dimpred

files = dimpred.find_images("my_images")     # image files in this folder, sorted by name
features = dimpred.extract_features(files)   # one row per image, in the order of files
embedding = dimpred.predict(features)        # images x 66 dimensions (default model)
S = dimpred.similarity(embedding)            # images x images, predicted similarity
labels = dimpred.load_model()["labels"]      # names of the 66 dimensions
```

With another model, the features have to come from the network of that model.
`extract_features` takes the network from the model you give it:

```python
print(dimpred.list_models())
model = dimpred.load_model("rn50x64_49d_ridge")     # the model of the DimPred paper
features = dimpred.extract_features(files, model)   # RN50x64 features
embedding = dimpred.predict(features, model)        # images x 49 dimensions
```

If the number of features does not fit the model (e.g. ViT-B/32 features and
an RN50x64 model), `predict` stops with an error that names the network and
the number of features the model expects. Features you extracted before (the
same network and layer, see [Feature extraction](#feature-extraction)) can be
passed directly as an array with images in rows.

### MATLAB

```matlab
addpath('/path/to/dimpred/matlab')
files = dimpred_find_images('my_images');     % cell column of full paths, sorted by name
features = dimpred_extract_features(files);   % runs Python, one row per image
embedding = dimpred_predict(features);        % images x 66 dimensions (default model)
S = dimpred_similarity(embedding);            % images x images, predicted similarity
model = dimpred_load_model;                   % default model, model.labels holds the dimension names
```

`dimpred_extract_features` runs `python -m dimpred ... --features-only` and
reads the result. It uses the Python in `cfg.python` if given, else the one
in the environment variable `DIMPRED_PYTHON`, else `python3` (on Windows,
where `python3` usually does not exist, set `cfg.python` or `DIMPRED_PYTHON`):

```matlab
cfg.python = '/path/to/env/bin/python';   % a Python with numpy, scipy, torch, open_clip_torch and pillow
cfg.device = 'cpu';                       % optional, default: cuda, then mps, then cpu
[features, files] = dimpred_extract_features(files, 'rn50x64_49d_ridge', cfg);
embedding = dimpred_predict(features, 'rn50x64_49d_ridge');
```

### Command line

```
python -m dimpred my_images                           # all images in the folder, writes dimpred_predictions.csv
python -m dimpred a.jpg b.jpg --model rn50x64_49d_ridge --out predictions.mat
python -m dimpred my_images --features-only --out features.mat
python -m dimpred --features features.mat --out predictions.csv
```

```
python -m dimpred IMAGE [IMAGE ...] [--model NAME] [--out FILE] [--features-only] [--device DEV] [--batch-size N]
python -m dimpred --features FILE.mat [--model NAME] [--out FILE]
```

IMAGE can be files or folders. Folders are expanded with `find_images`, and
the rows of the output are in the order in which the images were given.
`--model` takes the name of a shipped model or the path of a model file
(default: `vitb32_66d_elastic`). `--features FILE.mat` predicts from features
that were computed before (a .mat file with the variable `features`, images x
features, and optionally `files`), which does not need torch. Give the same
`--model` as for the extraction: the model name stored in a file written with
`--features-only` is not read. The tool prints a short summary with the
number of images, the model and the output file.
`python -m dimpred --help` lists all options, and the output formats are
described in [File formats](#file-formats).

### All functions

| Python | MATLAB | what it does |
|---|---|---|
| `dimpred.list_models()` | `names = dimpred_list_models` | names of the shipped models, sorted |
| `dimpred.load_model(model=None)` | `model = dimpred_load_model(model)` | load a model by name or file (default: `DEFAULT_MODEL`) |
| `dimpred.find_images(folder)` | `files = dimpred_find_images(folder)` | image files directly in a folder, sorted by name, full paths |
| `dimpred.extract_features(images, model=None, network=None, pretrained="openai", device=None, batch_size=32)` | `[features, files] = dimpred_extract_features(images, model, cfg)` | network features of image files |
| `dimpred.predict(features, model=None)` | `embedding = dimpred_predict(features, model)` | predicted dimension values |
| `dimpred.similarity(embedding, method="spose")` | `S = dimpred_similarity(embedding, method)` | predicted similarity, `"spose"` or `"dot"` |

`dimpred.DEFAULT_MODEL` is `"vitb32_66d_elastic"`. Wherever a model is
expected, you can pass nothing (default model), the name of a shipped model,
the path to a model file, or a model that was already loaded. A model is a
dict in Python and a struct in MATLAB with the fields `weights`,
`feature_mean`, `feature_scale`, `target_mean`, `labels`, `info` and `file`.
Each function explains its inputs and outputs in its help text
(`help(dimpred.predict)` in Python, `help dimpred_predict` in MATLAB).


## Images and limitations

The models were trained on the 1854 THINGS reference images, photographs of
single objects, one for each object concept, and tested on the image sets in
[How well the models work on new images](#how-well-the-models-work-on-new-images).
Some things to keep in mind:

- Each image is scaled and center-cropped to a square (see
  [Feature extraction](#feature-extraction)), so the network does not see
  the parts of a non-square image outside the central square.
- Images are converted to RGB, which drops the alpha channel. Transparent
  areas of a PNG then get the colors stored there, often black. Put such
  images on a background first.
- Predicted similarity within one homogeneous category is much less accurate
  than across categories: r = 0.19 to 0.37 with human similarity for the 120
  Peterson animals, 0.76 to 0.87 for the other image sets.
- SPoSE similarity depends on the other images in the set (see
  [Similarity](#similarity)).

Which model: the default `vitb32_66d_elastic` in general, and
`rn50x64_49d_ridge` to compare with the DimPred paper or with the published
49d predictions. The values of 49d and 66d models cannot be mixed, so use
the same model for all images you want to compare.


## Models

There are four shipped models (`dimpred/models/<name>.mat`):

| name | network (open_clip) | features | dims | regression | role |
|---|---|---|---|---|---|
| `vitb32_66d_elastic` | OpenAI CLIP ViT-B/32 (`ViT-B-32-quickgelu`) | 512 | 66 | elastic net | default; rebuilt model of Contier et al. (2024) |
| `rn50x64_49d_ridge` | OpenAI CLIP RN50x64 (`RN50x64`) | 1024 | 49 | fractional ridge | model of the DimPred paper, Philipp's published weights |
| `rn50x64_66d_ridge` | OpenAI CLIP RN50x64 (`RN50x64`) | 1024 | 66 | fractional ridge | same procedure as the paper, for the 66d embedding |
| `rn50x64_66d_elastic` | OpenAI CLIP RN50x64 (`RN50x64`) | 1024 | 66 | elastic net | as above, with elastic net |

All models use the weights `pretrained="openai"`, and their features are the
output of the image encoder of the network (open_clip's image embedding, not
normalized). All were trained on the 1854 THINGS reference images (the images
of the odd-one-out experiments) with Philipp Kaniuth's training code
(`training/fit.py`) and the settings of `training/call.py`: 3-fold
cross-validation repeated 3 times for choosing the regularization,
`random_state` 0. `rn50x64_49d_ridge` is not a refit but Philipp's published
fit of the paper. The 49d and 66d models predict different embeddings with
different dimensions, so their values cannot be mixed. The labels of the
dimensions are stored in each model.

### How well the models work on new images

Pearson correlation between predicted and human similarity (values below the
diagonal), on image sets that were never used for training. The same images
were used for all models, and the similarity measure is the one of the paper:
SPoSE similarity for 48new, 48nonref and the Peterson animals, Euclidean
distance for Kriegeskorte-92.

| model | 48new | 48nonref | Peterson animals | Kriegeskorte-92 | mean |
|---|---|---|---|---|---|
| `vitb32_66d_elastic` | 0.870 | 0.825 | 0.365 | 0.812 | 0.718 |
| `rn50x64_66d_elastic` | 0.868 | 0.829 | 0.278 | 0.794 | 0.692 |
| `rn50x64_66d_ridge` | 0.847 | 0.820 | 0.271 | 0.763 | 0.675 |
| `rn50x64_49d_ridge` | 0.848 | 0.810 | 0.186 | 0.771 | 0.654 |

48new: 48 new objects. 48nonref: 48 new images of THINGS objects. Both come
with human odd-one-out data from the DimPred paper. Peterson animals: 120
animal images (Peterson et al., 2018). Kriegeskorte-92: 92 images (Mur et al.,
2013). Homogeneous sets such as the Peterson animals remain difficult.

Elastic net was better than ridge for both networks on all four sets (the
ViT-B/32 ridge model is not among the shipped models, so it is not in the
table). The ViT-B/32 66d elastic net model was the best overall, and ViT-B/32
is also the smallest and fastest of the networks, so this model is the
default. Use `rn50x64_49d_ridge` if you want the model of the DimPred paper,
e.g. to compare with its results or with the published 49d predictions.

For comparison, the benchmark of the paper on the 1854 reference images
(cross-validated, 49d, ridge, computed from Philipp's published predictions)
gives r = 0.888 for OpenCLIP ViT-g-14, 0.887 for OpenCLIP RN50x64 and 0.866
for CLIP ViT-B/32.


## How the predictions are computed

In MATLAB notation

```
embedding = max(((features - feature_mean) ./ feature_scale) * weights + target_mean, 0)
```

or in Python

```python
z = (features - model["feature_mean"]) / model["feature_scale"]
embedding = np.maximum(z @ model["weights"] + model["target_mean"], 0)
```

- `features` (images x features) are the output of the image encoder of the
  network of the model.
- `feature_mean` and `feature_scale` are the mean and standard deviation of
  the features of the 1854 training images. New features are z-scored with
  these and not with the statistics of your own images, so the prediction of
  an image does not depend on which other images you pass.
- `weights` (features x dims) hold one regression per dimension.
- `target_mean` is the mean of each dimension across the 1854 training
  images. The regressions were fit on centered dimension values, so the mean
  has to be added back.
- Values below 0 are set to 0, because SPoSE dimensions are non-negative.

Some earlier, unofficial wrappers left out `target_mean`. This makes the
predictions too small and sets about 60% of them to zero, and the correlation
of predicted with human similarity for the 48nonref images drops from 0.81 to
0.76. Predictions made with such a wrapper should be made again. A quick
check of any implementation: features equal to `feature_mean` have to give
`target_mean` as prediction, e.g.
`dimpred.predict(model["feature_mean"], model)`.

The model files are plain MATLAB v5 .mat files (written with
`scipy.io.savemat`), so they can also be used without this package, with
MATLAB's `load` or with `scipy.io.loadmat`. They contain `weights` (features x
dims), `feature_mean` and `feature_scale` (1 x features), `target_mean`
(1 x dims), `labels` (cell array, dims x 1) and `info`, a struct with the
text fields `name`, `network`, `pretrained`, `layer`, `preprocessing`,
`embedding`, `regression`, `training_images`, `source`, `note` and `created`,
and the numbers `n_features` and `n_dims`.


## Similarity

`similarity(embedding)` computes SPoSE similarity, the similarity of the
model that the SPoSE dimensions come from. In the odd-one-out task, people see
three objects and pick the odd one out, i.e. they choose the two most similar
ones. The SPoSE model predicts this choice from the dot products of the
embeddings. The similarity of images i and j is the probability that i and j
are chosen as the most similar pair when a random third image k of the set is
added:

```
S[i, j] = mean over all k not in {i, j} of
          exp(e_i.e_j) / (exp(e_i.e_j) + exp(e_i.e_k) + exp(e_j.e_k))
```

with `S[i, i] = 1`. S is symmetric. Before `exp`, the largest dot product is
subtracted (the largest of all pairs, or, if the dot products span 700 or
more, the largest of the three in each triplet). This does not change the
probabilities, but large embedding values do not give inf or nan. Since the
third object comes from the images you pass, the similarity of two images
depends a little on the other images in the set, and at least 3 images are
needed. For each triplet the three pair probabilities add up to 1, so the
mean of all values off the diagonal is exactly 1/3 (a quick check of any
implementation).

`similarity(embedding, "dot")` (MATLAB: `dimpred_similarity(embedding, 'dot')`)
gives the plain dot product `embedding @ embedding.T`, which does not depend
on the other images. In the benchmarks, Kriegeskorte-92 was compared with the
Euclidean distance between the predicted embeddings, which you can compute
with `scipy.spatial.distance.pdist` in Python or `pdist` in MATLAB (Statistics
and Machine Learning Toolbox).

The computing time of the SPoSE similarity grows with the cube of the number
of images: a few seconds for 1000 images, but 10 times as many images take
about 1000 times as long. S and the matrices used on the way have n x n
values. For many thousands of images, use `"dot"`, or compute the similarity
for subsets (the values then depend on the subset).


## Feature extraction

`extract_features` uses open_clip with the weights `pretrained="openai"`.
Each image is opened with PIL, converted to RGB (so grayscale images work as
well), preprocessed with the preprocessing of the network (RN50x64: resize the
shortest side to 448 px, bicubic, center crop 448 x 448; ViT-B/32: the same
with 224), and passed through `model.encode_image` in float32 without
gradients, in batches of `batch_size` images. The result is a float32 array
of the features (images x features), not normalized. The device is `cuda` if
available, else `mps` (Apple GPU), else `cpu`. The first time a network is
used, open_clip downloads its weights (about 0.6 GB for ViT-B/32, 2.5 GB for
RN50x64).

For OpenAI's ViT models the open_clip configuration with `-quickgelu` has to
be used (`ViT-B-32-quickgelu`). The original CLIP ViTs use QuickGELU, and the
plain `ViT-B-32` configuration uses GELU with the same weights, which gives
different features (per-image r of about 0.98 with the original features
instead of 1.0). The models store the right name in `info["network"]`, and
`extract_features` uses it. When it loads RN50x64, open_clip warns about a
"QuickGELU mismatch", because its RN50x64 configuration does not set
QuickGELU while the OpenAI weights were trained with it. This does not matter
for the image features: the image encoder of RN50x64 is a ResNet with ReLU,
and QuickGELU is only used in the text part, which dimpred does not use.
open_clip's `RN50x64` gives Philipp's RN50x64 features (see below).

We validated the extraction against the features used for the paper:
open_clip `RN50x64` reproduces Philipp's RN50x64 features (r = 1.000000, max
abs difference 3e-5 on the cpu, 1.6e-4 on an Apple GPU), and
`ViT-B-32-quickgelu` reproduces the ViT-B/32 features used by Philipp and
Oliver Contier (r = 1.000000) and Philipp's published predictions for 48new,
48nonref, Kriegeskorte-92 and the Peterson animals (r = 1.000000). So
thingsvision is not needed. Different devices give slightly different
numbers; the tests allow a per-image correlation of 0.99999 and a difference
of 2e-3.

### Image order

The rows of the features and predictions are always in the order of the files
you give. `extract_features` does not take folders for this reason; use
`find_images` to list a folder. `find_images` returns the image files directly
in the folder (not in subfolders) with the extensions .jpg, .jpeg, .png,
.bmp, .tif, .tiff and .webp in any case, without hidden files, sorted by file
name with a plain string sort (the same order as `sort` in MATLAB). If your
images belong to a list in a specific order (e.g. the rows of a data matrix),
pass the files in that order and do not sort them.

### File formats

- Output of the command line tool, .csv (the default is
  `dimpred_predictions.csv`): header `image,<label 1>,...,<label n>` and one
  row per image. With `--features-only` the header is
  `image,feature_1,...,feature_p`.
- Output of the command line tool, .mat: the variables `files` (cell),
  `labels` (cell), `model` (name of the model) and `embedding` (images x
  dims), or with `--features-only` `features` (images x features) instead of
  `embedding`.
- Input for `--features`: a .mat file with `features` (images x features,
  double or single) and optionally `files` (cell array of names; without it,
  the image column holds the row numbers). A file written with
  `--features-only` can be used directly. From MATLAB, save it in the
  format `-v7` (e.g. `save('features.mat', 'features', 'files', '-v7')`),
  because scipy cannot read `-v7.3` files.
- Model files: see [How the predictions are computed](#how-the-predictions-are-computed).


## Which model was used where

| where | network | dims | regression |
|---|---|---|---|
| DimPred paper, all benchmarks | 53 networks/layers | 49d | ridge |
| DimPred paper, heatmaps (Fig. 7) | RN50x64 | 49d | ridge |
| DimPred paper revision, fMRI encoding (Fig. 8) | RN50x64 | 49d | ridge |
| Contier et al. 2024, THINGS-fMRI analyses | CLIP ViT-B/32 | 66d | elastic net (the paper text says ridge) |
| Contier et al. 2024, BOLD5000 replication (revision) | CLIP RN50 | 66d | ridge |
| NIH image set (2024) | RN50x64 | 49d and 66d | ridge |
| 49d predictions for all 26,107 THINGS images on OSF | RN50x64, CLIP RN50 | 49d | ridge |

The 66d predictions published with Contier et al. (2024)
(github.com/ViCCo-Group/dimension_encoding, `data/66d`) were computed with the
66d embedding before its correction in January 2023. The difference is small
(r = 0.996 with predictions after the correction). `vitb32_66d_elastic` uses
the corrected embedding. The rebuilt model reproduces Contier's predictions
with a median per-dimension r of 0.998.


## Training

### Philipp Kaniuth's code moved to training/

The training code of the DimPred paper was written by Philipp Kaniuth. It used
to be in `dimpred/` (`fit.py`, `call.py`) with `environment.yml` and
`setup.py` at the top level. The folder `dimpred/` is now the package that
applies the models, and the training code moved to `training/` (`fit.py`,
`call.py`, `environment.yml`); `setup.py` was replaced by `pyproject.toml`.
The code was moved unchanged, except for these small changes:

1. **Speed-up of the ridge regression** (`fracridge_cv` in `fit.py`). The
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
   for all 66 dimensions. The weights of the original code are stored in
   `tests/fixtures/philipp_original_ridge_rn50x64_66d.mat` and compared in
   `tests/test_training.py`. fracridge itself is unchanged; only its core
   function is used. For ridge, `train_model_with` now returns the weight
   matrix (features x dimensions) instead of a scikit-learn object, and
   `call.py` saves this matrix in the .joblib file. `get_predictions_for`
   works with both, so the models on OSF still work.
2. **scikit-learn compatibility.** `FracRidgeRegressorCV` no longer works with
   newer scikit-learn (from about 1.4 on; tested: it fails on 1.9.1 with
   "_preprocess_data() got an unexpected keyword argument 'normalize'" and
   worked on 1.3.2). After change 1 it is not needed anymore.
   `ElasticNetCV(n_alphas=...)` was replaced by `alphas=<int>` in scikit-learn
   1.7 and removed in 1.9. `fit.py` picks the right argument for the installed
   version (same alphas; verified, max difference 4e-8).
3. **call.py**: the import after the move (`from fit import ...`), and
   `determine_base_path`, which came from a `utils` module that was never in
   this repository, now reads the environment variable `DIMPRED_BASE_PATH`
   (default: the current folder).
4. **build_models.py** (new) recreates the shipped model files, and
   `training/data` holds the data it needs (see below).

`environment.yml` now creates the environment `dimpred-training` (with scipy
for `build_models.py`, without ipykernel). It was tested with Python 3.11,
scikit-learn 1.9.1 and fracridge 2.0, and with scikit-learn 1.3.2.

### Rebuilding the shipped models

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
`fit.py` with the settings of `call.py`, and saves the weights together with
the feature mean and standard deviation and the mean of each dimension.
`--features` is an optional folder where the features are cached
(`features_<network>.npy`; with cached features `--images` is not needed),
`--only NAME ...` builds only some of the models, and `--out` sets the output
folder. By default the files in `dimpred/models` are overwritten; use
`--out <folder>` to write them elsewhere. `rn50x64_49d_ridge` is not refit
but converted from Philipp's published weights and feature scaling (OSF,
`data/interim.7z`, folder `interim/heatmaps_base`, the files
`coefs_49d_ridge_OpenCLIP-RN50x64-openai_visual.npy` and
`Xstandardizer_49d_ridge_OpenCLIP-RN50x64-openai_visual.pkl`), whose folder
you pass with `--published`. A refit gives nearly the same weights
(per-dimension r = 1.0000, largest difference 2e-6, because the extracted
features differ slightly from Philipp's), but not the exact ones.

Features extracted on another device differ slightly (see
[Feature extraction](#feature-extraction)), and the elastic net can differ
slightly between scikit-learn versions (see change 2 above), so a rebuild on
another computer gives nearly but not exactly the shipped weights. Some tests
then fail: those that compare the shipped models with the expected
predictions (tolerance 1e-10), until the fixtures are made again with
`tests/fixtures/make_fixtures.py`, and the comparison of `rn50x64_66d_ridge`
with the weights of Philipp's original code (tolerance 1e-8).

`training/data` contains:

- `reference_image_names.txt`: the file names of the 1854 reference images
  in the order of the rows of the embeddings
- `spose_embedding_49d.txt`, `spose_embedding_66d.txt`: the SPoSE embeddings
  (1854 x 49 and 1854 x 66). The 66d embedding is the corrected version of
  January 2023.
- `labels_49d.txt`, `labels_66d.txt`: the labels of the dimensions

### The 1854 reference images

The models are trained on the 1854 THINGS reference images, the images of
the odd-one-out experiments, one per object concept. They are not part of this
repository (if you don't have them, ask us, e.g. in an issue on GitHub). They
have to be named as in `training/data/reference_image_names.txt`, and they
have to be in this order, which is the order of the rows of the embeddings.
Do not sort the file names instead: sorting in Python puts 21 concepts in a
different order (e.g. camera_lens comes before camera1 and camera2 in the
embedding; the others are concepts starting with chicken, crystal, hot, ice
and pepper), which silently pairs these images with the wrong dimension
values. The reference images are also a separate set from the images of the
THINGS database: of the 1663 reference concepts that also have an `_01b` image
in the THINGS database, 115 have a different photo there, and 191 of the 1854
reference images are not in THINGS at all.

### Using call.py

`call.py` is Philipp's script to train a model for one network and layer and
to predict the dimensions of an image set with it:

```
cd training
DIMPRED_BASE_PATH=<folder> python call.py <model> <module> <imageset> <n_dim> <regularization>
```

It expects the features as text files (images x features) in this layout
below `DIMPRED_BASE_PATH`:

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

Create the folder `dimpred/data/interim/dimpred` first; `call.py` does not
create it. If the model file already exists, it is loaded instead of fitted
again. `<regularization>` is `ridge` or `elastic`. The models on OSF were
saved with scikit-learn 1.0.2; newer versions load them with a version
warning, and the predictions work (tested with 1.3.2 and 1.9.1). For new
work, the shipped models and the functions above are easier to use.


## Tests

The tests were written before the package code, and they compare dimpred with
numbers that do not come from dimpred: Philipp's published results (features
and predictions of the paper model, predictions for the CC0 test images),
human similarity data, and an independent reference implementation.

```
python -m pytest tests                  # all tests
python -m pytest tests -m "not slow"    # without the slow ones
```

The slow tests extract features with torch, run MATLAB, train on the 1854
reference images or measure the speed of the similarity. Tests are skipped
if an optional program or package is missing (torch, open_clip, MATLAB,
scikit-learn or fracridge), if an optional large file that is not in the
repository is missing, or if they need an older scikit-learn (see below).
The MATLAB tests in `tests/test_matlab.py` look for MATLAB in the
environment variable `DIMPRED_MATLAB` (full path of the matlab program), then
in `/Applications/MATLAB_*/bin/matlab`, then on the system path. The training
tests (`tests/test_training.py`) need scikit-learn and fracridge (the training
environment; install pytest there first, `pip install pytest`). The two
comparisons with Philipp's original `FracRidgeRegressorCV` on a small problem
only run with scikit-learn up to about 1.3 (e.g. 1.3.2) and are skipped on
newer versions. The tests against the weights of Philipp's original code in
addition need the RN50x64 features of the 1854 reference images: set
`DIMPRED_REFERENCE_FEATURES` to `features_RN50x64.npy` (as written by
`build_models.py --features`) or to the folder that contains it.

The MATLAB tests can also be run in MATLAB directly, from the repository
folder:

```
matlab -batch "cd('tests/matlab'); run_dimpred_tests"
```

`run_dimpred_tests('fast')` leaves out the feature extraction, which starts
Python and loads the networks. The extraction tests need `DIMPRED_PYTHON`
(`setenv('DIMPRED_PYTHON', '/path/to/python')`); without it they are skipped.

What the tests check, in short:

- the shipped models give the expected predictions on 168 test images (tolerance 1e-10),
  and `rn50x64_49d_ridge` gives Philipp's published predictions (1e-10)
- the numbers in the model files: `target_mean` is the mean of each dimension
  of the embedding in `training/data`, the feature scaling of the rebuilt
  RN50x64 models is that of Philipp's published model, the labels are those of
  the embedding, and `rn50x64_66d_ridge` is the result of Philipp's original
  training code
- each part of the formula of `predict`, and mistakes of earlier versions
  (no `target_mean`, wrong z-scoring)
- `similarity` against small examples computed by hand, a slow reference that
  follows the definition, symmetry, the mean of 1/3, permutations, large
  values, and speed
- feature extraction of the three CC0 images against reference features, the
  ViT GELU/QuickGELU mistake, the order of the rows, and errors for missing
  files and folders
- correlation of predicted and human similarity for the 48nonref images
  (e.g. r = 0.810 for the paper model)
- the command line tool, the package without torch, the MATLAB functions
  against the same numbers and against the Python functions
- the speed-up of the training against a brute-force version, against
  Philipp's original code, and the order of the reference images in
  `training/data`

The fixtures are in `tests/fixtures`: `reference_data.mat` (Philipp's RN50x64
features and the ViT-B/32 features of 168 images, 48nonref and 120 Peterson
animals, his published predictions, the expected predictions of each shipped
model, the human similarity of the 48nonref images, and reference features and
published predictions of three CC0 images), `images/` (three THINGSplus CC0
images, free to redistribute), and `philipp_original_ridge_rn50x64_66d.mat`
(the weights and selected fractions of Philipp's original training code for
`rn50x64_66d_ridge`). `tests/fixtures/make_fixtures.py` documents how they were
made. It needs data that are not part of the repository and only has to be run
again if the shipped models change.


## Reproducing the paper

The code to reproduce the analyses and figures of the paper is in
https://github.com/ViCCo-Group/dimpred_paper. The data and all 53 fitted models
of the paper (49d, ridge, as pickled scikit-learn objects) are on OSF:
https://osf.io/jtekq (`data/interim.7z`). The RN50x64 49d ridge model of the
paper (used for the heatmaps and the fMRI encoding) is the shipped model
`rn50x64_49d_ridge`, identical to the published
`model_49d_ridge_OpenCLIP-RN50x64-openai_visual.joblib`, and it reproduces
Philipp's published predictions (see [Tests](#tests)). The other 52 models are
only available on OSF. They need scikit-learn and fracridge to load (the
training environment, `training/environment.yml`), and `training/call.py` can
use them (see [Using call.py](#using-callpy)).


## Repository layout

```
dimpred/            Python package
  __init__.py       the functions below and DEFAULT_MODEL
  list_models.py, load_model.py, find_images.py, extract_features.py, predict.py, similarity.py
  __main__.py       command line tool (python -m dimpred)
  models/*.mat      the shipped models
matlab/             MATLAB version (dimpred_list_models.m, dimpred_load_model.m, ...)
training/           Philipp Kaniuth's training code (fit.py, call.py, environment.yml),
                    build_models.py and the training data (data/)
tests/              pytest tests, MATLAB tests (tests/matlab) and fixtures (tests/fixtures)
pyproject.toml, README.md, LICENSE
```


## Citation

If you use DimPred, please cite:

```
@article{Kaniuth_2025,
	author={Kaniuth, Philipp and Mahner, Florian P and Perkuhn, Jonas and Hebart, Martin N},
	title={A high-throughput approach for the efficient prediction of perceived similarity of natural objects},
	journal={eLife},
	volume={14},
	pages={RP105394},
	year={2025},
	DOI={10.7554/eLife.105394},
	url={https://doi.org/10.7554/eLife.105394},
	publisher={eLife Sciences Publications, Ltd}
}
```

The dimensions come from Hebart et al. (2020), Nature Human Behaviour (49d),
and Hebart et al. (2023), eLife (THINGS-data, 66d). The default model is a
rebuilt version of the model of Contier, O., Baker, C. I., & Hebart, M. N.
(2024). Distributed representations of behaviour-derived object dimensions in
the human visual system. Nature Human Behaviour.


## Credits

Method and training code: Philipp Kaniuth. Paper: Philipp Kaniuth, Florian P.
Mahner, Jonas Perkuhn and Martin N. Hebart. Model export with dimension means
(MATLAB): Florian Mahner. The lightweight image-to-dimensions wrapper that this
package builds on: Luca Kämmer. Contier model and fMRI analyses: Oliver
Contier. Package, MATLAB version and tests: Martin Hebart.


## License

GNU Affero General Public License v3.0 (AGPL-3.0), see [LICENSE](LICENSE).
