# Details

The help of each function explains all its arguments (`help(dimpred.predict)`,
`help dimpred_predict`, `python -m dimpred --help`). This page collects what
is useful to know beyond that. The models and how well they work are in
[models.md](models.md), the training in [training/README.md](../training/README.md).

Contents:
[All functions](#all-functions) |
[Images and limitations](#images-and-limitations) |
[How the predictions are computed](#how-the-predictions-are-computed) |
[Model files](#model-files) |
[Similarity](#similarity) |
[Feature extraction](#feature-extraction) |
[Image order](#image-order) |
[Heatmaps (RISE)](#heatmaps-rise) |
[Command line](#command-line) |
[File formats](#file-formats) |
[MATLAB](#matlab) |
[Tests](#tests) |
[Repository layout](#repository-layout)

## All functions

| Python | MATLAB | what it does |
|---|---|---|
| `dimpred.list_models()` | `names = dimpred_list_models` | names of the shipped models, sorted |
| `dimpred.load_model(model=None)` | `model = dimpred_load_model(model)` | load a model by name or file (default: `DEFAULT_MODEL`) |
| `dimpred.find_images(folder)` | `files = dimpred_find_images(folder)` | image files directly in a folder, sorted by name, full paths |
| `dimpred.extract_features(images, model=None, network=None, pretrained="openai", device=None, batch_size=32)` | `[features, files] = dimpred_extract_features(images, model, cfg)` | network features of image files |
| `dimpred.predict(features, model=None)` | `embedding = dimpred_predict(features, model)` | predicted dimension values |
| `dimpred.similarity(embedding, method="spose", features=None, model=None)` | `S = dimpred_similarity(embedding, method, features, model)` | predicted similarity, `"spose"` or `"dot"`; with the features also the close pairs |
| `dimpred.rise(images, model=None, n_masks=6000, ...)` | `result = dimpred_rise(images, model, cfg)` | heatmaps of the predicted dimensions (RISE) |

The Python and MATLAB functions take the same arguments, except that the
MATLAB functions that run Python take their options in `cfg`, and give the
same numbers. `dimpred.DEFAULT_MODEL` is `"alignet_siglip2b_66d_kernel"`.
Wherever a model is expected, you can pass nothing (default model), the name
of a shipped model, the path to a model file, or a model that was already
loaded: a dict in Python and a struct in MATLAB with the fields `weights`,
`feature_mean`, `feature_scale`, `target_mean`, `labels`, `info` and `file`
(and, for a model with a local kernel, the fields of its kernel part and of
the close pairs, see [Model files](#model-files)).

## Images and limitations

The models were trained on the 1854 THINGS reference images, photographs of
single objects, one for each object concept, and tested on the image sets in
[models.md](models.md#how-well-they-predict-human-similarity). Some things to
keep in mind:

- Photos of single objects, as in THINGS, work best.
- The default model resizes the whole image to a square, so the aspect ratio
  of a non-square image is not kept. The CLIP models scale the image and crop
  the central square, so the network does not see the parts outside it (see
  [Feature extraction](#feature-extraction)).
- Images are converted to RGB, which drops the alpha channel. Transparent
  areas of a PNG then get the colors stored there, often black. Put such
  images on a background first.
- Predicted similarity within one homogeneous category is much less accurate
  than across categories. With the default model, r = 0.35 to 0.55 with human
  similarity for the Peterson sets of animals, automobiles, fruits, furniture
  and vegetables, and 0.76 to 0.90 for the mixed sets.
- SPoSE similarity depends on the other images in the set (see
  [Similarity](#similarity)).
- The values of 49d and 66d models cannot be mixed, so use the same model for
  all images you want to compare.

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

- `features` (images x features) are the features of the network of the
  model (see [Feature extraction](#feature-extraction)).
- `feature_mean` and `feature_scale` are the mean and standard deviation of
  the features of the 1854 training images. New features are z-scored with
  these and not with the statistics of your own images, so the prediction of
  an image does not depend on which other images you pass.
- `weights` (features x dims) hold one regression per dimension.
- `target_mean` is the mean of each dimension across the 1854 training
  images. The regressions were fit on centered dimension values, so the mean
  has to be added back.
- Values below 0 are set to 0, because SPoSE dimensions are non-negative.

**The local kernel (default model since 1.2.0).** `alignet_siglip2b_66d_kernel`
adds a correction from the training images that are similar to each image
in the network:

```
cos = (features ./ vecnorm(features, 2, 2)) * kernel_features'
embedding = max(z * weights + exp((cos - 1) / kernel_tau) * kernel_coefficients + target_mean, 0)
```

with `z` the z-scored features as above and `kernel_features` the features
of the 1854 training images, scaled to length 1. The correction is a
similarity-weighted sum of how much the model misses the training images
near the image. The ridge part and the kernel part were fit together, in
closed form (a Gaussian process with the covariance `beta *
exp((cos - 1) / tau)`; `training/fit.py`, `kernel_ridge_fit`), with tau =
0.5 and beta = 10 chosen on THINGS out of fold. Compared with the ridge
alone (`alignet_siglip2b_66d_ridge`), every dimension is predicted better
(mean r 0.839 instead of 0.810, the color dimensions 0.754 instead of 0.734),
also for categories of concepts left out of training (0.784 instead of
0.750). Each image is still predicted on its own.

Some earlier, unofficial wrappers left out `target_mean`. This makes the
predictions too small and sets about 60% of them to zero, and the correlation
of predicted with human similarity for the 48nonref images drops by about
0.05 (from 0.81 to 0.76 for the paper model, from 0.86 to 0.81 for
`alignet_siglip2b_66d_ridge`). Predictions made with such a wrapper should be
made again. A quick check of any implementation of a model without a kernel:
features equal to `feature_mean` have to give `target_mean` as prediction,
e.g. `dimpred.predict(model["feature_mean"], model)`.

### Model files

`dimpred/models/<name>.mat` are plain MATLAB v5 .mat files (written with
`scipy.io.savemat`) that the Python and the MATLAB version both read. They
can also be used without this package, with MATLAB's `load` or with
`scipy.io.loadmat`. They hold

| variable | size | meaning |
|---|---|---|
| `weights` | n_features x n_dims | regression weights |
| `feature_mean`, `feature_scale` | 1 x n_features | mean and std of the features of the 1854 training images |
| `target_mean` | 1 x n_dims | mean of each dimension in the training images |
| `labels` | n_dims x 1 cell | names of the dimensions |
| `info` | struct | the text fields name, network, pretrained, layer, preprocessing, embedding, regression, training_images, source, note and created, and the numbers n_features and n_dims |
| `kernel_features` | n_train x n_features | only with a local kernel: features of the training images, length 1 (single precision) |
| `kernel_coefficients` | n_train x n_dims | only with a local kernel |
| `kernel_tau` | 1 x 1 | only with a local kernel: width of the kernel |
| `close_pairs_weight`, `close_pairs_threshold` | 1 x 1 | only for the close pairs of `similarity` |

A file in this format can be given as `model` to all functions. A file
without one of the first six variables gives an error, it is never filled
with defaults. The kernel variables and the close-pair variables are
optional, but each group has to be complete.

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
subtracted, so large embedding values do not give inf or nan. Since the third
object comes from the images you pass, the similarity of two images depends a
little on the other images in the set, and at least 3 images are needed. The
mean of all values off the diagonal is exactly 1/3 (a quick check of any
implementation).

**Close pairs (since 1.2.0).** With the features of the images,
`similarity(embedding, features=features)` (MATLAB:
`dimpred_similarity(embedding, [], features)`) adds the network's own
similarity for pairs of images that are very close in the network. The dot
products above are replaced by

```
D[i, j] = e_i.e_j + close_pairs_weight * max(0, cos(f_i, f_j) - close_pairs_threshold)
```

with `cos` the cosine of the features. The weight (8) and the threshold
(0.448, the 90th percentile of the cosines between the 1854 training images,
so only the closest pairs change) come from the model (`model=`, default:
the default model; models without these settings give an error). The
dimensions describe the differences between kinds of objects well but the
fine differences within a kind less well, and the network knows these. On
the 8 independent test sets of the benchmark, the correlation with human
similarity is 0.699 with the close pairs and 0.596 without (0.572 in 1.1.0);
within the five Peterson categories 0.621 instead of 0.470. This is the
recommended way to predict similarity. Without the features, `similarity`
works as before.

`similarity(embedding, "dot")` (MATLAB: `dimpred_similarity(embedding, 'dot')`)
gives the plain dot product `embedding @ embedding.T` (with features: `D`
above), which does not depend on the other images. In the benchmarks, Kriegeskorte-92 and Cichy-118 were
compared with the Euclidean distance between the predicted embeddings, which
you can compute with `scipy.spatial.distance.pdist` in Python or `pdist` in
MATLAB (Statistics and Machine Learning Toolbox).

The computing time of the SPoSE similarity grows with the cube of the number
of images: a few seconds for 1000 images, about 1000 times as long for 10
times as many. For many thousands of images, use `"dot"`, or compute the
similarity for subsets (the values then depend on the subset).

## Feature extraction

The network comes from the model (`info.network`), so give the same model to
`extract_features` and `predict`. Each image is opened with PIL and converted
to RGB with 8 bits per channel (grayscale images get three equal channels,
16-bit grayscale images are scaled to 8 bit, and images with 32-bit pixels
give an error). The images go through the network in float32 without
gradients, in batches of `batch_size` images. The result is a float32 array
(images x features), not normalized. The device is `cuda` if available, else
`mps` (Apple GPU), else `cpu`.

- AligNet SigLIP2-B (default model): the whole image is resized to 224 x 224
  (no crop, the aspect ratio is not kept) with the bicubic interpolation of
  OpenCV (INTER_CUBIC), rewritten in numpy, values 0 to 1. It needs timm
  1.0.15 or newer. The features are `pre_logits` (768), the output of the
  attention pooling head. The weights (378 MB) are downloaded on first use
  into `~/.cache/dimpred` and checked with their sha256. Without internet
  access, download `alignet_siglip2_b.safetensors` from the release
  [alignet-weights-v1](https://github.com/ViCCo-Group/dimpred/releases/tag/alignet-weights-v1)
  and set the environment variable `DIMPRED_ALIGNET_WEIGHTS` to its path. The
  network is our PyTorch port of the TensorFlow model of the AligNet authors
  (`dimpred/alignet.py`); its features differ from the TensorFlow features of
  the 1854 reference images by at most 3e-4 (median 1e-5,
  [training/alignet](../training/alignet/README.md)).
- CLIP networks: open_clip with the weights `pretrained="openai"`, the
  preprocessing of the network (RN50x64: resize the shortest side to 448 px,
  bicubic, center crop 448 x 448; ViT-B/32: the same with 224), and the output
  of `model.encode_image`. The first time a network is used, open_clip
  downloads its weights (about 0.6 GB for ViT-B/32, 2.5 GB for RN50x64).
- For OpenAI's ViT models the open_clip configuration with `-quickgelu` has to
  be used (`ViT-B-32-quickgelu`). The original CLIP ViTs use QuickGELU, and
  the plain `ViT-B-32` configuration uses GELU with the same weights, which
  gives different features (per-image r of about 0.98 with the original
  features instead of 1.0). The models store the right name in
  `info["network"]`, and `extract_features` uses it.
- open_clip's warning about a "QuickGELU mismatch" for RN50x64 does not
  matter: QuickGELU is only used in the text part, which dimpred does not use.
- open_clip `RN50x64` and `ViT-B-32-quickgelu` reproduce the features of the
  DimPred paper and of Contier et al. (2024) (r = 1.000000, largest
  difference 3e-5 on the cpu), so thingsvision is not needed.
- Devices (cuda, mps, cpu) give slightly different features, differences of
  about 1e-4. The tests allow a per-image correlation of 0.99999 and a
  difference of 2e-3.
- Speed on an Apple M1 Max, per 1000 images: AligNet 6.4 s on the GPU (mps)
  and 33 to 38 s on the cpu (8 threads), plus 10 to 12 s for reading and
  resizing the images; CLIP ViT-B/32 2.0 s (mps) and 8.7 s (cpu).
- Features from your own code have to be computed in exactly this way.
- torch, open_clip and timm are only imported inside `extract_features` and
  `rise`, so the rest of the package works without them.

### Image order

The rows of the features and predictions are always in the order of the files
you give. `extract_features` does not take folders for this reason; use
`find_images` to list a folder. `find_images` returns the image files directly
in the folder (not in subfolders) with the extensions .jpg, .jpeg, .png,
.bmp, .tif, .tiff and .webp in any case, without hidden files, sorted by file
name with a plain string sort (the same order as `sort` in MATLAB). If your
images belong to a list in a specific order (e.g. the rows of a data matrix),
pass the files in that order and do not sort them.

## Heatmaps (RISE)

`rise(images, model)` (MATLAB `dimpred_rise`, command line `--rise`) gives,
for each image, a map of each dimension and a relevance map, with RISE
(Petsiuk, Das & Saenko, 2018, BMVC), as in Figure 7 of the DimPred paper.

- Masks: as `generate_masks` of the original RISE code
  (github.com/eclique/RISE): a grid of 8 x 8 cells (`grid`), each kept
  with probability 0.1 (`p`), upsampled bilinearly to 9 cells of
  ceil(input size / 8) pixels and cut to the input size at a random
  shift. 6000 masks (`n_masks`), the same masks for every image, drawn
  with seed 0 (`seed`). The grid, p and the number of masks are the
  settings of the paper; the masks are those of the original code after
  np.random.seed(0).
- The mask multiplies the network input image (values 0 to 1) before the
  normalization of the network, so masked pixels are black. The masked
  images then go through the network as in `extract_features`, and
  through `predict`.
- Map of a dimension: the predictions of the masked images, weighted by
  the masks. With `normalization="pixel"` (default), each pixel is divided
  by the sum of the masks at this pixel, which gives the weighted average.
  With `"original"`, all pixels are divided by n_masks x p, as in the
  original code. The masks do not cover all pixels equally often: with
  p = 0.1, their sum varies by a few percent over the image, about as much
  as the effect of the image content. With `"original"` this pattern is
  part of every map (the same for every image and model), with `"pixel"`
  it is removed.
- Relevance map: the dimension maps, weighted by the predicted values of
  the image without masks and divided by their sum
  (`compute_aggregate_saliency` of the code of the paper).
- The maps cover what the network sees: for the CLIP models the central
  square of the image (after resizing the shorter side to 448 px for
  RN50x64, 224 px for ViT-B/32), for AligNet the whole image, squeezed to
  a square. They are computed at the input size of the network and then
  resized to 224 x 224 (`map_size`). `view` is this image at the same size.
- Model: of the 66d models we compared, the maps of `rn50x64_66d_ridge` (a
  convolutional network) were the closest to Figure 7 of the paper. To
  replicate Figure 7, use `rn50x64_49d_ridge`, the model of the paper,
  whose maps were just as close. Any model works, AligNet is the fast
  option.
- Time on an Apple M1 Max (GPU, mps) with 6000 masks: about 10 min per
  image with RN50x64 and 49 s with AligNet; the cpu is much slower. The
  time is proportional to the number of masks. With the default
  normalization, 2000 masks still give stable maps: for the four images of
  Figure 7, the relevance maps of two separate sets of 3000 masks
  correlated with r = 0.95 to 0.99 (with `"original"` only 0.22 to 0.76).
- Memory: most of it is used by the network, which gets `batch_size`
  masked images at once (default 32; on an Apple M1 Max with mps, a peak
  of about 11 GB for RN50x64 and 2 GB for AligNet). The result needs about
  13 MB per image. A .mat file holds at most 2 GB per variable (162 images
  with 66 dimensions, 218 with 49), use .npz for more. `dimpred_rise` in
  MATLAB always gets a .mat file from Python, so for more images call it in
  a loop.
- Output: `relevance` (images x 224 x 224), `dimension_maps` (images x
  dimensions x 224 x 224), `embedding` (the predictions of the images
  without masks, the same as `predict(extract_features(...))`), `labels`,
  `files`, `model`, `view` (images x 224 x 224 x 3, uint8) and `settings`.
- `--png FOLDER` (MATLAB `cfg.png`) saves, for each image, the relevance map
  (`<name>_relevance.png`) and the maps of the 3 dimensions with the largest
  predicted values (`<name>_top<rank>_dim<k>.png`, k counts the dimensions
  from 1), on top of `view`, colored as in the paper (jet, 40% map and 60%
  image). In Python: `from dimpred.rise import overlay`.

## Command line

```
python -m dimpred IMAGE [IMAGE ...] [--model NAME] [--out FILE] [--features-only] [--device DEV] [--batch-size N]
python -m dimpred --features FILE.mat [--model NAME] [--out FILE]
python -m dimpred --rise IMAGE [IMAGE ...] [--model NAME] [--out FILE] [--n-masks N] [--png FOLDER] [--device DEV] [--batch-size N]
```

IMAGE can be files or folders. Folders are expanded to their images with
`find_images`, files stay in the given order, and the rows of the output are
in the order in which the images were given. `--model` takes the name of a
shipped model or the path of a model file (default:
`alignet_siglip2b_66d_kernel`). The second form predicts from features that
were computed before and does not need torch. Give the same `--model` as for
the extraction: the model name stored in a file written with `--features-only`
is not read. The third form saves the heatmaps of `rise` (see
[Heatmaps (RISE)](#heatmaps-rise)); `--png FOLDER` also saves PNG files of
the relevance map and of the 3 dimensions with the largest values.

### File formats

- Output of the command line tool, .csv (the default is
  `dimpred_predictions.csv`): header `image,<label 1>,...,<label n>` and one
  row per image. With `--features-only` the header is
  `image,feature_1,...,feature_p`.
- Output of the command line tool, .mat: the variables `files` (cell),
  `labels` (cell), `model` (name of the model) and `embedding` (images x
  dims), or with `--features-only` `features` (images x features) instead of
  `embedding`.
- Output of `--rise`, .mat (the default is `dimpred_heatmaps.mat`) or .npz:
  the variables of [Heatmaps (RISE)](#heatmaps-rise); in the .npz file,
  `settings` is a JSON text.
- Input for `--features`: a .mat file with `features` (images x features,
  double or single) and optionally `files` (cell array of names; without it,
  the image column holds the row numbers). A file written with
  `--features-only` can be used directly. From MATLAB, save it in the
  format `-v7` (e.g. `save('features.mat', 'features', 'files', '-v7')`),
  because scipy cannot read `-v7.3` files.
- Model files: see [Model files](#model-files).

## MATLAB

The MATLAB version needs R2016b or newer. The functions find the model files
in `../dimpred/models` relative to their own folder, so `matlab/` has to stay
inside the repository. All functions run in MATLAB alone except
`dimpred_extract_features` and `dimpred_rise`, which run `python -m dimpred`
(with `--features-only` or `--rise`) and read the result. The Python is
`cfg.python`, else the environment variable `DIMPRED_PYTHON`, else `python3`
(on Windows, set `cfg.python` or `DIMPRED_PYTHON`). Both functions put the
repository on the Python path, so Python and MATLAB use the same code and
models. `setenv` in MATLAB is passed on to Python, e.g. for
`DIMPRED_ALIGNET_WEIGHTS`. A model struct reaches Python as it is, also if it
was changed or made by hand. Unlike `extract_features` in Python,
`dimpred_extract_features` has no option for another network: the network
always comes from the model.

## Tests

The tests were written before the package code, and they compare dimpred with
numbers that do not come from dimpred: the published results of the DimPred
paper (features and predictions of the paper model, predictions for the CC0
test images), human similarity data, an independent extraction of the
features (open_clip directly, and TensorFlow for AligNet), the ridge fits of
our comparison of 15 networks (2026/10/01, a separate implementation), and an
independent implementation of the similarity.

```
python -m pytest tests -m "not slow"                    # without the slow ones, a few seconds
python -m pytest tests                                  # all tests
matlab -batch "cd('tests/matlab'); run_dimpred_tests"   # the MATLAB tests, from the repository folder
```

The slow tests extract features with torch, compute heatmaps with the real
networks, run MATLAB, train on the 1854 reference images or measure the speed
of the similarity. Tests that need something that is not installed are
skipped:

- torch and open_clip (feature extraction and heatmaps)
- MATLAB, found in the environment variable `DIMPRED_MATLAB` (full path of
  the matlab program), then in `/Applications/MATLAB_*/bin/matlab`, then on
  the system path
- the AligNet weights (`DIMPRED_ALIGNET_WEIGHTS`, the tests never download
  them)
- scikit-learn and fracridge for the training tests, and scikit-image for the
  comparison with the masks of the original RISE code
- optional large files that are not in the repository (see
  [training/README.md](../training/README.md#tests-of-the-training-code))

`run_dimpred_tests('fast')` leaves out the MATLAB tests that start Python
(feature extraction and heatmaps). These tests need `DIMPRED_PYTHON`
(`setenv('DIMPRED_PYTHON', '/path/to/python')`); without it they are skipped.

What the tests check, in short:

- the shipped models give the expected predictions on 168 test images
  (tolerance 1e-10), and `rn50x64_49d_ridge` gives the published predictions
  of the paper (1e-10)
- the numbers in the model files: `target_mean`, the feature scaling of the
  RN50x64 models (that of the paper), the labels, and the weights of the two
  ridge models (those of our comparison of 15 networks)
- each part of the formula of `predict`, and mistakes of earlier versions
  (no `target_mean`, wrong z-scoring)
- `similarity` against examples computed by hand and a slow reference that
  follows the definition
- feature extraction of the three CC0 images against reference features
  (open_clip, and TensorFlow for AligNet), the ViT GELU/QuickGELU mistake and
  the order of the rows
- `rise`: the masks against those of the original RISE code, both
  normalizations, the relevance map, the command line tool and the PNG files
- correlation of predicted and human similarity for the 48nonref images
  (e.g. r = 0.810 for the paper model, 0.863 for the default model)
- the command line tool, the package without torch, the MATLAB functions
  against the same numbers and against the Python functions
- the training code ([training/README.md](../training/README.md#tests-of-the-training-code))

The fixtures are in `tests/fixtures`:

- `reference_data.mat`: the RN50x64, ViT-B/32 and AligNet features of 168
  images (48nonref and the 120 Peterson animals), the published predictions
  of the paper, the expected predictions of each shipped model, the human
  similarity of the 48nonref images, and reference features and published
  predictions of three CC0 images
- `images/`: three THINGSplus CC0 images, free to redistribute
- `benchmark_ridge_fits.mat`: the norm and sum of the weights of each
  dimension of the two ridge models, as fitted in our comparison of 15
  networks
- `philipp_original_ridge_rn50x64_66d.mat`: the weights and selected
  fractions of the original training code for the fractional ridge of
  RN50x64 and the 66d embedding

`tests/fixtures/make_fixtures.py` documents how they were made. It needs data
that are not part of the repository and only has to be run again if the
shipped models change.

## Repository layout

```
dimpred/            Python package
  __init__.py       the functions below and DEFAULT_MODEL
  list_models.py, load_model.py, find_images.py, extract_features.py, predict.py, similarity.py, rise.py
  alignet.py        the network of the default model (AligNet SigLIP2-B in PyTorch)
  __main__.py       command line tool (python -m dimpred)
  models/*.mat      the shipped models
matlab/             MATLAB version (dimpred_list_models.m, dimpred_load_model.m, ..., dimpred_rise.m)
docs/               these pages
training/           the training code of the DimPred paper (fit.py, call.py, environment.yml),
                    build_models.py, the training data (data/) and the AligNet conversion (alignet/)
tests/              pytest tests, MATLAB tests (tests/matlab) and fixtures (tests/fixtures)
pyproject.toml, README.md, LICENSE
```

For scripts that imported `dimpred.fit` or `dimpred.call` before 1.0.0, see
[changes.md](changes.md#100).
