# DimPred

DimPred predicts, for any image, its values on the SPoSE dimensions of human
mental object representations (the 66 dimensions of Hebart et al., 2023, or
the 49 dimensions of Hebart et al., 2020), and from these dimensions the
perceived similarity between images. A deep neural network (by default the
image encoder of AligNet SigLIP2-B, Muttenthaler et al., 2025) turns each
image into a feature vector. One linear regression per dimension, trained on
the 1854 THINGS reference images, maps the features to the dimension values.
Heatmaps (RISE) show which parts of an image drive its predicted dimensions.
The method and its evaluation are described in

Kaniuth, P., Mahner, F. P., Perkuhn, J., & Hebart, M. N. (2025). A high-throughput
approach for the efficient prediction of perceived similarity of natural objects.
eLife 14:RP105394. https://doi.org/10.7554/eLife.105394

This repository contains a small Python package and a parallel MATLAB version
that apply the trained models, the trained models themselves, the training
code of the DimPred paper, and the tests. The Python and MATLAB functions have
the same names (`dimpred.predict` in Python, `dimpred_predict` in MATLAB), the
same arguments (except that the MATLAB functions that run Python take their
options in cfg, see [All functions](#all-functions)) and give the same numbers.

If you used an earlier version: the training code (`fit.py`, `call.py`) is in
[training/](training/README.md), and version 1.0.0 is the tag `v1.0.0`. What
changed in 1.1.0: [docs/changes.md](docs/changes.md).

Contents: [Installation](#installation) | [Quick start](#quick-start) |
[Models](#models) | [More](#more) | [Citation](#citation) | [Credits](#credits) |
[License](#license)

## Installation

Python 3.9 or newer. From a clone of this repository:

```
git clone https://github.com/ViCCo-Group/dimpred.git
cd dimpred
pip install -e .              # predictions and similarity from features (installs numpy and scipy)
pip install -e ".[extract]"   # in addition features and heatmaps from images (torch, open_clip_torch, timm, pillow)
pip install -e ".[test]"      # pytest, for running the tests
```

Without `extract` you can do everything except extracting features and
heatmaps from images. torch, open_clip and timm are only imported inside
`extract_features` and `rise`, so the rest of the package does not load them.
The default model downloads the weights of its network (378 MB) once into
`~/.cache/dimpred`. For offline use, download `alignet_siglip2_b.safetensors`
from the release [alignet-weights-v1](https://github.com/ViCCo-Group/dimpred/releases/tag/alignet-weights-v1)
and set `DIMPRED_ALIGNET_WEIGHTS` to its path.

MATLAB (R2016b or newer): add the folder `matlab/` to the path (`addpath`, see
[MATLAB](#matlab)) and keep it inside the repository, where its functions find
the models. Everything runs in MATLAB alone except `dimpred_extract_features`
and `dimpred_rise`, which run the Python command line tool. They need a Python
with numpy, scipy, torch, open_clip_torch, timm (1.0.15 or newer) and pillow,
in which dimpred does not have to be installed.

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

If the features do not fit the model (e.g. AligNet features and an RN50x64
model), `predict` stops with an error that names the network of the model.
Features you extracted before (same network and layer, see [Feature
extraction](docs/details.md#feature-extraction)) can be passed as an array.

Heatmaps with RISE (default model, 6000 masks: 49 s per image on the GPU of an
Apple M1 Max; `burrito.jpg` is in `tests/fixtures/images`):

```python
result = dimpred.rise("burrito.jpg")             # maps of each dimension and the relevance map
top = result["embedding"][0].argmax()            # the dimension with the largest predicted value
print(result["labels"][top])                     # food-related
from PIL import Image
from dimpred.rise import overlay
Image.fromarray(overlay(result["view"][0], result["relevance"][0])).save("burrito_relevance.png")
```

![The image as the network sees it, and its relevance map](docs/images/rise_example.png)

### MATLAB

```matlab
addpath('/path/to/dimpred/matlab')
files = dimpred_find_images('my_images');     % cell column of full paths, sorted by name
features = dimpred_extract_features(files);   % runs Python, one row per image
embedding = dimpred_predict(features);        % images x 66 dimensions (default model)
S = dimpred_similarity(embedding);            % images x images, predicted similarity
model = dimpred_load_model;                   % default model, model.labels holds the dimension names
```

The functions that run Python use the Python in `cfg.python` if given, else
the one in the environment variable `DIMPRED_PYTHON`, else `python3` (on
Windows, set `cfg.python` or `DIMPRED_PYTHON`):

```matlab
cfg.python = '/path/to/env/bin/python';   % a Python with numpy, scipy, torch, open_clip_torch, timm and pillow
cfg.device = 'cpu';                       % optional, default: cuda, then mps, then cpu
[features, files] = dimpred_extract_features(files, 'rn50x64_49d_ridge', cfg);
embedding = dimpred_predict(features, 'rn50x64_49d_ridge');
cfg.device = '';                          % heatmaps on the GPU if there is one
result = dimpred_rise('burrito.jpg', [], cfg);   % heatmaps with the default model, 6000 masks
[~, top] = max(result.embedding(1, :));   % result.labels{top} is the dimension with the largest value
imagesc(squeeze(result.relevance(1, :, :))), axis image off, colormap(jet)
```

### Command line

```
python -m dimpred my_images                           # all images in the folder, writes dimpred_predictions.csv
python -m dimpred a.jpg b.jpg --model rn50x64_49d_ridge --out predictions.mat
python -m dimpred my_images --features-only --out features.mat
python -m dimpred --features features.mat --out predictions.csv
python -m dimpred --rise burrito.jpg --png burrito_maps   # heatmaps, writes dimpred_heatmaps.mat and PNG files
```

Folders are expanded with `find_images`, and the rows of the output are in
the order in which the images were given. `--features` predicts from features
computed before (without torch); give the same `--model` as for the
extraction. `--png FOLDER` saves the relevance map and the maps of the 3
dimensions with the largest values of each image (the picture above shows the
image as the network sees it and `burrito_relevance.png`).
`python -m dimpred --help` lists all options (e.g. `--n-masks`), and
[File formats](docs/details.md#file-formats) describes the output.

### All functions

| Python | MATLAB | what it does |
|---|---|---|
| `dimpred.list_models()` | `names = dimpred_list_models` | names of the shipped models, sorted |
| `dimpred.load_model(model=None)` | `model = dimpred_load_model(model)` | load a model by name or file (default: `DEFAULT_MODEL`) |
| `dimpred.find_images(folder)` | `files = dimpred_find_images(folder)` | image files directly in a folder, sorted by name, full paths |
| `dimpred.extract_features(images, model=None, network=None, pretrained="openai", device=None, batch_size=32)` | `[features, files] = dimpred_extract_features(images, model, cfg)` | network features of image files |
| `dimpred.predict(features, model=None)` | `embedding = dimpred_predict(features, model)` | predicted dimension values |
| `dimpred.similarity(embedding, method="spose")` | `S = dimpred_similarity(embedding, method)` | predicted similarity, `"spose"` or `"dot"` |
| `dimpred.rise(images, model=None, n_masks=6000, ...)` | `result = dimpred_rise(images, model, cfg)` | heatmaps of the predicted dimensions (RISE) |

`dimpred.DEFAULT_MODEL` is `"alignet_siglip2b_66d_ridge"`. Wherever a model is
expected, you can pass nothing (default model), the name of a shipped model,
the path to a model file, or a model that was already loaded. A model is a
dict in Python and a struct in MATLAB with the fields `weights`,
`feature_mean`, `feature_scale`, `target_mean`, `labels`, `info` and `file`.
Each function explains its inputs and outputs in its help text
(`help(dimpred.predict)` in Python, `help dimpred_predict` in MATLAB).

## Models

Which model to use:

- To reproduce Contier et al. (2024): `vitb32_66d_elastic`, the rebuilt model
  of that paper (the default of 1.0.0).
- To reproduce the DimPred paper: `rn50x64_49d_ridge`, the published RN50x64
  model of the paper (its heatmaps and fMRI encoding). The code of the paper
  is in https://github.com/ViCCo-Group/dimpred_paper, its data and all 53
  models on OSF, https://osf.io/jtekq
  ([more](training/README.md#reproducing-the-dimpred-paper)).
- For the best prediction of the dimensions: the default model,
  `alignet_siglip2b_66d_ridge`.
- For heatmaps: `rn50x64_66d_ridge` (a convolutional network; of the 66d
  models, its maps were the closest to Figure 7 of the paper), or the default,
  which is much faster (6000 masks on the GPU of an Apple M1 Max: about 10 min
  per image instead of 49 s; 2000 masks still give stable maps).
- Use the same model for all images you want to compare (the values of 49d
  and 66d models cannot be mixed).

| name | network | features | dims | regression | use |
|---|---|---|---|---|---|
| `alignet_siglip2b_66d_ridge` | AligNet SigLIP2-B (`pre_logits`) | 768 | 66 | ridge | default, best prediction of the dimensions |
| `vitb32_66d_elastic` | OpenAI CLIP ViT-B/32 (`ViT-B-32-quickgelu`) | 512 | 66 | elastic net | rebuilt model of Contier et al. (2024) |
| `rn50x64_49d_ridge` | OpenAI CLIP RN50x64 (`RN50x64`) | 1024 | 49 | fractional ridge | model of the DimPred paper |
| `rn50x64_66d_ridge` | OpenAI CLIP RN50x64 | 1024 | 66 | ridge | heatmaps (convolutional network) |
| `rn50x64_66d_elastic` | OpenAI CLIP RN50x64 | 1024 | 66 | elastic net | RN50x64 66d with elastic net |

All models were trained on the 1854 THINGS reference images (the images of
the odd-one-out experiments) with the training code in
[training/](training/README.md). The ridge models of 1.1.0 use a ridge
regression that keeps the penalty chosen by cross-validation for the final
fit. `rn50x64_49d_ridge` is not a refit but the published fit of the paper.

### How well the models work on new images

Mean r per dimension (10-fold cross-validation on the 1854 training images),
and Pearson r between predicted and human similarity, averaged over 10 image
sets never used for training and over the 8 of them without 48new and 48nonref:

| model | per dimension | 10 test sets | 8 test sets |
|---|---|---|---|
| `alignet_siglip2b_66d_ridge` | 0.810 | 0.634 | 0.572 |
| `vitb32_66d_elastic` | - | 0.598 | 0.535 |
| `rn50x64_49d_ridge` | - | 0.562 | 0.496 |
| `rn50x64_66d_ridge` | 0.779 | 0.583 | 0.517 |
| `rn50x64_66d_elastic` | - | 0.578 | 0.511 |

The test sets are 48new and 48nonref (odd-one-out data of the DimPred paper),
six sets of Peterson et al. (2018), Kriegeskorte-92 and Cichy-118. AligNet's
teacher was fit to the THINGS odd-one-out data, so its per-dimension r and
its values on 48new and 48nonref (odd-one-out data, as in THINGS) are
probably too optimistic. On the 8 other sets it is also the best of the
shipped models. If you test predictions against THINGS odd-one-out data, use
`vitb32_66d_elastic`. All tables:
[docs/models.md](docs/models.md#how-well-they-predict-human-similarity).

## More

- [Images and limitations](docs/details.md#images-and-limitations), [how the predictions are computed](docs/details.md#how-the-predictions-are-computed), [similarity](docs/details.md#similarity)
- [Feature extraction](docs/details.md#feature-extraction), [image order](docs/details.md#image-order), [heatmaps (RISE)](docs/details.md#heatmaps-rise)
- [Command line](docs/details.md#command-line), [file formats](docs/details.md#file-formats), [MATLAB](docs/details.md#matlab)
- [The models](docs/models.md), [which model was used where](docs/models.md#which-model-was-used-where)
- [Training and reproducing the DimPred paper](training/README.md)
- [Tests](docs/details.md#tests), [repository layout](docs/details.md#repository-layout), [changes in 1.1.0](docs/changes.md)

## Citation

Please cite

- the DimPred paper (below),
- the paper of the dimensions: Hebart et al. (2023), eLife (66d), or Hebart
  et al. (2020), Nature Human Behaviour (49d),
- with the default model also Muttenthaler, L., Greff, K., Born, F.,
  Spitzer, B., Kornblith, S., Mozer, M. C., Müller, K.-R., Unterthiner, T.,
  & Lampinen, A. K. (2025). Aligning machine and human visual
  representations across abstraction levels. Nature 647, 349-355,
  https://doi.org/10.1038/s41586-025-09631-6,
- with `vitb32_66d_elastic` also Contier, O., Baker, C. I., & Hebart, M. N.
  (2024). Distributed representations of behaviour-derived object
  dimensions in the human visual system. Nature Human Behaviour 8,
  2179-2193, https://doi.org/10.1038/s41562-024-01980-y,
- with heatmaps also Petsiuk, V., Das, A., & Saenko, K. (2018). RISE:
  Randomized input sampling for explanation of black-box models. BMVC.

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

## Credits

Method and training code: Philipp Kaniuth. Model export with dimension means
(MATLAB) and the heatmaps of the DimPred paper: Florian Mahner. The wrapper
this package builds on: Luca Kämmer. Contier model: Oliver Contier. AligNet:
Lukas Muttenthaler and colleagues. Package, MATLAB version, AligNet port,
heatmaps and tests: Hebartlab.

## License

GNU Affero General Public License v3.0 (AGPL-3.0), see [LICENSE](LICENSE).
The AligNet weights that the default model downloads come with the license
of the AligNet models ([training/alignet](training/alignet/README.md)).
