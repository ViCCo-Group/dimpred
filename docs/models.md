# The models

Each model is one file in `dimpred/models` (see
[Model files](details.md#model-files)) and has a network, an embedding (49 or
66 SPoSE dimensions) and a regression. Which model to use is in the
[README](../README.md#models).

| model | network | features | dims | regression |
|---|---|---|---|---|
| `alignet_siglip2b_66d_kernel` (default) | AligNet SigLIP2-B (Muttenthaler et al., 2025), `pre_logits` | 768 | 66 | ridge + local kernel |
| `alignet_siglip2b_66d_ridge` | AligNet SigLIP2-B, `pre_logits` | 768 | 66 | ridge |
| `vitb32_66d_elastic` | OpenAI CLIP ViT-B/32 (open_clip `ViT-B-32-quickgelu`) | 512 | 66 | elastic net |
| `rn50x64_49d_ridge` | OpenAI CLIP RN50x64 (open_clip `RN50x64`) | 1024 | 49 | fractional ridge |
| `rn50x64_66d_ridge` | OpenAI CLIP RN50x64 | 1024 | 66 | ridge |
| `rn50x64_66d_elastic` | OpenAI CLIP RN50x64 | 1024 | 66 | elastic net |

- `alignet_siglip2b_66d_kernel` predicts the dimensions and similarity
  best (tables below): the ridge of `alignet_siglip2b_66d_ridge` plus a
  local kernel, fit together ([details](details.md#how-the-predictions-are-computed)),
  and the settings of the close pairs of `similarity`. Its network is the
  PyTorch port in `dimpred/alignet.py`
  ([training/alignet](../training/alignet/README.md)).
- `alignet_siglip2b_66d_ridge` was the default model of dimpred 1.1.0.
- `vitb32_66d_elastic` is the model of Contier et al. (2024), rebuilt with
  the corrected 66d embedding (see [below](#which-model-was-used-where)). It
  was the default model of dimpred 1.0.0.
- `rn50x64_49d_ridge` is the model of the DimPred paper: its published
  weights, which reproduce its published predictions.
- `rn50x64_66d_ridge` uses a convolutional network, e.g. for the heatmaps
  of `dimpred.rise` (RISE). In dimpred 1.0.0 it was fitted with the
  fractional ridge of the paper; since 1.1.0 it is fitted with the ridge
  with a directly chosen penalty.
- "ridge" chooses the penalty of each dimension by cross-validation and
  keeps it for the final fit. "fractional ridge" is the regression of the
  DimPred paper. "elastic net" is scikit-learn's ElasticNetCV.

## How the models were trained

All models were trained on the 1854 THINGS reference images (the images of
the odd-one-out experiments) with the training code of the DimPred paper in
`training/` and the settings of `training/call.py`: 3-fold cross-validation
repeated 3 times for choosing the regularization, `random_state` 0. The
features are those of `dimpred.extract_features`: for the CLIP models
(weights `pretrained="openai"`) the output of the image encoder (open_clip's
image embedding, not normalized), for AligNet its `pre_logits`.
`rn50x64_49d_ridge` is not a refit but the published fit of the paper. The
49d and 66d models predict different embeddings with different dimensions,
so their values cannot be mixed. The labels of the dimensions are stored in
each model. How the models are built, and why the ridge of 1.1.0 differs from
the fractional ridge of the paper: [training/README.md](../training/README.md).

## How well they predict human similarity

Pearson r between predicted and human similarity (values below the
diagonal), on image sets never used for training:

| model | 48new | 48nonref | P-animals | P-automobiles | P-fruits | P-furniture | P-various | P-vegetables | K-92 | C-118 | mean 10 | mean 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `alignet_siglip2b_66d_kernel`, with the close pairs | 0.906 | 0.885 | 0.723 | 0.698 | 0.542 | 0.539 | 0.781 | 0.602 | 0.865 | 0.842 | 0.738 | 0.699 |
| `alignet_siglip2b_66d_kernel` | 0.909 | 0.886 | 0.457 | 0.573 | 0.394 | 0.447 | 0.767 | 0.477 | 0.838 | 0.818 | 0.657 | 0.596 |
| `alignet_siglip2b_66d_ridge` | 0.900 | 0.863 | 0.485 | 0.552 | 0.345 | 0.380 | 0.759 | 0.421 | 0.833 | 0.798 | 0.634 | 0.572 |
| `vitb32_66d_elastic` | 0.870 | 0.825 | 0.365 | 0.506 | 0.346 | 0.410 | 0.713 | 0.378 | 0.812 | 0.751 | 0.598 | 0.535 |
| `rn50x64_49d_ridge` | 0.848 | 0.810 | 0.186 | 0.541 | 0.362 | 0.331 | 0.691 | 0.313 | 0.771 | 0.770 | 0.562 | 0.496 |
| `rn50x64_66d_ridge` | 0.866 | 0.826 | 0.281 | 0.530 | 0.339 | 0.384 | 0.705 | 0.345 | 0.788 | 0.767 | 0.583 | 0.517 |
| `rn50x64_66d_elastic` | 0.868 | 0.829 | 0.278 | 0.516 | 0.348 | 0.352 | 0.728 | 0.303 | 0.794 | 0.769 | 0.578 | 0.511 |

- 48new (48 new objects) and 48nonref (48 new images of THINGS objects):
  human odd-one-out data of the DimPred paper.
- P: the six sets of 120 images with similarity ratings of Peterson et al.
  (2018).
- K-92 (Mur et al., 2013) and C-118 (Cichy et al., 2019): dissimilarity from
  arrangements, compared with the Euclidean distance of the predictions, as
  in the paper. All other sets use the SPoSE similarity of the predictions.
- mean 8: the 8 sets without 48new and 48nonref.
- "with the close pairs": `similarity(embedding, features=features)`, the
  recommended way since 1.2.0. It helps most within the Peterson categories
  (mean of the five: 0.621 instead of 0.470 without, 0.437 for
  `alignet_siglip2b_66d_ridge`). All settings were chosen on THINGS, not on
  these sets.
- Homogeneous sets such as the Peterson fruits remain difficult for all
  models.

**AligNet and THINGS.** AligNet was trained with a teacher network that was
fitted to the human odd-one-out judgments of THINGS, the same data the
SPoSE dimensions come from. Its numbers on 48new and 48nonref (odd-one-out
judgments, as in THINGS) and per dimension are therefore probably too
optimistic. On the 8 other sets it is also best: +0.037 over
`vitb32_66d_elastic` (mean 8), better on 6 of the 8 sets (not on the
Peterson fruits and furniture). Of the networks we compared, AligNet
DINOv2-B predicted the 8 other sets slightly better (+0.028) but the
dimensions less well; we use SigLIP2-B. If you test predictions against
THINGS odd-one-out data, use `vitb32_66d_elastic`.

**Per dimension**, cross-validated on the 1854 training images (mean r over
the 66 dimensions, 10 folds): 0.839 for `alignet_siglip2b_66d_kernel`
(better than the ridge for all 66 dimensions; 0.784 against 0.750 when whole
THINGS categories are left out of training), 0.810 for `alignet_siglip2b_66d_ridge` and
0.779 for `rn50x64_66d_ridge` (0.758 with the fractional ridge of 1.0.0;
on the 10 test sets the fractional ridge was better, 0.595 against 0.583).
`rn50x64_49d_ridge`: 0.717, the mean over its 49 dimensions (the fractional
ridge of the paper fitted again in each fold), so not directly comparable
with the 66d values. For the elastic net models there is no such value.

## Which model was used where

| where | network | dims | regression |
|---|---|---|---|
| DimPred paper, all benchmarks | 53 networks/layers | 49d | fractional ridge |
| DimPred paper, heatmaps (Fig. 7) | RN50x64 | 49d | fractional ridge |
| DimPred paper revision, fMRI encoding (Fig. 8) | RN50x64 | 49d | fractional ridge |
| Contier et al. 2024, THINGS-fMRI analyses | CLIP ViT-B/32 | 66d | elastic net (the paper text says ridge) |
| Contier et al. 2024, BOLD5000 replication (revision) | CLIP RN50 | 66d | ridge |
| NIH image set (2024) | RN50x64 | 49d and 66d | ridge |
| 49d predictions for all 26,107 THINGS images on OSF | RN50x64, CLIP RN50 | 49d | ridge |

Here "ridge" means the ridge regression used at that time, not the ridge
of the current models.

The 66d predictions published with Contier et al. (2024) (`data/66d` in
github.com/ViCCo-Group/dimension_encoding) were made with the 66d embedding
before its correction in January 2023. Predictions with the corrected
embedding correlate with them at r = 0.996. `vitb32_66d_elastic` uses the
corrected embedding; its predictions correlate with the published ones at
a median per-dimension r of 0.998.

The code of the paper is in https://github.com/ViCCo-Group/dimpred_paper,
its data and all 53 fitted models are on OSF, https://osf.io/jtekq
(see [Reproducing the DimPred paper](../training/README.md#reproducing-the-dimpred-paper)).
