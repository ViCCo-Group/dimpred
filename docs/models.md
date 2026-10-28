# The models

Each model is one file in `dimpred/models` (see
[Model files](details.md#model-files)) and has a network, an embedding (49 or
66 SPoSE dimensions of objects, or the 36 material dimensions, see
[Material dimensions](#material-dimensions)) and a regression. Which model to
use is in the [README](../README.md#models).

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

## Material dimensions

Since 1.3.0, three models predict the 36 material dimensions of Schmidt,
Hebart, Schmid & Fleming (2025, PNAS): the SPoSE embedding of the 600 STUFF
images (200 materials x 3 images), learned from 1.87 million triplet 2AFC
judgments (which of two images is more similar to a reference). Their
similarity is `similarity(embedding, method="2afc")`
([details](details.md#similarity)).

| model | network | features | dims | regression |
|---|---|---|---|---|
| `alignet_siglip2b_36d_kernel` | AligNet SigLIP2-B, `pre_logits` | 768 | 36 | ridge + local kernel |
| `alignet_siglip2b_36d_ridge` | AligNet SigLIP2-B, `pre_logits` | 768 | 36 | ridge |
| `rn50x64_36d_ridge` | OpenAI CLIP RN50x64 | 1024 | 36 | kernel ridge of the paper |

- `alignet_siglip2b_36d_kernel` predicts best. Penalties, tau (0.5) and
  beta (10) were chosen in inner folds of whole materials. It has no
  close-pair settings: on STUFF, the close pairs did not improve the
  similarity.
- `rn50x64_36d_ridge` is the model of the paper for new images, rebuilt
  from its MATLAB code (lambda 1000, then a scale factor per dimension). It
  gives the published predictions of the 48 out-of-sample images (within
  2e-6).
- All three were trained on the 600 STUFF images with
  `training/build_material_models.py`
  ([training/README.md](../training/README.md#the-material-dimensions-130)).

How well they predict (2026/10/26). The first four columns are
cross-validated on the 600 images, leaving out one material (its 3 images)
at a time: the mean r per dimension, the r between the predicted similarity
of all 600 x 600 pairs and that of the human embedding, the r with the fully
sampled human similarity of 60 of the images, and the share of 174,000
held-out triplets predicted correctly. The last column is the r with the
human similarity of 48 new photos of the STUFF materials (the
out-of-sample images of the paper), for the models trained on all 600
images. All similarities with `method="2afc"`.

| model | per dimension | 600 x 600 | 60 images | triplets | 48 new images |
|---|---|---|---|---|---|
| `alignet_siglip2b_36d_kernel` | 0.820 | 0.896 | 0.856 | 69.9% | 0.842 [0.787, 0.882] |
| `alignet_siglip2b_36d_ridge` | 0.806 | 0.887 | 0.851 | 69.6% | 0.835 |
| `rn50x64_36d_ridge` (paper) | 0.769 | 0.857 | 0.838 | 68.9% | 0.818 |
| SigLIP2-B/16, ridge + local kernel (not shipped) | 0.799 | 0.885 | 0.861 | 69.5% | 0.826 |
| AligNet SigLIP2-B alone (its cosine) | - | 0.765 | 0.752 | 66.6% | 0.765 |
| the human embedding itself | 1 | 1 | 0.943 | 71.9% | - |
| noise ceiling | - | - | 0.984 | 73.8% | 0.975 |

The 95% interval of the 48 new images comes from bootstrapping images.
Compared with the model of the paper, `alignet_siglip2b_36d_kernel` is
better on 34 of the 36 dimensions; on the human similarity of the 48 and
the 60 images it is better by about 0.02 (paired intervals -0.005 to 0.060
and -0.012 to 0.048). Leaving out single images instead of whole materials
(new photos of known materials), the mean r per dimension is 0.851 for the
kernel model and 0.836 for the ridge. With the similarity of the paper's
code (only one of the two images as the reference), the paper model gives
0.788 for the 48 images, the published r = 0.79.

On objects and reachspaces, the material models predict human similarity
much worse than their network alone ([Other domains](#other-domains)).

On renderings of 100 materials on one shape (Lagunas et al., 2019, triplets
with a reference), the 2afc similarity of the predicted dimensions was close
to chance, while the Euclidean distance between them agreed with the
majority answer in 72% of the triplets (`alignet_siglip2b_36d_kernel`),
about as often as AligNet's own similarity (73%).

## Other domains

How well the models predict human similarity in domains they were not
trained for (2026/10/28). Similarities as in the sections above (SPoSE for
the object models, the kernel model with the close pairs; 2afc for the
material models).

### Reachspaces

The 990 images of near-scale scenes (reachspaces) and the human odd-one-out
data of Josephs, Hebart & Konkle (2023, Cognition): the similarity of 45
images from fully sampled triplets (26,588 judgments), 1000 triplets with
about 28 judgments each, and all 1.24 million triplets of the 990 images.
None of these images were used for training.

| model | 45 images, r | 1000 repeated triplets | all triplets |
|---|---|---|---|
| `alignet_siglip2b_66d_kernel`, with the close pairs | 0.900 [0.847, 0.940] | 59.3% [57.9, 60.8] | 56.3% |
| `alignet_siglip2b_66d_ridge` | 0.881 [0.829, 0.920] | 58.3% [56.8, 59.8] | 55.5% |
| `vitb32_66d_elastic` | 0.817 [0.738, 0.877] | 56.9% [55.3, 58.6] | 54.4% |
| `rn50x64_49d_ridge` | 0.771 [0.674, 0.845] | 55.3% [53.8, 57.0] | 53.6% |
| `rn50x64_66d_ridge` | 0.823 [0.720, 0.892] | 56.6% [55.0, 58.3] | 54.2% |
| `rn50x64_66d_elastic` | 0.835 [0.752, 0.893] | 55.8% [54.2, 57.4] | 54.4% |
| AligNet SigLIP2-B alone (its cosine) | 0.793 | 59.5% | 56.3% |
| the same, with the SPoSE similarity | 0.871 | 59.5% | 56.4% |
| the embedding of Josephs et al. (fitted to 90% of the triplets) | 0.954 | 62.5% | 60.4% |
| noise ceiling | 0.980 | 63.9% | - |

- Similarity as above (SPoSE, the kernel model with the close pairs),
  computed within the set of images people saw together. Triplets: share
  of triplets in which the pair the people kept together has the largest
  predicted similarity (chance 33.3%).
- 95% intervals from bootstrapping images (45 images) or triplets (1000
  triplets). Noise ceilings: split halves of the judgments (45 images),
  each judgment predicted by the majority of the other judgments of the
  same triplet (1000 triplets; 138 catch trials with an image shown twice
  left out; the paper reports 66.3%).
- The images are 400 x 300 and not square. Square center crops gave the
  same results (largest difference 0.003 for the AligNet models).
- Each model is better than the cosine of its own network (RN50x64: +0.26
  to +0.33 on the 45 images, +5.6 to +6.9 points on the 1000 triplets;
  ViT-B/32: +0.25 and +5.8 points), but part of this is the SPoSE
  similarity (AligNet: 0.793 to 0.871 from the SPoSE similarity alone).
  Against AligNet with the SPoSE similarity, `alignet_siglip2b_66d_kernel`
  is better on the 45 images (+0.029 [0.004, 0.054]) and the same on the
  triplets (-0.2 points [-0.9, 0.5] on the 1000, -0.1 [-0.3, 0.1] on all).

The material models are far below their network here, e.g.
`alignet_siglip2b_36d_kernel` 0.375 on the 45 images (AligNet: 0.852 with
the 2afc similarity of its cosines; difference -0.48 [-0.65, -0.32]) and
47.6% on the 1000 triplets (AligNet 59.4%).

### Objects and materials

The material models are for images of materials. On objects they predict
human similarity much worse than their network alone: 0.517 on K-92
(AligNet 0.862, both with the 2afc similarity) and 50.3% on the THINGS
odd-one-out triplets (AligNet 61.7%). The other way round,
`alignet_siglip2b_66d_kernel` predicts the STUFF similarity about as well
as AligNet (0.750 on the 48 new images, AligNet 0.766 with the SPoSE
similarity; 66.1% of all STUFF triplets, AligNet 66.2%), but less well
than the material models (0.842).
