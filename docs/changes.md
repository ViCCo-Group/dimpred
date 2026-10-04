# Changes

## 1.2.0

- The default model is now `alignet_siglip2b_66d_kernel`: the same network
  (AligNet SigLIP2-B) and the same ridge as `alignet_siglip2b_66d_ridge`,
  plus a local kernel, a correction from the training images that are
  similar to the image in the network, fit together with the ridge
  ([details](details.md#how-the-predictions-are-computed)). It predicts every
  one of the 66 dimensions better (mean r per dimension 0.839 instead of
  0.810, the color dimensions too) and also does so for concept categories
  left out of training. The model file is larger (6.6 MB), because it holds
  the features of the 1854 training images. To get the numbers of 1.1.0, give
  `alignet_siglip2b_66d_ridge`.
- `similarity` (`dimpred_similarity`) takes the features of the images as an
  optional input. With them, the network's own similarity is added for pairs
  of images that are very close in the network (close pairs; weight and
  threshold come from the model). This is the recommended way to predict
  similarity: on the 8 independent test sets, r = 0.699 instead of 0.596 for
  the dimensions alone and 0.572 for 1.1.0, above all within categories
  ([details](details.md#similarity)). Without the features, `similarity`
  works as in 1.1.0.
- Model files can hold a kernel part (`kernel_features`,
  `kernel_coefficients`, `kernel_tau`) and the close-pair settings
  (`close_pairs_weight`, `close_pairs_threshold`). Models without them work
  as before.
- In MATLAB, a model struct given to `dimpred_extract_features` or
  `dimpred_rise` reaches Python with its kernel part and close-pair settings.
- Training: `training/fit.py` has `kernel_ridge_fit`, and
  `training/build_models.py` builds the new model (regression
  `"ridge + kernel"`).

## 1.1.0

- The default model is now `alignet_siglip2b_66d_ridge` (network AligNet
  SigLIP2-B, Muttenthaler et al., 2025; 1.0.0: `vitb32_66d_elastic`). To get
  the numbers of 1.0.0, give `vitb32_66d_elastic`. The first time the default
  model is used, it downloads the weights of its network (378 MB) into
  `~/.cache/dimpred` (for offline use, set `DIMPRED_ALIGNET_WEIGHTS`). It
  needs timm 1.0.15 or newer, which `pip install -e ".[extract]"` installs.
- `rn50x64_66d_ridge` was fitted again, with a ridge regression that keeps
  the penalty chosen by cross-validation
  ([training/README.md](../training/README.md#the-ridge-regression-of-110)).
  Its predictions changed. The model of 1.0.0 is in the tag `v1.0.0`.
- In the training code (`training/fit.py`, `call.py`), `"ridge"` is now this
  ridge regression. The fractional ridge of the DimPred paper is
  `"fracridge"`.
- New: heatmaps with RISE (`dimpred.rise`, `python -m dimpred --rise`,
  `dimpred_rise`), see [Heatmaps (RISE)](details.md#heatmaps-rise).
- In MATLAB, `dimpred_extract_features` gives Python a model struct as it
  is (1.0.0: it loaded `model.file` again, so a network changed by hand in
  `model.info` was not used, and a struct without a file did not work).
- The README is shorter. The details are in [details.md](details.md) and
  [models.md](models.md), the training in
  [training/README.md](../training/README.md).

## 1.0.0

The version in the tag `v1.0.0`: the Python package and the MATLAB version
with four shipped models (default `vitb32_66d_elastic`). The earlier version
of this repository only contained the training code of the DimPred paper
(`dimpred/fit.py`, `dimpred/call.py`), and the fitted models had to be
downloaded from OSF. This code moved to `training/`, and `dimpred/` became the
package that applies the models, with the models included. Scripts that
imported `dimpred.fit` or `dimpred.call` have to import `fit` from
`training/` instead (run them there or add `training/` to the Python path).
