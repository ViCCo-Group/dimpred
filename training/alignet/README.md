# AligNet SigLIP2-B in PyTorch

The default model of dimpred uses AligNet SigLIP2-B (Muttenthaler et al.,
2025, Nature 647, 349-355, github.com/google-deepmind/alignet). Its authors
released it as a TensorFlow SavedModel only. We converted the weights to
PyTorch, so that dimpred needs neither TensorFlow nor OpenCV. The network
is in `dimpred/alignet.py`. This folder has the scripts to redo the
conversion. You only need them to check or repeat it.

## Files

- `dump_tf_variables.py`: reads the 218 anonymous variables of the
  SavedModel and saves them with their Flax names (`tf_variables.npz`,
  378 MB, not in the repository; `mapping.tsv`).
- `convert_to_torch.py`: maps them to the PyTorch network of
  `dimpred/alignet.py` and saves `alignet_siglip2_b.safetensors`
  (`mapping_torch.tsv` lists which Flax arrays each tensor comes from and
  how). As a check, the image encoder is also loaded with timm's own loader
  for big_vision checkpoints, which gives the same 162 tensors.
- `stablehlo_text.py`: writes the forward pass of the SavedModel as text,
  which we used to check the order of the variables and the network.
- `ALIGNET_MODELS_LICENSE.md`: the license file of the AligNet models.

## Redo the conversion

```
# download and unpack (350,866,208 bytes)
curl -O https://storage.googleapis.com/alignet/models/SigLIP2-B-alignet.tar.gz
tar xzf SigLIP2-B-alignet.tar.gz
python dump_tf_variables.py SigLIP2-B-alignet    # in an environment with tensorflow
python convert_to_torch.py                       # in an environment with torch and timm (1.0.15 or newer)
```

The result has 377,786,744 bytes and the same 168 tensors and metadata as
the file that dimpred downloads (release `alignet-weights-v1` of
github.com/ViCCo-Group/dimpred, sha256
`2ce461e04ac11271c736d32477d873f7d14fe6932c73d8757fec2854c6482647`). Its
sha256 can still differ, because safetensors writes the metadata entries in
an arbitrary order (on 2026/10/02, a new conversion gave identical tensors
and metadata, but another order). So compare the tensors:

```
python -c "from safetensors.torch import load_file; import torch; a = load_file('alignet_siglip2_b.safetensors'); b = load_file('<released file>'); print(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a))"
```

## What we checked

- The outputs of the PyTorch network equal those of TensorFlow for the same
  input arrays (2880 images of the DimPred benchmark): largest difference
  3e-5 on the cpu and 1.5e-5 on an Apple GPU for `pre_logits` (the features
  dimpred uses), per-image r = 1.000000. This is float32 rounding: on 96
  images, TensorFlow and PyTorch are about equally far from PyTorch in
  float64.
- From the image file to the features (dimpred's reading and resizing): the
  1854 THINGS reference images differ from the TensorFlow features by at
  most 3e-4 (median 1e-5 per image). The few larger differences come from
  pixel values that are exactly x.5 before rounding in the resize.
- The model `alignet_siglip2b_66d_ridge`, fitted on these features, differs
  from the fit on the TensorFlow features by at most 2.5e-7 in the weights
  and 2.5e-6 in the predictions.

## Things to know

- GELU is the tanh approximation, as in big_vision (timm's default for this
  network is the exact GELU, which changes the features by up to 1.7e-2).
  timm before 1.0.15 uses the exact GELU in the attention pooling head
  even with `act_layer="gelu_tanh"` (features change by up to 3e-3), so
  `dimpred/alignet.py` checks all activation functions and gives an error
  with older versions.
- The input is the whole image resized to 224 x 224 with the bicubic
  interpolation of OpenCV (INTER_CUBIC), rewritten in numpy (not compared
  with OpenCV itself; OpenCV's fixed-point weights can change single pixel
  values by 1), no crop, values 0 to 1, without mean or std normalization,
  as in the AligNet training code. PIL's bicubic resize is
  not a substitute: it smooths when it shrinks an image (per-image r of the
  features about 0.986).
- The outputs `layer_0` to `layer_11` of the SavedModel are not the residual
  stream after each block, as the AligNet README says, but the output of the
  self-attention branch of the block.

## License

The license file of the AligNet models (ALIGNET_MODELS_LICENSE.md) puts
software under Apache 2.0 and all other materials under CC-BY 4.0, and
lists SigLIP2, the model AligNet SigLIP2-B is based on, under Apache 2.0.
Both licenses allow redistribution with attribution and a note of the
changes, which this README and the metadata of the safetensors file give.
We changed the parameter names and array layouts, not the values. The safetensors file holds the source,
the paper, the license and a note on the conversion in its metadata.

Hebartlab, 2026/10/02
