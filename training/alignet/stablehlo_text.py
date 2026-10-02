"""
Write the StableHLO of the AligNet SavedModel as text (stablehlo_module.mlir in this folder),
with TensorFlow's own MLIR tools (no jax needed). In it, @_wrapped_jax_export_main
has %arg2 ... %arg219 = jax2tf_arg_0 ... jax2tf_arg_217 (the weights) and %arg220 =
the images.

Usage (TensorFlow environment):
    python stablehlo_text.py <SigLIP2-B-alignet folder>

Writes stablehlo_module.mlir (0.5 MB) into this folder. Only needed to check
the mapping of the variables, not for the conversion itself.

Hebartlab, 2026/10/01
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

# History:
# 2026/10/02: copied to dimpred/training/alignet; the SavedModel folder is
#   given as argument, the output goes into this folder
# 2026/10/01: written for the port from TensorFlow

HERE = os.path.dirname(os.path.abspath(__file__))

tf.config.set_visible_devices([], "GPU")
f = tf.saved_model.load(sys.argv[1]).signatures["serving_default"]
text = tf.mlir.experimental.run_pass_pipeline(tf.mlir.experimental.convert_function(f),
                                              "tf-xla-call-module-deserialization")
with open(os.path.join(HERE, "stablehlo_module.mlir"), "w") as fid:
    fid.write(text)
