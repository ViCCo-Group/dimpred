"""
Read the 218 anonymous variables of the AligNet SigLIP2-B SavedModel and save
them with their Flax parameter names (tf_variables.npz, mapping.tsv).

The SavedModel is a jax2tf export: variable i of the serving function is the
input jax2tf_arg_i of the XlaCallModule op (jax2tf_arg_218 is the image).
jax2tf flattens the Flax parameter tree with sorted keys, so the order is
MAPHead_0, Transformer (encoder_norm, then encoderblock_0, _1, _10, _11, _2,
..., _9, as strings sort), embedding, pos_embedding, triplet_head, and last
the ImageNet head. The order was checked in the StableHLO module: the
execution order of the blocks uses variables 17-32, 33-48, 81-96, ...,
193-208, 49-64, 65-80, and the shapes match. The names of the ImageNet head
are not in the checkpoint; we call it i1k_head (a LayerNorm, then a Dense
layer, as in the StableHLO).

Usage (TensorFlow environment):
    python dump_tf_variables.py <SigLIP2-B-alignet folder>

Writes tf_variables.npz (378 MB, not part of the repository) and mapping.tsv
into this folder.

Hebartlab, 2026/10/01
"""
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# History:
# 2026/10/02: copied to dimpred/training/alignet; the SavedModel folder is
#   given as argument
# 2026/10/01: written for the port from TensorFlow

HERE = os.path.dirname(os.path.abspath(__file__))


def flax_names():
    """Flax names of the 218 variables, in the order of jax2tf_arg_0 ... jax2tf_arg_217."""

    def dense(prefix):
        return [prefix + "/bias", prefix + "/kernel"]

    def layernorm(prefix):
        return [prefix + "/bias", prefix + "/scale"]

    def attention(prefix):
        names = []
        for part in ["key", "out", "query", "value"]:
            names += dense(prefix + "/" + part)
        return names

    names = []
    # MAP head (attention pooling)
    p = "model/_Model_0/MAPHead_0"
    names += layernorm(p + "/LayerNorm_0")
    names += dense(p + "/MlpBlock_0/Dense_0") + dense(p + "/MlpBlock_0/Dense_1")
    names += attention(p + "/MultiHeadDotProductAttention_0")
    names += [p + "/probe"]
    # Transformer: final norm, then the 12 blocks in sorted string order
    p = "model/_Model_0/Transformer"
    names += layernorm(p + "/encoder_norm")
    for block in sorted(f"encoderblock_{i}" for i in range(12)):
        b = p + "/" + block
        names += layernorm(b + "/LayerNorm_0") + layernorm(b + "/LayerNorm_1")
        names += dense(b + "/MlpBlock_0/Dense_0") + dense(b + "/MlpBlock_0/Dense_1")
        names += attention(b + "/MultiHeadDotProductAttention_0")
    # patch embedding and position embedding
    names += dense("model/_Model_0/embedding")
    names += ["model/_Model_0/pos_embedding"]
    # heads
    names += dense("triplet_head")
    names += layernorm("i1k_head/norm") + dense("i1k_head/dense")
    return names


def expected_shape(name):
    """Shape of a Flax parameter of the SigLIP B/16 image tower and the heads."""
    leaf = name.split("/")[-1]
    parent = name.split("/")[-2]
    if name.endswith("embedding/kernel"):
        return (16, 16, 3, 768)
    if name.endswith("pos_embedding"):
        return (1, 196, 768)
    if leaf == "probe":
        return (1, 1, 768)
    if name.startswith("triplet_head"):
        return (768, 1024) if leaf == "kernel" else (1024,)
    if name.startswith("i1k_head/dense"):
        return (768, 1000) if leaf == "kernel" else (1000,)
    if parent in ["query", "key", "value"]:
        return (768, 12, 64) if leaf == "kernel" else (12, 64)
    if parent == "out":
        return (12, 64, 768) if leaf == "kernel" else (768,)
    if parent == "Dense_0":
        return (768, 3072) if leaf == "kernel" else (3072,)
    if parent == "Dense_1":
        return (3072, 768) if leaf == "kernel" else (768,)
    return (768,)  # LayerNorm scale and bias, embedding bias


if __name__ == "__main__":
    import tensorflow as tf

    MODEL_DIR = sys.argv[1]
    tf.config.set_visible_devices([], "GPU")
    m = tf.saved_model.load(MODEL_DIR)
    f = m.signatures["serving_default"]

    # variable i is captured input i, which is jax2tf_arg_i (input 0 of the
    # function is the image, jax2tf_arg_218)
    assert all(v.handle is c for v, c in zip(f.variables, f.captured_inputs))
    fdef = [fun for fun in f.graph.as_graph_def().library.function if len(fun.signature.input_arg) == 219][0]
    args = [a.name for a in fdef.signature.input_arg]
    assert args[0] == "images" and args[1:] == [f"jax2tf_arg_{i}_readvariableop_resource" for i in range(218)]

    # key of each variable in the checkpoint (variables/<k>/...), found by value
    reader = tf.train.load_checkpoint(os.path.join(MODEL_DIR, "variables", "variables"))
    ckpt = {k: reader.get_tensor(k) for k in reader.get_variable_to_shape_map() if k.startswith("variables/")}

    names = flax_names()
    assert len(names) == len(f.variables) == 218 and len(set(names)) == 218
    arrays = {}
    rows = []
    for i, (name, v) in enumerate(zip(names, f.variables)):
        x = v.numpy()
        assert x.dtype == np.float32
        assert x.shape == expected_shape(name), (i, name, x.shape, expected_shape(name))
        arrays[name] = x
        keys = [k.split("/")[1] for k, y in ckpt.items() if y.shape == x.shape and np.array_equal(y, x)]
        rows.append(f"{i}\t{','.join(keys)}\t{name}\t{list(x.shape)}")
    np.savez(os.path.join(HERE, "tf_variables.npz"), **arrays)
    with open(os.path.join(HERE, "mapping.tsv"), "w") as fid:
        fid.write("jax2tf_arg\tcheckpoint_variable\tflax_name\tshape\n")
        fid.write("\n".join(rows) + "\n")
    print(f"saved {len(arrays)} variables, {sum(x.size for x in arrays.values())} numbers")
