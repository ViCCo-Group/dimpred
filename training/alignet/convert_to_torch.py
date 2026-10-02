"""
Convert the AligNet SigLIP2-B weights (tf_variables.npz, from
dump_tf_variables.py) to the PyTorch network in alignet.py and save them as
alignet_siglip2_b.safetensors.

Flax stores Dense kernels as in x out and attention kernels as
in x heads x head_dim (query, key, value) or heads x head_dim x out (out);
torch Linear weights are out x in. The patch embedding is a 16 x 16 conv,
HWIO in Flax, OIHW in torch. query, key and value are stacked into one qkv
Linear in the blocks, key and value into one kv Linear in the MAP head (as
in timm).

As a check, the same weights are also loaded with timm's own loader for
big_vision checkpoints (timm.models.vision_transformer._load_weights), which
has to give exactly the same tensors.

Usage (torch environment, after dump_tf_variables.py):
    python convert_to_torch.py

Writes alignet_siglip2_b.safetensors and mapping_torch.tsv into this
folder. The tensors equal those of the released file (WEIGHTS_SHA256 in
dimpred/alignet.py); the sha256 of a new file can differ because of the
order of the metadata, see README.md.

Hebartlab, 2026/10/01
"""
import os
import sys
import tempfile

import numpy as np
import torch

# History:
# 2026/10/02: copied to dimpred/training/alignet; the network comes from
#   dimpred/alignet.py (the same AligNet as in the port)
# 2026/10/01: written for the port from TensorFlow

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # the dimpred repository
from dimpred.alignet import AligNet  # noqa: E402

TOWER = "model/_Model_0/"


def dense(w, prefix):
    """torch weight and bias of a Flax Dense layer."""
    return w[prefix + "/kernel"].T, w[prefix + "/bias"]


def attention_weights(w, prefix, part):
    """torch weight (out x in) and bias of query, key, value or out of a Flax attention layer."""
    kernel, bias = w[f"{prefix}/{part}/kernel"], w[f"{prefix}/{part}/bias"]
    if part == "out":
        return kernel.reshape(768, 768).T, bias  # heads x head_dim x out
    return kernel.reshape(768, 768).T, bias.reshape(768)  # in x heads x head_dim


def map_weights(w):
    """
    state_dict = map_weights(w)

    torch state dict of AligNet from the Flax-named arrays, and the list of
    (torch name, Flax names, operation) for the mapping table.
    """
    sd = {}
    table = []

    def put(name, value, sources, operation):
        sd[name] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
        table.append((name, sources, operation))

    e = "encoder."
    t = TOWER
    put(e + "patch_embed.proj.weight", w[t + "embedding/kernel"].transpose(3, 2, 0, 1), [t + "embedding/kernel"],
        "transpose HWIO -> OIHW")
    put(e + "patch_embed.proj.bias", w[t + "embedding/bias"], [t + "embedding/bias"], "")
    put(e + "pos_embed", w[t + "pos_embedding"], [t + "pos_embedding"], "")
    for i in range(12):
        b = f"{t}Transformer/encoderblock_{i}"
        mha = b + "/MultiHeadDotProductAttention_0"
        tb = f"{e}blocks.{i}."
        put(tb + "norm1.weight", w[b + "/LayerNorm_0/scale"], [b + "/LayerNorm_0/scale"], "")
        put(tb + "norm1.bias", w[b + "/LayerNorm_0/bias"], [b + "/LayerNorm_0/bias"], "")
        qkv = [attention_weights(w, mha, part) for part in ["query", "key", "value"]]
        put(tb + "attn.qkv.weight", np.concatenate([q[0] for q in qkv]),
            [f"{mha}/{p}/kernel" for p in ["query", "key", "value"]], "reshape 768 x 768, transpose, stack q, k, v")
        put(tb + "attn.qkv.bias", np.concatenate([q[1] for q in qkv]),
            [f"{mha}/{p}/bias" for p in ["query", "key", "value"]], "reshape 768, stack q, k, v")
        weight, bias = attention_weights(w, mha, "out")
        put(tb + "attn.proj.weight", weight, [mha + "/out/kernel"], "reshape 768 x 768, transpose")
        put(tb + "attn.proj.bias", bias, [mha + "/out/bias"], "")
        put(tb + "norm2.weight", w[b + "/LayerNorm_1/scale"], [b + "/LayerNorm_1/scale"], "")
        put(tb + "norm2.bias", w[b + "/LayerNorm_1/bias"], [b + "/LayerNorm_1/bias"], "")
        for r in range(2):
            weight, bias = dense(w, f"{b}/MlpBlock_0/Dense_{r}")
            put(f"{tb}mlp.fc{r + 1}.weight", weight, [f"{b}/MlpBlock_0/Dense_{r}/kernel"], "transpose")
            put(f"{tb}mlp.fc{r + 1}.bias", bias, [f"{b}/MlpBlock_0/Dense_{r}/bias"], "")
    put(e + "norm.weight", w[t + "Transformer/encoder_norm/scale"], [t + "Transformer/encoder_norm/scale"], "")
    put(e + "norm.bias", w[t + "Transformer/encoder_norm/bias"], [t + "Transformer/encoder_norm/bias"], "")

    # MAP head (attention pooling with a learned query, the probe)
    h = t + "MAPHead_0"
    mha = h + "/MultiHeadDotProductAttention_0"
    p = e + "attn_pool."
    put(p + "latent", w[h + "/probe"], [h + "/probe"], "")
    weight, bias = attention_weights(w, mha, "query")
    put(p + "q.weight", weight, [mha + "/query/kernel"], "reshape 768 x 768, transpose")
    put(p + "q.bias", bias, [mha + "/query/bias"], "reshape 768")
    kv = [attention_weights(w, mha, part) for part in ["key", "value"]]
    put(p + "kv.weight", np.concatenate([k[0] for k in kv]), [f"{mha}/{x}/kernel" for x in ["key", "value"]],
        "reshape 768 x 768, transpose, stack k, v")
    put(p + "kv.bias", np.concatenate([k[1] for k in kv]), [f"{mha}/{x}/bias" for x in ["key", "value"]],
        "reshape 768, stack k, v")
    weight, bias = attention_weights(w, mha, "out")
    put(p + "proj.weight", weight, [mha + "/out/kernel"], "reshape 768 x 768, transpose")
    put(p + "proj.bias", bias, [mha + "/out/bias"], "")
    put(p + "norm.weight", w[h + "/LayerNorm_0/scale"], [h + "/LayerNorm_0/scale"], "")
    put(p + "norm.bias", w[h + "/LayerNorm_0/bias"], [h + "/LayerNorm_0/bias"], "")
    for r in range(2):
        weight, bias = dense(w, f"{h}/MlpBlock_0/Dense_{r}")
        put(f"{p}mlp.fc{r + 1}.weight", weight, [f"{h}/MlpBlock_0/Dense_{r}/kernel"], "transpose")
        put(f"{p}mlp.fc{r + 1}.bias", bias, [f"{h}/MlpBlock_0/Dense_{r}/bias"], "")

    # heads
    weight, bias = dense(w, "triplet_head")
    put("triplet_head.weight", weight, ["triplet_head/kernel"], "transpose")
    put("triplet_head.bias", bias, ["triplet_head/bias"], "")
    put("i1k_norm.weight", w["i1k_head/norm/scale"], ["i1k_head/norm/scale"], "")
    put("i1k_norm.bias", w["i1k_head/norm/bias"], ["i1k_head/norm/bias"], "")
    weight, bias = dense(w, "i1k_head/dense")
    put("i1k_head.weight", weight, ["i1k_head/dense/kernel"], "transpose")
    put("i1k_head.bias", bias, ["i1k_head/dense/bias"], "")
    return sd, table


def check_with_timm_loader(w, sd):
    """Load the image tower with timm's big_vision loader and compare with our mapping."""
    from timm.models.vision_transformer import _load_weights

    net = AligNet()
    with tempfile.TemporaryDirectory() as folder:
        fname = os.path.join(folder, "big_vision.npz")
        np.savez(fname, **{k[len(TOWER):]: v for k, v in w.items() if k.startswith(TOWER)})
        with torch.no_grad():
            _load_weights(net.encoder, fname)
    timm_sd = net.encoder.state_dict()
    ours = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    assert set(timm_sd) == set(ours), set(timm_sd) ^ set(ours)
    for k in timm_sd:
        assert torch.equal(timm_sd[k], ours[k]), k
    print(f"timm's big_vision loader gives the same {len(timm_sd)} tensors")


if __name__ == "__main__":
    from safetensors.torch import save_file

    w = dict(np.load(os.path.join(HERE, "tf_variables.npz")))
    sd, table = map_weights(w)

    # every Flax array is used exactly once
    used = [s for _, sources, _ in table for s in sources]
    assert sorted(used) == sorted(w), set(w) ^ set(used)
    assert len(used) == len(set(used)) == 218

    # the network has exactly these tensors
    net = AligNet()
    assert set(net.state_dict()) == set(sd), set(net.state_dict()) ^ set(sd)
    for k, v in net.state_dict().items():
        assert v.shape == sd[k].shape, (k, v.shape, sd[k].shape)
    net.load_state_dict(sd)
    check_with_timm_loader(w, sd)

    metadata = {
        "format": "pt",
        "model": "AligNet SigLIP2-B (SigLIP 2 ViT-B/16, 224 px), post-trained on the AligNet triplets",
        "source": "https://storage.googleapis.com/alignet/models/SigLIP2-B-alignet.tar.gz (TensorFlow SavedModel)",
        "paper": "Muttenthaler et al. (2025). Aligning machine and human visual representations across "
                 "abstraction levels. Nature 647, 349-355",
        "code": "https://github.com/google-deepmind/alignet",
        "license": "Apache-2.0 (SigLIP 2 weights; AligNet checkpoints follow the license of the original model)",
        "network": "timm vit_base_patch16_siglip_224, act_layer gelu_tanh, num_classes 0 (key prefix encoder.), "
                   "plus triplet_head Linear(768, 1024), i1k_norm LayerNorm(768, eps 1e-6), i1k_head Linear(768, 1000)",
        "input": "float32 n x 3 x 224 x 224, values 0 to 1 (cv2 INTER_CUBIC resize of the whole image, / 255)",
        "changes": "converted from the TensorFlow SavedModel to PyTorch by Hebartlab, 2026/10/01: parameter names "
                   "and array layouts changed, values unchanged (float32); outputs checked against TensorFlow",
    }
    out = os.path.join(HERE, "alignet_siglip2_b.safetensors")
    save_file(sd, out, metadata=metadata)
    print(f"saved {out}: {len(sd)} tensors, {sum(v.numel() for v in sd.values())} numbers, "
          f"{os.path.getsize(out)} bytes")

    with open(os.path.join(HERE, "mapping_torch.tsv"), "w") as fid:
        fid.write("torch_name\tshape\tflax_names\toperation\n")
        for name, sources, operation in table:
            fid.write(f"{name}\t{list(sd[name].shape)}\t{', '.join(sources)}\t{operation}\n")
