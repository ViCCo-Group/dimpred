import os

import numpy as np

from .load_model import load_model

# History:
# 2026/10/02: the network and its preprocessing in load_network, split into
#   the image with values 0 to 1 and the normalization with the network, so
#   that rise can mask the image in between (the features do not change);
#   check_images and choose_device for rise
# 2026/10/02: AligNet SigLIP2-B (dimpred/alignet.py), the network of the new
#   default model
# 2026/09/30: written for the first release of the package


def extract_features(images, model=None, network=None, pretrained="openai", device=None, batch_size=32):
    """
    features = extract_features(images, model=None, network=None, pretrained="openai", device=None, batch_size=32)

    Network features of images, as needed by predict. Each image is read with
    PIL, converted to RGB with 8 bits per channel (grayscale images get three
    equal channels, an alpha channel is dropped, 16-bit grayscale images are
    scaled to 8 bit, and images with 32-bit integer or float pixels give an
    error; see read_image), preprocessed as the network expects it, and
    passed through the network in float32. The features are those used for
    training the models, not normalized:
        AligNet SigLIP2-B (default model): the whole image is resized to
            224 x 224 (no crop), and the features are pre_logits, the
            output of the attention pooling head (dimpred/alignet.py)
        CLIP networks (open_clip): resize to 224 px (ViT-B/32) or 448 px
            (RN50x64) and center crop, and the features are the output of
            the image encoder (model.encode_image)

    The network is set by the model (model["info"]["network"]), so pass the
    same model to extract_features and to predict. For OpenAI's CLIP ViT
    models, the network name has to end in "-quickgelu": open_clip's plain
    "ViT-B-32" uses GELU instead of QuickGELU with the same weights and gives
    different features (r ~ 0.98 with the correct ones).

    Folders are not accepted, list their images with find_images first. In
    this way, row i of the features always belongs to the i-th file you passed.
    All files are checked before the network is loaded, so a wrong path gives
    an error at once. A file that cannot be read as an image gives an error
    that names the file.

    This function needs torch, open_clip_torch and pillow
    (pip install torch open_clip_torch pillow). The rest of dimpred works
    without them. The first time a network is used, its weights are
    downloaded, which can take a while: by open_clip for the CLIP networks,
    by dimpred for AligNet (378 MB into ~/.cache/dimpred; or set the
    environment variable DIMPRED_ALIGNET_WEIGHTS to the path of
    alignet_siglip2_b.safetensors, see dimpred/alignet.py).

    Input:
        images:     list of image files, or a single file (str)
        model:      model name, path of a model file, or a model from
                    load_model; its network and pretrained weights are used
                    (default: dimpred.DEFAULT_MODEL, network AligNet SigLIP2-B)
        network:    "AligNet SigLIP2-B" or the open_clip name of a network,
                    e.g. "RN50x64"; if given, network and pretrained are
                    used instead of the model
        pretrained: open_clip name of the weights (default: "openai"); only
                    used together with an open_clip network
        device:     "cuda", "mps" or "cpu" (default: cuda if available, else
                    mps if available, else cpu). Different devices give
                    slightly different features (differences of about 1e-4).
        batch_size: number of images passed through the network at once
                    (default: 32); use a smaller value if you run out of
                    memory

    Output:
        features: float32 array, n_images x n_features (one row per image,
                  in the order of images; AligNet SigLIP2-B: 768 features,
                  RN50x64: 1024, ViT-B-32-quickgelu: 512)

    Example:
        files = dimpred.find_images("my_images")
        features = dimpred.extract_features(files, model="rn50x64_49d_ridge")
        embedding = dimpred.predict(features, "rn50x64_49d_ridge")

    Hebartlab, 2026/09/30

    See also: find_images, predict, load_model, load_network
    """

    # Check input before loading the network, which takes a few seconds
    images = check_images(images, "extract_features")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError(f"batch_size has to be a whole number of at least 1, not {batch_size!r}.")

    # Network and weights: from the model, unless the network is given
    if network is None:
        info = load_model(model)["info"]
        network, pretrained = info["network"], info["pretrained"]

    # The network, with its preprocessing in two steps (see load_network)
    to_unit, encode = load_network(network, pretrained, device)

    # Extract features, batch by batch. torch was imported by load_network.
    import torch
    features = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([to_unit(read_image(fname)) for fname in images[start:start + batch_size]])
        features.append(encode(batch))

    return np.concatenate(features).astype(np.float32)


def load_network(network, pretrained="openai", device=None):
    """
    to_unit, encode = load_network(network, pretrained="openai", device=None)

    Load a network and its preprocessing, split into two steps:
        to_unit(image)  the image as the network sees it, before the
                        normalization of the network: resized (and for the
                        CLIP networks cropped), values 0 to 1
        encode(batch)   the normalization of the network (CLIP networks: the
                        mean and std of their training images; AligNet:
                        none), on the cpu, then the network on the device;
                        returns the features
    extract_features computes encode(torch.stack([to_unit(image), ...])).
    rise masks the images between the two steps, so that masked pixels are
    black. The two steps together are exactly the preprocessing of the
    network (the open_clip preprocessing is a list of steps whose last step
    is the normalization), so the features are the same as without the
    split. If the preprocessing of an open_clip network does not end with a
    normalization, to_unit is the whole preprocessing.

    This function needs torch, open_clip_torch and pillow. AligNet needs
    timm and safetensors, which come with open_clip.

    Input:
        network:    "AligNet SigLIP2-B" or the open_clip name of a network,
                    e.g. "RN50x64"
        pretrained: open_clip name of the weights (default: "openai"); not
                    used for AligNet
        device:     "cuda", "mps" or "cpu" (default: cuda if available, else
                    mps if available, else cpu)

    Output:
        to_unit: function, PIL image (RGB) to a float32 tensor, 3 x height x
                 width, values 0 to 1
        encode:  function, float32 tensor n x 3 x height x width (on the
                 cpu) to the features, float32 array n x n_features

    Example:
        to_unit, encode = load_network("RN50x64")
        features = encode(torch.stack([to_unit(read_image(f)) for f in files]))

    Hebartlab, 2026/10/02

    See also: extract_features, read_image, rise
    """

    # These packages are only needed here, so we import them here and the
    # rest of dimpred works without them (importing torch also takes a
    # while). AligNet needs timm and safetensors, which come with open_clip.
    try:
        import open_clip
        import torch
        import PIL  # noqa: F401 (needed by read_image)
        from . import alignet
    except ImportError as error:
        raise ImportError(f"Feature extraction needs torch, open_clip_torch and pillow ({error}). Install them "
                          f"with: pip install torch open_clip_torch pillow") from error

    device = choose_device(device)

    # Load the network and its preprocessing. For AligNet: our port, with the
    # weights from get_weights (downloaded the first time), its preprocessing
    # (values 0 to 1, no normalization) and its output pre_logits. For the
    # open_clip networks: the preprocessing for test images (not the one with
    # augmentation for training) and the output of the image encoder.
    if network == alignet.NETWORK:
        net = alignet.load_alignet(device=device)
        to_unit = alignet.preprocess

        def normalize(batch):
            return batch

        def network_output(batch):
            return net(batch)["pre_logits"]
    else:
        net, _, preprocess = open_clip.create_model_and_transforms(network, pretrained=pretrained, device=device)
        net.eval()
        network_output = net.encode_image
        # The preprocessing is a torchvision Compose (resize, center crop,
        # conversion to RGB, to a tensor with values 0 to 1, Normalize).
        # Applying its steps one after the other is what Compose does, so the
        # split does not change any value.
        steps = list(getattr(preprocess, "transforms", []))
        if steps and type(steps[-1]).__name__ == "Normalize":
            normalize = steps[-1]

            def to_unit(image):
                for step in steps[:-1]:
                    image = step(image)
                return image
        else:
            to_unit = preprocess

            def normalize(batch):
                return batch

    def encode(batch):
        # the normalization on the cpu, as in the preprocessing of each image
        # before, so that the features do not change
        with torch.no_grad():
            batch = normalize(batch).to(device)
            return network_output(batch).float().cpu().numpy()

    return to_unit, encode


def check_images(images, function):
    """
    images = check_images(images, function)

    Check the image files before the network is loaded, which takes a few
    seconds, so that a wrong path gives an error at once. A single file is
    turned into a list. Folders are not accepted (the error says to use
    find_images, so that the order of the images is always the order of the
    files). function is the name of the calling function, for the message.
    """

    if isinstance(images, (str, os.PathLike)):
        images = [images]
    images = [os.fspath(fname) for fname in images]
    if not images:
        raise ValueError("No images given.")
    for fname in images:
        if os.path.isdir(fname):
            raise ValueError(f"{fname} is a folder. {function} only takes image files. Please get the images "
                             f"of a folder with find_images first, e.g. {function}(find_images(folder)).")
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Image not found: {fname}")
    return images


def choose_device(device):
    """
    device = choose_device(device)

    The device for the network: the first of cuda, mps and cpu that is
    available, unless one is given. A given device is checked here, because
    torch would only fail later, with a message that does not say what to
    do. Needs torch.
    """

    import torch

    available = ["cpu"]
    if torch.backends.mps.is_available():
        available.insert(0, "mps")
    if torch.cuda.is_available():
        available.insert(0, "cuda")
    if device is None:
        return available[0]
    if str(device).split(":")[0] not in available:  # e.g. "cuda:1" is the second gpu
        raise ValueError(f"The device '{device}' is not available on this computer. Available devices: "
                         f"{', '.join(available)}.")
    return device


def read_image(fname):
    """
    image = read_image(fname)

    Read an image file as an RGB image with 8 bits per channel (PIL image),
    as the networks expect it. For most images, PIL's convert("RGB") does
    this: grayscale images get three equal channels, an alpha channel is
    dropped, CMYK and palette images are converted. 16-bit grayscale images
    (e.g. some PNG and TIFF files) are scaled to 8 bit first (values / 257),
    because convert("RGB") would set all values above 255 to 255, which makes
    the image almost white. Images with 32-bit integer or float pixels have
    no fixed range of values, so they give an error.

    PIL reads the pixels only when they are needed, and its error for a
    broken file (e.g. "image file is truncated") does not name the file. We
    read the pixels here and add the file name, so that a broken file can be
    found among thousands.

    Input:
        fname: path of the image file

    Output:
        image: PIL image in mode RGB

    Hebartlab, 2026/09/30

    See also: extract_features
    """

    from PIL import Image  # imported here, as in extract_features

    try:
        with Image.open(fname) as image:
            image.load()  # read the pixels now, so that errors in the file show up here
            if image.mode in ["I", "F"]:
                raise ValueError(f"The image {fname} has 32-bit integer or float pixels (PIL mode {image.mode}), "
                                 f"which have no fixed range of values. Please save it as an 8-bit image.")
            if image.mode in ["I;16", "I;16B", "I;16L", "I;16N"]:
                pixels = np.asarray(image, dtype=float) / 257  # 0 to 65535 becomes 0 to 255
                image = Image.fromarray(pixels.round().astype(np.uint8))
            return image.convert("RGB")
    except (OSError, SyntaxError) as error:  # PIL gives SyntaxError for some broken files
        raise OSError(f"Could not read the image {fname} ({error}).") from error
