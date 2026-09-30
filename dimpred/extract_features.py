import os

import numpy as np

from .load_model import load_model

# History:
# 2026/09/30: written for the first release of the package


def extract_features(images, model=None, network=None, pretrained="openai", device=None, batch_size=32):
    """
    features = extract_features(images, model=None, network=None, pretrained="openai", device=None, batch_size=32)

    Network features of images, as needed by predict. Each image is read with
    PIL, converted to RGB with 8 bits per channel (grayscale images get three
    equal channels, an alpha channel is dropped, 16-bit grayscale images are
    scaled to 8 bit, and images with 32-bit integer or float pixels give an
    error; see read_image), preprocessed as the network expects it (for
    RN50x64: resize to 448 px and center crop, for ViT-B/32: 224 px), and
    passed through the image encoder of the network (open_clip,
    model.encode_image, in float32). The features are the output of the image
    encoder, not normalized, as used for training the models.

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
    without them. The first time a network is used, open_clip downloads its
    weights, which can take a while.

    Input:
        images:     list of image files, or a single file (str)
        model:      model name, path of a model file, or a model from
                    load_model; its network and pretrained weights are used
                    (default: dimpred.DEFAULT_MODEL, network ViT-B-32-quickgelu)
        network:    open_clip name of the network, e.g. "RN50x64"; if given,
                    network and pretrained are used instead of the model
        pretrained: open_clip name of the weights (default: "openai"); only
                    used together with network
        device:     "cuda", "mps" or "cpu" (default: cuda if available, else
                    mps if available, else cpu). Different devices give
                    slightly different features (differences of about 1e-4).
        batch_size: number of images passed through the network at once
                    (default: 32); use a smaller value if you run out of
                    memory

    Output:
        features: float32 array, n_images x n_features (one row per image,
                  in the order of images; RN50x64: 1024 features,
                  ViT-B-32-quickgelu: 512)

    Example:
        files = dimpred.find_images("my_images")
        features = dimpred.extract_features(files, model="rn50x64_49d_ridge")
        embedding = dimpred.predict(features, "rn50x64_49d_ridge")

    Martin Hebart, 2026/09/30

    See also: find_images, predict, load_model
    """

    # Check input before loading the network, which takes a few seconds
    if isinstance(images, (str, os.PathLike)):
        images = [images]
    images = [os.fspath(fname) for fname in images]
    if not images:
        raise ValueError("No images given.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError(f"batch_size has to be a whole number of at least 1, not {batch_size!r}.")
    for fname in images:
        if os.path.isdir(fname):
            raise ValueError(f"{fname} is a folder. extract_features only takes image files. Please get the images "
                             f"of a folder with find_images first, e.g. extract_features(find_images(folder)).")
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Image not found: {fname}")

    # Network and weights: from the model, unless the network is given
    if network is None:
        info = load_model(model)["info"]
        network, pretrained = info["network"], info["pretrained"]

    # These packages are only needed here, so we import them here and the
    # rest of dimpred works without them (importing torch also takes a while)
    try:
        import open_clip
        import torch
        import PIL  # noqa: F401 (needed by read_image)
    except ImportError as error:
        raise ImportError(f"Feature extraction needs torch, open_clip_torch and pillow ({error}). Install them "
                          f"with: pip install torch open_clip_torch pillow") from error

    # Device: the first that is available, unless one is given. A given device
    # is checked here, because torch would only fail later, with a message
    # that does not say what to do.
    available = ["cpu"]
    if torch.backends.mps.is_available():
        available.insert(0, "mps")
    if torch.cuda.is_available():
        available.insert(0, "cuda")
    if device is None:
        device = available[0]
    elif str(device).split(":")[0] not in available:  # e.g. "cuda:1" is the second gpu
        raise ValueError(f"The device '{device}' is not available on this computer. Available devices: "
                         f"{', '.join(available)}.")

    # Load the network and its preprocessing (the one for test images, not
    # the one with augmentation for training)
    net, _, preprocess = open_clip.create_model_and_transforms(network, pretrained=pretrained, device=device)
    net.eval()

    # Extract features, batch by batch
    features = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = []
            for fname in images[start:start + batch_size]:
                batch.append(preprocess(read_image(fname)))
            batch = torch.stack(batch).to(device)
            features.append(net.encode_image(batch).float().cpu().numpy())

    return np.concatenate(features).astype(np.float32)


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

    Martin Hebart, 2026/09/30

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
