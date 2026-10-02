"""
RISE heatmaps of the predicted dimensions: which parts of an image drive
its predicted dimension values.

RISE (Randomized Input Sampling for Explanation; Petsiuk, V., Das, A., &
Saenko, K., 2018, BMVC) multiplies an image with many random masks, passes
the masked images through the network and the model, and averages the
predictions of each dimension, weighted by the masks. The DimPred paper
(Kaniuth et al., 2025, Figure 7) used it to show which image regions are
relevant for the predicted dimensions and similarity. The masks here are
those of the original RISE code (github.com/eclique/RISE, explanations.py).

Functions:
    rise            heatmaps of images (one per dimension, and the relevance map)
    generate_masks  the random masks of RISE
    random_grids    the random part of the masks (grids and shifts)
    upsample_grids  the masks from their grids and shifts
    mask_sums       sums of the masks, weighted by the predictions
    saliency_maps   maps from these sums ("pixel" or "original" normalization)
    relevance_map   the dimension maps, weighted by the predicted dimension values
    overlay         a map in jet colors on top of the image (for PNG files)

Hebartlab, 2026/10/02

See also: extract_features, predict
"""

import os
import time

import numpy as np

from .extract_features import check_images, load_network, read_image
from .load_model import load_model
from .predict import predict

# History:
# 2026/10/02: written for dimpred 1.1.0, from the comparison with the maps
#   of the DimPred paper (RISE with 6000 masks, an 8 x 8 grid and p = 0.1)

# The two ways to normalize the maps (see saliency_maps)
NORMALIZATIONS = ("pixel", "original")


def rise(images, model=None, n_masks=6000, grid=8, p=0.1, normalization="pixel", seed=0, map_size=224,
         device=None, batch_size=32, verbose=False):
    """
    result = rise(images, model=None, n_masks=6000, grid=8, p=0.1, normalization="pixel", seed=0, map_size=224,
                  device=None, batch_size=32, verbose=False)

    Heatmaps that show which parts of each image drive its predicted
    dimensions, with RISE (Petsiuk et al., 2018), as in Figure 7 of the
    DimPred paper. Each image is multiplied with n_masks random masks (soft
    masks with values 0 to 1, the same masks for every image), the
    dimensions of each masked image are predicted, and the map of a
    dimension is the average of its predictions, weighted by the masks.
    Pixels that give high values of a dimension when they are kept get high
    values in its map. The relevance map combines the maps of all
    dimensions, weighted by the predicted dimension values of the image
    without masks and divided by their sum, as in the DimPred paper. It
    shows which parts of the image matter for the dimensions that describe
    the image best, and so for its predicted similarity to other images.

    The masks are those of the original RISE code (see generate_masks):
    grid x grid cells, each kept with probability p, upsampled bilinearly
    and shifted at random. The mask multiplies the network input
    image with values 0 to 1, before the normalization of the network, so
    masked pixels are black. The masked images then go through the steps of
    extract_features and predict.

    Normalization of the maps (see saliency_maps):
        "pixel"     at each pixel, the average of the predictions weighted
                    by the masks, sum(prediction * mask) / sum(mask)
                    (default). The masks do not cover all pixels equally
                    often (with p = 0.1, the sum of the masks varies by a
                    few percent over the image, about as much as the effect
                    of the image content), and this removes that pattern.
        "original"  sum(prediction * mask) / (n_masks * p), as in the
                    original RISE code. The maps then also contain the
                    pattern of the masks, which is the same for every image
                    and model.

    The maps are computed on the network input image, at the input size of
    the network, and then resized to map_size x map_size (bilinear, with
    antialiasing). They cover what the network sees: for the CLIP models
    the central square of the image (after resizing its shorter side to 448
    px for RN50x64, 224 px for ViT-B/32), for AligNet the whole image,
    resized to a square (the aspect ratio is not kept). The output view is
    this image at the size of the maps, so that maps can be shown on top of
    it (see overlay).

    Time: each image and each mask is one pass through the network. With
    6000 masks, this takes about 10 min per image with RN50x64 and less than
    1 min with AligNet on the GPU of an Apple M1 Max (mps), and much longer
    on the cpu. With normalization "pixel", fewer masks (e.g. 2000) still
    give stable maps. With "original", the maps of different masks differ
    much more (see docs/details.md).

    Model: for heatmaps we recommend rn50x64_66d_ridge (a convolutional
    network; of the 66d models we compared, its maps were the closest to
    Figure 7 of the DimPred paper). Figure 7 itself was made with
    rn50x64_49d_ridge. Any model works; the default model (AligNet) is the
    fast option.

    Memory: most of it is used by the network, which gets batch_size masked
    images at once (with batch_size 32 on an Apple M1 Max with mps: about
    11 GB for RN50x64, 2 GB for AligNet); a smaller batch_size needs less.
    RISE itself adds the sums of one image at the input size of the network
    (n_dims x height x width, float64; 106 MB for RN50x64), and the result
    needs about 13 MB per image (66 dimensions, maps of 224 x 224). The
    memory does not grow with n_masks.

    This function needs torch, open_clip_torch and pillow, as
    extract_features.

    Input:
        images:        list of image files, or a single file (str); folders
                       are not accepted (use find_images)
        model:         model name, path of a model file, or a model from
                       load_model (default: dimpred.DEFAULT_MODEL)
        n_masks:       number of masks (default: 6000)
        grid:          number of cells of the masks in each direction
                       (default: 8)
        p:             probability that a cell is kept (default: 0.1)
        normalization: "pixel" (default) or "original", see above
        seed:          seed of the random masks (default: 0); the same seed
                       gives the same masks
        map_size:      height and width of the maps (default: 224), or None
                       for the input size of the network
        device:        "cuda", "mps" or "cpu" (default: cuda if available,
                       else mps if available, else cpu)
        batch_size:    masked images passed through the network at once
                       (default: 32); use a smaller value if you run out of
                       memory
        verbose:       print a line after each image (default: False)

    Output:
        result: dict with
            relevance:      float32, n_images x map_size x map_size, the
                            relevance map of each image
            dimension_maps: float32, n_images x n_dims x map_size x
                            map_size, the map of each dimension
            embedding:      float64, n_images x n_dims, the predicted
                            dimension values of the images without masks
                            (predict(extract_features(images, model), model))
            labels:         list of n_dims str, the names of the dimensions
            files:          list of the image files
            model:          name of the model
            view:           uint8, n_images x map_size x map_size x 3, the
                            image as the network sees it (RGB)
            settings:       dict with n_masks, grid, p, seed, normalization,
                            input_size ([height, width] of the network
                            input) and map_size

    Example:
        result = dimpred.rise(["cat.jpg"], "rn50x64_66d_ridge", n_masks=2000)
        top = result["embedding"][0].argmax()   # the dimension with the largest value
        print(result["labels"][top])
        from PIL import Image
        from dimpred.rise import overlay
        Image.fromarray(overlay(result["view"][0], result["relevance"][0])).save("cat_relevance.png")

    Hebartlab, 2026/10/02

    See also: generate_masks, saliency_maps, relevance_map, overlay, extract_features, predict
    """

    # Check input before loading the network, which takes a few seconds
    images = check_images(images, "rise")
    check_whole_number("n_masks", n_masks, 1)
    check_whole_number("grid", grid, 1)
    check_whole_number("seed", seed, 0)
    if seed >= 2 ** 32:  # the range of np.random.RandomState
        raise ValueError(f"seed has to be smaller than 2**32, not {seed!r}.")
    if map_size is not None:
        check_whole_number("map_size", map_size, 1)
    check_whole_number("batch_size", batch_size, 1)
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)) or not 0 < p <= 1:
        raise ValueError(f"p, the probability that a cell of the masks is kept, has to be larger than 0 and at "
                         f"most 1, not {p!r}.")
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization {normalization!r}. Please use 'pixel' (the average of the "
                         f"predictions at each pixel, weighted by the masks) or 'original' (as in the original "
                         f"RISE code).")

    # The model and its network, with the preprocessing in two steps: the
    # image with values 0 to 1 (to_unit), and the normalization with the
    # network (encode). We mask the image between the two.
    model = load_model(model)
    info = model["info"]
    to_unit, encode = load_network(info["network"], info.get("pretrained", "openai"), device)
    import torch  # imported by load_network
    n_dims = model["weights"].shape[1]

    # Read and preprocess every image once before the masks, so that a broken
    # file gives an error at once, and not after the maps of the images
    # before it (minutes each). The images are read again below, one at a
    # time, so that only one image is in memory. The masks multiply the
    # image with values 0 to 1, so the preprocessing has to give these.
    for fname in images:
        image = to_unit(read_image(fname))
        if float(image.min()) < 0 or float(image.max()) > 1:
            raise ValueError(f"The preprocessing of the network {info['network']} does not give an image with "
                             f"values from 0 to 1 before its normalization, so rise cannot mask it (masked pixels "
                             f"have to be black).")

    relevance, dimension_maps, embedding, view = [], [], [], []
    grids, shifts, input_size = None, None, None
    for i_image, fname in enumerate(images):
        start_time = time.time()

        # The image as the network sees it, values 0 to 1, 3 x height x width
        image = to_unit(read_image(fname))

        # The masks: the same for all images. They depend on the input size
        # of the network, which is the same for all images, so we draw the
        # random grids and shifts once (they are small) and make the masks
        # of each batch from them.
        if grids is None:
            input_size = tuple(int(n) for n in image.shape[1:])
            grids, shifts = random_grids(n_masks, input_size, grid, p, seed)

        # The prediction of the image without masks, as
        # predict(extract_features(fname, model), model)
        image_embedding = predict(encode(image[None]), model)[0]

        # The masked images, batch by batch: the predictions of the
        # dimensions, summed over the masks with the masks as weights
        sums = np.zeros((n_dims, *input_size))
        coverage = np.zeros(input_size)
        for start in range(0, n_masks, batch_size):
            masks = upsample_grids(grids[start:start + batch_size], shifts[start:start + batch_size], input_size)
            masked = image[None] * torch.from_numpy(masks)[:, None]  # n x 3 x height x width
            batch_sums, batch_coverage = mask_sums(predict(encode(masked), model), masks)
            sums += batch_sums
            coverage += batch_coverage

        # The maps at the input size of the network, then at map_size
        maps = saliency_maps(sums, coverage, n_masks, p, normalization)
        relevance.append(resize_maps(relevance_map(maps, image_embedding), map_size).astype(np.float32))
        dimension_maps.append(resize_maps(maps, map_size).astype(np.float32))
        embedding.append(image_embedding)
        pixels = resize_maps(image.numpy().astype(np.float64), map_size)  # 3 x map_size x map_size, 0 to 1
        view.append(np.clip(np.round(pixels * 255), 0, 255).astype(np.uint8).transpose(1, 2, 0))

        if verbose:
            print(f"RISE: image {i_image + 1} of {len(images)} ({os.path.basename(fname)}) took "
                  f"{time.time() - start_time:.0f} s", flush=True)

    settings = dict(n_masks=n_masks, grid=grid, p=p, seed=seed, normalization=normalization,
                    input_size=list(input_size), map_size=map_size)
    return dict(relevance=np.stack(relevance), dimension_maps=np.stack(dimension_maps),
                embedding=np.stack(embedding), labels=list(model["labels"]), files=images,
                model=info.get("name", model.get("file", "")), view=np.stack(view), settings=settings)


def generate_masks(n_masks, input_size, grid=8, p=0.1, seed=0):
    """
    masks = generate_masks(n_masks, input_size, grid=8, p=0.1, seed=0)

    The random masks of RISE, as generate_masks of the original code
    (github.com/eclique/RISE, explanations.py, class RISE). For each mask,
    grid x grid cells, each kept (1) with probability p and else 0, are
    upsampled bilinearly to (grid + 1) * cell_size pixels, with cell_size
    = ceil(input_size / grid), and a window of the input size is cut out at
    a random shift of 0 to cell_size - 1 pixels in each direction. This
    gives soft masks with values from 0 to 1, whose edges do not lie on a
    fixed grid. The random numbers come from np.random.RandomState(seed), in
    the order of the original code (all grids first, then the two shifts of
    each mask), so the masks are those of the original code after
    np.random.seed(seed).

    Input:
        n_masks:    number of masks
        input_size: (height, width) of the network input image
        grid:       number of cells in each direction (default: 8)
        p:          probability that a cell is kept (default: 0.1)
        seed:       seed of the random numbers (default: 0)

    Output:
        masks: float32 array, n_masks x height x width, values 0 to 1

    Example:
        masks = generate_masks(6000, (448, 448))  # the masks of rise for RN50x64 (4.8 GB)

    Hebartlab, 2026/10/02

    See also: random_grids, upsample_grids, rise
    """

    grids, shifts = random_grids(n_masks, input_size, grid, p, seed)
    return upsample_grids(grids, shifts, input_size)


def random_grids(n_masks, input_size, grid=8, p=0.1, seed=0):
    """
    grids, shifts = random_grids(n_masks, input_size, grid=8, p=0.1, seed=0)

    The random part of generate_masks: the grids and the shifts of all
    masks, in the order of the random numbers of the original RISE code.
    They are small, so rise draws them once and makes the masks of each
    batch from them (upsample_grids). All masks at once would need a lot of
    memory (6000 masks of 448 x 448 pixels: 4.8 GB).

    Input:
        n_masks, input_size, grid, p, seed: as in generate_masks

    Output:
        grids:  bool array, n_masks x grid x grid (True: the cell is kept)
        shifts: int array, n_masks x 2, the shift of each mask in rows and
                in columns (0 to cell_size - 1)

    Hebartlab, 2026/10/02

    See also: generate_masks, upsample_grids
    """

    # np.random.RandomState is the generator behind np.random.seed and
    # np.random.rand, which the original code uses
    random = np.random.RandomState(seed)
    cell_size = [int(np.ceil(n / grid)) for n in input_size]
    grids = random.rand(n_masks, grid, grid) < p
    shifts = [[random.randint(0, cell_size[0]), random.randint(0, cell_size[1])] for _ in range(n_masks)]
    return grids, np.array(shifts, dtype=int).reshape(n_masks, 2)


def upsample_grids(grids, shifts, input_size):
    """
    masks = upsample_grids(grids, shifts, input_size)

    The masks of generate_masks from their grids and shifts (random_grids).
    The original code upsamples each grid with skimage's
    resize(grid, (grid + 1) * cell_size, order=1, mode="reflect",
    anti_aliasing=False), which is scipy.ndimage.zoom(grid, ..., order=1,
    mode="mirror", grid_mode=True): bilinear interpolation between the
    centers of the cells, mirrored at the cells of the border. Then it cuts
    out the window of the input size at the shift. Bilinear interpolation
    works on the rows and on the columns one after the other, so the
    upsampled grid is rows @ grid @ columns.T, where column j of rows is
    cell j alone, upsampled along the rows (zoom of the identity matrix).
    We compute this only for the rows and columns of the window, which is
    6 to 9 times faster than zoom for each mask and gives the same values
    (up to 1e-16).

    Input:
        grids:      n_masks x grid x grid (bool or 0 and 1)
        shifts:     n_masks x 2, shift in rows and in columns
        input_size: (height, width) of the network input image

    Output:
        masks: float32 array, n_masks x height x width, values 0 to 1

    Hebartlab, 2026/10/02

    See also: generate_masks, random_grids
    """

    from scipy import ndimage  # only needed here

    grids = np.asarray(grids, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=int).reshape(-1, 2)
    n_cells = grids.shape[1]

    def interpolation(n_pixels):
        # (n_cells + 1) * cell_size x n_cells: column j is cell j upsampled
        cell_size = int(np.ceil(n_pixels / n_cells))
        up_size = (n_cells + 1) * cell_size
        return ndimage.zoom(np.eye(n_cells), (up_size / n_cells, 1), order=1, mode="mirror", grid_mode=True)

    height, width = input_size
    rows = interpolation(height)[shifts[:, :1] + np.arange(height)]       # n_masks x height x n_cells
    columns = interpolation(width)[shifts[:, 1:] + np.arange(width)]      # n_masks x width x n_cells
    return (rows @ grids @ columns.transpose(0, 2, 1)).astype(np.float32)


def mask_sums(predictions, masks):
    """
    sums, coverage = mask_sums(predictions, masks)

    The sums that the maps of RISE are made of: for each dimension, the sum
    of the masks weighted by the predictions of the masked images, and the
    sum of the masks. rise adds them up batch by batch.

    Input:
        predictions: n_masks x n_dims, the predicted dimensions of the
                     masked images
        masks:       n_masks x height x width

    Output:
        sums:     float64, n_dims x height x width: sum over the masks of
                  prediction * mask
        coverage: float64, height x width: sum of the masks

    Hebartlab, 2026/10/02

    See also: saliency_maps, rise
    """

    predictions = np.asarray(predictions, dtype=np.float64)
    masks = np.asarray(masks)
    flat = masks.reshape(masks.shape[0], -1).astype(np.float64)
    sums = (predictions.T @ flat).reshape(predictions.shape[1], *masks.shape[1:])
    coverage = flat.sum(axis=0).reshape(masks.shape[1:])
    return sums, coverage


def saliency_maps(sums, coverage, n_masks, p, normalization="pixel"):
    """
    maps = saliency_maps(sums, coverage, n_masks, p, normalization="pixel")

    The map of each dimension from the sums of mask_sums:
        "pixel"     sums / coverage: at each pixel, the average of the
                    predictions, weighted by the masks. Pixels that no mask
                    kept (coverage 0, only with very few masks) are nan.
        "original"  sums / (n_masks * p), as in the original RISE code
                    (sal = p.T @ masks / N / p1). n_masks * p is the
                    expected value of coverage, so the maps also contain how
                    often each pixel was kept.

    Input:
        sums, coverage: from mask_sums, summed over all masks
        n_masks:        number of masks
        p:              probability that a cell of the masks is kept
        normalization:  "pixel" (default) or "original"

    Output:
        maps: float64, n_dims x height x width

    Hebartlab, 2026/10/02

    See also: mask_sums, relevance_map, rise
    """

    if normalization == "pixel":
        with np.errstate(invalid="ignore", divide="ignore"):
            maps = sums / coverage
        maps[..., coverage == 0] = np.nan
        return maps
    if normalization == "original":
        return sums / (n_masks * p)
    raise ValueError(f"Unknown normalization {normalization!r}. Please use 'pixel' or 'original'.")


def relevance_map(dimension_maps, embedding):
    """
    relevance = relevance_map(dimension_maps, embedding)

    The relevance map of an image: the maps of all dimensions, weighted by
    the predicted dimension values of the image and divided by their sum,
    as compute_aggregate_saliency in the code of the DimPred paper. If all
    predicted values are 0, the map is nan.

    Input:
        dimension_maps: n_dims x height x width
        embedding:      n_dims, the predicted dimension values of the image

    Output:
        relevance: float64, height x width

    Hebartlab, 2026/10/02

    See also: saliency_maps, rise
    """

    dimension_maps = np.asarray(dimension_maps, dtype=np.float64)
    embedding = np.asarray(embedding, dtype=np.float64)
    total = embedding.sum()
    if total == 0:
        return np.full(dimension_maps.shape[1:], np.nan)
    return np.tensordot(embedding, dimension_maps, axes=1) / total


def resize_maps(maps, size):
    """
    resized = resize_maps(maps, size)

    Resize maps (... x height x width) to size x size, bilinear with
    antialiasing (torch.nn.functional.interpolate). Maps that have this size
    already, or size None, are returned unchanged.
    """

    import torch

    maps = np.asarray(maps, dtype=np.float64)
    if size is None or maps.shape[-2:] == (size, size):
        return maps
    flat = torch.from_numpy(np.ascontiguousarray(maps.reshape(-1, 1, *maps.shape[-2:])))
    resized = torch.nn.functional.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False,
                                              antialias=True)
    return resized.numpy().reshape(*maps.shape[:-2], size, size)


def overlay(view, saliency):
    """
    image = overlay(view, saliency)

    A map on top of the image, as in Figure 7 of the DimPred paper: the map
    is scaled from its minimum to its maximum, shown in jet colors (dark
    blue for the minimum, dark red for the maximum) and mixed with the
    image, 40% map and 60% image. A map with the same value everywhere
    shows the image only, and so do pixels that are nan.

    Input:
        view:     uint8 array, height x width x 3 (RGB), e.g. result["view"][i] of rise
        saliency: map with the same height and width, e.g. result["relevance"][i]

    Output:
        image: uint8 array, height x width x 3 (RGB), e.g. for
               PIL.Image.fromarray(image).save("relevance.png")

    Hebartlab, 2026/10/02

    See also: rise
    """

    view = np.asarray(view)
    saliency = np.asarray(saliency, dtype=np.float64)
    known = np.isfinite(saliency)
    if not known.any() or not saliency[known].max() > saliency[known].min():
        return view.astype(np.uint8)
    scaled = (saliency - saliency[known].min()) / (saliency[known].max() - saliency[known].min())
    image = 0.4 * 255 * jet(np.where(known, scaled, 0)) + 0.6 * view
    image[~known] = view[~known]
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def jet(values):
    """
    colors = jet(values)

    The jet colormap (as jet in MATLAB): for values from 0 to 1, the colors
    go from dark blue over blue, cyan, yellow and red to dark red. colors
    has one more dimension than values, with red, green and blue (0 to 1).
    """

    values = np.clip(np.asarray(values, dtype=np.float64), 0, 1)[..., None]
    return np.clip(1.5 - np.abs(4 * values - np.array([3.0, 2.0, 1.0])), 0, 1)


def check_whole_number(name, value, minimum):
    """Error if value is not a whole number of at least minimum (bool is not a number here)."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} has to be a whole number of at least {minimum}, not {value!r}.")
