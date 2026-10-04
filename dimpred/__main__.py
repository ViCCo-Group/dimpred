"""
Command line tool of dimpred: predict the SPoSE dimensions of images.

Usage:
    python -m dimpred IMAGE [IMAGE ...] [--model NAME] [--out FILE] [--features-only] [--device DEV] [--batch-size N]
    python -m dimpred --features FILE.mat [--model NAME] [--out FILE]
    python -m dimpred --rise IMAGE [IMAGE ...] [--model NAME] [--out FILE] [--n-masks N] [--png FOLDER]
                      [--device DEV] [--batch-size N]

The first form extracts the network features of the images and predicts
their dimensions (needs torch and open_clip). IMAGE can be image files or
folders. The images in a folder are taken in sorted order (see find_images),
everything else stays in the given order. With --features-only, the features
are saved instead of the predictions (the MATLAB function
dimpred_extract_features uses this).

The second form predicts the dimensions from precomputed features, saved in
a .mat file as the variable "features" (n_images x n_features) and,
optionally, "files" (cell array with one name per image). This does not need
torch.

The third form computes RISE heatmaps of the images (see dimpred.rise): for
each image, the map of each dimension and the relevance map. This is slow:
with 6000 masks, about 10 min per image with RN50x64 and less than 1 min with
AligNet on the GPU of an Apple M1 Max. Fewer masks (e.g. --n-masks 2000)
still give stable maps. For heatmaps we recommend --model rn50x64_66d_ridge
(a convolutional network); AligNet (default model) is the fast option.

Options:
    --model:          name of a shipped model or path of a model file
                      (default: alignet_siglip2b_66d_kernel)
    --out:            output file, .csv (default: dimpred_predictions.csv) or
                      .mat; with --rise .mat (default: dimpred_heatmaps.mat)
                      or .npz
    --features-only:  save the features instead of the predictions
    --rise:           compute RISE heatmaps instead of predictions
    --n-masks:        number of masks for --rise (default: 6000)
    --png:            with --rise, also save PNG files in this folder
    --device:         cuda, mps or cpu (default: the first that is available)
    --batch-size:     images passed through the network at once (default: 32)

Output: a .csv file with the header "image,<label 1>,...,<label n>" (or
"image,feature_1,..." with --features-only) and one row per image, or a
.mat file with the variables files, labels, model (name of the model) and
embedding (n_images x n_dims) or, with --features-only, features
(n_images x n_features).
With --rise: a .mat or .npz file with relevance (n_images x 224 x 224),
dimension_maps (n_images x n_dims x 224 x 224), embedding (n_images x n_dims,
the predictions of the images without masks), labels, files, model, view
(n_images x 224 x 224 x 3, uint8, the image as the network sees it) and
settings (a struct in the .mat file, a JSON text in the .npz file). With
--png FOLDER, for each image <name>_relevance.png (the relevance map on the
view of the image) and <name>_top<rank>_dim<k>.png for the 3 dimensions with
the largest predicted values (k counts the dimensions from 1). If two images
have the same name, the names start with the number of the image (1_, 2_, ...).

Examples:
    python -m dimpred my_images --out predictions.csv
    python -m dimpred cat.jpg dog.jpg --model rn50x64_49d_ridge --out predictions.mat
    python -m dimpred --features my_features.mat --out predictions.csv
    python -m dimpred --rise cat.jpg --model rn50x64_66d_ridge --n-masks 2000 --png cat_maps

Hebartlab, 2026/09/30

See also: dimpred.extract_features, dimpred.predict, dimpred.rise
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import scipy.io

import dimpred

# History:
# 2026/10/04: new default model alignet_siglip2b_66d_kernel
# 2026/10/02: an empty MKL_NUM_THREADS (as from MATLAB) is removed before torch is imported
# 2026/10/02: --rise, --n-masks and --png for the RISE heatmaps (dimpred.rise)
# 2026/10/02: new default model alignet_siglip2b_66d_ridge
# 2026/09/30: written for the first release of the package


def main(argv=None):
    """
    main(argv=None)

    Run the command line tool: read the arguments (argv, default: the
    arguments of the command line), get the features (from the images or
    from a features file), predict the dimensions and save the result, or,
    with --rise, compute and save the heatmaps. Errors in the input end the
    program with one message and exit code 1 (wrong arguments with exit
    code 2), without a Python traceback.
    """

    # MATLAB starts Python with MKL_NUM_THREADS set to an empty text, and
    # torch then warns on every run that the value is invalid. Without the
    # variable, torch uses its default number of threads, as it does anyway.
    # torch is only imported later, in extract_features and rise.
    if os.environ.get("MKL_NUM_THREADS") == "":
        del os.environ["MKL_NUM_THREADS"]

    parser = argparse.ArgumentParser(
        prog="python -m dimpred",
        description="Predict the SPoSE dimensions of images, from the images or from precomputed features, or "
                    "compute RISE heatmaps of the images (--rise).")
    parser.add_argument("images", nargs="*", metavar="IMAGE",
                        help="image files or folders (the images of a folder are taken in sorted order)")
    parser.add_argument("--features", metavar="FILE.mat",
                        help="predict from the features in this file (variable 'features', optional 'files')")
    parser.add_argument("--model", default=dimpred.DEFAULT_MODEL, metavar="NAME",
                        help=f"name of a shipped model ({', '.join(dimpred.list_models())}) or path of a model "
                             f"file (default: {dimpred.DEFAULT_MODEL})")
    parser.add_argument("--out", metavar="FILE",
                        help="output file, .csv or .mat (default: dimpred_predictions.csv); with --rise .mat or "
                             ".npz (default: dimpred_heatmaps.mat)")
    parser.add_argument("--features-only", action="store_true",
                        help="save the network features instead of the predictions")
    parser.add_argument("--rise", action="store_true",
                        help="compute RISE heatmaps of the images instead of predictions (slow: with 6000 masks "
                             "about 10 min per image with RN50x64, less than 1 min with AligNet, on the GPU of an "
                             "Apple M1 Max)")
    parser.add_argument("--n-masks", type=int, metavar="N",
                        help="number of masks for --rise (default: 6000); 2000 masks still give stable maps")
    parser.add_argument("--png", metavar="FOLDER",
                        help="with --rise, also save PNG files of the relevance map and of the 3 dimensions "
                             "with the largest values of each image in this folder")
    parser.add_argument("--device", metavar="DEV", help="cuda, mps or cpu (default: the first that is available)")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N",
                        help="images passed through the network at once (default: 32)")
    args = parser.parse_args(argv)

    # Check input (parser.error prints the usage and exits with code 2)
    if not args.images and not args.features:
        parser.error("please give image files or folders, or a features file with --features")
    if args.images and args.features:
        parser.error("please give either images or --features, not both")
    if args.features and args.features_only:
        parser.error("--features-only only works with images")
    if args.rise and (args.features or args.features_only):
        parser.error("--rise only works with images, not with --features or --features-only")
    if not args.rise and (args.n_masks is not None or args.png is not None):
        parser.error("--n-masks and --png only work with --rise")
    if args.rise:
        if args.out is None:
            args.out = "dimpred_heatmaps.mat"
        if not args.out.lower().endswith((".mat", ".npz")):
            parser.error(f"with --rise, --out has to be a .mat or a .npz file, not {args.out}")
        if args.n_masks is None:
            args.n_masks = 6000
    else:
        if args.out is None:
            args.out = "dimpred_predictions.csv"
        if not args.out.lower().endswith((".csv", ".mat")):
            parser.error(f"--out has to be a .csv or a .mat file, not {args.out}")

    # Errors in the input are printed as one message, without a traceback
    try:
        model = dimpred.load_model(args.model)
        model_name = model["info"].get("name", args.model)

        if args.features:
            features, files = read_features_file(args.features)
        else:
            # Images: folders are expanded to their images, files are kept as
            # given. The files are checked before the message below is
            # printed, so that a wrong path gives only the error. With flush,
            # the message comes before any later error, also if the output
            # goes to a file.
            files = []
            for path in args.images:
                if os.path.isdir(path):
                    files += dimpred.find_images(path)
                elif os.path.isfile(path):
                    files.append(path)
                else:
                    raise FileNotFoundError(f"Image not found: {path}")
            if args.rise:
                run_rise(args, files, model, model_name)
                return
            print(f"Extracting the features of {images_text(len(files))} with {model['info']['network']}", flush=True)
            features = dimpred.extract_features(files, model, device=args.device, batch_size=args.batch_size)

        # Predict, or keep the features
        if args.features_only:
            variable, values = "features", features
            labels = [f"feature_{i}" for i in range(1, features.shape[1] + 1)]
        else:
            variable, values = "embedding", dimpred.predict(features, model)
            labels = model["labels"]

        save_output(args.out, files, labels, model_name, variable, values)

    except (ValueError, OSError, ImportError) as error:
        sys.exit(f"dimpred: error: {error}")

    what = "features" if args.features_only else f"predicted {values.shape[1]} dimensions"
    print(f"Saved the {what} of {images_text(len(files))} (model {model_name}) to {args.out}")


def run_rise(args, files, model, model_name):
    """
    run_rise(args, files, model, model_name)

    The route of --rise: compute the heatmaps of the images with
    dimpred.rise, save them (save_heatmaps) and, with --png, the PNG files
    (save_pngs). The output file and the PNG folder are checked before the
    heatmaps are computed, which can take hours, so that a wrong path gives
    an error at once. A broken image does too, because dimpred.rise reads
    all images before the first mask.
    """

    # The output folder has to exist, and a .mat file can hold at most 2 GB
    # per variable (about 160 images with 66 dimensions at 224 x 224)
    out_folder = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(out_folder):
        raise FileNotFoundError(f"The folder of the output file {args.out} does not exist: {out_folder}")
    n_values = len(files) * model["weights"].shape[1] * 224 * 224
    if args.out.lower().endswith(".mat") and 4 * n_values >= 2 ** 31:
        raise ValueError(f"The maps of {images_text(len(files))} are too large for a .mat file (at most 2 GB "
                         f"per variable). Please save them as .npz, or run fewer images at a time.")
    if args.png is not None:
        os.makedirs(args.png, exist_ok=True)

    print(f"Computing RISE heatmaps of {images_text(len(files))} with {model['info']['network']} and "
          f"{args.n_masks} masks", flush=True)
    result = dimpred.rise(files, model, n_masks=args.n_masks, device=args.device, batch_size=args.batch_size,
                          verbose=True)
    save_heatmaps(args.out, result)
    message = f"Saved the heatmaps of {images_text(len(files))} (model {model_name}) to {args.out}"
    if args.png is not None:
        save_pngs(args.png, result)
        message += f", PNG files in {args.png}"
    print(message)


def save_heatmaps(fname, result):
    """
    save_heatmaps(fname, result)

    Save the result of dimpred.rise as .mat or .npz, depending on the
    extension of fname, with the variables relevance, dimension_maps,
    embedding, labels, files, model, view and settings. In the .mat file,
    labels and files are cell arrays and settings is a struct (its numbers
    as double). In the .npz file, labels and files are arrays of text and
    settings is a JSON text (json.loads(str(data["settings"]))), so that
    np.load needs no allow_pickle. np.savez gets an open file, because it
    would add .npz to a name that ends with .NPZ.
    """

    arrays = {key: result[key] for key in ["relevance", "dimension_maps", "embedding", "view"]}
    if fname.lower().endswith(".mat"):
        settings = {key: float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
                    for key, value in result["settings"].items()}
        settings["input_size"] = [float(n) for n in settings["input_size"]]
        scipy.io.savemat(fname, dict(arrays,
                                     files=np.array(result["files"], dtype=object).reshape(-1, 1),
                                     labels=np.array(result["labels"], dtype=object).reshape(-1, 1),
                                     model=result["model"], settings=settings))
    else:
        with open(fname, "wb") as f:
            np.savez(f, **arrays, files=np.array(result["files"], dtype=str),
                     labels=np.array(result["labels"], dtype=str), model=np.array(result["model"]),
                     settings=np.array(json.dumps(result["settings"])))


def save_pngs(folder, result):
    """
    save_pngs(folder, result)

    PNG files of the result of dimpred.rise: for each image the relevance
    map and the maps of the 3 dimensions with the largest predicted values,
    each on top of the view of the image (dimpred.rise.overlay), named
    <name>_relevance.png and <name>_top<rank>_dim<k>.png (k counts the
    dimensions from 1). If two images have the same name, the names start
    with the number of the image (1_, 2_, ...), so that no file is
    overwritten.
    """

    from PIL import Image

    from dimpred.rise import overlay  # the module dimpred/rise.py (dimpred.rise is the function)

    names = [os.path.splitext(os.path.basename(fname))[0] for fname in result["files"]]
    if len(set(names)) < len(names):
        names = [f"{i + 1}_{name}" for i, name in enumerate(names)]
    for i, name in enumerate(names):
        view = result["view"][i]
        Image.fromarray(overlay(view, result["relevance"][i])).save(os.path.join(folder, f"{name}_relevance.png"))
        # the 3 largest values first, ties in the order of the dimensions
        top = np.argsort(-result["embedding"][i], kind="stable")[:3]
        for rank, d in enumerate(top):
            Image.fromarray(overlay(view, result["dimension_maps"][i, d])).save(
                os.path.join(folder, f"{name}_top{rank + 1}_dim{d + 1}.png"))


def read_features_file(fname):
    """
    features, files = read_features_file(fname)

    Read precomputed features from a .mat file with the variable "features"
    (n_images x n_features) and, optionally, "files" (cell array with one name
    per image). Without "files", the images are named by their row numbers
    (1, 2, ...). loadmat with simplify_cells returns one file name as str and
    the features of one image as a vector, which we turn back into a list and
    a row.
    """

    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Features file not found: {fname}")
    try:
        data = scipy.io.loadmat(fname, simplify_cells=True)
    except Exception as error:  # scipy raises many different errors for files it cannot read
        raise ValueError(f"Could not read the features file {fname} ({error}). Please save it in MATLAB with "
                         f"save(fname, 'features', 'files', '-v7').") from error
    if "features" not in data:
        variables = [name for name in data if not name.startswith("__")]
        raise ValueError(f"The file {fname} has no variable 'features' (n_images x n_features). "
                         f"Variables in the file: {', '.join(variables) or 'none'}")

    features = np.atleast_2d(np.asarray(data["features"], dtype=float))
    if "files" in data:
        files = [str(name) for name in np.atleast_1d(data["files"])]
    else:
        files = [str(row) for row in range(1, features.shape[0] + 1)]  # no names, use the row numbers
    if len(files) != features.shape[0]:
        raise ValueError(f"The file {fname} has {features.shape[0]} rows of features, but {len(files)} file names.")
    return features, files


def save_output(fname, files, labels, model_name, variable, values):
    """
    save_output(fname, files, labels, model_name, variable, values)

    Save the predictions or features (values, one row per file) as .csv or
    .mat, depending on the extension of fname. The .csv file has the header
    "image,<label 1>,...,<label n>" and one row per image, and it is written
    in UTF-8, so that file names with any characters can be saved. The .mat
    file has the variables files, labels (cell arrays), model (name of the
    model) and values, saved under the name in variable ("embedding" or
    "features").
    """

    if fname.lower().endswith(".mat"):
        scipy.io.savemat(fname, {
            "files": np.array(files, dtype=object).reshape(-1, 1),  # cell arrays in MATLAB
            "labels": np.array(labels, dtype=object).reshape(-1, 1),
            "model": model_name,
            variable: values,
        })
    else:
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image"] + list(labels))
            for image, row in zip(files, values):
                writer.writerow([image] + row.tolist())  # tolist gives all digits


def images_text(n):
    """Text for the printed messages: "1 image" or "n images"."""

    if n == 1:
        return "1 image"
    return f"{n} images"


if __name__ == "__main__":
    main()
