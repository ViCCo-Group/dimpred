"""
Command line tool of dimpred: predict the SPoSE dimensions of images.

Usage:
    python -m dimpred IMAGE [IMAGE ...] [--model NAME] [--out FILE] [--features-only] [--device DEV] [--batch-size N]
    python -m dimpred --features FILE.mat [--model NAME] [--out FILE]

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

Options:
    --model:          name of a shipped model or path of a model file
                      (default: vitb32_66d_elastic)
    --out:            output file, .csv (default: dimpred_predictions.csv) or .mat
    --features-only:  save the features instead of the predictions
    --device:         cuda, mps or cpu (default: the first that is available)
    --batch-size:     images passed through the network at once (default: 32)

Output: a .csv file with the header "image,<label 1>,...,<label n>" (or
"image,feature_1,..." with --features-only) and one row per image, or a
.mat file with the variables files, labels, model (name of the model) and
embedding (n_images x n_dims) or, with --features-only, features
(n_images x n_features).

Examples:
    python -m dimpred my_images --out predictions.csv
    python -m dimpred cat.jpg dog.jpg --model rn50x64_49d_ridge --out predictions.mat
    python -m dimpred --features my_features.mat --out predictions.csv

Martin Hebart, 2026/09/30

See also: dimpred.extract_features, dimpred.predict
"""

import argparse
import csv
import os
import sys

import numpy as np
import scipy.io

import dimpred

# History:
# 2026/09/30: written for the first release of the package


def main(argv=None):
    """
    main(argv=None)

    Run the command line tool: read the arguments (argv, default: the
    arguments of the command line), get the features (from the images or
    from a features file), predict the dimensions and save the result.
    Errors in the input end the program with one message and exit code 1
    (wrong arguments with exit code 2), without a Python traceback.
    """

    parser = argparse.ArgumentParser(
        prog="python -m dimpred",
        description="Predict the SPoSE dimensions of images, from the images or from precomputed features.")
    parser.add_argument("images", nargs="*", metavar="IMAGE",
                        help="image files or folders (the images of a folder are taken in sorted order)")
    parser.add_argument("--features", metavar="FILE.mat",
                        help="predict from the features in this file (variable 'features', optional 'files')")
    parser.add_argument("--model", default=dimpred.DEFAULT_MODEL, metavar="NAME",
                        help=f"name of a shipped model ({', '.join(dimpred.list_models())}) or path of a model "
                             f"file (default: {dimpred.DEFAULT_MODEL})")
    parser.add_argument("--out", default="dimpred_predictions.csv", metavar="FILE",
                        help="output file, .csv or .mat (default: dimpred_predictions.csv)")
    parser.add_argument("--features-only", action="store_true",
                        help="save the network features instead of the predictions")
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
