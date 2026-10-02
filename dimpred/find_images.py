import os

# History:
# 2026/09/30: written for the first release of the package

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def find_images(folder):
    """
    files = find_images(folder)

    All image files in a folder, sorted by file name. Use this to get the
    images of a folder for extract_features, which only takes files, so that
    the order of the rows of the features is always the order of the files
    you passed.

    Only files directly in the folder are returned (subfolders are not
    searched), with the extensions .jpg .jpeg .png .bmp .tif .tiff .webp in
    upper or lower case. Hidden files (names starting with ".") are left
    out, e.g. the "._" files that macOS writes on external drives. The names
    are sorted with a plain string sort, so upper case comes before lower
    case and "img10.jpg" before "img2.jpg" (the same order as sort in
    MATLAB). If you need another order, sort the list yourself.

    Input:
        folder: path of the folder

    Output:
        files: list of the full paths of the images (str)

    Example:
        files = dimpred.find_images("my_images")
        features = dimpred.extract_features(files)

    Hebartlab, 2026/09/30

    See also: extract_features
    """

    # Check input
    if not os.path.isdir(folder):
        if os.path.exists(folder):
            raise ValueError(f"{folder} is a file, not a folder. find_images needs a folder, single images can be "
                             f"passed to extract_features directly.")
        raise ValueError(f"The folder {folder} does not exist.")

    # Get the image files, sorted by name
    folder = os.path.abspath(folder)
    files = []
    for name in sorted(os.listdir(folder)):
        fname = os.path.join(folder, name)
        if name.lower().endswith(EXTENSIONS) and not name.startswith(".") and os.path.isfile(fname):
            files.append(fname)

    if not files:
        raise ValueError(f"No images found in {folder} (looked for files ending in {' '.join(EXTENSIONS)}).")
    return files
