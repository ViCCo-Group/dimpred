import os

# History:
# 2026/09/30: written for the first release of the package


def list_models():
    """
    names = list_models()

    Names of the models that come with dimpred. Each model is a .mat file in
    the folder models of the package, and its name is the file name without
    .mat. Any of these names can be passed as model to load_model, predict,
    extract_features and to the command line tool (--model).

    Input:
        none

    Output:
        names: list of str, sorted alphabetically, e.g.
               ['rn50x64_49d_ridge', 'rn50x64_66d_elastic', 'rn50x64_66d_ridge', 'vitb32_66d_elastic']

    Example:
        for name in dimpred.list_models():
            print(name, dimpred.load_model(name)["info"]["note"])

    Martin Hebart, 2026/09/30

    See also: load_model
    """

    # The name of a model is the name of its file without .mat
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    names = [os.path.splitext(fname)[0] for fname in os.listdir(folder) if fname.endswith(".mat")]
    return sorted(names)
