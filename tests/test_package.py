"""
Tests of the package as a whole: dimpred without torch.

torch and open_clip are only needed for extract_features, and they are
imported inside that function. So the rest of dimpred has to work when they
are not installed, and it must not import them when they are installed:
importing torch takes several seconds and a lot of memory, which would make
every call of the command line tool slow. Each test runs a new Python
process, because the test process itself may have imported torch already.

Martin Hebart, 2026/09/30

See also: test_cli.py, test_extract_features_errors.py
"""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from helpers import DEFAULT_MODEL, IMAGES, TOL_MODEL, assert_close, output_of, python_env

# History:
# 2026/09/30: new file; the test without torch moved here from test_predict.py,
#   and a test that importing dimpred does not import torch

TORCH_INSTALLED = importlib.util.find_spec("torch") is not None  # does not import torch

# Uses all functions that do not need torch
USE_DIMPRED = (
    "import numpy as np\n"
    "import dimpred\n"
    "names = dimpred.list_models()\n"
    "model = dimpred.load_model()\n"
    "prediction = dimpred.predict(np.load('features.npy'), model)\n"
    "np.save('prediction.npy', prediction)\n"
    "dimpred.similarity(prediction)\n"
    f"dimpred.find_images({IMAGES!r})\n"
)


def run_python(code, cwd):
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=python_env(), capture_output=True, text=True,
                          timeout=300)


def test_package_works_without_torch(ref, tmp_path):
    # torch and open_clip cannot be imported in this process, as if they
    # were not installed
    np.save(tmp_path / "features.npy", ref["features_vitb32"][:5])
    block_torch = (
        "import sys\n"
        "for name in ('torch', 'torchvision', 'open_clip'):\n"
        "    sys.modules[name] = None\n"
    )
    result = run_python(block_torch + USE_DIMPRED, tmp_path)
    assert result.returncode == 0, f"dimpred does not work without torch\n{output_of(result)}"
    assert_close(np.load(tmp_path / "prediction.npy"), ref["expected_" + DEFAULT_MODEL][:5], TOL_MODEL,
                 "prediction without torch")


@pytest.mark.skipif(not TORCH_INSTALLED, reason="torch is not installed, so dimpred cannot import it")
def test_dimpred_does_not_import_torch(ref, tmp_path):
    np.save(tmp_path / "features.npy", ref["features_vitb32"][:5])
    check = (
        "import sys\n"
        "with open('imported.txt', 'w') as f:\n"
        "    f.write(','.join(m for m in ('torch', 'open_clip') if m in sys.modules))\n"
    )
    result = run_python(USE_DIMPRED + check, tmp_path)
    assert result.returncode == 0, f"dimpred failed\n{output_of(result)}"
    imported = (tmp_path / "imported.txt").read_text()
    assert not imported, (
        f"import dimpred and the functions without feature extraction imported {imported}. They should "
        f"only be imported inside extract_features.")
