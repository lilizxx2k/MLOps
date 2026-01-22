# MLOps/tests/__init__.py
import os
import sys

_TEST_ROOT = os.path.dirname(__file__)                    # MLOps/tests
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)               # MLOps
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")            # MLOps/src
_MLOPS_ROOT = os.path.join(_SRC_ROOT, "mlops")             # MLOps/src/mlops

sys.path.insert(0, _SRC_ROOT)

_PATH_DATA = os.path.join(_MLOPS_ROOT)
