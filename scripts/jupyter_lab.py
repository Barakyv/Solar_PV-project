"""Start JupyterLab on the repository, for a live run of the stage notebooks.

    python scripts/jupyter_lab.py [jupyterlab options]

On the college machine libzmq aborts every kernel unless TMP is too long for an AF_UNIX address
(RH-05). This sets it for the server and every kernel it starts, only where it is needed, and
removes the empty folders libzmq leaves there when the server stops.
"""

import subprocess
import sys

from _notebook import clean_zmq_tmp, enable_kernel

from solarpv.config.paths import REPO_ROOT

if not enable_kernel():
    sys.exit("no Jupyter kernel can start on this machine (RH-05)")
try:
    returncode = subprocess.call([sys.executable, "-m", "jupyterlab", *sys.argv[1:]],
                                 cwd=REPO_ROOT)
finally:
    clean_zmq_tmp()
sys.exit(returncode)
