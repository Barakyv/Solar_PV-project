import sys

from solarpv.config.cli import no_options
from solarpv.eda import quality
from solarpv.eda.quality import MANIFEST, run

no_options("python -m solarpv.eda.quality", quality.__doc__)  # X62: --help runs nothing
run()
print(f"wrote {MANIFEST}")
sys.exit(0)
