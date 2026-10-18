import sys

from solarpv.eda.quality import MANIFEST, run

run()
print(f"wrote {MANIFEST}")
sys.exit(0)
