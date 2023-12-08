import os
import meierlab
from pathlib import Path

path = Path(meierlab.__file__).parent.parent.parent.absolute()
ATLAS_FILE= path / "tests/data/atlas.csv"
SUB_FILE= path / "tests/data/sub-atlas.tsv"

os.environ["ATLAS_FILE"] = str(ATLAS_FILE)
os.environ["SUB_FILE"] = str(SUB_FILE)