#!/usr/bin/env python3
"""Package the recovered satellite data for upload into the Colab notebook.

Creates ``satellite_cnn_colab_upload.zip`` containing ``metadata_clean.csv``
plus every recovered 128x128x1 ``.npy`` crop under
``satellite_cnn_recovered/images/``. Upload that single zip to Google Colab and
point the notebook's ``DATA_DIR`` at the extracted folder (or upload the files
individually via the notebook's prompts).

Usage:
    python pack_satellite_for_colab.py [output.zip]
"""

from __future__ import annotations

import zipfile
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC_DIR = REPO / "satellite_cnn_recovered"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "satellite_cnn_colab_upload.zip"

def main() -> None:
    meta_csv = SRC_DIR / "metadata_clean.csv"
    if not meta_csv.exists():
        print(f"metadata not found: {meta_csv}")
        return
    images = sorted((SRC_DIR / "images").glob("*.npy"))
    if not images:
        print("no images found")
        return
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(meta_csv, "metadata_clean.csv")
        for img in images:
            z.write(img, f"images/{img.name}")
    print(f"Wrote {OUT} ({len(images)} images + metadata_clean.csv)")

if __name__ == "__main__":
    main()
