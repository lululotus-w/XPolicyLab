#!/usr/bin/env python3
"""Download Pi0.5 checkpoint from GCS using gcsfs with progress bar."""

import gcsfs
import os
from pathlib import Path
from tqdm import tqdm

# Source and destination
GCS_PATH = "gs://openpi-assets/checkpoints/pi05_base/params"
LOCAL_PATH = "/mnt/nfs01/whl/home_cache/openpi/openpi-assets/checkpoints/pi05_base/params"

def download_checkpoint():
    """Download checkpoint with progress tracking."""
    fs = gcsfs.GCSFileSystem()

    # Create local directory
    os.makedirs(LOCAL_PATH, exist_ok=True)

    # List all files in GCS
    print(f"Listing files from {GCS_PATH}...")
    gcs_files = fs.ls(GCS_PATH.replace("gs://", ""))
    print(f"Found {len(gcs_files)} files/directories")

    # Get all files recursively
    all_files = []
    for item in gcs_files:
        if fs.isfile(item):
            all_files.append(item)
        elif fs.isdir(item):
            all_files.extend(fs.ls(item, recursive=True))

    # Filter only files
    all_files = [f for f in all_files if fs.isfile(f)]
    print(f"Total files to download: {len(all_files)}")

    # Calculate total size
    total_size = sum(fs.size(f) for f in all_files)
    print(f"Total size: {total_size / (1024**3):.2f} GB")

    # Download with progress
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for gcs_file in all_files:
            # Get relative path
            rel_path = gcs_file.replace("openpi-assets/checkpoints/pi05_base/params/", "")
            local_file = os.path.join(LOCAL_PATH, rel_path)

            # Create parent directory
            os.makedirs(os.path.dirname(local_file), exist_ok=True)

            # Download file
            file_size = fs.size(gcs_file)
            pbar.set_postfix_str(f"File: {os.path.basename(local_file)}")

            with fs.open(gcs_file, 'rb') as src:
                with open(local_file, 'wb') as dst:
                    chunk_size = 8 * 1024 * 1024  # 8MB chunks
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        dst.write(chunk)
                        pbar.update(len(chunk))

    print(f"\n✓ Download complete: {LOCAL_PATH}")

if __name__ == "__main__":
    download_checkpoint()
