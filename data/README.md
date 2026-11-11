# Data Directory

This folder contains only lightweight, reproducible metadata in the GitHub-ready project package.

## Included

- manifests/bsds300_source_split.csv: deterministic source-level split manifest.

## Excluded from Git

- raw/: the official BSDS300 archive and extracted images.
- Any locally created temporary training data.

The raw source data is intentionally excluded because it is externally owned and must be obtained from the official source under its own terms.

## Prepare the data

From the project root:

    python -m src.prepare_data

The script downloads the official BSDS300 archive if it is absent, safely extracts it into data/raw/BSDS300, and writes the manifest. It expects the official dataset counts of 200 train images and 100 test images.

The resulting project split is:

| Split | Images |
| --- | ---: |
| Train | 160 |
| Validation | 40 |
| Test | 100 |

The project uses the manifest to prevent source-image leakage between the splits. See [../DATASET_NOTICE.md](../DATASET_NOTICE.md) before using the downloaded data or project demo assets.
