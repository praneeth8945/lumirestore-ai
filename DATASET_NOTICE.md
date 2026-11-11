# Dataset Notice and Attribution

## Source dataset

LumiRestore AI uses clean natural images from the Berkeley Segmentation Dataset and Benchmark 300 (BSDS300) to create synthetic paired low-light examples.

- Official project page: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- Official image archive used by the preparation script: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz

The project preparation script downloads the archive on demand into data/raw and creates a deterministic manifest. Raw source images are excluded from the GitHub-oriented project package.

## Required citation

    D. Martin, C. Fowlkes, D. Tal, and J. Malik.
    A Database of Human Segmented Natural Images and its Application to
    Evaluating Segmentation Algorithms and Measuring Ecological Statistics.
    Proceedings of the International Conference on Computer Vision, 2001,
    pages 416-423.

## Use and redistribution note

The official BSDS page describes the dataset as available for non-commercial research and educational purposes. This repository does not grant any rights in BSDS300 beyond those supplied by its owners.

Before downloading, redistributing, publishing, or using the dataset or any derived demonstration asset, review the official source terms and make sure your intended use is permitted. In particular:

- Do not add the raw archive or extracted raw image corpus to a public Git repository.
- The package may contain a small low-light/reference/reconstruction demonstration triplet derived during a project run. Treat those files as source-derived and subject to the BSDS300 context and terms.
- Remove source-derived demo assets before a use that cannot comply with the dataset's educational or non-commercial terms.

## Project-specific transformation

BSDS300 images are not used as pre-existing low-light labels. LumiRestore AI generates paired low-light observations by applying camera-inspired gamma darkening, exposure reduction, smooth shading, colour casts, signal-dependent noise, and quantisation to clean crops. This generated pairing is specific to this project and does not change the source data's attribution or terms.
