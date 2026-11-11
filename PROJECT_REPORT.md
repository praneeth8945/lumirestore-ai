# LumiRestore AI: Project Report

## 1. Title

**LumiRestore AI: Low-Light Image Reconstruction Using a Compact Residual U-Net**

## 2. Abstract

LumiRestore AI is a local image-processing system for reconstructing low-light RGB photographs. The project combines an interpretable adaptive-exposure baseline with a compact residual U-Net that predicts spatial and colour corrections. Clean natural images from BSDS300 are converted into deterministic paired low-light examples through a camera-inspired synthetic degradation process. The model is evaluated on a source-disjoint test split using PSNR, SSIM, MAE, and MSE. A Streamlit interface and command-line tool make the trained checkpoint usable without a GPU.

The project is designed as a reproducible academic implementation. Its reported metrics apply only to the defined synthetic test protocol and must not be generalized to all real low-light cameras or environments.

## 3. Problem statement

Low-light photographs can contain weak illumination, nonuniform shading, colour shifts, reduced contrast, noise, and quantisation artefacts. Conventional brightness adjustment can improve visibility but often cannot preserve local structure and colour simultaneously.

**The problem is to reconstruct a visually faithful clean RGB image from a synthetically generated low-light observation using a CPU-friendly learning-based model, while maintaining a source-disjoint evaluation protocol and transparent reporting.**

## 4. Objectives

1. Build a reproducible paired low-light reconstruction dataset from clean natural images.
2. Train a compact residual U-Net that runs locally on a CPU.
3. Preserve input image dimensions, including non-multiples of eight.
4. Handle larger inputs through overlap-and-feather tiled inference.
5. Evaluate reconstruction quality against paired clean images using PSNR, SSIM, MAE, and MSE.
6. Deliver both an interactive application and command-line workflow.
7. Avoid fabricated results by preserving configuration, split manifests, training history, and evaluation reports.

## 5. Dataset and split protocol

The source images are from the official Berkeley Segmentation Dataset and Benchmark 300 (BSDS300). The original 200-image train partition is deterministically shuffled with seed 2024 and divided into 160 training and 40 validation images. The official 100-image BSDS300 test partition is retained as the test set.

| Partition | Source images | Purpose |
| --- | ---: | --- |
| Training | 160 | Optimize model weights |
| Validation | 40 | Select the best checkpoint using reconstruction PSNR |
| Test | 100 | Final held-out evaluation |

The manifest stores source IDs and relative paths, and validation checks ensure that no source image crosses split boundaries.

Each clean crop is degraded on the fly using gamma darkening, reduced exposure, smooth illumination gradients, per-channel colour casts, signal-dependent noise, and 8-bit quantisation. This is a camera-inspired synthetic procedure; it is not a calibrated physical camera simulation.

## 6. Methodology

### 6.1 Adaptive-exposure baseline

The input luminance is estimated using standard RGB luminance weights. A global gamma is computed from mean luminance and constrained to a safe range. This generates a stable brightness baseline before neural reconstruction.

### 6.2 Residual U-Net

The neural component is a three-level encoder-decoder with skip connections:

- Base channel count: 24.
- Channel scales: 24, 48, 96, and 192.
- Reflection-padded 3 x 3 convolutions.
- Group Normalization and LeakyReLU activations.
- Residual blocks in the encoder, bottleneck, and decoder fusion stages.
- Bilinear decoder upsampling.
- A bounded tanh output head that predicts an RGB residual.

The final result is the adaptive-exposure baseline plus the learned residual. The output is clamped to the valid RGB interval.

### 6.3 Loss and optimization

The training loss is:

    0.80 x Charbonnier loss + 0.20 x (1 - SSIM)

The default run uses AdamW with learning rate 0.002, weight decay 0.00001, cosine annealing, gradient clipping at norm 1.0, a maximum of 12 epochs, and early stopping after four validation epochs without a PSNR improvement.

### 6.4 Inference

Images at or below 384 pixels in both dimensions are restored in one pass. Larger images are split into overlapping tiles. A Hann-window feathering mask blends tile outputs to reduce seam artifacts. The original dimensions are preserved.

## 7. Evaluation protocol

The held-out test split contains one deterministic 128 x 128 crop per source image. The same synthetic degradation mechanism produces the low-light input and the source crop acts as the clean reference.

| Metric | Meaning |
| --- | --- |
| PSNR | Pixel-level fidelity; higher is better |
| SSIM | Structural similarity; higher is better |
| MAE | Mean absolute error; lower is better |
| MSE | Mean squared error; lower is better |

The report also compares the learned reconstruction with both the raw low-light input and the adaptive-exposure-only baseline.

## 8. Results

The final full run completed on 2026-09-24 using seed 2024 and all 12 configured epochs. The best validation PSNR was 26.6795 dB. The values below are taken directly from reports/evaluation_metrics.json and apply only to the fixed, source-disjoint BSDS300 synthetic test protocol.

| Test measure | Input | Adaptive exposure | Residual U-Net |
| --- | ---: | ---: | ---: |
| PSNR (dB) | 9.9428 | 21.1092 | 26.1303 |
| SSIM | 0.3090 | 0.8603 | 0.9346 |
| MAE | Not reported | Not reported | 0.050036 |
| MSE | Not reported | Not reported | 0.003992 |

The checkpoint contains 2,331,147 parameters. Training ran for 438.27 seconds. With eight PyTorch CPU threads and a warmed 128 x 128 test patch, median inference was 27.782 ms and p95 inference was 32.749 ms across 12 timed forward passes. The learned reconstruction improved on the raw input by 16.1875 dB PSNR and 0.6256 SSIM. These timing figures are a local build-environment measurement, not a promise for every laptop.

## 9. System requirements

- Python 3.11, 3.12, or 3.13.
- CPU-only PyTorch 2.6 and Torchvision 0.21.
- NumPy, Pillow, and Streamlit.
- A modern laptop CPU such as an Intel Core Ultra 7 is sufficient for inference and the compact training configuration.
- Internet access is needed only for dependency installation and the on-demand BSDS300 archive download during training.

The included model checkpoint allows local reconstruction without downloading the training corpus.

## 10. Deliverables

- Trained PyTorch state-dict checkpoint.
- Streamlit upload interface.
- Command-line inference tool with optional paired-reference metrics.
- Deterministic data preparation script and split manifest.
- Training and evaluation script.
- Automated unit tests.
- Dataset notice, model card, README, and this report.

## 11. Limitations and responsible use

This project is trained on synthetic low-light degradations. It cannot guarantee scene-accurate recovery from real images with clipped highlights, severe motion blur, nonuniform sensor noise, or information lost below the sensor noise floor. The output should not be used as forensic evidence, medical imaging analysis, or a safety-critical decision source.

The source dataset and its derivative demo assets retain their original terms. Raw BSDS300 images are intentionally excluded from GitHub-oriented packaging.

## 12. Future work

- Train with real paired low-light and normal-light datasets.
- Add perceptual and colour-consistency losses.
- Compare against zero-reference enhancement methods.
- Add uncertainty estimates or low-confidence warnings.
- Export an ONNX or quantized inference variant after validating output fidelity.
- Add a benchmark of full-resolution latency on several local CPUs.

## 13. References

1. D. Martin, C. Fowlkes, D. Tal, and J. Malik. A Database of Human Segmented Natural Images and its Application to Evaluating Segmentation Algorithms and Measuring Ecological Statistics. ICCV, 2001, pages 416-423.
2. Official BSDS project page: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
