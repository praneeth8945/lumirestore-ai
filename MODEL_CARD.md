# Model Card: LumiRestore Residual U-Net

## Model summary

| Field | Value |
| --- | --- |
| Model name | LumiRestore Residual U-Net |
| Task | Low-light RGB image reconstruction |
| Framework | PyTorch |
| Inference device | CPU |
| Input | RGB photograph; original dimensions are preserved |
| Output | RGB reconstructed image, same dimensions as input |
| Checkpoint format | PyTorch state dictionary with metadata |
| Architecture | Three-level residual U-Net over adaptive-exposure baseline |

## Intended use

The model is intended for academic experimentation and local visual enhancement of ordinary RGB photographs with low-light characteristics similar to the training degradation process. It is suitable for a user-facing demo, image-processing coursework, and reproducible reconstruction experiments.

## Out-of-scope use

The model is not intended for:

- Forensic reconstruction or evidence recovery.
- Medical, biometric, satellite, autonomous-driving, or safety-critical imaging.
- Claims that it recovers scene information not present in the input.
- Quantitative quality claims on unpaired user photographs.
- Commercial redistribution of BSDS300 data or source-derived assets without checking source terms.

## Model design

The model first creates a classical global adaptive-exposure estimate. A compact residual U-Net then predicts a bounded RGB correction:

    reconstructed = clamp(adaptive_exposure(input) + bounded_residual, 0, 1)

The network uses 24 base channels and has encoder scales of 24, 48, 96, and 192 channels. It contains reflection-padded convolutions, Group Normalization, LeakyReLU activations, residual blocks, bilinear upsampling, and skip connections.

Inputs are internally replicate-padded to a multiple of eight and cropped back to their original spatial dimensions. For large inputs, inference uses overlapping 384-pixel tiles with 48-pixel overlap and Hann-window feathering.

## Training data

The clean-image source is BSDS300. The project creates paired low-light targets synthetically from clean RGB crops using gamma darkening, exposure reduction, smooth shading, mild channel casts, signal-dependent noise, and 8-bit quantisation.

The deterministic split contains 160 training images, 40 validation images, and 100 held-out test images. No source image occurs in more than one split.

The data source and terms are documented in [DATASET_NOTICE.md](DATASET_NOTICE.md).

## Training procedure

| Setting | Value |
| --- | --- |
| Seed | 2024 |
| Patch size | 128 x 128 |
| Default maximum epochs | 12 |
| Optimizer | AdamW |
| Initial learning rate | 0.002 |
| Weight decay | 0.00001 |
| Scheduler | Cosine annealing |
| Gradient clipping | Maximum norm 1.0 |
| Checkpoint selection | Highest validation PSNR |
| Early stopping | Four validation epochs without PSNR improvement |
| Loss | 0.80 Charbonnier plus 0.20 x (1 - SSIM) |

## Evaluation

The shipped checkpoint was trained for 12 epochs using seed 2024. Evaluation used 100 fixed source-disjoint BSDS300 test crops with the repository's camera-inspired synthetic degradation. The test run was generated on 2026-09-24.

| System | PSNR (dB) | SSIM | MAE | MSE |
| --- | ---: | ---: | ---: | ---: |
| Low-light input | 9.9428 | 0.3090 | — | — |
| Adaptive exposure | 21.1092 | 0.8603 | — | — |
| Residual U-Net | 26.1303 | 0.9346 | 0.050036 | 0.003992 |

The model has 2,331,147 parameters. It improved by 16.1875 dB PSNR and 0.6256 SSIM over the raw low-light input under this protocol. With eight PyTorch CPU threads and a warmed 128 x 128 patch, the measured median inference time was 27.782 ms and p95 was 32.749 ms over 12 forward passes. This is a build-environment measurement, not a hardware guarantee.

| Metric | Reporting rule |
| --- | --- |
| PSNR | Compare against the paired clean reference only |
| SSIM | Compare against the paired clean reference only |
| MAE and MSE | Report for the residual U-Net output only |
| CPU latency | State image or patch size and summary statistic |

The project explicitly distinguishes the raw input, adaptive-exposure baseline, and learned reconstruction. Results on the synthetic BSDS300 protocol should not be presented as universal real-camera performance.

## Known limitations

- Training data is synthetic low-light data rather than raw, camera-calibrated pairs.
- Global exposure correction may be insufficient for images with extreme local lighting variation.
- Strong real sensor noise, compression, blur, unusual colour science, or clipped regions may produce artefacts.
- Blending can reduce aggressive corrections, but it is not a quality guarantee.
- PSNR and SSIM are invalid without an aligned reference image.

## Monitoring and user controls

The Streamlit app exposes a blend control between the original input and model output. The command-line tool records input size, elapsed inference time, tiled inference status, and luminance before and after restoration. These controls help users inspect behaviour rather than treating the output as ground truth.

## Ethical and data considerations

The model should be used as an enhancement tool, not as proof of what a scene contained. It should not be used to identify people, infer hidden details, or support high-stakes decisions. Users are responsible for confirming that any input images and any use of BSDS300-derived demonstration assets comply with applicable rights and source terms.
