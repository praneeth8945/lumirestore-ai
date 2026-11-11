"""Run LumiRestore reconstruction from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
from torchvision.transforms import functional as transform_functional

from src.metrics import psnr_per_image, ssim_per_image
from src.predict import load_model_bundle, luminance_mean, restore_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct one low-light photograph.")
    parser.add_argument("--input", required=True, help="Low-light input image.")
    parser.add_argument("--output", default="outputs/reconstructed.png", help="Output PNG path.")
    parser.add_argument("--report", default="outputs/reconstruction_report.json", help="JSON report path.")
    parser.add_argument("--blend", type=float, default=1.0, help="Model blend in the [0, 1] range.")
    parser.add_argument(
        "--reference",
        help="Optional aligned clean reference image. Enables PSNR and SSIM reporting.",
    )
    arguments = parser.parse_args()

    output_path = Path(arguments.output)
    report_path = Path(arguments.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(arguments.input) as opened_input:
        low_light = ImageOps.exif_transpose(opened_input).convert("RGB")
    result = restore_image(low_light, load_model_bundle(), blend=arguments.blend)
    result.image.save(output_path)
    report: dict[str, object] = {
        "model": "ResidualUNet",
        "input": arguments.input,
        "output": str(output_path),
        "input_size": result.input_size,
        "blend": arguments.blend,
        "tiled_inference": result.tiled,
        "tile_count": result.tile_count,
        "elapsed_ms": round(result.elapsed_ms, 3),
        "luminance_before": round(luminance_mean(low_light), 3),
        "luminance_after": round(luminance_mean(result.image), 3),
    }

    if arguments.reference:
        with Image.open(arguments.reference) as opened_reference:
            reference = ImageOps.exif_transpose(opened_reference).convert("RGB")
        if reference.size != result.image.size:
            raise ValueError("Reference image dimensions must exactly match the restored output.")
        output_tensor = transform_functional.pil_to_tensor(result.image).float().unsqueeze(0) / 255.0
        reference_tensor = transform_functional.pil_to_tensor(reference).float().unsqueeze(0) / 255.0
        report["reference_metrics"] = {
            "psnr": round(float(psnr_per_image(output_tensor, reference_tensor)[0]), 6),
            "ssim": round(float(ssim_per_image(output_tensor, reference_tensor)[0]), 6),
            "mae": round(float((output_tensor - reference_tensor).abs().mean()), 6),
        }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Reconstruction saved to: {output_path}")
    print(f"CPU inference: {result.elapsed_ms:.1f} ms")
    print(f"Analysis report: {report_path}")


if __name__ == "__main__":
    main()
