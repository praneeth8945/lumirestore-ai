"""Fast, self-contained validation tests for LumiRestore AI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.config import BASE_CHANNELS, MANIFEST_PATH, PATCH_SIZE
from src.dataset import degrade_low_light, validate_manifest
from src.metrics import psnr_per_image, reconstruction_loss, ssim_per_image
from src.model import ResidualUNet, adaptive_exposure
from src.predict import load_model_bundle, restore_image, restore_tensor


class LumiRestoreCoreTests(unittest.TestCase):
    def test_manifest_has_source_disjoint_expected_counts(self) -> None:
        counts = validate_manifest(MANIFEST_PATH, check_files=False)
        self.assertEqual(counts, {"train": 160, "validation": 40, "test": 100})

    def test_degradation_is_seed_reproducible_and_darker(self) -> None:
        clean = torch.full((3, PATCH_SIZE, PATCH_SIZE), 0.55)
        generator_one = torch.Generator().manual_seed(123)
        generator_two = torch.Generator().manual_seed(123)
        low_one = degrade_low_light(clean, generator_one)
        low_two = degrade_low_light(clean, generator_two)
        self.assertTrue(torch.equal(low_one, low_two))
        self.assertLess(float(low_one.mean()), float(clean.mean()))

    def test_metrics_match_identity_case(self) -> None:
        image = torch.rand(2, 3, 64, 64)
        self.assertAlmostEqual(float(ssim_per_image(image, image).mean()), 1.0, places=5)
        self.assertGreater(float(psnr_per_image(image, image).mean()), 80.0)
        self.assertLess(float(reconstruction_loss(image, image)), 0.01)

    def test_model_preserves_shape_and_is_finite(self) -> None:
        model = ResidualUNet(base_channels=BASE_CHANNELS).eval()
        image = torch.rand(1, 3, 133, 179)
        with torch.inference_mode():
            output = model(image)
        self.assertEqual(tuple(output.shape), tuple(image.shape))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_adaptive_exposure_brightens_a_dark_tensor(self) -> None:
        image = torch.full((1, 3, 64, 64), 0.10)
        corrected = adaptive_exposure(image)
        self.assertGreater(float(corrected.mean()), float(image.mean()))

    def test_checkpoint_and_tiled_inference_contract(self) -> None:
        model = ResidualUNet(base_channels=BASE_CHANNELS)
        checkpoint = {
            "schema_version": 1,
            "architecture": "ResidualUNet",
            "base_channels": BASE_CHANNELS,
            "patch_size": PATCH_SIZE,
            "model_state_dict": model.state_dict(),
            "best_validation_psnr": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"
            torch.save(checkpoint, checkpoint_path)
            bundle = load_model_bundle(checkpoint_path)
            input_tensor = torch.rand(3, 421, 503)
            output_tensor, tiled, tile_count = restore_tensor(
                input_tensor,
                bundle,
                tile_size=192,
                overlap=24,
            )
            self.assertTrue(tiled)
            self.assertGreater(tile_count, 1)
            self.assertEqual(tuple(output_tensor.shape), tuple(input_tensor.shape))
            image = Image.fromarray((input_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8"))
            result = restore_image(image, bundle)
        self.assertEqual(result.image.size, image.size)
        self.assertGreater(result.elapsed_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
