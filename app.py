"""Interactive Streamlit application for LumiRestore AI.

Run from the project root with:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import warnings
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import ASSETS_DIR, MODEL_PATH
from src.metrics import psnr_per_image, ssim_per_image
from src.predict import (
    ModelBundle,
    luminance_mean,
    load_model_bundle,
    restore_image,
)


MAX_INPUT_PIXELS = 16_000_000
SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
DEMO_INPUT_PATH = ASSETS_DIR / "demo_low_light.png"
DEMO_REFERENCE_PATH = ASSETS_DIR / "demo_reference.png"


def safe_open_image(source: str | Path | BinaryIO) -> Image.Image:
    """Decode an upload safely, apply EXIF rotation, and return RGB pixels."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as opened:
                if opened.width * opened.height > MAX_INPUT_PIXELS:
                    raise ValueError(
                        "This image exceeds the 16-megapixel app limit. "
                        "Resize it and try again."
                    )
                opened.load()
                normalized = ImageOps.exif_transpose(opened).convert("RGB")
                return normalized.copy()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("The image is too large to process safely.") from error
    except UnidentifiedImageError as error:
        raise ValueError("Please upload a valid PNG, JPEG, or WebP image.") from error
    except OSError as error:
        raise ValueError("This image could not be decoded. Try exporting it again.") from error


def image_fingerprint(image: Image.Image | None) -> str | None:
    """Create a compact identity for invalidating stale app results."""
    if image is None:
        return None
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size}".encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Return a single [1, 3, H, W] RGB tensor in the [0, 1] range."""
    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def reference_metrics(output: Image.Image, reference: Image.Image) -> dict[str, float]:
    """Calculate reference metrics only for pixel-aligned images."""
    if output.size != reference.size:
        raise ValueError("Reference dimensions must exactly match the input image.")
    output_tensor = image_to_tensor(output)
    reference_tensor = image_to_tensor(reference)
    return {
        "psnr_db": round(float(psnr_per_image(output_tensor, reference_tensor)[0]), 4),
        "ssim": round(float(ssim_per_image(output_tensor, reference_tensor)[0]), 4),
        "mae": round(float((output_tensor - reference_tensor).abs().mean()), 6),
    }


def image_as_png(image: Image.Image) -> bytes:
    """Encode a display image as a download-ready PNG."""
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


@st.cache_resource(show_spinner="Loading the CPU reconstruction model…")
def get_model_bundle() -> ModelBundle:
    """Cache the checkpoint across Streamlit reruns."""
    return load_model_bundle(MODEL_PATH)


def render_sidebar(bundle: ModelBundle) -> None:
    """Keep key operating limits visible without crowding the result view."""
    with st.sidebar:
        st.header("LumiRestore AI")
        st.caption("Residual U-Net low-light image reconstruction")
        st.divider()
        st.write("**Model**")
        st.caption(f"{bundle.architecture} · {bundle.base_channels} base channels")
        st.write("**Local inference**")
        st.caption("CPU-ready; large images are reconstructed in overlapping tiles.")
        st.write("**Upload limit**")
        st.caption("PNG, JPEG, or WebP up to 16 megapixels. Images are converted to RGB locally.")
        st.divider()
        st.caption(
            "PSNR and SSIM are shown only when a clean, pixel-aligned reference image is supplied."
        )


def choose_images() -> tuple[Image.Image | None, Image.Image | None, str]:
    """Render source controls and return normalized input/reference images."""
    has_demo = DEMO_INPUT_PATH.exists()
    choices = ["Upload an image"]
    if has_demo:
        choices.insert(0, "Use the packaged demo")
    source_choice = st.radio("Choose an input", choices, horizontal=True)

    if source_choice == "Use the packaged demo":
        try:
            input_image = safe_open_image(DEMO_INPUT_PATH)
            reference_image = None
            use_reference = st.checkbox(
                "Evaluate against the packaged aligned reference",
                value=DEMO_REFERENCE_PATH.exists(),
                disabled=not DEMO_REFERENCE_PATH.exists(),
                help="This is meaningful only because the demo input and reference are aligned.",
            )
            if use_reference:
                reference_image = safe_open_image(DEMO_REFERENCE_PATH)
            return input_image, reference_image, "packaged_demo"
        except ValueError as error:
            st.error(f"The packaged demo could not be loaded: {error}")
            return None, None, "packaged_demo"

    uploaded_input = st.file_uploader(
        "Low-light input image",
        type=SUPPORTED_IMAGE_TYPES,
        help="PNG, JPEG, or WebP; maximum 16 megapixels after decoding.",
    )
    if uploaded_input is None:
        st.info("Upload a low-light photograph, or switch to the packaged demo.")
        return None, None, "upload"
    try:
        input_image = safe_open_image(uploaded_input)
    except ValueError as error:
        st.error(str(error))
        return None, None, "upload"

    with st.expander("Optional clean reference for PSNR / SSIM", expanded=False):
        st.caption(
            "Use a clean image of the exact same scene at the exact same pixel dimensions. "
            "A visually similar but unaligned image will not produce meaningful metrics."
        )
        uploaded_reference = st.file_uploader(
            "Aligned clean reference image",
            type=SUPPORTED_IMAGE_TYPES,
            key="reference_upload",
        )
    if uploaded_reference is None:
        return input_image, None, "upload"
    try:
        reference_image = safe_open_image(uploaded_reference)
    except ValueError as error:
        st.warning(f"Reference ignored: {error}")
        return input_image, None, "upload"
    if reference_image.size != input_image.size:
        st.warning(
            "Reference ignored: its dimensions do not match the input image exactly."
        )
        return input_image, None, "upload"
    return input_image, reference_image, "upload"


def make_metadata(
    *,
    bundle: ModelBundle,
    input_image: Image.Image,
    source: str,
    blend: float,
    result_image: Image.Image,
    elapsed_ms: float,
    tiled: bool,
    tile_count: int,
    metrics: dict[str, float] | None,
) -> dict[str, object]:
    """Build a JSON-serializable record for the download button."""
    return {
        "application": "LumiRestore AI",
        "model": {
            "architecture": bundle.architecture,
            "base_channels": bundle.base_channels,
            "patch_size": bundle.patch_size,
            "best_validation_psnr_db": round(bundle.best_validation_psnr, 4),
        },
        "input_source": source,
        "input_size": {"width": input_image.width, "height": input_image.height},
        "output_size": {"width": result_image.width, "height": result_image.height},
        "blend": round(float(blend), 2),
        "inference": {
            "elapsed_ms": round(float(elapsed_ms), 3),
            "tiled": bool(tiled),
            "tile_count": int(tile_count),
        },
        "luminance": {
            "before": round(luminance_mean(input_image), 3),
            "after": round(luminance_mean(result_image), 3),
        },
        "aligned_reference_metrics": metrics,
    }


def render_result(record: dict[str, object]) -> None:
    """Render a stored restoration result and its download controls."""
    input_image = record["input_image"]
    output_image = record["output_image"]
    metadata = record["metadata"]
    assert isinstance(input_image, Image.Image)
    assert isinstance(output_image, Image.Image)
    assert isinstance(metadata, dict)

    st.subheader("Reconstruction")
    original_column, restored_column = st.columns(2)
    with original_column:
        st.image(input_image, caption="Original low-light input", use_container_width=True)
    with restored_column:
        st.image(output_image, caption="LumiRestore reconstruction", use_container_width=True)

    inference = metadata["inference"]
    luminance = metadata["luminance"]
    assert isinstance(inference, dict)
    assert isinstance(luminance, dict)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Input luminance", f"{luminance['before']:.1f} / 100")
    metric_columns[1].metric("Output luminance", f"{luminance['after']:.1f} / 100")
    metric_columns[2].metric("CPU inference", f"{inference['elapsed_ms']:.1f} ms")
    tile_label = str(inference["tile_count"])
    metric_columns[3].metric("Tiles processed", tile_label)

    metrics = metadata.get("aligned_reference_metrics")
    if isinstance(metrics, dict):
        st.subheader("Aligned-reference evaluation")
        evaluation_columns = st.columns(3)
        evaluation_columns[0].metric("PSNR", f"{metrics['psnr_db']:.2f} dB")
        evaluation_columns[1].metric("SSIM", f"{metrics['ssim']:.4f}")
        evaluation_columns[2].metric("MAE", f"{metrics['mae']:.4f}")
    else:
        st.caption("No aligned clean reference was supplied, so PSNR and SSIM were not calculated.")

    st.download_button(
        "Download reconstruction (PNG)",
        data=image_as_png(output_image),
        file_name="lumirestore_reconstruction.png",
        mime="image/png",
        use_container_width=True,
    )
    st.download_button(
        "Download run metadata (JSON)",
        data=json.dumps(metadata, indent=2),
        file_name="lumirestore_metadata.json",
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    """Run the Streamlit UI."""
    st.set_page_config(
        page_title="LumiRestore AI",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("LumiRestore AI")
    st.caption(
        "Low-light image reconstruction with a compact Residual U-Net. "
        "Everything runs locally on your CPU."
    )

    if not MODEL_PATH.exists():
        st.error("The trained model checkpoint is not available yet.")
        st.code("python -m src.train_model", language="bash")
        st.info("Run the training command from the project folder, then restart this app.")
        return

    try:
        bundle = get_model_bundle()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        st.error("The model checkpoint could not be loaded.")
        st.caption(f"Details: {error}")
        st.info("Recreate it with `python -m src.train_model`, then restart the app.")
        return

    render_sidebar(bundle)
    input_image, reference_image, source = choose_images()
    if input_image is None:
        return

    st.caption(
        f"Normalized input: {input_image.width} × {input_image.height} pixels · RGB · "
        f"{input_image.width * input_image.height / 1_000_000:.2f} MP"
    )
    blend = st.slider(
        "Model restoration strength",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="0 keeps the input unchanged; 1 uses the full model reconstruction.",
    )

    request_signature = (
        image_fingerprint(input_image),
        image_fingerprint(reference_image),
        round(float(blend), 2),
        source,
    )
    if st.button("Restore image", type="primary", use_container_width=True):
        try:
            with st.spinner("Reconstructing image locally…"):
                restoration = restore_image(input_image, bundle, blend=blend)
                metrics = (
                    reference_metrics(restoration.image, reference_image)
                    if reference_image is not None
                    else None
                )
            metadata = make_metadata(
                bundle=bundle,
                input_image=input_image,
                source=source,
                blend=blend,
                result_image=restoration.image,
                elapsed_ms=restoration.elapsed_ms,
                tiled=restoration.tiled,
                tile_count=restoration.tile_count,
                metrics=metrics,
            )
            st.session_state["lumirestore_result"] = {
                "request_signature": request_signature,
                "input_image": input_image.copy(),
                "output_image": restoration.image.copy(),
                "metadata": metadata,
            }
        except (RuntimeError, ValueError, OSError) as error:
            st.error("The image could not be reconstructed.")
            st.caption(f"Details: {error}")
            return

    record = st.session_state.get("lumirestore_result")
    if isinstance(record, dict) and record.get("request_signature") == request_signature:
        render_result(record)
    else:
        st.info("Set the restoration strength, then select **Restore image**.")


if __name__ == "__main__":
    main()
