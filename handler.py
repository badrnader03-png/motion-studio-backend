import base64
import glob
import io
import os
import traceback
from typing import Any

import runpod
from PIL import Image, ImageOps


MODEL_ID = os.getenv("MODEL_NAME", "black-forest-labs/FLUX.2-dev")
CACHE_ROOT = os.getenv(
    "HF_HUB_CACHE",
    "/runpod-volume/huggingface-cache/hub",
)
MAX_INPUT_SIDE = int(os.getenv("MAX_INPUT_SIDE", "1024"))
MAX_OUTPUT_SIDE = int(os.getenv("MAX_OUTPUT_SIDE", "1024"))

_PIPELINE = None


def resolve_cached_model(model_id: str) -> str:
    """Use RunPod's cached Hugging Face snapshot when available."""
    if "/" not in model_id:
        return model_id

    org, name = model_id.split("/", 1)
    model_root = os.path.join(CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as file:
            revision = file.read().strip()

        candidate = os.path.join(snapshots_dir, revision)
        if os.path.isdir(candidate):
            print(f"[model] Using cached snapshot: {candidate}", flush=True)
            return candidate

    candidates = sorted(
        glob.glob(os.path.join(snapshots_dir, "*")),
        key=os.path.getmtime,
        reverse=True,
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            print(f"[model] Using cached snapshot fallback: {candidate}", flush=True)
            return candidate

    print(
        f"[model] Cached snapshot unavailable; using Hugging Face ID: {model_id}",
        flush=True,
    )
    return model_id


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    print("[model] Importing torch and Flux2Pipeline...", flush=True)
    import torch
    from diffusers import Flux2Pipeline

    model_path = resolve_cached_model(MODEL_ID)
    local_only = os.path.isdir(model_path)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    print(f"[model] Loading FLUX.2 pipeline from: {model_path}", flush=True)
    pipeline = Flux2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
        token=token if not local_only else None,
        low_cpu_mem_usage=True,
    )

    # FLUX.2 is large. CPU offload lets it run on an 80/96 GB GPU with
    # sufficient system RAM, at the cost of some latency.
    pipeline.enable_model_cpu_offload()

    if getattr(pipeline, "vae", None) is not None:
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()

    pipeline.set_progress_bar_config(disable=True)
    _PIPELINE = pipeline

    print("[model] FLUX.2 pipeline ready", flush=True)
    return _PIPELINE


def decode_image(value: str) -> Image.Image:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image must be a non-empty base64 string or data URL.")

    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value

    try:
        raw = base64.b64decode(encoded, validate=False)
        image = Image.open(io.BytesIO(raw))
    except Exception as error:
        raise ValueError(f"Invalid image data: {error}") from error

    image = ImageOps.exif_transpose(image).convert("RGB")

    if max(image.size) > MAX_INPUT_SIDE:
        image.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.Resampling.LANCZOS)

    width = max(64, image.width - image.width % 16)
    height = max(64, image.height - image.height % 16)

    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    return image


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def output_dimensions(reference: Image.Image) -> tuple[int, int]:
    """Keep the reference image aspect ratio, bounded for predictable cost."""
    width, height = reference.size
    scale = min(1.0, MAX_OUTPUT_SIDE / max(width, height))

    width = max(512, int(width * scale))
    height = max(512, int(height * scale))

    # Keep dimensions divisible by 16.
    width = max(512, width - width % 16)
    height = max(512, height - height % 16)

    return width, height


def validate_job_input(
    job_input: dict[str, Any],
) -> tuple[Image.Image, Image.Image, str]:
    prompt = str(job_input.get("prompt", "")).strip()
    if len(prompt) < 3:
        raise ValueError("prompt is required.")

    base_value = job_input.get("base_image")
    reference_value = job_input.get("reference_image")

    if not base_value:
        raise ValueError("base_image is required.")
    if not reference_value:
        raise ValueError("reference_image is required.")

    return decode_image(base_value), decode_image(reference_value), prompt


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch

        print("[job] Request received", flush=True)

        job_input = job.get("input") or {}
        base_image, reference_image, prompt = validate_job_input(job_input)

        seed = int(job_input.get("seed", 0))
        steps = max(20, min(int(job_input.get("steps", 30)), 50))
        guidance_scale = max(
            1.0,
            min(float(job_input.get("guidance_scale", 2.5)), 5.0),
        )

        width, height = output_dimensions(reference_image)

        full_prompt = (
            "Create one single final photorealistic image, not a collage and not a split screen. "
            "Image 1 is the sole identity reference. Preserve the same adult person's face, "
            "facial geometry, skin tone, hair identity, and recognizable characteristics from Image 1. "
            "Image 2 is the visual reference for the requested pose, clothing, framing, composition, "
            "and environment only. Do not copy the identity or face from Image 2. "
            "Do not merge the two faces. Maintain natural anatomy and realistic proportions. "
            f"User edit instruction: {prompt}"
        )

        pipeline = get_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        print(
            f"[job] Starting FLUX.2 generation: {width}x{height}, "
            f"steps={steps}, guidance={guidance_scale}, seed={seed}",
            flush=True,
        )

        with torch.inference_mode():
            result = pipeline(
                image=[base_image, reference_image],
                prompt=full_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                caption_upsample_temperature=0.15,
                generator=generator,
            ).images[0]

        print("[job] Generation completed", flush=True)

        return {
            "ok": True,
            "image_base64": encode_image(result),
            "mime_type": "image/png",
            "model": MODEL_ID,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
        }

    except Exception as error:
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "model": MODEL_ID,
        }


if __name__ == "__main__":
    print("[worker] Starting Motion Studio FLUX.2 RunPod worker", flush=True)
    runpod.serverless.start({"handler": handler})
