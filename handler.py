import base64
import glob
import io
import os
import traceback
from typing import Any

import runpod
from PIL import Image, ImageOps


MODEL_ID = os.getenv("MODEL_NAME", "Qwen/Qwen-Image-Edit-2511")
CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
MAX_SIDE = int(os.getenv("MAX_INPUT_SIDE", "768"))

_PIPELINE = None


def resolve_cached_model(model_id: str) -> str:
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

    candidates = sorted(glob.glob(os.path.join(snapshots_dir, "*")))
    for candidate in candidates:
        if os.path.isdir(candidate):
            print(f"[model] Using cached snapshot fallback: {candidate}", flush=True)
            return candidate

    print(f"[model] Cached snapshot unavailable; using Hugging Face ID: {model_id}", flush=True)
    return model_id


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    print("[model] Importing torch and diffusers...", flush=True)
    import torch
    from diffusers import QwenImageEditPlusPipeline

    model_path = resolve_cached_model(MODEL_ID)
    local_only = os.path.isdir(model_path)

    print(f"[model] Loading pipeline from: {model_path}", flush=True)
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    )

    # Supported memory optimization for this pipeline.
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    _PIPELINE = pipeline
    print("[model] Pipeline ready", flush=True)
    return _PIPELINE


def decode_image(value: str) -> Image.Image:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image must be a non-empty base64 string or data URL.")

    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    raw = base64.b64decode(encoded, validate=False)

    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image).convert("RGB")

    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    width = max(64, image.width - image.width % 16)
    height = max(64, image.height - image.height % 16)

    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    return image


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def validate_job_input(job_input: dict[str, Any]) -> tuple[Image.Image, Image.Image, str]:
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
        steps = max(8, min(int(job_input.get("steps", 20)), 40))
        true_cfg_scale = max(1.0, min(float(job_input.get("true_cfg_scale", 4.0)), 8.0))
        negative_prompt = str(job_input.get("negative_prompt", " "))

        full_prompt = (
            "Image 1 is the BASE image and the only identity reference. "
            "Preserve the person's facial identity and distinguishing characteristics from Image 1. "
            "Image 2 is the REFERENCE image. Copy only the elements requested by the user. "
            "Do not blend the identities and do not copy the face from Image 2. "
            f"User instruction: {prompt}"
        )

        pipeline = get_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        print("[job] Starting image generation", flush=True)

        with torch.inference_mode():
            result = pipeline(
                image=[base_image, reference_image],
                prompt=full_prompt,
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            ).images[0]

        print("[job] Generation completed", flush=True)

        return {
            "ok": True,
            "image_base64": encode_image(result),
            "mime_type": "image/png",
            "seed": seed,
            "steps": steps,
        }

    except Exception as error:
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(error),
            "error_type": type(error).__name__,
        }


if __name__ == "__main__":
    print("[worker] Starting Motion Studio RunPod worker", flush=True)
    runpod.serverless.start({"handler": handler})
