import base64
import glob
import io
import json
import os
import traceback
from pathlib import Path
from typing import Any

import runpod
from PIL import Image, ImageOps

MODEL_ID = os.getenv("MODEL_NAME", "Qwen/Qwen-Image-Edit-2511")
CACHE_ROOT = os.getenv("HF_HUB_CACHE", "/runpod-volume/huggingface-cache/hub")
MAX_INPUT_SIDE = int(os.getenv("MAX_INPUT_SIDE", "1024"))
POLICY_PATH = Path(os.getenv("POLICY_PATH", "/app/policy.json"))
_PIPELINE = None


def load_policy() -> dict[str, Any]:
    if not POLICY_PATH.exists():
        return {}
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def check_policy(job_input: dict[str, Any]) -> None:
    policy = load_policy()

    if policy.get("require_adult_confirmation", False):
        if not bool(job_input.get("adult_confirmed", False)):
            raise ValueError("adult_confirmed is required.")

    prompt = " ".join(str(job_input.get("prompt", "")).lower().split())
    for term in policy.get("custom_blocked_terms", []):
        if " ".join(str(term).lower().split()) in prompt:
            raise ValueError("Request blocked by custom policy.")


def resolve_cached_model(model_id: str) -> str:
    if "/" not in model_id:
        return model_id

    org, name = model_id.split("/", 1)
    root = os.path.join(CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(root, "refs", "main")
    snapshots = os.path.join(root, "snapshots")

    if os.path.isfile(refs_main):
        revision = Path(refs_main).read_text(encoding="utf-8").strip()
        candidate = os.path.join(snapshots, revision)
        if os.path.isdir(candidate):
            print(f"[model] Using cached snapshot: {candidate}", flush=True)
            return candidate

    candidates = sorted(
        glob.glob(os.path.join(snapshots, "*")),
        key=os.path.getmtime,
        reverse=True,
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            print(f"[model] Using cached snapshot fallback: {candidate}", flush=True)
            return candidate

    return model_id


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    import torch
    from diffusers import QwenImageEditPlusPipeline

    model_path = resolve_cached_model(MODEL_ID)
    local_only = os.path.isdir(model_path)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    print(f"[model] Loading pipeline from: {model_path}", flush=True)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
        token=token if not local_only else None,
        low_cpu_mem_usage=True,
    )

    pipe.enable_model_cpu_offload()

    if getattr(pipe, "vae", None) is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    pipe.set_progress_bar_config(disable=True)
    _PIPELINE = pipe

    print("[model] Pipeline ready", flush=True)
    return _PIPELINE


def decode_image(value: str) -> Image.Image:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image must be a non-empty base64 string or data URL.")

    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    raw = base64.b64decode(encoded, validate=False)
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")

    if max(image.size) > MAX_INPUT_SIDE:
        image.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.Resampling.LANCZOS)

    width = max(64, image.width - image.width % 16)
    height = max(64, image.height - image.height % 16)

    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    return image


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def reference_dimensions(reference: Image.Image) -> tuple[int, int]:
    # The result should follow Image 2 framing and aspect ratio.
    width, height = reference.size
    scale = min(1.0, 1024 / max(width, height))
    width = max(512, int(width * scale))
    height = max(512, int(height * scale))
    width -= width % 16
    height -= height % 16
    return width, height


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch

        job_input = job.get("input") or {}
        check_policy(job_input)

        user_prompt = str(job_input.get("prompt", "")).strip()
        if len(user_prompt) < 3:
            raise ValueError("prompt is required.")

        if not job_input.get("base_image"):
            raise ValueError("base_image is required.")
        if not job_input.get("reference_image"):
            raise ValueError("reference_image is required.")

        image_1 = decode_image(job_input["base_image"])
        image_2 = decode_image(job_input["reference_image"])

        seed = int(job_input.get("seed", 42))
        steps = max(30, min(int(job_input.get("steps", 50)), 60))
        true_cfg_scale = max(
            2.5,
            min(float(job_input.get("true_cfg_scale", 4.0)), 6.0),
        )
        width, height = reference_dimensions(image_2)

        # Keep this direct. Long contradictory prompts can cause image drift.
        prompt = (
            "Use Image 1 as the sole source for the person's identity and body. "
            "Keep the exact same adult person from Image 1: same face, facial features, "
            "hair, skin tone, body size, body shape, and body proportions. "
            "Use Image 2 only for the exact pose, outfit, camera angle, framing, "
            "lighting, and background. Replace the person in Image 2 with the person "
            "from Image 1. Produce one photorealistic image. "
            "Do not copy the face or body identity from Image 2. "
            f"Additional instruction: {user_prompt}"
        )

        negative_prompt = (
            "anime, cartoon, illustration, painting, drawing, 3d render, collage, "
            "split screen, two people, duplicate person, face blend, identity mix, "
            "different face, changed body shape, changed body size, deformed anatomy, "
            "extra limbs, blurry, low quality"
        )

        print(
            f"[job] Generating {width}x{height}; steps={steps}; "
            f"cfg={true_cfg_scale}; seed={seed}",
            flush=True,
        )

        pipe = get_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.inference_mode():
            result = pipe(
                image=[image_1, image_2],
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
                num_images_per_prompt=1,
            ).images[0]

        print("[job] Generation complete", flush=True)

        return {
            "ok": True,
            "image_base64": encode_image(result),
            "mime_type": "image/png",
            "model": MODEL_ID,
            "seed": seed,
            "steps": steps,
            "true_cfg_scale": true_cfg_scale,
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
    print("[worker] Starting Motion Studio Qwen fixed worker", flush=True)
    runpod.serverless.start({"handler": handler})
