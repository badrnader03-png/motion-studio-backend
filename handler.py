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
POSE_MODEL_ID = os.getenv("POSE_MODEL_NAME", "lllyasviel/Annotators")
CACHE_ROOT = os.getenv("HF_HUB_CACHE", "/runpod-volume/huggingface-cache/hub")
MAX_INPUT_SIDE = int(os.getenv("MAX_INPUT_SIDE", "1024"))
POLICY_PATH = Path(os.getenv("POLICY_PATH", "/app/policy.json"))

_PIPELINE = None
_POSE_DETECTOR = None


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
            print(f"[cache] Using snapshot: {candidate}", flush=True)
            return candidate

    candidates = sorted(
        glob.glob(os.path.join(snapshots, "*")),
        key=os.path.getmtime,
        reverse=True,
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            print(f"[cache] Using snapshot fallback: {candidate}", flush=True)
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

    print(f"[model] Loading Qwen pipeline from: {model_path}", flush=True)

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

    print("[model] Qwen pipeline ready", flush=True)
    return _PIPELINE


def get_pose_detector():
    global _POSE_DETECTOR
    if _POSE_DETECTOR is not None:
        return _POSE_DETECTOR

    from controlnet_aux import OpenposeDetector

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(f"[pose] Loading OpenPose annotator: {POSE_MODEL_ID}", flush=True)

    _POSE_DETECTOR = OpenposeDetector.from_pretrained(
        POSE_MODEL_ID,
        cache_dir=CACHE_ROOT,
        token=token,
    )

    print("[pose] OpenPose annotator ready", flush=True)
    return _POSE_DETECTOR


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
    width, height = reference.size
    scale = min(1.0, 1024 / max(width, height))
    width = max(512, int(width * scale))
    height = max(512, int(height * scale))
    width = max(512, width - width % 16)
    height = max(512, height - height % 16)
    return width, height


def build_pose_map(reference: Image.Image) -> Image.Image:
    detector = get_pose_detector()

    print("[pose] Extracting body, hand and face pose from Image 2", flush=True)
    pose_map = detector(
        reference,
        include_body=True,
        include_hand=True,
        include_face=True,
        hand_and_face=True,
    )

    if not isinstance(pose_map, Image.Image):
        pose_map = Image.fromarray(pose_map)

    return pose_map.convert("RGB").resize(reference.size, Image.Resampling.NEAREST)


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
        pose_map = build_pose_map(image_2)

        seed = int(job_input.get("seed", 42))
        steps = max(35, min(int(job_input.get("steps", 50)), 60))
        true_cfg_scale = max(
            3.0,
            min(float(job_input.get("true_cfg_scale", 4.5)), 6.5),
        )

        width, height = reference_dimensions(image_2)

        prompt = (
            "Perform a precise three-reference image edit. "
            "Reference Image 1 contains the only person identity and body that may appear in the result. "
            "Preserve that adult person's exact face, facial geometry, hair identity, skin tone, "
            "body volume, body silhouette, limb thickness, torso proportions, waist, hips, chest, "
            "height appearance and all recognizable identity details from Reference Image 1. "
            "Reference Image 2 defines the exact target outfit, exact scene, camera position, framing, "
            "perspective and lighting. "
            "Reference Image 3 is an OpenPose skeleton extracted from Reference Image 2. "
            "Follow Reference Image 3 closely for the location and bending of the head, torso, shoulders, "
            "elbows, wrists, hips, knees, ankles and hands. "
            "Recreate the composition of Reference Image 2, replacing its person with the exact person "
            "from Reference Image 1. Produce one photorealistic photograph, not a collage. "
            "Never copy the face or body identity from Reference Image 2. Never average or merge identities. "
            f"Additional user instruction: {user_prompt}"
        )

        negative_prompt = (
            "anime, cartoon, illustration, painting, drawing, 3d render, collage, split screen, "
            "two people, duplicate person, face blend, identity mix, different person, different face, "
            "changed body volume, changed body shape, changed proportions, slimmed body, enlarged body, "
            "wrong pose, standing pose when target is sitting, missing limbs, extra limbs, extra fingers, "
            "deformed anatomy, blurry, low quality"
        )

        print(
            f"[job] Qwen 3-reference generation {width}x{height}; "
            f"steps={steps}; true_cfg={true_cfg_scale}; seed={seed}",
            flush=True,
        )

        pipe = get_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.inference_mode():
            result = pipe(
                image=[image_1, image_2, pose_map],
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

        output = {
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

        if bool(job_input.get("return_pose_map", False)):
            output["pose_map_base64"] = encode_image(pose_map)

        return output

    except Exception as error:
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "model": MODEL_ID,
        }


if __name__ == "__main__":
    print("[worker] Starting Motion Studio Qwen Pose v3", flush=True)
    runpod.serverless.start({"handler": handler})
