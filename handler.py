import base64, glob, io, json, os, traceback
from pathlib import Path
from typing import Any
import runpod
from PIL import Image, ImageOps

MODEL_ID = os.getenv("MODEL_NAME", "Qwen/Qwen-Image-Edit-2511")
CACHE_ROOT = os.getenv("HF_HUB_CACHE", "/runpod-volume/huggingface-cache/hub")
MAX_INPUT_SIDE = int(os.getenv("MAX_INPUT_SIDE", "1024"))
POLICY_PATH = Path(os.getenv("POLICY_PATH", "/app/policy.json"))
_PIPELINE = None

def load_policy():
    if not POLICY_PATH.exists():
        return {}
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))

def check_policy(job_input):
    policy = load_policy()
    prompt = " ".join(str(job_input.get("prompt","")).lower().split())
    if policy.get("require_adult_confirmation") and not job_input.get("adult_confirmed"):
        raise ValueError("adult_confirmed is required.")
    for term in policy.get("custom_blocked_terms", []):
        if " ".join(str(term).lower().split()) in prompt:
            raise ValueError("Request blocked by custom policy.")

def resolve_cached_model(model_id):
    if "/" not in model_id:
        return model_id
    org, name = model_id.split("/", 1)
    root = os.path.join(CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(root, "refs", "main")
    snapshots = os.path.join(root, "snapshots")
    if os.path.isfile(refs_main):
        rev = Path(refs_main).read_text(encoding="utf-8").strip()
        candidate = os.path.join(snapshots, rev)
        if os.path.isdir(candidate):
            return candidate
    for candidate in sorted(glob.glob(os.path.join(snapshots, "*")), reverse=True):
        if os.path.isdir(candidate):
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
    return _PIPELINE

def decode_image(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image must be a non-empty base64 string or data URL.")
    encoded = value.split(",",1)[1] if value.startswith("data:") and "," in value else value
    raw = base64.b64decode(encoded, validate=False)
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    if max(image.size) > MAX_INPUT_SIDE:
        image.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.Resampling.LANCZOS)
    w = max(64, image.width - image.width % 16)
    h = max(64, image.height - image.height % 16)
    return image if image.size == (w,h) else image.resize((w,h), Image.Resampling.LANCZOS)

def encode_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handler(job: dict[str, Any]):
    try:
        import torch
        job_input = job.get("input") or {}
        check_policy(job_input)
        prompt = str(job_input.get("prompt","")).strip()
        if len(prompt) < 3:
            raise ValueError("prompt is required.")
        if not job_input.get("base_image"):
            raise ValueError("base_image is required.")
        if not job_input.get("reference_image"):
            raise ValueError("reference_image is required.")
        base_image = decode_image(job_input["base_image"])
        reference_image = decode_image(job_input["reference_image"])
        seed = int(job_input.get("seed",0))
        steps = max(12, min(int(job_input.get("steps",24)), 40))
        cfg = max(1.0, min(float(job_input.get("true_cfg_scale",4.0)), 8.0))
        full_prompt = (
            "Create one single photorealistic final image. "
            "Image 1 is the only identity and body reference. Preserve exactly the adult person's "
            "face, hair identity, skin tone, body size, body shape, proportions, and distinguishing features from Image 1. "
            "Image 2 is only the reference for clothing, pose, framing, camera angle, and background. "
            "Copy the outfit and pose from Image 2 without copying its face or identity. "
            "Do not blend identities. Do not slim, enlarge, or reshape the body from Image 1. "
            f"User instruction: {prompt}"
        )
        pipe = get_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            result = pipe(
                image=[base_image, reference_image],
                prompt=full_prompt,
                generator=generator,
                true_cfg_scale=cfg,
                negative_prompt="changed identity, different face, blended face, duplicate person, changed body shape, changed body size, distorted anatomy, collage, split screen",
                num_inference_steps=steps,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            ).images[0]
        return {"ok": True, "image_base64": encode_image(result), "mime_type":"image/png", "model":MODEL_ID, "seed":seed, "steps":steps}
    except Exception as error:
        traceback.print_exc()
        return {"ok":False, "error":str(error), "error_type":type(error).__name__, "model":MODEL_ID}

if __name__ == "__main__":
    print("[worker] Starting Motion Studio Qwen v2 worker", flush=True)
    runpod.serverless.start({"handler": handler})
