import base64
import io
import os
import tempfile
import traceback
from typing import Any

import runpod
import torch
from PIL import Image, ImageOps
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.utils.export_utils import export_to_video


MODEL_ID = os.getenv(
    "MODEL_NAME",
    "TestOrganizationPleaseIgnore/WAMU_v3_WAN2.2_I2V_LIGHTNING"
)

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

FIXED_FPS = 16
MIN_FRAMES = 8
MAX_FRAMES = 160

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, "
    "画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, "
    "残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, "
    "毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, "
    "三条腿, 背景人很多, 倒着走"
)

_PIPELINE = None


def decode_image(value: str) -> Image.Image:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image must be a base64 string or data URL")

    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]

    raw = base64.b64decode(value, validate=False)

    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    return image


def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size

    if width == height:
        return image.resize(
            (SQUARE_DIM, SQUARE_DIM),
            Image.Resampling.LANCZOS
        )

    aspect_ratio = width / height

    max_aspect_ratio = MAX_DIM / MIN_DIM
    min_aspect_ratio = MIN_DIM / MAX_DIM

    image_to_resize = image

    if aspect_ratio > max_aspect_ratio:
        target_w = MAX_DIM
        target_h = MIN_DIM

        crop_width = int(round(height * max_aspect_ratio))
        left = (width - crop_width) // 2

        image_to_resize = image.crop(
            (left, 0, left + crop_width, height)
        )

    elif aspect_ratio < min_aspect_ratio:
        target_w = MIN_DIM
        target_h = MAX_DIM

        crop_height = int(round(width / min_aspect_ratio))
        top = (height - crop_height) // 2

        image_to_resize = image.crop(
            (0, top, width, top + crop_height)
        )

    else:
        if width > height:
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))

    return image_to_resize.resize(
        (final_w, final_h),
        Image.Resampling.LANCZOS
    )


def resize_last_image(
    image: Image.Image,
    reference: Image.Image
) -> Image.Image:

    ref_width, ref_height = reference.size
    width, height = image.size

    scale = max(
        ref_width / width,
        ref_height / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    left = (new_width - ref_width) // 2
    top = (new_height - ref_height) // 2

    return image.crop(
        (
            left,
            top,
            left + ref_width,
            top + ref_height
        )
    )


def get_num_frames(duration: float) -> int:
    frames = int(round(duration * FIXED_FPS))

    frames = max(
        MIN_FRAMES,
        min(MAX_FRAMES, frames)
    )

    return frames + 1


def get_pipeline():
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    print(
        f"[model] Loading Wan 2.2 pipeline: {MODEL_ID}",
        flush=True
    )

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=token,
        low_cpu_mem_usage=True,
    )

    # RunPod version:
    # Unlike the original ZeroGPU Space, we do not use spaces.aoti_load().
    # CPU offload helps reduce VRAM requirements.
    pipe.enable_model_cpu_offload()

    if getattr(pipe, "vae", None) is not None:
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

    pipe.set_progress_bar_config(disable=False)

    _PIPELINE = pipe

    print("[model] Wan 2.2 ready", flush=True)

    return _PIPELINE


def encode_video(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    video_path = None

    try:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for Wan 2.2."
            )

        job_input = job.get("input") or {}

        image_value = (
            job_input.get("image")
            or job_input.get("input_image")
        )

        if not image_value:
            raise ValueError("image is required")

        prompt = str(
            job_input.get(
                "prompt",
                "make this image come alive, cinematic motion, smooth animation"
            )
        ).strip()

        if not prompt:
            raise ValueError("prompt is required")

        negative_prompt = str(
            job_input.get(
                "negative_prompt",
                DEFAULT_NEGATIVE_PROMPT
            )
        )

        duration = float(
            job_input.get("duration", 3.5)
        )

        duration = max(
            0.5,
            min(10.0, duration)
        )

        steps = int(
            job_input.get("steps", 4)
        )

        steps = max(
            1,
            min(30, steps)
        )

        guidance_scale = float(
            job_input.get(
                "guidance_scale",
                1.0
            )
        )

        guidance_scale_2 = float(
            job_input.get(
                "guidance_scale_2",
                1.0
            )
        )

        seed = int(
            job_input.get("seed", 42)
        )

        fps = FIXED_FPS

        image = decode_image(image_value)
        image = resize_image(image)

        last_image = None

        if job_input.get("last_image"):
            last_image = decode_image(
                job_input["last_image"]
            )

            last_image = resize_last_image(
                last_image,
                image
            )

        num_frames = get_num_frames(
            duration
        )

        print(
            "[job] "
            f"{image.width}x{image.height} "
            f"frames={num_frames} "
            f"fps={fps} "
            f"steps={steps} "
            f"seed={seed}",
            flush=True
        )

        pipe = get_pipeline()

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(seed)

        kwargs = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": image.height,
            "width": image.width,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "guidance_scale_2": guidance_scale_2,
            "num_inference_steps": steps,
            "generator": generator,
            "output_type": "np",
        }

        if last_image is not None:
            kwargs["last_image"] = last_image

        with torch.inference_mode():
            result = pipe(**kwargs)

        frames = result.frames[0]

        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        )

        video_path = tmp.name
        tmp.close()

        export_to_video(
            frames,
            video_path,
            fps=fps,
            quality=int(
                job_input.get("quality", 6)
            )
        )

        video_base64 = encode_video(
            video_path
        )

        return {
            "ok": True,
            "video_base64": video_base64,
            "mime_type": "video/mp4",
            "model": MODEL_ID,
            "seed": seed,
            "steps": steps,
            "duration": duration,
            "fps": fps,
            "num_frames": num_frames,
            "width": image.width,
            "height": image.height,
        }

    except Exception as error:

        traceback.print_exc()

        return {
            "ok": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "model": MODEL_ID,
        }

    finally:

        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass


if __name__ == "__main__":

    print(
        "[worker] Starting Motion Studio Wan 2.2 I2V",
        flush=True
    )

    runpod.serverless.start(
        {
            "handler": handler
        }
    )
