# Motion Studio Engine

RunPod Serverless backend for Qwen Image Edit 2511.

## RunPod cached model

Set the RunPod endpoint model field to:

```text
Qwen/Qwen-Image-Edit-2511
```

## Expected input

```json
{
  "input": {
    "base_image": "data:image/jpeg;base64,...",
    "reference_image": "data:image/jpeg;base64,...",
    "prompt": "Keep identity from image 1 and copy requested details from image 2.",
    "steps": 20,
    "seed": 0
  }
}
```

## Output

```json
{
  "ok": true,
  "image_base64": "...",
  "mime_type": "image/png"
}
```
