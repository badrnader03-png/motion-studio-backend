# Motion Studio — FLUX.2 Multi-Reference Backend

This backend replaces Qwen Image Edit with:

```text
black-forest-labs/FLUX.2-dev
```

It accepts the same frontend payload:

```json
{
  "input": {
    "base_image": "data:image/jpeg;base64,...",
    "reference_image": "data:image/jpeg;base64,...",
    "prompt": "Use the pose and outfit from image 2 while preserving image 1 identity",
    "steps": 30,
    "seed": 0,
    "guidance_scale": 2.5
  }
}
```

## Why FLUX.2

FLUX.2 supports native multi-reference editing. The backend sends:

- Image 1: identity reference.
- Image 2: clothing, pose, composition, and environment reference.

## Required Hugging Face setup

`black-forest-labs/FLUX.2-dev` is gated.

1. Open the model page while logged into Hugging Face.
2. Accept the model license and access conditions.
3. Create a Hugging Face token with read access.
4. Add it to RunPod as a **Secret**, not a plain environment variable:

```text
HF_TOKEN = hf_...
```

## RunPod endpoint settings

```text
Model: black-forest-labs/FLUX.2-dev
Branch: main
Dockerfile path: /Dockerfile
Build context: .
GPU priority:
1. 96 GB
2. 80 GB
Max workers: 1
```

FLUX.2 is much larger than the previous model. An 80 GB or 96 GB GPU plus CPU offloading and large system RAM is recommended.

Add:

```text
MODEL_NAME = black-forest-labs/FLUX.2-dev
```

The existing Hugging Face `index.html` does not need to change.

## Content controls

This backend does not add a separate custom prompt filter. The application owner may define product-level rules, while RunPod, Hugging Face, the model license, the model's acceptable-use policy, and applicable law still apply.
