# Motion Studio Qwen v2

عدّل `policy.json` للتحكم في قواعد المنتج.

واجهة الطلب يجب أن ترسل:

```json
{
  "input": {
    "base_image": "...",
    "reference_image": "...",
    "prompt": "...",
    "adult_confirmed": true,
    "steps": 24,
    "seed": 0
  }
}
```

تبقى شروط RunPod وترخيص الموديل والقانون سارية.
