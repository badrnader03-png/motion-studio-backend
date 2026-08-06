# Motion Studio — Qwen Pose v3

النسخة دي تضيف مرحلة جديدة قبل التوليد:

1. Image 1 = الهوية والجسم.
2. Image 2 = الملابس والمشهد والكاميرا.
3. OpenPose يستخرج وضعية Image 2 تلقائيًا.
4. Qwen يستقبل الصور الثلاث مع تعليمات صارمة.

## ارفع إلى GitHub

```text
Dockerfile
handler.py
requirements.txt
policy.json
README.md
```

## إعدادات RunPod

```text
MODEL_NAME=Qwen/Qwen-Image-Edit-2511
POSE_MODEL_NAME=lllyasviel/Annotators
```

اترك:

```text
Dockerfile path: /Dockerfile
Build context: .
```

## إعدادات التجربة

```text
Steps: 50
Seed: 42
```

اكتب في البرومبت فقط:

```text
Copy the exact outfit and composition from Image 2.
```

## ملاحظة صريحة

هذه النسخة تستخدم خريطة الوضعية كمرجع ثالث داخل Qwen. هي أقوى من البرومبت وحده، لكنها ليست ControlNet صلبًا؛ لذلك تحسن الالتزام بالوضعية ولا تضمن نسخها بكسل-ببكسل.
