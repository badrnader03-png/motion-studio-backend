# Motion Studio — Qwen fixed

الموديل المستخدم:

```text
Qwen/Qwen-Image-Edit-2511
```

## مهم جدًا

اكتب في مربع البرومبت **التعديل المطلوب فقط**، مثل:

```text
Copy the exact outfit and pose from Image 2.
```

لا تضع كلمات الـ Negative Prompt مثل:

```text
cartoon, painting, bad anatomy
```

داخل مربع البرومبت؛ الكود يرسلها تلقائيًا في `negative_prompt`.

## الإعدادات المقترحة

```text
Steps: 50
Seed: 42
```

Image 1 = الهوية والجسم  
Image 2 = الملابس والوضعية والخلفية
