Improved reForge is aimed at enhancing performance and optimization. Feel free to open an issue if you have any optimization ideas.

There are also some minor config adjustments to improve generation speed, while still staying within a safe range without reducing image quality.

<hr>

- [x] Removed some unecessary features.
- [x] Improved lora-block-weight model accuracy.
- [x] Seed when generating multiple images with batch count now randomized and no longer incremental. 
- [x] Added Sage Attention and Flash Attention.
  It's not really a big performance gain, but there is a noticeable speed improvement, around 0.7-0.8 it/s.<br>
  [SageAttention and FlashAttention setup](docs/sage_flash_attention_setup.md)<br>
  Refer: [https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2866](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2866)
- [x] Now you can drop an image and load the metadata directly in the `txt2img` tab, without needing to go through `PNG Info → Send to txt2img`.
- [x] Added Batch Diversity extension.
  - Purpose: helps batches generated from the same prompt produce more varied angles, framing, and composition.
  - Default behavior: prompt-safe; it diversifies independent seeds and initial latent noise without adding lighting, background, style, or setting tags.
  - Optional behavior: framing tags can be enabled manually when latent-only diversity is not enough.
  - Recommended start: enable `Batch Diversity`, use `Latent + seeds`, set strength around `0.35`, keep framing tags off.
  - Affected files: `extensions/sd-webui-batch-diversity/scripts/batch_diversity.py`, `extensions/sd-webui-batch-diversity/README.md`.
- [x] Added Inline Negative Prompt extension.
  - Purpose: allows per-line negative tags in batch prompt-list workflows by moving `(n:...)` markers from the positive prompt to the matching negative prompt.
  - Example: `from below, low angle, (n:from below)` becomes positive `low angle` and negative `<static negative>, from below`.
  - Duplicate handling: removes only exact top-level duplicate tags from the positive prompt.
  - Affected files: `extensions/sd-webui-inline-negative/scripts/inline_negative.py`, `extensions/sd-webui-inline-negative/README.md`.
- [x] Added Inline Resolution extension.
  - Purpose: allows per-line width/height in batch prompt-list workflows via `(r:width, height)` markers; lines without a marker fall back to the WebUI base resolution.
  - Example: `1girl, city, (r:1024, 1380)` renders that line at `1024x1380`; the marker is removed before generation.
  - Batching: forces internal `batch_size = 1` when mixed sizes are detected, since one tensor batch cannot contain images of different dimensions.
  - Hires fix: explicit hires resize values are kept; otherwise `hr_scale` is reapplied to the per-line first-pass resolution.
  - Affected files: `extensions/sd-webui-inline-resolution/scripts/inline_resolution.py`, `extensions/sd-webui-inline-resolution/README.md`.
