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
- [x] Added Inline Negative Prompt extension.
  - Purpose: allows per-line negative tags in batch prompt-list workflows by moving `(n:...)` markers from the positive prompt to the matching negative prompt.
  - Example: `from below, low angle, (n:from below)` becomes positive `low angle` and negative `<static negative>, from below`.
- [x] Added Inline Resolution extension.
  - Purpose: allows per-line width/height in batch prompt-list workflows via `(r:width, height)` markers; lines without a marker fall back to the WebUI base resolution.
  - Example: `wide shot, city, (r:1024, 1380)` renders that line at `1024x1380`; the marker is removed before generation.

## Using Character and Style LoRAs

impForge provides native SDXL LoRA role splitting to reduce interference between character and style LoRAs. It is intended for SDXL models, especially IllustriousXL.

Use `role=style` for a style LoRA and `role=char` for a character LoRA:

```text
<lora:style:0.9:0.9:role=style>
<lora:character:0.95:0.95:role=char>
```

A practical starting point when combining both LoRAs is:

```text
<lora:style:0.8:0.8:role=style>
<lora:character:1.0:1.0:role=char_strong>
```

Available roles:

| Role | Purpose |
| --- | --- |
| `style` | Preserves the style while reducing character and composition leakage. |
| `style_pure` | Applies stricter style isolation. Use it when the style LoRA strongly changes the character. |
| `char` | Default character profile for identity, hair, eyes, clothing, and accessories. |
| `char_strong` | Strengthens character identity when details disappear after adding a style LoRA. |
| `char_max` | Maximum identity profile. Use cautiously because it may introduce artifacts or overpower the style. |

Start with `style` and `char`. Move to `char_strong` only when character details remain too weak. Lower the style strength before trying `char_max`.

### Custom and Legacy Block Weights

Existing 12-value `lbw=` prompts remain supported:

```text
<lora:style:0.9:0.9:lbw=0,0,0,0,0,0,0,1,1,1,1,1>
<lora:character:0.95:0.95:lbw=0,0,0,0,0,0,0.9,0.9,0.9,0,0,0>
```

Both 12-value legacy lists and 13-value extended lists are accepted. Use either `role=` or `lbw=` for one LoRA. If both are provided, `lbw=` takes precedence.

Malformed lists, non-finite values, and unknown roles produce an error instead of silently falling back to another profile.

### Batch Restriction

Every prompt in one processing batch must use the same LoRA names, strengths, and roles. Run prompts with different LoRA stacks as separate jobs. This prevents later prompts from silently using the first prompt's LoRA configuration.

Role splitting improves isolation but cannot recover details that the LoRA did not learn reliably. Exact tattoos, small symbols, and asymmetric accessories may still require higher resolution, regional conditioning, inpainting, or LoRA retraining. More detailed documentation is available in [SDXL LoRA Role Split](docs/sdxl_lora_role_split.md).
