# SDXL LoRA Role Split

Use LoRA Block Weight with role presets to reduce interference between style and character LoRAs on SDXL and IllustriousXL.

Role splitting cannot guarantee 95–100% identity accuracy. It cannot recover details that were weakly learned, inconsistently captioned, smaller than the effective latent resolution, or strongly contradicted by the checkpoint. Exact tattoos, text-like symbols, and asymmetric accessories may still require a close-up, regional conditioning, inpainting, or LoRA retraining.

Style LoRA:

```text
<lora:style_lora:0.7:0.7:role=style>
```

Character LoRA:

```text
<lora:character_lora:1.0:1.0:role=char>
```

If character clothes or hair still drift:

```text
<lora:character_lora:1.1:1.1:role=char_strong>
```

Equivalent preset names also work:

```text
<lora:style_lora:0.7:0.7:lbw=STYLE_SDXL>
<lora:character_lora:1.0:1.0:lbw=CHAR_SDXL>
<lora:character_lora:1.1:1.1:lbw=CHAR_SDXL_STRONG>
```

Explicit roles are recommended. Folder inference recognizes components such as `character_loras`, `my_styles`, `char`, and `style`, but explicit prompt roles remain authoritative.

Valid roles are `char`, `char_strong`, `char_max`, `style`, and `style_pure`. Raw `lbw=` lists must contain exactly 12 legacy values or 13 extended values. Invalid roles, non-finite values, and malformed lists now produce a readable error instead of silently falling back.

All prompts within one processing batch must use the same LoRA names, strengths, and roles because the model has one active LoRA patch stack at a time. Split prompts with different stacks into separate jobs.

Generation metadata includes the resolved preset, source, TE and UNet strength, and patch coverage. Unknown patch keys remain unchanged for compatibility and are reported in logs. A high unknown count means the LoRA key layout needs additional engine mapping before role weighting can reliably isolate it.

Increase character strength one step at a time. If `char_strong` or `char_max` adds artifacts without restoring the detail, more weight is not the fix. Test the character LoRA alone at its training resolution and use fixed seeds when comparing it with a style LoRA.

This cannot perfectly remove concepts baked into a LoRA, but it reduces style-LoRA content bleed and character-LoRA style bleed.
