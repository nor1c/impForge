# SDXL LoRA Role Split

Use LoRA Block Weight with role presets to separate style LoRAs from character LoRAs.

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

This cannot perfectly remove concepts baked into a LoRA, but it reduces style-LoRA content bleed and character-LoRA style bleed.
