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

Generation metadata includes the resolved preset, source, TE and UNet strength, the step window, and patch coverage. Patch keys are reported as `scaled` (the block has its own preset slot), `folded` (a block without its own slot), `passthrough` (outside the UNet blocks, for example `time_embed`), and `unknown`. A high unknown count means the LoRA key layout needs additional engine mapping before role weighting can reliably isolate it.

### Effective Weight and the Text Encoder

A patch entry already carries the requested strength, so the applied weight is `strength x preset ratio`. The character presets used to exceed 1.0 on the early blocks, meaning `role=char` at `0.9` applied as `1.03` — an extrapolation past anything the LoRA was trained at, and a number the user never wrote.

Two changes address this.

**Effective weight is capped at 1.0.** The relative shape of a preset is preserved, only the extrapolation is removed. At `0.9` this touches `BASE` and `IN04` by around 3%; at `1.0` up to 13% across five blocks. Disable **Cap effective LoRA block weight at 1.0** in Settings → Extra Networks for the previous behaviour.

**`BASE` is held below 1.0.** The text-encoder slots split by depth:

| Slot | Layers | Role |
| --- | --- | --- |
| `BASE` | CLIP-L 0-7, CLIP-G 0-20 | Reads the prompt semantically — context, framing, pose wording |
| `BASE_LATE` | CLIP-L 8-11, CLIP-G 21-31 | Where a trigger word binds to identity |

Amplifying `BASE` strengthened the LoRA's reading of the prompt along with its dataset associations, so pose wording had to fight an over-weighted interpretation of itself. Text-encoder conditioning is computed once before sampling, so a step window cannot relieve this path — the preset is the only place to address it.

| Preset | `BASE` was | now | `BASE_LATE` |
| --- | --- | --- | --- |
| `CHAR_SDXL` | 1.15 | 0.85 | 0.70 (unchanged) |
| `CHAR_SDXL_STRONG` | 1.30 | 0.90 | 0.80 (unchanged) |
| `CHAR_SDXL_MAX` | 1.40 | 0.95 | 0.90 (unchanged) |

`BASE_LATE` is deliberately untouched, so trigger-word binding is unaffected. `IN07`, `IN08`, and `M00` are also left alone below 1.0: in SDXL these carry identity and layout together, so lowering them costs likeness as much as it frees composition.

### Folded Blocks

The 12/13-value preset vocabulary is fixed for backwards compatibility, and only the attention-carrying blocks have a slot in it. The remaining UNet blocks — `IN00`-`IN03`, `IN06`, and `OUT06`-`OUT08` — carry conv weights that LoCon and LyCORIS LoRAs patch. Leaving them unscaled let those LoRAs keep full strength there and bypass role weighting entirely.

A folded block takes the **mean of its resolution level's slots**, capped at 1.0. Folding onto a single neighbour would have distorted it in both directions: `IN00`-`IN03` would inherit `IN04`, which is 1.15 in `CHAR_SDXL` and would amplify a block past full strength, while `OUT06`-`OUT08` would inherit `OUT05` at 0.15 and be stripped of the surface detail they contribute. The mean keeps a folded block representative of its level, and the cap keeps folding a restriction rather than a boost.

Blocks that own a slot are unaffected and keep their above-1.0 preset weights.

## Step Windows

Composition is decided during the first few high-sigma steps; identity, face, and fine detail are built afterwards. A character LoRA that is active from step 0 therefore competes with the prompt over the pose.

Lowering the strength or pushing the deep-block weights down does not separate the two, because in SDXL the deep blocks (`IN07`, `IN08`, `M00`) carry identity and layout together, so attenuating them costs identity by as much as it frees up composition.

Delaying the LoRA in time avoids that trade-off. The composition phase runs without the LoRA, then the LoRA applies at full strength for the remaining steps:

```text
<lora:character_lora:0.9:0.9:role=char:start=0.15>
```

`start=` is the first step the LoRA applies to, `stop=` is the first step it no longer applies to, and `step=start-stop` sets both:

```text
<lora:character_lora:0.9:0.9:role=char:stop=0.6>
<lora:character_lora:0.9:0.9:role=char:step=0.15-0.85>
```

Start at `start=0.15`. If identity weakens, lower it rather than raising the strength. If the pose still does not respond, raise it. `start=0` is the same as no window.

### Fractions vs Absolute Steps

A bound containing a decimal point is a fraction of the current pass; a whole number is a step index. Fractions are recommended because the step counter restarts for every sampling run and hires fix resamples at a different step count, so `start=0.15` lands proportionally in both while `start=4` means something different in each. Both bounds of one tag must use the same form; mixing them is rejected.

### Behaviour Across Passes

| Pass | Behaviour |
| --- | --- |
| Base txt2img | Windows apply as written. |
| Hires fix | Fractional windows scale to the pass. Absolute windows are skipped and the LoRA runs at full strength, since a step index calibrated for the base pass would delay it again during the pass that builds final detail. |
| ADetailer | Windows are skipped. Face inpainting does not decide composition, so muting the character LoRA there would only remove identity from the fix. |
| Lowvram | Windows are declined with a warning. Weights are patched per forward pass from a captured patch list rather than baked once, so muting is not reliably observed. |

Step windows gate UNet patches only. Text-encoder conditioning is computed once before sampling begins, so it cannot follow a step window and keeps the requested TE strength.

Crossing a window boundary restores the affected weights from the patcher backup and re-bakes them. For a large LoRA this is a brief pause on that step, logged when it exceeds a second.

Each LoRA is capped at 1.0 individually, but a stack is not: several LoRAs contributing to the same block still add up. A character plus one style LoRA lands around 1.2-1.3 and works normally, so the warning fires above 1.35 — roughly a full extra LoRA's worth of weight on one block. That is the point where prompt-driven pose usually stops responding.

Every load also logs the peak effective weight per LoRA, so the gap between the strength written in the prompt and the weight applied stays visible:

```text
[LBW] character.safetensors: written UNet:0.9 TE:0.9 -> peak effective 1.00 at IN04
```

Increase character strength one step at a time. If `char_strong` or `char_max` adds artifacts without restoring the detail, more weight is not the fix. Test the character LoRA alone at its training resolution and use fixed seeds when comparing it with a style LoRA.

This cannot perfectly remove concepts baked into a LoRA, but it reduces style-LoRA content bleed and character-LoRA style bleed.
