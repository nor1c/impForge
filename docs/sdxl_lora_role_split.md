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

Generation metadata includes the resolved preset, source, TE and UNet strength, and patch coverage. Patch keys are reported as `scaled` (the block has its own preset slot), `folded` (a block without its own slot), `passthrough` (outside the UNet blocks, for example `time_embed`), and `unknown`. A high unknown count means the LoRA key layout needs additional engine mapping before role weighting can reliably isolate it.

### Effective Weight and the Text Encoder

A patch entry already carries the requested strength, so the applied weight is `strength x preset ratio`. The character presets used to exceed 1.0 on the early blocks, meaning `role=char` at `0.9` applied as `1.03` — an extrapolation past anything the LoRA was trained at, and a number the user never wrote.

Two changes address this.

**Effective weight is capped at 1.0.** The relative shape of a preset is preserved, only the extrapolation is removed. At `0.9` this touches `BASE` and `IN04` by around 3%; at `1.0` up to 13% across five blocks. Disable **Cap effective LoRA block weight at 1.0** in Settings → Extra Networks for the previous behaviour.

**`BASE` is held below 1.0.** The text-encoder slots split by depth:

| Slot | Layers | Role |
| --- | --- | --- |
| `BASE` | CLIP-L 0-7, CLIP-G 0-20 | Reads the prompt semantically — context, framing, pose wording |
| `BASE_LATE` | CLIP-L 8-11, CLIP-G 21-31 | Where a trigger word binds to identity |

Amplifying `BASE` strengthened the LoRA's reading of the prompt along with its dataset associations, so pose wording had to fight an over-weighted interpretation of itself. Text-encoder conditioning is computed once before sampling rather than per step, so the preset is the only place to address it.

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

