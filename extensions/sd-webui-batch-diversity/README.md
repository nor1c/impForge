# Batch Diversity

Prompt-safe batch diversity extension for Forge/ReForge WebUI.

This extension is intended for SDXL danbooru models such as IllustriousXL. By default it does not change the prompt. It diversifies batch outputs by enforcing independent seeds and perturbing the initial latent noise in a deterministic, metadata-recorded way.

Optional danbooru framing tags are available, but disabled by default. The built-in default list only contains framing/view/body-orientation tags and avoids lighting, background, style, quality, artist, and setting tags.

## What It Is For

Use this when generating multiple images with the same prompt produces images that are too similar in angle, pose, framing, or composition.

This is common with strongly trained SDXL anime/danbooru checkpoints, especially when the prompt, LoRAs, character tags, or quality tags strongly constrain the model. Changing only the seed is sometimes not enough, so this extension adds extra diversity at the seed and latent-noise level.

The default behavior is prompt-safe: it does not add lighting, background, style, or natural-language prompt modifiers.

## How It Works

Batch Diversity uses WebUI script hooks:

- `process()` runs before sampling and can rewrite the job's seed lists.
- `process_before_every_sampling()` runs just before the sampler starts and can replace the initial noise tensor.

The extension has three main mechanisms:

- Independent seeds: replaces later batch seeds with hash-derived independent seeds instead of relying on nearby or same-base seed behavior.
- Latent noise jitter: blends deterministic extra noise into each image's initial latent noise.
- Spatial latent shift: rolls each latent noise map by a deterministic offset, which can move composition without changing prompt text.

The modified noise is normalized back to the original sample's mean and standard deviation. This keeps the noise scale stable and reduces the chance of quality collapse.

## Recommended Start

- Enable Batch Diversity: on
- Mode: Latent + seeds
- Diversity strength: 0.35
- Independent seeds: on
- Latent noise jitter: on
- Spatial latent shift: on
- Danbooru framing tags: off

This is the safest default for IllustriousXL-style SDXL models because it does not modify your prompt.

## Settings

### Enable Batch Diversity

Turns the extension on for the current generation.

If disabled, the extension does nothing.

### Mode

`Latent + seeds` is the recommended default. It uses independent seeds plus latent noise diversification.

`Latent only` keeps the existing seed list and only modifies initial latent noise. Use this when you intentionally want to keep seed behavior unchanged.

`Latent + danbooru framing` enables the same seed/noise behavior and can also append optional danbooru framing tags if the tag checkbox is enabled.

### Diversity Strength

Controls how strongly the initial latent noise is changed.

Suggested values:

- `0.20-0.30`: subtle diversity
- `0.35`: recommended starting point
- `0.45-0.60`: stronger composition changes
- `0.70+`: experimental, can reduce prompt stability

If images are still too similar, increase this gradually. If images start ignoring your prompt or quality drops, lower it.

### Independent Seeds

Creates independent hash-derived seeds for every image in the batch.

When enabled, variation-seed mode is neutralized by setting variation strength to `0`, because variation mode intentionally keeps the same base seed and can make a batch more similar.

### Latent Noise Jitter

Adds deterministic extra noise to the starting latent. This is usually the most important prompt-free diversity mechanism.

Keep this enabled unless you only want spatial shifting.

### Spatial Latent Shift

Rolls each image's latent noise map differently. This can alter framing and composition without adding tags.

Keep this enabled for composition diversity.

### Optional Danbooru Framing Tags

Disabled by default.

When enabled with `Latent + danbooru framing`, the extension appends a few tags from the allowed tag list to each prompt. The default list only uses framing/view/body-orientation tags, such as `from above`, `profile`, `cowboy shot`, or `looking away`.

Do not add lighting, background, setting, artist, quality, or style tags unless you explicitly want those to change.

### Danbooru Tags Per Image

Controls how many tags are selected from the allowed list per image.

Recommended value: `1` or `2`.

Higher values can over-constrain the prompt or create conflicting tags.

### Allowed Danbooru Framing Tags

One tag per line. Only used when optional danbooru framing is enabled.

The default list intentionally avoids tags like `dramatic lighting`, `outdoors`, `bedroom`, `city`, `masterpiece`, or artist/style tags.

## Presets

### Prompt-Safe Default

- Mode: Latent + seeds
- Diversity strength: 0.35
- Independent seeds: on
- Latent noise jitter: on
- Spatial latent shift: on
- Danbooru framing tags: off

Use this first.

### Stronger Diversity

- Mode: Latent + seeds
- Diversity strength: 0.50
- Independent seeds: on
- Latent noise jitter: on
- Spatial latent shift: on
- Danbooru framing tags: off

Use this if the default is still too similar.

### Keep Seed Behavior

- Mode: Latent only
- Diversity strength: 0.30
- Independent seeds: off
- Latent noise jitter: on
- Spatial latent shift: on

Use this if another script or workflow controls seeds.

### Danbooru Framing Variation

- Mode: Latent + danbooru framing
- Diversity strength: 0.35
- Independent seeds: on
- Latent noise jitter: on
- Spatial latent shift: on
- Danbooru framing tags: on
- Danbooru tags per image: 1 or 2

Use this only when prompt-free methods are not enough and you accept small framing-tag changes.

## Metadata

The extension writes its settings into PNG infotext:

- `Batch Diversity`
- `Batch Diversity strength`
- `Batch Diversity independent seeds`
- `Batch Diversity latent jitter`
- `Batch Diversity spatial shift`
- `Batch Diversity danbooru tags` when optional tags are used

This makes generations easier to inspect later.

## Limitations

This extension improves diversity, but it cannot guarantee completely different images in every situation.

Outputs may still be similar if:

- The prompt heavily fixes composition.
- A LoRA strongly forces a pose, angle, or framing.
- ControlNet, IP-Adapter, reference preprocessors, or img2img input constrain composition.
- CFG is very high.
- The model has a strong default composition bias for the tag combination.

If this happens, try increasing diversity strength, lowering CFG slightly, disabling variation seed mode, or enabling optional danbooru framing tags with carefully chosen framing-only tags.

## Notes For SDXL Danbooru Models

For IllustriousXL-style checkpoints, danbooru tags usually work better than natural language. That is why this extension does not use phrases like `dramatic lighting`, `low angle shot`, or `cinematic composition` by default.

The default approach changes the latent starting point rather than adding prompt text, so your lighting, background, style, and setting remain controlled by your original prompt.
