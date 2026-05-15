# Inline Negative Prompt

Inline Negative Prompt is a small WebUI extension for per-prompt negative tags.
It lets you write negative-prompt tags inside a positive prompt by wrapping them
with `(n:...)`. During generation, the extension removes those markers from the
positive prompt and appends their contents to that image's negative prompt.

This is useful for batch prompt-list workflows where each line uses a different
positive prompt, but the UI still uses one shared static negative prompt.

## Problem It Solves

When generating from a prompt file or list, you may need one line to avoid a
specific angle, pose, framing, or artifact without affecting the other lines.
For example, only one prompt may need to avoid `from below`, while the rest of
the batch should keep using the normal shared negative prompt.

Without this extension, adding `from below` to the normal negative prompt affects
every line. With this extension, you can attach it only to the line that needs it.

## How It Works

Input prompt line:

```text
1girl, sky, sunset, (n:from below, low angle)
```

Static negative prompt:

```text
low quality, blurry
```

Generation-time result:

```text
positive: 1girl, sky, sunset
negative: low quality, blurry, from below, low angle
```

The prompt visible in the UI is not rewritten permanently. The transform happens
inside the generation process for the current request.

## Enable It

1. Restart the WebUI after installing or adding the extension.
2. Open the `Inline Negative Prompt` accordion in the txt2img or img2img UI.
3. Enable `Enable Inline Negative Prompt`.
4. Add `(n:...)` markers to the prompt lines that need per-line negative tags.

The extension is disabled by default.

## Syntax

- `(n:tag)` moves one tag to the negative prompt.
- `(n:tag1, tag2)` moves multiple comma-separated tags.
- Multiple `(n:...)` markers in one prompt are allowed.
- The marker prefix is case-insensitive, so `(N:from below)` also works.

Examples:

```text
1girl, street, (n:from below)
1girl, cinematic lighting, (n:low angle, dutch angle)
1girl, (n:from behind), garden, (n:back view)
```

## Duplicate Handling

If a tag is present in both the positive prompt and an inline negative marker,
the extension removes only exact top-level duplicate tags from the positive
prompt.

Example:

```text
from below, low angle, (n:from below)
```

Generation-time result:

```text
positive: low angle
negative: <your static negative>, from below
```

Non-exact matches are preserved:

```text
from below view, (n:from below)
```

Generation-time result:

```text
positive: from below view
negative: <your static negative>, from below
```

## Batch Prompt Lists

Each prompt line is processed independently. This means one line can add an
inline negative without affecting other lines in the same batch/list.

Example prompt list:

```text
1girl, standing, city
1girl, sitting, cafe, (n:from below)
1girl, walking, forest
```

Only the second line receives `from below` in its negative prompt.

## Hires Fix

The extension also processes hires fix prompts when hires fix prompt fields are
present. Inline negative markers in hires prompts are moved to the corresponding
hires negative prompt.

## Metadata

When inline negative tags are moved, the generation metadata includes:

```text
Inline Negative Prompt: True
Inline Negative Prompt tags: <moved tags>
```

This makes it easier to confirm that the extension was active for a generated
image.

## Limitations

- The syntax is intentionally simple: use direct `(n:...)` groups.
- Duplicate removal only checks exact top-level comma-separated tags.
- It does not modify model behavior like `sd-webui-negpip`; it only rewrites
  prompt text before conditioning.
- It does not require or depend on `sd-webui-negpip`.

## Files

- `scripts/inline_negative.py` contains the extension logic and UI.
