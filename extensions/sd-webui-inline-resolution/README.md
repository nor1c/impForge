# Inline Resolution

Inline Resolution is a small WebUI extension that lets you set per-prompt
width/height directly inside a prompt by wrapping the desired size with
`(r:...)`. The marker is removed from the positive prompt at generation time
and that line is rendered at the requested resolution. Prompt lines without a
marker use the WebUI's base width/height as a fallback.

This is useful for batch prompt-list workflows where some lines need a
different aspect ratio (portrait, landscape, square) without separate runs.

## Problem It Solves

When generating from a prompt list (one prompt per line) with a single shared
resolution slider, you cannot change the canvas size for individual lines.
For example, you might want one line to be `1024x1380` for a portrait shot
and another line to be `1380x1024` for a wide shot. Without this extension,
you need to run them as separate jobs.

With this extension, you can write the resolution inline:

```text
1girl, standing, city, (r:1024, 1380)
1girl, landscape view, (r:1380, 1024)
1girl, walking, forest
```

The first two lines use the inline resolutions. The last line falls back to
the WebUI's base width/height because it has no `(r:...)` marker.

## Enable It

1. Restart the WebUI after installing or adding the extension.
2. Open the `Inline Resolution` accordion in the txt2img or img2img UI.
3. Enable `Enable Inline Resolution`.
4. Add `(r:...)` markers to the lines that need a custom resolution.

The extension is disabled by default.

## Syntax

- `(r:width, height)` - comma-separated, e.g. `(r:1024, 1380)`.
- `(r:1024x1380)` is also accepted as a convenience.
- Both values must be positive integers.
- Any positive integer is accepted as-is. The extension does not round.
- Only the first valid `(r:...)` marker per prompt is used; later markers are
  left untouched so you can see something is off.

Examples:

```text
1girl, city, (r:1024, 1380)
1girl, landscape, (r:1380, 1024)
1girl, portrait, (r:1025, 1380)
1girl, square, (r:1024x1024)
```

## Base Resolution Fallback

If a prompt line does not contain a valid `(r:...)` marker, the extension
uses the WebUI width/height for that image. This means you can mix lines
with custom sizes and lines that follow the global setting in the same
prompt list.

## Batch Behavior

A single tensor batch must contain images of identical dimensions. When the
extension detects that at least one prompt asks for a resolution different
from the WebUI base size, it switches to per-image execution by setting:

- `batch_size` to `1`
- `n_iter` to the total number of prompts

The total number of generated images stays the same; only the internal
batching changes. Generation may take slightly longer because images run
sequentially instead of in parallel batches.

If every prompt's `(r:...)` matches the WebUI base size (or no marker is
used), batching is left alone.

## Hires Fix

When hires fix is enabled with explicit resize values (`Resize width to` /
`Resize height to`), those values are honoured as the final hires target.

When hires fix is enabled with `Upscale by` (no explicit resize), the
extension recomputes the hires target per line by scaling the per-line
first-pass resolution by `hr_scale`. For example, if a line is `1024x1380`
and `Upscale by` is `2.0`, the hires target becomes `2048x2760`.

The same `(r:...)` marker is also stripped from hires prompt fields if
present, so the hires pass sees a clean prompt.

## Metadata

When inline resolutions are applied, generation metadata includes:

```text
Inline Resolution: True
Inline Resolution size: <width>x<height>
```

This makes it easier to confirm which images were generated using the inline
resolution markers.

## Notes

- No core code changes are required.
- Independent of `sd-webui-inline-negative`; both can be enabled together.
- Does not validate dimension multiples; the underlying model may still
  apply its own internal alignment.

## Files

- `scripts/inline_resolution.py` contains the extension logic and UI.
