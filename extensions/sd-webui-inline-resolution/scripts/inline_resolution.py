"""
Inline Resolution extension.

Lets users specify per-prompt width/height inside the positive prompt using a
``(r:width, height)`` marker. The marker is stripped from the positive prompt
and the corresponding image is generated at the requested resolution. Lines
without a marker use the WebUI's base resolution.

Designed for batch prompt-list workflows where different lines need different
canvas sizes (e.g. portrait vs landscape) without juggling separate runs.

This extension is isolated from core code: it only mutates ``p.all_prompts``
plus a few per-batch fields (``p.width``, ``p.height``, ``p.batch_size``,
``p.n_iter``, ``p.rng``). It does not patch any core modules.

Note: when at least one prompt requests a resolution different from the base
canvas, the extension forces ``batch_size = 1`` because images of different
dimensions cannot be sampled together in a single tensor batch.
"""

import re

import gradio as gr

from modules import scripts, rng


MARKER_PREFIX = "r:"
# (r:WIDTH, HEIGHT) - both positive integers, with optional whitespace.
_MARKER_BODY_RE = re.compile(r"^\s*(\d+)\s*[,xX]\s*(\d+)\s*$")
opt_f = 8


def _find_marker(prompt):
    """
    Return ``(start, end, width, height)`` for the first valid ``(r:...)``
    marker in ``prompt``. Returns ``None`` if nothing valid is found.

    Only outermost parenthesised groups are considered. The contents must be
    exactly ``WIDTH, HEIGHT`` (or ``WIDTHxHEIGHT``) with positive integers.
    Anything else, including malformed content, is left untouched so we do
    not accidentally swallow unrelated weighted-prompt syntax.
    """

    if not prompt:
        return None

    i = 0
    n = len(prompt)
    while i < n:
        ch = prompt[i]
        if ch == "(":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                cj = prompt[j]
                if cj == "\\" and j + 1 < n:
                    j += 2
                    continue
                if cj == "(":
                    depth += 1
                elif cj == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1

            if depth != 0:
                break

            inner = prompt[i + 1:j]
            stripped = inner.lstrip()
            if stripped.lower().startswith(MARKER_PREFIX):
                offset = inner.lower().find(MARKER_PREFIX)
                body = inner[offset + len(MARKER_PREFIX):]
                m = _MARKER_BODY_RE.match(body)
                if m:
                    width = int(m.group(1))
                    height = int(m.group(2))
                    if width > 0 and height > 0:
                        return (i, j + 1, width, height)
                # Malformed body - skip past this group without consuming it.
            i = j + 1
            continue

        if ch == "\\" and i + 1 < n:
            i += 2
            continue

        i += 1

    return None


def _strip_marker(prompt, span):
    """Remove the marker span and tidy up stray separators."""

    start, end = span
    cleaned = prompt[:start] + prompt[end:]
    return _tidy_separators(cleaned)


def _tidy_separators(prompt):
    if not prompt:
        return prompt

    segments = []
    depth = 0
    start = 0
    i = 0
    n = len(prompt)
    while i < n:
        ch = prompt[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
        elif ch == "," and depth == 0:
            segments.append(prompt[start:i])
            start = i + 1
        i += 1
    segments.append(prompt[start:])

    cleaned = ", ".join(seg.strip() for seg in segments if seg.strip())
    return cleaned


def _apply(prompt):
    """
    Return ``(cleaned_prompt, (width, height) | None)``.

    Only the first valid ``(r:...)`` marker is honoured per prompt; subsequent
    markers are left in place so the user notices and fixes their prompt.
    """

    if not prompt:
        return prompt, None

    found = _find_marker(prompt)
    if not found:
        return prompt, None

    start, end, width, height = found
    cleaned = _strip_marker(prompt, (start, end))
    return cleaned, (width, height)


class InlineResolutionScript(scripts.Script):
    sorting_priority = 0.6

    def title(self):
        return "Inline Resolution"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Inline Resolution", open=False):
            enabled = gr.Checkbox(
                label="Enable Inline Resolution",
                value=False,
                info=(
                    "Use (r:width, height) inside a prompt line to override "
                    "that line's resolution. Lines without (r:...) use the "
                    "WebUI's base width/height. When mixed sizes are present "
                    "the extension forces batch size to 1 internally."
                ),
            )
            gr.Markdown(
                "Syntax: `(r:1024, 1380)`, `(r:1380, 1024)`, or `(r:1024x1380)`. "
                "Any positive integer dimensions are accepted as written; the "
                "extension does not round or reshape the values."
            )
        return (enabled,)

    def process(self, p, enabled):
        if not enabled:
            return

        if getattr(p, "_inline_resolution_applied", False):
            # Defensive: never re-process the same processing object twice.
            return

        all_prompts = list(getattr(p, "all_prompts", []) or [])
        if not all_prompts:
            return

        all_negative_prompts = list(getattr(p, "all_negative_prompts", []) or [])
        all_hr_prompts = getattr(p, "all_hr_prompts", None)
        all_hr_prompts = list(all_hr_prompts) if all_hr_prompts else None
        all_hr_negative_prompts = getattr(p, "all_hr_negative_prompts", None)
        all_hr_negative_prompts = list(all_hr_negative_prompts) if all_hr_negative_prompts else None

        base_w = int(getattr(p, "width", 0) or 0)
        base_h = int(getattr(p, "height", 0) or 0)

        sizes = []
        any_custom = False
        new_prompts = []
        new_hr_prompts = list(all_hr_prompts) if all_hr_prompts else None

        for i, prompt in enumerate(all_prompts):
            cleaned, size = _apply(prompt)
            new_prompts.append(cleaned)
            if size is not None:
                w, h = size
                sizes.append((w, h))
                if w != base_w or h != base_h:
                    any_custom = True
            else:
                sizes.append((base_w, base_h))

            if new_hr_prompts is not None and i < len(new_hr_prompts):
                hr_cleaned, _ = _apply(new_hr_prompts[i])
                new_hr_prompts[i] = hr_cleaned

        if not sizes:
            return

        # If nobody asked for a custom resolution, leave everything alone.
        if not any_custom:
            # Still strip any (r:...) markers we may have removed defensively.
            prompts_changed = new_prompts != all_prompts
            hr_changed = new_hr_prompts is not None and new_hr_prompts != all_hr_prompts
            if prompts_changed:
                p.all_prompts = new_prompts
            if hr_changed:
                p.all_hr_prompts = new_hr_prompts
            return

        total = len(new_prompts)

        # Mixed resolutions cannot share a tensor batch. Force per-image batches.
        # Preserve the total image count by folding batch_size into n_iter.
        original_batch_size = int(getattr(p, "batch_size", 1) or 1)
        original_n_iter = int(getattr(p, "n_iter", 1) or 1)
        target_total = original_batch_size * original_n_iter
        if target_total != total:
            # When the user supplies a prompt list, total already equals the
            # number of prompts. Trust that count.
            target_total = total

        p.batch_size = 1
        p.n_iter = target_total

        # Pad/truncate the per-image arrays to match the new layout.
        def _pad_or_truncate(seq, length, fill):
            if seq is None:
                return None
            seq = list(seq)
            if len(seq) < length:
                seq.extend([fill] * (length - len(seq)))
            elif len(seq) > length:
                seq = seq[:length]
            return seq

        p.all_prompts = _pad_or_truncate(new_prompts, target_total, new_prompts[-1] if new_prompts else "")
        p.all_negative_prompts = _pad_or_truncate(
            all_negative_prompts,
            target_total,
            all_negative_prompts[-1] if all_negative_prompts else "",
        )
        if hasattr(p, "all_seeds") and p.all_seeds is not None:
            p.all_seeds = _pad_or_truncate(list(p.all_seeds), target_total, int(p.all_seeds[-1]) if len(p.all_seeds) else 0)
        if hasattr(p, "all_subseeds") and p.all_subseeds is not None:
            p.all_subseeds = _pad_or_truncate(list(p.all_subseeds), target_total, int(p.all_subseeds[-1]) if len(p.all_subseeds) else 0)

        if new_hr_prompts is not None:
            p.all_hr_prompts = _pad_or_truncate(new_hr_prompts, target_total, new_hr_prompts[-1] if new_hr_prompts else "")
        if all_hr_negative_prompts is not None:
            p.all_hr_negative_prompts = _pad_or_truncate(
                all_hr_negative_prompts,
                target_total,
                all_hr_negative_prompts[-1] if all_hr_negative_prompts else "",
            )

        sizes = _pad_or_truncate(sizes, target_total, (base_w, base_h))

        p._inline_resolution_sizes = sizes
        p._inline_resolution_base = (base_w, base_h)
        p._inline_resolution_applied = True

        # Hires fix coupling: if hires fix is enabled with explicit resize
        # values, those are kept as-is. Otherwise hr_scale runs against the
        # per-line first-pass resolution we set in before_process_batch().

        try:
            if p.extra_generation_params is None:
                p.extra_generation_params = {}
            p.extra_generation_params["Inline Resolution"] = True
        except Exception:
            pass

    def before_process_batch(self, p, *args, **kwargs):
        sizes = getattr(p, "_inline_resolution_sizes", None)
        if not sizes:
            return

        n = int(getattr(p, "iteration", 0) or 0)
        if n >= len(sizes):
            return

        width, height = sizes[n]
        if width <= 0 or height <= 0:
            return

        # Only update if the size actually differs to avoid pointless RNG/model
        # churn.
        current_w = int(getattr(p, "width", 0) or 0)
        current_h = int(getattr(p, "height", 0) or 0)
        if width == current_w and height == current_h:
            return

        p.width = int(width)
        p.height = int(height)

        # Recreate the latent RNG so it matches the new dimensions. The core
        # loop already built p.rng before this hook fires.
        try:
            from modules import shared
            latent_channels = getattr(shared.sd_model, "latent_channels", 4)
        except Exception:
            latent_channels = 4

        try:
            seeds = getattr(p, "seeds", None) or [int(getattr(p, "seed", 0) or 0)]
            subseeds = getattr(p, "subseeds", None) or [int(getattr(p, "subseed", 0) or 0)]
            p.rng = rng.ImageRNG(
                (latent_channels, p.height // opt_f, p.width // opt_f),
                seeds,
                subseeds=subseeds,
                subseed_strength=getattr(p, "subseed_strength", 0),
                seed_resize_from_h=getattr(p, "seed_resize_from_h", 0),
                seed_resize_from_w=getattr(p, "seed_resize_from_w", 0),
            )
        except Exception:
            # If RNG rebuild fails for any reason, fall back to whatever the
            # core loop produced; the generation will still run, just at the
            # core-provided shape.
            pass

        # Hires fix recalculation, only when no explicit resize is provided.
        if hasattr(p, "enable_hr") and getattr(p, "enable_hr", False):
            hr_resize_x = int(getattr(p, "hr_resize_x", 0) or 0)
            hr_resize_y = int(getattr(p, "hr_resize_y", 0) or 0)
            hr_scale = float(getattr(p, "hr_scale", 0) or 0)
            if hr_resize_x == 0 and hr_resize_y == 0 and hr_scale > 0:
                p.hr_upscale_to_x = int(p.width * hr_scale)
                p.hr_upscale_to_y = int(p.height * hr_scale)

        try:
            if p.extra_generation_params is None:
                p.extra_generation_params = {}
            p.extra_generation_params["Inline Resolution size"] = f"{p.width}x{p.height}"
        except Exception:
            pass
