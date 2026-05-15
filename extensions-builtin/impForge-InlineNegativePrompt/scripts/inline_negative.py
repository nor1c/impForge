"""
Inline Negative Prompt extension.

Lets users embed per-line negative tags inside the positive prompt using a
`(n:...)` marker. The contents of every marker are stripped from the positive
prompt and appended to the corresponding negative prompt at generation time.

Designed for batch workflows that send a list of prompts line-by-line while
keeping a single shared static negative prompt.

This extension only mutates the per-image prompt arrays
(`p.all_prompts`, `p.all_negative_prompts`, and the hires variants), so it
does not require any changes to core code.
"""

import re

import gradio as gr

from modules import scripts
from modules.ui_components import InputAccordion


MARKER_PREFIX = "n:"


def _find_inline_negative_groups(prompt):
    """
    Scan ``prompt`` for top-level parenthesised groups whose content starts
    with ``n:``. Returns a list of ``(start, end, inner)`` tuples where
    ``start``/``end`` are slice indices into ``prompt`` covering the full
    ``(n:...)`` marker (including the parentheses) and ``inner`` is the raw
    text between ``n:`` and the closing parenthesis.

    Only the outer-most parenthesis pairs are considered. Nested parentheses
    inside an inline-negative group are kept as part of ``inner`` so things
    like ``(n:(low angle:1.2))`` round-trip cleanly.
    """

    groups = []
    if not prompt:
        return groups

    i = 0
    n = len(prompt)
    while i < n:
        ch = prompt[i]
        if ch == "(":
            # Find the matching closing paren, accounting for nesting and
            # simple escape sequences (\() so we don't blow up on weighted
            # prompt syntax used elsewhere.
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
                # Unmatched paren, leave the rest of the string alone.
                break

            inner_full = prompt[i + 1:j]
            stripped = inner_full.lstrip()
            if stripped.lower().startswith(MARKER_PREFIX):
                # Preserve whatever follows the prefix verbatim.
                offset = inner_full.lower().find(MARKER_PREFIX)
                inner = inner_full[offset + len(MARKER_PREFIX):]
                groups.append((i, j + 1, inner))

            i = j + 1
            continue

        if ch == "\\" and i + 1 < n:
            i += 2
            continue

        i += 1

    return groups


def _split_tags(text):
    """Split a comma-separated tag string into trimmed tags, dropping empties."""

    return [t.strip() for t in text.split(",") if t.strip()]


def _remove_exact_duplicate_tags(prompt, tags_to_remove):
    """
    Remove top-level comma-separated tags from ``prompt`` that are an exact
    case-insensitive match for any tag in ``tags_to_remove``. Only tags at
    the outer level (depth 0 with respect to parentheses) are considered, so
    we do not accidentally rewrite the inside of weighted groups.
    """

    if not prompt or not tags_to_remove:
        return prompt

    targets = {t.strip().lower() for t in tags_to_remove if t and t.strip()}
    if not targets:
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
            segments.append((start, i, i + 1, prompt[start:i]))
            start = i + 1
        i += 1
    segments.append((start, n, n, prompt[start:]))

    remove_spans = []
    for start, end, sep_end, seg in segments:
        if seg.strip().lower() in targets:
            remove_spans.append((start, sep_end))

    if not remove_spans:
        return prompt

    cleaned = prompt
    for start, end in reversed(remove_spans):
        cleaned = cleaned[:start] + cleaned[end:]

    return _tidy_separators(cleaned)


def _append_negative(existing, addition):
    addition = (addition or "").strip().strip(",")
    if not addition:
        return existing
    existing = (existing or "").rstrip()
    if not existing:
        return addition
    if existing.endswith(","):
        return f"{existing} {addition}"
    return f"{existing}, {addition}"


def _apply(prompt, negative):
    """
    Apply the inline-negative transform to a single ``(prompt, negative)``
    pair. Returns the rewritten ``(prompt, negative, extracted_tags)``
    tuple. ``extracted_tags`` is the flat list of tags moved out of the
    positive prompt for metadata reporting.
    """

    if not prompt:
        return prompt, negative, []

    groups = _find_inline_negative_groups(prompt)
    if not groups:
        return prompt, negative, []

    extracted = []
    # Build the cleaned prompt by removing each marker span (right-to-left so
    # earlier indices remain valid).
    cleaned = prompt
    for start, end, inner in reversed(groups):
        cleaned = cleaned[:start] + cleaned[end:]
        extracted_tags = _split_tags(inner)
        # Preserve original ordering of extracted tags overall.
        extracted = extracted_tags + extracted

    # Tidy up stray separators left behind by removal (e.g. ", , ").
    cleaned = _tidy_separators(cleaned)

    # Remove exact duplicate top-level positive tags.
    cleaned = _remove_exact_duplicate_tags(cleaned, extracted)

    # Append all extracted tags to the negative prompt as a single block.
    if extracted:
        negative = _append_negative(negative, ", ".join(extracted))

    return cleaned, negative, extracted


def _tidy_separators(prompt):
    """Collapse separator litter left by removed markers without flattening lines."""

    if not prompt:
        return prompt

    text = prompt

    # Marker removal commonly leaves ", ," on the same line. Collapse only
    # same-line comma clutter so existing newlines and paragraph spacing remain
    # untouched.
    while True:
        cleaned = re.sub(r",[ \t]*,", ",", text)
        if cleaned == text:
            break
        text = cleaned

    # Remove spaces/tabs left directly before line breaks, but preserve the
    # line breaks themselves.
    text = re.sub(r"[ \t]+(\r?\n)", r"\1", text)

    # If a removed first tag left a leading comma, drop that comma without
    # touching blank lines after it.
    text = re.sub(r"^([ \t]*),[ \t]*", r"\1", text)

    return text


class InlineNegativePromptScript(scripts.Script):
    sorting_priority = 0.5

    def title(self):
        return "Inline Negative Prompt"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(True, label="Inline Negative Prompt", elem_id="inline_negative_prompt_enable", open=False) as enabled:
            gr.Markdown(
                "Move tags wrapped in `(n:...)` from the positive prompt into "
                "the negative prompt at generation time. Useful for per-line "
                "negatives in batch prompt-list workflows."
            )
            gr.Markdown(
                "Syntax: `(n:from below)` or `(n:from below, low angle)`. "
                "Exact duplicate top-level tags are removed from the positive "
                "prompt. Static negative prompt is left untouched aside from "
                "having the extracted tags appended."
            )
        return (enabled,)

    def process(self, p, enabled):
        if not enabled:
            return

        moved_any = False
        all_extracted = []

        all_prompts = list(getattr(p, "all_prompts", []) or [])
        all_negative_prompts = list(getattr(p, "all_negative_prompts", []) or [])

        # Pad the negative list defensively; setup_prompts() should already
        # keep these in sync but we never want an IndexError to surface.
        if len(all_negative_prompts) < len(all_prompts):
            pad = [""] * (len(all_prompts) - len(all_negative_prompts))
            all_negative_prompts.extend(pad)

        for i, prompt in enumerate(all_prompts):
            negative = all_negative_prompts[i] if i < len(all_negative_prompts) else ""
            new_prompt, new_negative, extracted = _apply(prompt, negative)
            if extracted:
                moved_any = True
                all_extracted.extend(extracted)
                all_prompts[i] = new_prompt
                all_negative_prompts[i] = new_negative

        if moved_any:
            p.all_prompts = all_prompts
            p.all_negative_prompts = all_negative_prompts

        # Hires fix prompts are populated by setup_prompts() too. Apply the
        # same transform when present so the second pass also benefits.
        all_hr_prompts = getattr(p, "all_hr_prompts", None)
        all_hr_negative_prompts = getattr(p, "all_hr_negative_prompts", None)
        if all_hr_prompts and all_hr_negative_prompts:
            hr_prompts = list(all_hr_prompts)
            hr_negatives = list(all_hr_negative_prompts)
            if len(hr_negatives) < len(hr_prompts):
                hr_negatives.extend([""] * (len(hr_prompts) - len(hr_negatives)))

            hr_changed = False
            for i, prompt in enumerate(hr_prompts):
                negative = hr_negatives[i] if i < len(hr_negatives) else ""
                new_prompt, new_negative, extracted = _apply(prompt, negative)
                if extracted:
                    hr_changed = True
                    hr_prompts[i] = new_prompt
                    hr_negatives[i] = new_negative

            if hr_changed:
                p.all_hr_prompts = hr_prompts
                p.all_hr_negative_prompts = hr_negatives

        if moved_any:
            try:
                if p.extra_generation_params is None:
                    p.extra_generation_params = {}
                p.extra_generation_params["Inline Negative Prompt"] = True
                # Deduplicate while preserving order for readability.
                seen = set()
                ordered = []
                for tag in all_extracted:
                    key = tag.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    ordered.append(tag)
                if ordered:
                    p.extra_generation_params["Inline Negative Prompt tags"] = ", ".join(ordered)
            except Exception:
                # Metadata is best-effort, never fail generation over it.
                pass
