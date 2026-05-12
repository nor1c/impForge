"""
Native LoRA Block Weight engine for reForge/impForge.

Why this exists
---------------
The external sd-webui-lora-block-weight extension has a fatal bug in its
reForge code path (lbwrf): when multiple LoRAs are stacked and not all of
them patch the same state_dict key, its zip() over (lvals, mu, lwei, ...)
misaligns per-LoRA block weights and applies them to the wrong LoRAs. That
causes the cross-contamination and artifacts the user sees (e.g., style LoRA
block weights being applied to character LoRA patches and vice versa).

This engine fixes the problem by applying block weights *per-LoRA, inline*
during load, using a snapshot-diff: we record the ModelPatcher's patch state
before a LoRA is added and after, scale only the new entries by that LoRA's
block-ratio vector, then commit. This makes cross-LoRA alignment impossible
to mess up because each LoRA's patches are scaled only by its own ratios.

SDXL 12-block layout (canonical order used by lbw=): 
    BASE, IN04, IN05, IN07, IN08, M00, OUT00, OUT01, OUT02, OUT03, OUT04, OUT05

The 13th slot BASE_LATE is an engine extension for finer text-encoder
control: CLIP encoders have a shallow half (layers 0 .. ~2/3 depth) that
carries semantic/identity concepts and a deep tail (last ~1/3 of layers)
that carries visual/style-adjacent concepts. Scaling those halves
independently lets a character LoRA dominate semantic identity (BASE) while
style LoRAs keep their visual-concept hooks (BASE_LATE) or vice versa.

Backward compatibility: users can still pass 12-value lbw= lists. In that
case BASE_LATE is auto-filled from BASE, so nothing changes for existing
prompts.

Only the blocks that actually contain transformer / cross-attention layers
get slots — IN00-IN03, IN06, IN09-IN11, OUT06-OUT11 are either pure
convolutions, time-embeddings, or do not exist in SDXL's UNet.
"""
from __future__ import annotations
import os
import re
import logging

logger = logging.getLogger(__name__)

# Fraction of each CLIP encoder's deepest layers treated as "late". The
# research consensus is that deep CLIP layers encode visual/style-adjacent
# concepts while shallow layers encode semantic identity. 1/3 is the
# community-tested split that isolates the two without being too surgical.
CLIP_LATE_FRACTION = 1.0 / 3.0

# ---------------------------------------------------------------------------
# Presets (SDXL 13-block: BASE, IN04, IN05, IN07, IN08, M00,
#                          OUT00, OUT01, OUT02, OUT03, OUT04, OUT05, BASE_LATE)
# For backward compat these are stored as 13-tuples. 12-tuple callers still
# work — parse_lbw_spec auto-fills BASE_LATE from BASE.
# ---------------------------------------------------------------------------
SDXL_PRESETS = {
    # Flat reference
    "ALL":  (1.0,) * 13,
    "NONE": (0.0,) * 13,
    # Style LoRA: keep OUT (style/detail), attenuate IN/M (content). Allows tiny
    # bleed in IN07/IN08/M00 so composition can still benefit from style cues.
    # BASE_LATE (last slot) slightly open so style LoRAs can still influence
    # visual concept embeddings in the deep CLIP layers.
    "STYLE_SDXL":       (0, 0, 0, 0.15, 0.25, 0.55, 0.75, 0.95, 1.0, 1.0, 0.9, 0.7, 0.35),
    # Style LoRA, zero content leakage: fully zero IN + M00 and zero BASE_LATE.
    "STYLE_SDXL_PURE":  (0, 0, 0, 0,    0,    0,    0.75, 0.95, 1.0, 1.0, 0.9, 0.7, 0.0),
    # Character LoRA: amplify IN/M (identity/clothing/hair), fade OUT (style).
    # BASE_LATE kept lower than BASE so char LoRA doesn't rewrite style-
    # adjacent deep-CLIP concepts that the style LoRAs need.
    "CHAR_SDXL":        (1.15, 1.15, 1.1, 1.1, 1.05, 1.0, 0.8, 0.65, 0.5, 0.35, 0.25, 0.15, 0.7),
    # Character LoRA, stronger identity lock.
    "CHAR_SDXL_STRONG": (1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.8),
    # Character LoRA, maximum identity lock.
    "CHAR_SDXL_MAX":    (1.4, 1.4, 1.3, 1.25, 1.2, 1.1, 1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.9),
}

# Canonical slot order. BASE_LATE is slot 12 (13th) so 12-value lists remain
# backward-compatible.
SDXL_BLOCK_ORDER = (
    "BASE", "IN04", "IN05", "IN07", "IN08", "M00",
    "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05",
    "BASE_LATE",
)

# Role shorthand -> preset name
ROLE_ALIASES = {
    "style":            "STYLE_SDXL",
    "sty":              "STYLE_SDXL",
    "style_sdxl":       "STYLE_SDXL",
    "style_pure":       "STYLE_SDXL_PURE",
    "sty_pure":         "STYLE_SDXL_PURE",
    "style_sdxl_pure":  "STYLE_SDXL_PURE",
    "char":             "CHAR_SDXL",
    "character":        "CHAR_SDXL",
    "content":          "CHAR_SDXL",
    "char_sdxl":        "CHAR_SDXL",
    "char_strong":      "CHAR_SDXL_STRONG",
    "character_strong": "CHAR_SDXL_STRONG",
    "char_sdxl_strong": "CHAR_SDXL_STRONG",
    "char_max":         "CHAR_SDXL_MAX",
    "character_max":    "CHAR_SDXL_MAX",
    "char_sdxl_max":    "CHAR_SDXL_MAX",
}

# ---------------------------------------------------------------------------
# Key -> block matcher
# ---------------------------------------------------------------------------
# Patch keys from ldm_patched have the form
#   diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight
#   diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight
#   diffusion_model.output_blocks.0.1.transformer_blocks.2.attn2.to_v.weight
# CLIP keys:
#   clip_l.transformer.text_model.encoder.layers.N.self_attn.q_proj.weight
#   clip_g.transformer.text_model.encoder.layers.N.self_attn.q_proj.weight
#   transformer.text_model.encoder.layers.N.* (fallback)

_re_input_block  = re.compile(r"diffusion_model\.input_blocks\.(\d+)\.")
_re_middle_block = re.compile(r"diffusion_model\.middle_block\.")
_re_output_block = re.compile(r"diffusion_model\.output_blocks\.(\d+)\.")
_re_clip_layer   = re.compile(r"\.encoder\.layers\.(\d+)\.")
_re_clip_l       = re.compile(r"(?:^|\.)clip_l\.")
_re_clip_g       = re.compile(r"(?:^|\.)clip_g\.")

# SDXL CLIP layer counts (for late-layer threshold math).
_CLIP_L_LAYERS = 12   # CLIP-L has 12 transformer layers (0..11)
_CLIP_G_LAYERS = 32   # CLIP-G has 32 transformer layers (0..31)

# Map SDXL input_blocks index -> 12-block slot name (or None if no slot).
# SDXL UNet has 9 input_blocks (0..8). Transformer blocks live at 4,5,7,8.
_INPUT_SLOT = {
    4: "IN04", 5: "IN05", 7: "IN07", 8: "IN08",
}
# Map SDXL output_blocks index -> slot. OUT00..OUT05 exist (transformer).
_OUTPUT_SLOT = {
    0: "OUT00", 1: "OUT01", 2: "OUT02",
    3: "OUT03", 4: "OUT04", 5: "OUT05",
}


def _clip_slot(key: str) -> str:
    """Return BASE_LATE for deep CLIP layers (last ~1/3), BASE otherwise.
    Uses CLIP-L/CLIP-G layer counts when the encoder can be identified; for
    generic 'transformer.*' keys falls back to a CLIP-L-style split."""
    m = _re_clip_layer.search(key)
    if m is None:
        # No layer index at all (e.g. final_layer_norm, text_projection).
        # These are shared across the whole encoder; map to BASE.
        return "BASE"
    layer = int(m.group(1))
    if _re_clip_g.search(key):
        num_layers = _CLIP_G_LAYERS
    elif _re_clip_l.search(key):
        num_layers = _CLIP_L_LAYERS
    else:
        # Generic diffusers / kohya key without clip_l/clip_g prefix.
        # Assume CLIP-L layout (12 layers) — this is the conservative default
        # because most character LoRAs only train CLIP-L.
        num_layers = _CLIP_L_LAYERS
    late_threshold = num_layers - max(1, int(round(num_layers * CLIP_LATE_FRACTION)))
    return "BASE_LATE" if layer >= late_threshold else "BASE"


def key_to_slot(key: str) -> str | None:
    """Return the SDXL 13-block slot name for a patch key, or None if the key
    belongs to a block we don't individually scale (conv stem, time embed,
    etc. — those stay at strength 1.0)."""
    if key.startswith("diffusion_model."):
        m = _re_middle_block.search(key)
        if m:
            return "M00"
        m = _re_input_block.search(key)
        if m:
            return _INPUT_SLOT.get(int(m.group(1)))
        m = _re_output_block.search(key)
        if m:
            return _OUTPUT_SLOT.get(int(m.group(1)))
        return None  # conv_in, time_embed, label_emb, out.0, etc.
    # CLIP / text encoder patches: split by layer depth.
    return _clip_slot(key)


# ---------------------------------------------------------------------------
# Ratio parsing
# ---------------------------------------------------------------------------
def _expand_12_to_13(vals_12):
    """Auto-fill BASE_LATE (slot 12) from BASE (slot 0) for backward compat
    with existing 12-value lbw= lists."""
    return tuple(vals_12) + (vals_12[0],)


def parse_lbw_spec(spec) -> tuple[float, ...] | None:
    """Parse an lbw= argument. Accepts:
      - preset name (e.g. 'CHAR_SDXL')
      - role alias  (e.g. 'char', 'style_pure')
      - 12-value comma list (legacy — BASE_LATE auto-filled from BASE)
      - 13-value comma list (explicit BASE_LATE)
    Returns a 13-tuple of floats, or None if unparseable."""
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None

    # Preset name (case-insensitive)
    key = s.upper()
    if key in SDXL_PRESETS:
        return SDXL_PRESETS[key]

    # Role alias
    if s.lower() in ROLE_ALIASES:
        return SDXL_PRESETS[ROLE_ALIASES[s.lower()]]

    # Comma-separated raw values
    if "," in s:
        try:
            vals = tuple(float(x.strip()) for x in s.split(",") if x.strip() != "")
        except ValueError:
            return None
        if len(vals) == 13:
            return vals
        if len(vals) == 12:
            return _expand_12_to_13(vals)
        # Graceful: accept shorter/longer by padding/truncating to 13.
        if len(vals) < 12:
            vals = vals + (1.0,) * (12 - len(vals))
            return _expand_12_to_13(vals)
        # len between 13 and longer -> truncate
        vals = vals[:13]
        if len(vals) == 13:
            return vals
        return _expand_12_to_13(vals[:12])

    return None


def parse_role(spec: str) -> tuple[float, ...] | None:
    if spec is None:
        return None
    key = str(spec).strip().lower()
    preset = ROLE_ALIASES.get(key)
    if preset is None:
        return None
    return SDXL_PRESETS[preset]


# ---------------------------------------------------------------------------
# Path-based auto-routing
# ---------------------------------------------------------------------------
# If a LoRA lives under a path component matching one of these directory
# names, it gets the corresponding default role preset unless the prompt
# specifies lbw=/role= explicitly.
DEFAULT_PATH_RULES = [
    # (directory-name fragment, preset key)
    ("characters", "CHAR_SDXL"),
    ("character",  "CHAR_SDXL"),
    ("char",       "CHAR_SDXL"),
    ("styles",     "STYLE_SDXL"),
    ("style",      "STYLE_SDXL"),
    ("sty",        "STYLE_SDXL"),
]


def infer_ratios_from_path(filename: str) -> tuple[float, ...] | None:
    """Infer a 13-block ratio tuple from the LoRA filename / path. Returns
    None if no directory rule matches."""
    if not filename:
        return None
    parts = filename.replace("\\", "/").lower().split("/")
    for fragment, preset_key in DEFAULT_PATH_RULES:
        if fragment in parts:
            return SDXL_PRESETS[preset_key]
    return None


# ---------------------------------------------------------------------------
# Patch scaling
# ---------------------------------------------------------------------------
def scale_patches_inplace(patches_dict, added_keys, ratios):
    """Scale the newly-added entries (those in added_keys) of a ModelPatcher's
    .patches dict by the per-block ratios.

    `patches_dict`   — the ModelPatcher.patches dict (or CLIP patcher.patches)
    `added_keys`     — iterable of (key, new_entry_index) pairs identifying
                        which patch tuples were added by *this* LoRA
    `ratios`         — 13-tuple of floats (SDXL block weights + BASE_LATE)

    The patch tuple format is (strength_patch, value, strength_model, offset, function).
    We scale strength_patch by the block's ratio. strength_model is left
    untouched. A ratio of 1.0 is a no-op; ratios of 0.0 effectively disable
    that block's contribution from this LoRA.
    """
    if ratios is None:
        return 0

    # Accept 12-tuples for resilience; expand to 13 on the fly.
    if len(ratios) == 12:
        ratios = _expand_12_to_13(ratios)

    scaled = 0
    for key, idx in added_keys:
        slot = key_to_slot(key)
        if slot is None:
            continue  # conv stem etc. — leave at flat strength
        slot_idx = SDXL_BLOCK_ORDER.index(slot)
        ratio = ratios[slot_idx]
        if ratio == 1.0:
            continue
        entry = patches_dict[key][idx]
        # entry is a tuple: (strength_patch, value, strength_model, offset, function)
        new_strength = entry[0] * ratio
        patches_dict[key][idx] = (new_strength,) + tuple(entry[1:])
        scaled += 1
    return scaled


def snapshot_patches(patches_dict):
    """Capture the current length of each key's patch list so we can compute
    a diff after a LoRA is applied."""
    return {k: len(v) for k, v in patches_dict.items()}


def diff_added_keys(patches_dict, before):
    """Yield (key, index) for every patch tuple added since the snapshot."""
    for k, vals in patches_dict.items():
        prev = before.get(k, 0)
        cur = len(vals)
        for i in range(prev, cur):
            yield (k, i)


# ---------------------------------------------------------------------------
# Public entry point used by networks.py
# ---------------------------------------------------------------------------
def apply_block_weights(unet_patcher, clip_patcher, before_unet, before_clip,
                         ratios, lora_name="<unknown>"):
    """After load_lora_for_models has added a LoRA's patches, scale only the
    newly-added entries by `ratios`. Safe even if one of the patchers is None
    or if ratios is None (no-op)."""
    if ratios is None:
        return

    scaled_total = 0
    if unet_patcher is not None and hasattr(unet_patcher, "patches"):
        added = list(diff_added_keys(unet_patcher.patches, before_unet or {}))
        scaled_total += scale_patches_inplace(unet_patcher.patches, added, ratios)

    if clip_patcher is not None:
        clip_patches = getattr(clip_patcher, "patches", None)
        if clip_patches is None and hasattr(clip_patcher, "patcher"):
            clip_patches = getattr(clip_patcher.patcher, "patches", None)
        if clip_patches is not None:
            added = list(diff_added_keys(clip_patches, before_clip or {}))
            scaled_total += scale_patches_inplace(clip_patches, added, ratios)

    if scaled_total:
        # Compact summary per LoRA — one line instead of per-key spam.
        logger.info(
            "[LBW] %s scaled %d patches with %s",
            os.path.basename(str(lora_name)),
            scaled_total,
            ",".join(f"{r:g}" for r in ratios),
        )


def is_native_engine_active():
    """Sentinel so the legacy sd-webui-lora-block-weight extension can detect
    that native block weighting is handling things and step aside to avoid
    double-apply (which causes artifacts and low contrast)."""
    return True
