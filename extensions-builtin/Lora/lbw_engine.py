from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CLIP_LATE_FRACTION = 1.0 / 3.0

# BASE covers the early text-encoder layers (CLIP-L 0-7, CLIP-G 0-20), which
# read the prompt semantically: context, framing, and pose wording. BASE_LATE
# covers the final layers, which is where a trigger word binds to identity.
#
# The character presets used to push BASE above 1.0 to strengthen identity, but
# that amplified the dataset's habitual posing along with it, so prompt wording
# had to fight an over-weighted reading of itself. Text-encoder conditioning is
# computed once before sampling rather than per step, so the preset is the only
# place to address it. BASE is therefore held below 1.0 while
# BASE_LATE keeps its original value, so trigger-word binding is unchanged.
SDXL_PRESETS = {
    'ALL': (1.0,) * 13,
    'NONE': (0.0,) * 13,
    'STYLE_SDXL': (0, 0, 0, 0.15, 0.25, 0.55, 0.75, 0.95, 1.0, 1.0, 0.9, 0.7, 0.35),
    'STYLE_SDXL_PURE': (0, 0, 0, 0, 0, 0, 0.75, 0.95, 1.0, 1.0, 0.9, 0.7, 0.0),
    'CHAR_SDXL': (0.85, 1.15, 1.1, 1.1, 1.05, 1.0, 0.8, 0.65, 0.5, 0.35, 0.25, 0.15, 0.7),
    'CHAR_SDXL_STRONG': (0.9, 1.25, 1.2, 1.15, 1.1, 1.05, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.8),
    'CHAR_SDXL_MAX': (0.95, 1.4, 1.3, 1.25, 1.2, 1.1, 1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.9),
}

SDXL_BLOCK_ORDER = (
    'BASE', 'IN04', 'IN05', 'IN07', 'IN08', 'M00',
    'OUT00', 'OUT01', 'OUT02', 'OUT03', 'OUT04', 'OUT05',
    'BASE_LATE',
)

ROLE_ALIASES = {
    'style': 'STYLE_SDXL',
    'sty': 'STYLE_SDXL',
    'style_sdxl': 'STYLE_SDXL',
    'style_pure': 'STYLE_SDXL_PURE',
    'sty_pure': 'STYLE_SDXL_PURE',
    'style_sdxl_pure': 'STYLE_SDXL_PURE',
    'char': 'CHAR_SDXL',
    'character': 'CHAR_SDXL',
    'content': 'CHAR_SDXL',
    'char_sdxl': 'CHAR_SDXL',
    'char_strong': 'CHAR_SDXL_STRONG',
    'character_strong': 'CHAR_SDXL_STRONG',
    'char_sdxl_strong': 'CHAR_SDXL_STRONG',
    'char_max': 'CHAR_SDXL_MAX',
    'character_max': 'CHAR_SDXL_MAX',
    'char_sdxl_max': 'CHAR_SDXL_MAX',
}

_re_input_block = re.compile(r'(?:^|\.)diffusion_model\.input_blocks\.(\d+)\.')
_re_middle_block = re.compile(r'(?:^|\.)diffusion_model\.middle_block\.')
_re_output_block = re.compile(r'(?:^|\.)diffusion_model\.output_blocks\.(\d+)\.')
_re_clip_layer = re.compile(r'(?:encoder\.layers|resblocks)\.(\d+)\.')
_re_clip_l = re.compile(r'(?:^|\.)(?:clip_l|text_encoder)(?:\.|$)')
_re_clip_g = re.compile(r'(?:^|\.)(?:clip_g|text_encoder_2)(?:\.|$)')
_re_clip_family = re.compile(r'(?:^|\.)(?:clip_[lg]|text_encoder(?:_2)?|transformer\.text_model)(?:\.|$)')

_CLIP_L_LAYERS = 12
_CLIP_G_LAYERS = 32

# SDXL UNet layout (channel_mult [1,2,4], num_res_blocks 2, transformer_depth
# [0,0,2,2,10,10]):
#   IN00 conv_in, IN01-02 level0, IN03 downsample, IN04-05 level1,
#   IN06 downsample, IN07-08 level2, M00 middle,
#   OUT00-02 level2, OUT03-05 level1, OUT06-08 level0.
#
# Only the attention-carrying blocks have a dedicated preset slot, because the
# 12/13-value preset vocabulary is fixed for backwards compatibility. Blocks
# without their own slot are folded onto the slots of the same resolution level
# so that conv-touching LoRAs (LoCon/LyCORIS) are still weighted. Leaving them
# unscaled meant they kept full strength and bypassed role splitting entirely.
#
# A folded block takes the mean of its level's slots rather than a single
# neighbour, and the result is clamped to 1.0. Folding onto one neighbour let a
# block inherit an above-1.0 character weight (IN04 is 1.15 in CHAR_SDXL) or an
# aggressive tail weight (OUT05 is 0.15), so a block with no slot of its own
# could end up amplified or gutted. The mean keeps it representative of the
# level and the clamp keeps folding a restriction, never a boost.
_INPUT_SLOT = {
    0: 'IN04', 1: 'IN04', 2: 'IN04', 3: 'IN04',  # level0 + downsample -> earliest slot
    4: 'IN04', 5: 'IN05',
    6: 'IN07',                                    # downsample into level2
    7: 'IN07', 8: 'IN08',
}
_OUTPUT_SLOT = {
    0: 'OUT00', 1: 'OUT01', 2: 'OUT02',
    3: 'OUT03', 4: 'OUT04', 5: 'OUT05',
    6: 'OUT05', 7: 'OUT05', 8: 'OUT05',           # level0 -> latest slot
}

# Blocks that own a preset slot outright. Anything mapped through the folding
# above is reported separately so coverage stays auditable in logs.
_DIRECT_INPUT_SLOTS = frozenset((4, 5, 7, 8))
_DIRECT_OUTPUT_SLOTS = frozenset((0, 1, 2, 3, 4, 5))

# Slots a folded block averages over, keyed by the slot it maps to. Input
# level0/level1 share IN04-IN05; the level2 entry point shares IN07-IN08.
# Output level0 has no slots of its own and averages the level1 tail.
_FOLD_GROUP = {
    'IN04': ('IN04', 'IN05'),
    'IN07': ('IN07', 'IN08'),
    'OUT05': ('OUT03', 'OUT04', 'OUT05'),
}

FOLD_CEILING = 1.0


@dataclass(frozen=True)
class KeyClassification:
    slot: str | None
    category: str


def _clip_slot(key):
    match = _re_clip_layer.search(key)
    if match is None:
        return 'BASE'
    layer = int(match.group(1))
    if _re_clip_g.search(key):
        num_layers = _CLIP_G_LAYERS
    elif _re_clip_l.search(key):
        num_layers = _CLIP_L_LAYERS
    else:
        num_layers = _CLIP_G_LAYERS if layer >= _CLIP_L_LAYERS else _CLIP_L_LAYERS
    late_threshold = num_layers - max(1, int(round(num_layers * CLIP_LATE_FRACTION)))
    return 'BASE_LATE' if layer >= late_threshold else 'BASE'


def classify_key(key):
    if 'diffusion_model.' in key:
        if _re_middle_block.search(key):
            return KeyClassification('M00', 'scaled')
        match = _re_input_block.search(key)
        if match:
            index = int(match.group(1))
            slot = _INPUT_SLOT.get(index)
            if slot is None:
                return KeyClassification(None, 'passthrough')
            return KeyClassification(slot, 'scaled' if index in _DIRECT_INPUT_SLOTS else 'folded')
        match = _re_output_block.search(key)
        if match:
            index = int(match.group(1))
            slot = _OUTPUT_SLOT.get(index)
            if slot is None:
                return KeyClassification(None, 'passthrough')
            return KeyClassification(slot, 'scaled' if index in _DIRECT_OUTPUT_SLOTS else 'folded')
        return KeyClassification(None, 'passthrough')
    if _re_clip_family.search(key):
        return KeyClassification(_clip_slot(key), 'scaled')
    return KeyClassification(None, 'unknown')


def key_to_slot(key):
    return classify_key(key).slot


def _expand_12_to_13(values):
    return tuple(values) + (values[0],)


def parse_lbw_spec(spec):
    if spec is None:
        return None
    value = str(spec).strip()
    if not value:
        return None
    preset = value.upper()
    if preset in SDXL_PRESETS:
        return SDXL_PRESETS[preset]
    alias = ROLE_ALIASES.get(value.lower())
    if alias:
        return SDXL_PRESETS[alias]
    if ',' not in value:
        return None
    parts = [part.strip() for part in value.split(',')]
    if len(parts) not in (12, 13) or any(not part for part in parts):
        return None
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return None
    if any(not math.isfinite(item) for item in values):
        return None
    return _expand_12_to_13(values) if len(values) == 12 else values


def parse_role(spec):
    if spec is None:
        return None
    preset = ROLE_ALIASES.get(str(spec).strip().lower())
    return SDXL_PRESETS[preset] if preset else None


DEFAULT_PATH_RULES = (
    (re.compile(r'(?:^|[_\-.])(?:characters?|char)(?:$|[_\-.])'), 'CHAR_SDXL'),
    (re.compile(r'(?:^|[_\-.])(?:styles?|sty)(?:$|[_\-.])'), 'STYLE_SDXL'),
)


def infer_ratios_from_path(filename):
    if not filename:
        return None
    parts = filename.replace('\\', '/').lower().split('/')[:-1]
    for part in parts:
        for pattern, preset in DEFAULT_PATH_RULES:
            if pattern.search(part):
                return SDXL_PRESETS[preset]
    return None


def new_stats():
    return {'scaled': 0, 'folded': 0, 'passthrough': 0, 'unknown': 0, 'total': 0, 'unknown_keys': []}


def resolve_ratio(classification, ratios):
    """The weight to apply for one classified key.

    Directly-scaled blocks use their own slot. Folded blocks take the mean of
    their resolution level's slots, capped at FOLD_CEILING so that folding can
    only ever restrict a block, never amplify it past full strength.
    """
    slot = classification.slot
    ratio = ratios[SDXL_BLOCK_ORDER.index(slot)]
    if classification.category != 'folded':
        return ratio
    group = _FOLD_GROUP.get(slot)
    if group:
        ratio = sum(ratios[SDXL_BLOCK_ORDER.index(name)] for name in group) / len(group)
    return min(ratio, FOLD_CEILING)


EFFECTIVE_CEILING = 1.0


def clamp_effective(value, ceiling=EFFECTIVE_CEILING):
    """Hold an effective weight within +/- ceiling, keeping its sign.

    A patch entry already carries the requested strength, so multiplying it by a
    preset ratio above 1.0 produces an effective weight the LoRA was never
    trained at. That extrapolation is what makes a character LoRA impose its
    dataset's habitual posing, and it also means the number written in the
    prompt is not the number being applied.
    """
    if ceiling is None:
        return value
    if value > ceiling:
        return ceiling
    if value < -ceiling:
        return -ceiling
    return value


def scale_patches_inplace(patches_dict, added_keys, ratios, ceiling=EFFECTIVE_CEILING):
    stats = new_stats()
    if ratios is None:
        return stats
    if len(ratios) == 12:
        ratios = _expand_12_to_13(ratios)
    for key, index in added_keys:
        stats['total'] += 1
        classification = classify_key(key)
        if classification.category not in ('scaled', 'folded'):
            stats[classification.category] += 1
            if classification.category == 'unknown' and len(stats['unknown_keys']) < 5:
                stats['unknown_keys'].append(key)
            continue
        ratio = resolve_ratio(classification, ratios)
        entry = patches_dict[key][index]
        scaled = clamp_effective(entry[0] * ratio, ceiling)
        patches_dict[key][index] = (scaled,) + tuple(entry[1:])
        stats[classification.category] += 1
    return stats


def snapshot_patches(patches_dict):
    return {key: len(values) for key, values in patches_dict.items()}


def diff_added_keys(patches_dict, before):
    for key, values in patches_dict.items():
        for index in range(before.get(key, 0), len(values)):
            yield key, index


def _merge_stats(target, source):
    for key in ('scaled', 'folded', 'passthrough', 'unknown', 'total'):
        target[key] += source[key]
    target['unknown_keys'].extend(source['unknown_keys'][:5 - len(target['unknown_keys'])])


def apply_block_weights(unet_patcher, clip_patcher, before_unet, before_clip, ratios, lora_name='<unknown>', ceiling=EFFECTIVE_CEILING):
    stats = new_stats()
    if ratios is None:
        return stats
    if unet_patcher is not None and hasattr(unet_patcher, 'patches'):
        added = list(diff_added_keys(unet_patcher.patches, before_unet or {}))
        _merge_stats(stats, scale_patches_inplace(unet_patcher.patches, added, ratios, ceiling))
    if clip_patcher is not None:
        clip_patches = getattr(clip_patcher, 'patches', None)
        if clip_patches is None and hasattr(clip_patcher, 'patcher'):
            clip_patches = getattr(clip_patcher.patcher, 'patches', None)
        if clip_patches is not None:
            added = list(diff_added_keys(clip_patches, before_clip or {}))
            _merge_stats(stats, scale_patches_inplace(clip_patches, added, ratios, ceiling))
    logger.info(
        '[LBW] %s scaled=%d folded=%d passthrough=%d unknown=%d total=%d',
        os.path.basename(str(lora_name)),
        stats['scaled'],
        stats['folded'],
        stats['passthrough'],
        stats['unknown'],
        stats['total'],
    )
    if stats['unknown_keys']:
        logger.warning('[LBW] %s has unsupported patch keys: %s', os.path.basename(str(lora_name)), ', '.join(stats['unknown_keys']))
    return stats


# Each LoRA is capped at 1.0 individually, so a warning is only meaningful for
# what stacking adds on top. A character plus one style LoRA at sane strengths
# lands around 1.19-1.29 and is a normal, working combination, so the threshold
# sits above it: warn when a block carries roughly a full extra LoRA's worth of
# weight, not whenever two LoRAs overlap.
STACK_WARN_THRESHOLD = 1.35


def effective_weights(unet_strength, te_strength, ratios, ceiling=EFFECTIVE_CEILING):
    """Per-block effective weight for one LoRA, as actually applied.

    The strength written in the prompt is multiplied by the preset ratio, so
    what runs is rarely the number the user typed. Returning the applied values
    keeps that visible in logs and metadata.
    """
    if ratios is None:
        ratios = SDXL_PRESETS['ALL']
    if len(ratios) == 12:
        ratios = _expand_12_to_13(ratios)
    weights = {}
    for block, ratio in zip(SDXL_BLOCK_ORDER, ratios):
        # BASE and BASE_LATE are the text-encoder slots; the rest are UNet.
        strength = te_strength if block in ('BASE', 'BASE_LATE') else unet_strength
        weights[block] = clamp_effective(abs(float(strength)) * abs(float(ratio)), ceiling)
    return weights


def summarize_stack_load(entries, threshold=STACK_WARN_THRESHOLD, ceiling=EFFECTIVE_CEILING):
    """Report per-block weight totals across a whole LoRA stack.

    ``entries`` is an iterable of (name, unet_strength, te_strength, ratios).
    Each LoRA is capped individually, but the stack total is not: several LoRAs
    contributing to the same block still add up past full strength, which is
    what makes a stack override prompt-driven composition. That total is not
    visible from the individual strengths written in the prompt, so it is worth
    surfacing.

    Returns (totals, overloaded) where totals maps block name -> summed weight
    and overloaded lists (block, total) pairs above the threshold.
    """
    totals = dict.fromkeys(SDXL_BLOCK_ORDER, 0.0)
    for _name, unet_strength, te_strength, ratios in entries:
        for block, weight in effective_weights(unet_strength, te_strength, ratios, ceiling).items():
            totals[block] += weight
    overloaded = [(block, total) for block, total in totals.items() if total > threshold]
    return totals, overloaded

def log_stack_load(entries, threshold=STACK_WARN_THRESHOLD, ceiling=EFFECTIVE_CEILING):
    for name, unet_strength, te_strength, ratios in entries:
        weights = effective_weights(unet_strength, te_strength, ratios, ceiling)
        peak_block = max(weights, key=weights.get)
        logger.info(
            '[LBW] %s: written UNet:%g TE:%g -> peak effective %.2f at %s',
            os.path.basename(str(name)), unet_strength, te_strength,
            weights[peak_block], peak_block,
        )
    totals, overloaded = summarize_stack_load(entries, threshold, ceiling)
    if overloaded:
        detail = ', '.join(f'{block}={total:.2f}' for block, total in overloaded)
        logger.warning(
            '[LBW] stacked LoRA weight exceeds %.2f on %d block(s): %s. '
            'Blocks above 1.0 tend to override prompt-driven pose and composition; '
            'lower a strength or remove a LoRA if poses stop responding.',
            threshold, len(overloaded), detail,
        )
    return totals, overloaded

def is_native_engine_active():
    return True
