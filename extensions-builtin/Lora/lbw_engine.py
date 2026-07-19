from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CLIP_LATE_FRACTION = 1.0 / 3.0

SDXL_PRESETS = {
    'ALL': (1.0,) * 13,
    'NONE': (0.0,) * 13,
    'STYLE_SDXL': (0, 0, 0, 0.15, 0.25, 0.55, 0.75, 0.95, 1.0, 1.0, 0.9, 0.7, 0.35),
    'STYLE_SDXL_PURE': (0, 0, 0, 0, 0, 0, 0.75, 0.95, 1.0, 1.0, 0.9, 0.7, 0.0),
    'CHAR_SDXL': (1.15, 1.15, 1.1, 1.1, 1.05, 1.0, 0.8, 0.65, 0.5, 0.35, 0.25, 0.15, 0.7),
    'CHAR_SDXL_STRONG': (1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.8),
    'CHAR_SDXL_MAX': (1.4, 1.4, 1.3, 1.25, 1.2, 1.1, 1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.9),
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
_INPUT_SLOT = {4: 'IN04', 5: 'IN05', 7: 'IN07', 8: 'IN08'}
_OUTPUT_SLOT = {0: 'OUT00', 1: 'OUT01', 2: 'OUT02', 3: 'OUT03', 4: 'OUT04', 5: 'OUT05'}


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
            slot = _INPUT_SLOT.get(int(match.group(1)))
            return KeyClassification(slot, 'scaled' if slot else 'passthrough')
        match = _re_output_block.search(key)
        if match:
            slot = _OUTPUT_SLOT.get(int(match.group(1)))
            return KeyClassification(slot, 'scaled' if slot else 'passthrough')
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


def scale_patches_inplace(patches_dict, added_keys, ratios):
    stats = {'scaled': 0, 'passthrough': 0, 'unknown': 0, 'total': 0, 'unknown_keys': []}
    if ratios is None:
        return stats
    if len(ratios) == 12:
        ratios = _expand_12_to_13(ratios)
    for key, index in added_keys:
        stats['total'] += 1
        classification = classify_key(key)
        if classification.category != 'scaled':
            stats[classification.category] += 1
            if classification.category == 'unknown' and len(stats['unknown_keys']) < 5:
                stats['unknown_keys'].append(key)
            continue
        ratio = ratios[SDXL_BLOCK_ORDER.index(classification.slot)]
        entry = patches_dict[key][index]
        patches_dict[key][index] = (entry[0] * ratio,) + tuple(entry[1:])
        stats['scaled'] += 1
    return stats


def snapshot_patches(patches_dict):
    return {key: len(values) for key, values in patches_dict.items()}


def diff_added_keys(patches_dict, before):
    for key, values in patches_dict.items():
        for index in range(before.get(key, 0), len(values)):
            yield key, index


def _merge_stats(target, source):
    for key in ('scaled', 'passthrough', 'unknown', 'total'):
        target[key] += source[key]
    target['unknown_keys'].extend(source['unknown_keys'][:5 - len(target['unknown_keys'])])


def apply_block_weights(unet_patcher, clip_patcher, before_unet, before_clip, ratios, lora_name='<unknown>'):
    stats = {'scaled': 0, 'passthrough': 0, 'unknown': 0, 'total': 0, 'unknown_keys': []}
    if ratios is None:
        return stats
    if unet_patcher is not None and hasattr(unet_patcher, 'patches'):
        added = list(diff_added_keys(unet_patcher.patches, before_unet or {}))
        _merge_stats(stats, scale_patches_inplace(unet_patcher.patches, added, ratios))
    if clip_patcher is not None:
        clip_patches = getattr(clip_patcher, 'patches', None)
        if clip_patches is None and hasattr(clip_patcher, 'patcher'):
            clip_patches = getattr(clip_patcher.patcher, 'patches', None)
        if clip_patches is not None:
            added = list(diff_added_keys(clip_patches, before_clip or {}))
            _merge_stats(stats, scale_patches_inplace(clip_patches, added, ratios))
    logger.info(
        '[LBW] %s scaled=%d passthrough=%d unknown=%d total=%d',
        os.path.basename(str(lora_name)),
        stats['scaled'],
        stats['passthrough'],
        stats['unknown'],
        stats['total'],
    )
    if stats['unknown_keys']:
        logger.warning('[LBW] %s has unsupported patch keys: %s', os.path.basename(str(lora_name)), ', '.join(stats['unknown_keys']))
    return stats


def is_native_engine_active():
    return True
