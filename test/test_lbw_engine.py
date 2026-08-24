import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / 'extensions-builtin' / 'Lora' / 'lbw_engine.py'
SPEC = importlib.util.spec_from_file_location('impforge_lbw_engine', MODULE_PATH)
lbw_engine = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lbw_engine
SPEC.loader.exec_module(lbw_engine)


@pytest.mark.parametrize(('key', 'slot', 'category'), [
    ('diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight', 'IN04', 'scaled'),
    ('diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight', 'M00', 'scaled'),
    ('diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_q.weight', 'OUT05', 'scaled'),
    ('diffusion_model.input_blocks.0.0.weight', 'IN04', 'folded'),
    ('diffusion_model.input_blocks.6.0.op.weight', 'IN07', 'folded'),
    ('diffusion_model.output_blocks.8.0.in_layers.2.weight', 'OUT05', 'folded'),
    ('diffusion_model.time_embed.0.weight', None, 'passthrough'),
    ('clip_l.transformer.text_model.encoder.layers.11.self_attn.q_proj.weight', 'BASE_LATE', 'scaled'),
    ('clip_g.transformer.text_model.encoder.layers.31.self_attn.q_proj.weight', 'BASE_LATE', 'scaled'),
    ('unsupported.adapter.weight', None, 'unknown'),
])
def test_classify_key(key, slot, category):
    result = lbw_engine.classify_key(key)
    assert result.slot == slot
    assert result.category == category


def test_parse_lbw_spec_is_strict_and_legacy_compatible():
    legacy = lbw_engine.parse_lbw_spec(','.join(['1'] * 12))
    assert legacy == (1.0,) * 13
    assert lbw_engine.parse_lbw_spec(','.join(['1'] * 11)) is None
    assert lbw_engine.parse_lbw_spec(','.join(['1'] * 14)) is None
    assert lbw_engine.parse_lbw_spec(','.join(['nan'] * 13)) is None
    assert lbw_engine.parse_lbw_spec('char_strong') == lbw_engine.SDXL_PRESETS['CHAR_SDXL_STRONG']


def test_scale_only_requested_patch_entries_and_report_coverage():
    patches = {
        'diffusion_model.input_blocks.4.layer': [(2.0, 'old'), (2.0, 'new')],
        'diffusion_model.input_blocks.1.layer': [(3.0, 'folded')],
        'diffusion_model.time_embed.layer': [(3.0, 'pass')],
        'unsupported.layer': [(4.0, 'unknown')],
    }
    added = [
        ('diffusion_model.input_blocks.4.layer', 1),
        ('diffusion_model.input_blocks.1.layer', 0),
        ('diffusion_model.time_embed.layer', 0),
        ('unsupported.layer', 0),
    ]
    ratios = list(lbw_engine.SDXL_PRESETS['ALL'])
    ratios[lbw_engine.SDXL_BLOCK_ORDER.index('IN04')] = 0.5
    # ceiling=None keeps the raw products visible; clamping is covered below.
    stats = lbw_engine.scale_patches_inplace(patches, added, tuple(ratios), ceiling=None)
    assert patches['diffusion_model.input_blocks.4.layer'][0][0] == 2.0
    assert patches['diffusion_model.input_blocks.4.layer'][1][0] == 1.0
    # IN01 has no preset slot of its own; it folds onto the mean of its level
    # (IN04 0.5, IN05 1.0) so conv-only LoRAs cannot bypass role weighting at
    # full strength.
    assert patches['diffusion_model.input_blocks.1.layer'][0][0] == pytest.approx(2.25)
    assert patches['diffusion_model.time_embed.layer'][0][0] == 3.0
    assert stats == {
        'scaled': 1,
        'folded': 1,
        'passthrough': 1,
        'unknown': 1,
        'total': 4,
        'unknown_keys': ['unsupported.layer'],
    }


@pytest.mark.parametrize('preset', ['CHAR_SDXL', 'CHAR_SDXL_STRONG', 'CHAR_SDXL_MAX', 'STYLE_SDXL', 'STYLE_SDXL_PURE'])
def test_folding_never_amplifies_a_block(preset):
    """Folding is meant to restrict blocks without a slot, never to boost them.

    The character presets exceed 1.0 on the early slots, so folding onto a
    single neighbour would have handed an unslotted block an above-full weight.
    """
    ratios = lbw_engine.SDXL_PRESETS[preset]
    folded_keys = [
        'diffusion_model.input_blocks.0.0.weight',
        'diffusion_model.input_blocks.1.0.in_layers.2.weight',
        'diffusion_model.input_blocks.3.0.op.weight',
        'diffusion_model.input_blocks.6.0.op.weight',
        'diffusion_model.output_blocks.6.0.in_layers.2.weight',
        'diffusion_model.output_blocks.8.0.in_layers.2.weight',
    ]
    for key in folded_keys:
        classification = lbw_engine.classify_key(key)
        assert classification.category == 'folded'
        ratio = lbw_engine.resolve_ratio(classification, ratios)
        assert 0.0 <= ratio <= lbw_engine.FOLD_CEILING, f'{key} on {preset} resolved to {ratio}'


def test_folded_output_tail_is_not_gutted_by_the_last_slot():
    """OUT06-08 average the level1 tail instead of inheriting OUT05 alone.

    CHAR_SDXL puts OUT05 at 0.15, so folding onto that single slot would have
    stripped the final surface-detail blocks down to near nothing.
    """
    ratios = lbw_engine.SDXL_PRESETS['CHAR_SDXL']
    classification = lbw_engine.classify_key('diffusion_model.output_blocks.7.0.in_layers.2.weight')
    ratio = lbw_engine.resolve_ratio(classification, ratios)
    out05 = ratios[lbw_engine.SDXL_BLOCK_ORDER.index('OUT05')]
    assert ratio > out05
    # Mean of OUT03 0.35, OUT04 0.25, OUT05 0.15.
    assert ratio == pytest.approx(0.25)


def test_directly_scaled_blocks_keep_weights_above_one():
    """The fold ceiling must not leak into blocks that own their slot."""
    ratios = lbw_engine.SDXL_PRESETS['CHAR_SDXL']
    classification = lbw_engine.classify_key(
        'diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight'
    )
    assert classification.category == 'scaled'
    assert lbw_engine.resolve_ratio(classification, ratios) == pytest.approx(1.15)


CHARACTER_PRESETS = ('CHAR_SDXL', 'CHAR_SDXL_STRONG', 'CHAR_SDXL_MAX')


@pytest.mark.parametrize('preset', CHARACTER_PRESETS)
def test_base_slot_is_held_below_full_strength(preset):
    """BASE reads the prompt semantically, so it must not be amplified.

    Text-encoder conditioning is computed once before sampling rather than per
    step. Amplifying it made the LoRA's dataset posing outweigh the wording of
    the prompt.
    """
    base = lbw_engine.SDXL_PRESETS[preset][lbw_engine.SDXL_BLOCK_ORDER.index('BASE')]
    assert base < 1.0, f'{preset} BASE is {base}'


@pytest.mark.parametrize(('preset', 'expected'), [
    ('CHAR_SDXL', 0.70),
    ('CHAR_SDXL_STRONG', 0.80),
    ('CHAR_SDXL_MAX', 0.90),
])
def test_base_late_is_unchanged(preset, expected):
    """Trigger-word binding lives in the late CLIP layers and stays as it was."""
    late = lbw_engine.SDXL_PRESETS[preset][lbw_engine.SDXL_BLOCK_ORDER.index('BASE_LATE')]
    assert late == pytest.approx(expected)


def test_character_presets_keep_their_relative_order():
    index = lbw_engine.SDXL_BLOCK_ORDER.index('BASE')
    values = [lbw_engine.SDXL_PRESETS[name][index] for name in CHARACTER_PRESETS]
    assert values == sorted(values), values


@pytest.mark.parametrize('preset', CHARACTER_PRESETS)
@pytest.mark.parametrize('strength', [0.5, 0.8, 0.9, 1.0, 1.2])
def test_no_effective_weight_exceeds_one_when_clamped(preset, strength):
    weights = lbw_engine.effective_weights(strength, strength, lbw_engine.SDXL_PRESETS[preset])
    peak = max(weights.values())
    assert peak <= lbw_engine.EFFECTIVE_CEILING, f'{preset} @{strength} peaked at {peak}'


def test_disabling_the_clamp_reproduces_the_old_behaviour():
    patches = {'diffusion_model.input_blocks.4.layer': [(0.9, 'delta')]}
    added = [('diffusion_model.input_blocks.4.layer', 0)]
    ratios = lbw_engine.SDXL_PRESETS['CHAR_SDXL']

    lbw_engine.scale_patches_inplace(patches, added, ratios, ceiling=None)
    assert patches['diffusion_model.input_blocks.4.layer'][0][0] == pytest.approx(1.035)

    patches = {'diffusion_model.input_blocks.4.layer': [(0.9, 'delta')]}
    lbw_engine.scale_patches_inplace(patches, added, ratios)
    assert patches['diffusion_model.input_blocks.4.layer'][0][0] == pytest.approx(1.0)


def test_clamp_preserves_sign_for_negative_strengths():
    assert lbw_engine.clamp_effective(-1.4) == pytest.approx(-1.0)
    assert lbw_engine.clamp_effective(-0.6) == pytest.approx(-0.6)
    assert lbw_engine.clamp_effective(1.4) == pytest.approx(1.0)
    assert lbw_engine.clamp_effective(1.4, ceiling=None) == pytest.approx(1.4)


def test_infer_role_from_common_directory_names():
    assert lbw_engine.infer_ratios_from_path('models/character_loras/person.safetensors') == lbw_engine.SDXL_PRESETS['CHAR_SDXL']
    assert lbw_engine.infer_ratios_from_path('models/my_styles/look.safetensors') == lbw_engine.SDXL_PRESETS['STYLE_SDXL']
    assert lbw_engine.infer_ratios_from_path('models/misc/style_name.safetensors') is None
