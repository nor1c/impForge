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
    ('diffusion_model.input_blocks.0.0.weight', None, 'passthrough'),
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
        'diffusion_model.input_blocks.0.layer': [(3.0, 'pass')],
        'unsupported.layer': [(4.0, 'unknown')],
    }
    added = [
        ('diffusion_model.input_blocks.4.layer', 1),
        ('diffusion_model.input_blocks.0.layer', 0),
        ('unsupported.layer', 0),
    ]
    ratios = list(lbw_engine.SDXL_PRESETS['ALL'])
    ratios[lbw_engine.SDXL_BLOCK_ORDER.index('IN04')] = 0.5
    stats = lbw_engine.scale_patches_inplace(patches, added, tuple(ratios))
    assert patches['diffusion_model.input_blocks.4.layer'][0][0] == 2.0
    assert patches['diffusion_model.input_blocks.4.layer'][1][0] == 1.0
    assert patches['diffusion_model.input_blocks.0.layer'][0][0] == 3.0
    assert stats == {
        'scaled': 1,
        'passthrough': 1,
        'unknown': 1,
        'total': 3,
        'unknown_keys': ['unsupported.layer'],
    }


def test_infer_role_from_common_directory_names():
    assert lbw_engine.infer_ratios_from_path('models/character_loras/person.safetensors') == lbw_engine.SDXL_PRESETS['CHAR_SDXL']
    assert lbw_engine.infer_ratios_from_path('models/my_styles/look.safetensors') == lbw_engine.SDXL_PRESETS['STYLE_SDXL']
    assert lbw_engine.infer_ratios_from_path('models/misc/style_name.safetensors') is None
