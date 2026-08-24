import importlib.util
import sys
import types
from pathlib import Path

import pytest

LORA_DIR = Path(__file__).parents[1] / 'extensions-builtin' / 'Lora'


def _load(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, LORA_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


lbw_engine = _load('lbw_engine', 'lbw_engine.py')
lbw_schedule = _load('lbw_schedule', 'lbw_schedule.py')


class FakeModel:
    def __init__(self):
        self.device = None
        self.lowvram_patch_counter = 0


class FakePatcher:
    """Minimal ModelPatcher stand-in covering the parts the schedule touches."""

    def __init__(self, model=None):
        self.patches = {}
        self.backup = {}
        self.model = model or FakeModel()
        self.rebaked = []

    def clone(self):
        """Mirrors ModelPatcher.clone: shared model and backup, copied lists."""
        other = FakePatcher(self.model)
        other.patches = {key: values[:] for key, values in self.patches.items()}
        other.backup = self.backup
        return other

    def patch_weight_to_device(self, key, device_to=None):
        self.rebaked.append(key)


@pytest.mark.parametrize(('start', 'stop', 'step', 'expected'), [
    (None, None, None, None),
    ('0', None, None, None),
    ('4', None, None, (4, None, False)),
    (None, '6', None, (None, 6, False)),
    ('2', '9', None, (2, 9, False)),
    (None, None, '3-7', (3, 7, False)),
    ('0.15', None, None, (0.15, None, True)),
    (None, '0.6', None, (None, 0.6, True)),
    ('0.1', '0.7', None, (0.1, 0.7, True)),
    (None, None, '0.15-0.8', (0.15, 0.8, True)),
    # A zero start still counts as a window when a stop bounds it.
    ('0', '5', None, (0, 5, False)),
])
def test_parse_step_window(start, stop, step, expected):
    assert lbw_engine.parse_step_window(start, stop, step) == expected


@pytest.mark.parametrize(('start', 'stop', 'step'), [
    ('-1', None, None),
    ('abc', None, None),
    ('5', '5', None),
    ('7', '3', None),
    (None, None, '4'),
    (None, None, ''),
    # Fractions must stay within the run.
    ('1.5', None, None),
    ('-0.2', None, None),
    ('0.9', '0.3', None),
    # Mixing an absolute bound with a fractional one is ambiguous.
    ('4', '0.8', None),
    ('0.2', '12', None),
])
def test_parse_step_window_rejects_bad_input(start, stop, step):
    with pytest.raises(ValueError):
        lbw_engine.parse_step_window(start, stop, step)


@pytest.mark.parametrize(('window', 'step', 'active'), [
    (None, 0, True),
    ((4, None, False), 3, False),
    ((4, None, False), 4, True),
    ((4, None, False), 30, True),
    ((None, 6, False), 5, True),
    ((None, 6, False), 6, False),
    ((2, 5, False), 1, False),
    ((2, 5, False), 2, True),
    ((2, 5, False), 5, False),
])
def test_window_is_active(window, step, active):
    assert lbw_engine.window_is_active(window, step) is active


@pytest.mark.parametrize(('total_steps', 'boundary'), [
    (28, 4),   # base pass
    (14, 2),   # hires fix resamples at fewer steps
    (40, 6),
])
def test_fractional_window_scales_to_each_pass(total_steps, boundary):
    """One fraction has to mean the same proportion whatever the step count."""
    window = lbw_engine.parse_step_window('0.15', None, None)
    assert lbw_engine.window_is_active(window, boundary - 1, total_steps) is False
    assert lbw_engine.window_is_active(window, boundary, total_steps) is True


def test_fractional_window_fails_open_without_a_step_count():
    """A missing step count must never silently mute a requested LoRA."""
    window = lbw_engine.parse_step_window('0.5', None, None)
    assert lbw_engine.window_is_active(window, 0, None) is True
    assert lbw_engine.window_is_active(window, 0, 0) is True


def test_absolute_window_ignores_the_step_count():
    window = lbw_engine.parse_step_window('4', None, None)
    assert lbw_engine.window_is_active(window, 3, 28) is False
    assert lbw_engine.window_is_active(window, 4, 14) is True


@pytest.mark.parametrize(('window', 'fractional'), [
    (None, False),
    ((4, None, False), False),
    ((0.15, None, True), True),
])
def test_window_is_fractional(window, fractional):
    assert lbw_engine.window_is_fractional(window) is fractional


def test_schedule_zeroes_before_start_and_restores_after():
    patcher = FakePatcher()
    before = lbw_schedule.lbw_engine.snapshot_patches(patcher.patches)
    patcher.patches['unet.layer'] = [(0.9, 'delta')]

    lbw_schedule.collect(patcher, before, (4, None, False), label='char')
    assert lbw_schedule.get_schedule(patcher) is not None

    # Before the start step the patch is muted.
    changed = lbw_schedule.apply_step(patcher, 0)
    assert changed == {'unet.layer'}
    assert patcher.patches['unet.layer'][0][0] == 0.0

    # Repeating the same step is a no-op, so no redundant weight re-bake.
    assert lbw_schedule.apply_step(patcher, 1) == set()

    # From the start step the original strength is restored in full.
    changed = lbw_schedule.apply_step(patcher, 4)
    assert changed == {'unet.layer'}
    assert patcher.patches['unet.layer'][0][0] == 0.9


def test_schedule_ignores_unscheduled_patches():
    patcher = FakePatcher()
    patcher.patches['unet.layer'] = [(0.9, 'delta')]
    assert lbw_schedule.get_schedule(patcher) is None
    assert lbw_schedule.apply_step(patcher, 0) == set()
    assert patcher.patches['unet.layer'][0][0] == 0.9


def test_stop_window_mutes_late_steps():
    patcher = FakePatcher()
    before = {}
    patcher.patches['unet.layer'] = [(1.0, 'delta')]
    lbw_schedule.collect(patcher, before, (None, 3, False))

    lbw_schedule.apply_step(patcher, 2)
    assert patcher.patches['unet.layer'][0][0] == 1.0
    lbw_schedule.apply_step(patcher, 3)
    assert patcher.patches['unet.layer'][0][0] == 0.0


def test_clear_removes_schedule():
    patcher = FakePatcher()
    patcher.patches['unet.layer'] = [(1.0, 'delta')]
    lbw_schedule.collect(patcher, {}, (2, None, False))
    lbw_schedule.clear(patcher)
    assert lbw_schedule.get_schedule(patcher) is None


def test_fractional_schedule_follows_the_current_step_count():
    """The same schedule has to hold at 28 steps and again at 14."""
    patcher = FakePatcher()
    patcher.patches['unet.layer'] = [(0.9, 'delta')]
    lbw_schedule.collect(patcher, {}, (0.25, None, True), label='char')

    lbw_schedule.apply_step(patcher, 6, total_steps=28)
    assert patcher.patches['unet.layer'][0][0] == 0.0
    lbw_schedule.apply_step(patcher, 7, total_steps=28)
    assert patcher.patches['unet.layer'][0][0] == 0.9

    lbw_schedule.apply_step(patcher, 2, total_steps=14)
    assert patcher.patches['unet.layer'][0][0] == 0.0
    lbw_schedule.apply_step(patcher, 4, total_steps=14)
    assert patcher.patches['unet.layer'][0][0] == 0.9


def test_schedule_is_declined_under_lowvram():
    """LowVramPatch reads a captured patch list, so muting is not observable."""
    patcher = FakePatcher()
    patcher.model.lowvram_patch_counter = 12
    patcher.patches['unet.layer'] = [(0.9, 'delta')]

    assert lbw_schedule.collect(patcher, {}, (4, None, False), label='char') is None
    assert lbw_schedule.get_schedule(patcher) is None
    # The LoRA keeps its full strength rather than being silently muted.
    lbw_schedule.apply_step(patcher, 0)
    assert patcher.patches['unet.layer'][0][0] == 0.9


def test_schedule_reaches_patch_lists_of_clones():
    """ModelPatcher.clone() shallow-copies each key's list of patch entries."""
    patcher = FakePatcher()
    patcher.patches['unet.layer'] = [(0.9, 'delta')]
    lbw_schedule.collect(patcher, {}, (4, None, False), label='char')

    clone = patcher.clone()
    assert clone.patches['unet.layer'] is not patcher.patches['unet.layer']

    lbw_schedule.apply_step(clone, 0)
    assert clone.patches['unet.layer'][0][0] == 0.0
    # The originally collected patcher must be muted too, so whichever list the
    # sampler ends up reading sees the same strength.
    assert patcher.patches['unet.layer'][0][0] == 0.0

    lbw_schedule.apply_step(clone, 4)
    assert clone.patches['unet.layer'][0][0] == 0.9
    assert patcher.patches['unet.layer'][0][0] == 0.9


def test_rebake_restores_backup_before_repatching():
    patcher = FakePatcher()
    backup = types.SimpleNamespace(weight='original', inplace_update=False)
    patcher.backup['unet.layer'] = backup
    patcher.patches['unet.layer'] = [(0.0, 'delta')]

    restored = {}

    class FakeUtils:
        @staticmethod
        def set_attr_param(model, key, weight):
            restored[key] = weight

        @staticmethod
        def copy_to_param(model, key, weight):
            restored[key] = weight

    fake_modules = types.ModuleType('ldm_patched.modules')
    fake_modules.utils = FakeUtils
    fake_root = types.ModuleType('ldm_patched')
    fake_root.modules = fake_modules
    sys.modules['ldm_patched'] = fake_root
    sys.modules['ldm_patched.modules'] = fake_modules
    sys.modules['ldm_patched.modules.utils'] = FakeUtils
    try:
        lbw_schedule.rebake(patcher, {'unet.layer'})
    finally:
        for name in ('ldm_patched.modules.utils', 'ldm_patched.modules', 'ldm_patched'):
            sys.modules.pop(name, None)

    assert restored == {'unet.layer': 'original'}
    # Backup is dropped so patch_weight_to_device takes a fresh one.
    assert 'unet.layer' not in patcher.backup
    assert patcher.rebaked == ['unet.layer']


def test_rebake_skips_keys_without_baked_weights():
    patcher = FakePatcher()
    patcher.patches['unet.layer'] = [(0.0, 'delta')]
    lbw_schedule.rebake(patcher, {'unet.layer'})
    assert patcher.rebaked == []


def test_restore_unmutes_entries_left_off_at_end_of_sampling():
    patcher = FakePatcher()
    patcher.patches['unet.layer'] = [(1.0, 'delta')]
    lbw_schedule.collect(patcher, {}, (None, 3, False))

    # Sampling ends past the stop step, leaving the entry muted.
    lbw_schedule.apply_step(patcher, 5)
    assert patcher.patches['unet.layer'][0][0] == 0.0

    changed = lbw_schedule.restore(patcher)
    assert changed == {'unet.layer'}
    assert patcher.patches['unet.layer'][0][0] == 1.0
    # Already restored, so a second call is a no-op.
    assert lbw_schedule.restore(patcher) == set()


def test_restore_is_safe_without_a_schedule():
    patcher = FakePatcher()
    patcher.patches['unet.layer'] = [(1.0, 'delta')]
    assert lbw_schedule.restore(patcher) == set()
    assert patcher.patches['unet.layer'][0][0] == 1.0


def test_stack_load_flags_overloaded_blocks():
    entries = [
        ('style_a', 0.35, 0.35, lbw_engine.SDXL_PRESETS['STYLE_SDXL_PURE']),
        ('style_b', 0.45, 0.45, lbw_engine.SDXL_PRESETS['STYLE_SDXL_PURE']),
        ('style_c', 0.4, 0.4, lbw_engine.SDXL_PRESETS['STYLE_SDXL_PURE']),
        ('char_a', 0.5, 0.5, lbw_engine.SDXL_PRESETS['CHAR_SDXL_MAX']),
        ('char_b', 0.9, 0.9, lbw_engine.SDXL_PRESETS['CHAR_SDXL']),
    ]
    totals, overloaded = lbw_engine.summarize_stack_load(entries)
    overloaded_blocks = {block for block, _total in overloaded}
    # Five stacked LoRAs pile onto the deep blocks even with each one capped.
    assert 'M00' in overloaded_blocks
    assert totals['M00'] > lbw_engine.STACK_WARN_THRESHOLD


def test_stack_load_is_quiet_for_a_reasonable_stack():
    """One character plus one style LoRA is a normal, working combination."""
    entries = [
        ('style', 0.7, 0.7, lbw_engine.SDXL_PRESETS['STYLE_SDXL']),
        ('char', 0.9, 0.9, lbw_engine.SDXL_PRESETS['CHAR_SDXL']),
    ]
    _totals, overloaded = lbw_engine.summarize_stack_load(entries)
    assert overloaded == []


def test_stack_load_is_quiet_for_a_single_lora():
    """Clamping already holds one LoRA at 1.0, so nothing should be flagged."""
    for preset in ('CHAR_SDXL', 'CHAR_SDXL_STRONG', 'CHAR_SDXL_MAX'):
        for strength in (0.8, 0.9, 1.0):
            entries = [('char', strength, strength, lbw_engine.SDXL_PRESETS[preset])]
            totals, overloaded = lbw_engine.summarize_stack_load(entries)
            assert overloaded == [], f'{preset} @{strength} flagged {overloaded}'
            assert max(totals.values()) <= lbw_engine.EFFECTIVE_CEILING


def test_effective_weights_expose_what_actually_runs():
    """The number written in the prompt is not the number applied."""
    ratios = lbw_engine.SDXL_PRESETS['CHAR_SDXL']
    raw = lbw_engine.effective_weights(0.9, 0.9, ratios, ceiling=None)
    capped = lbw_engine.effective_weights(0.9, 0.9, ratios)
    # IN04 is 1.15 in the preset, so 0.9 would otherwise apply as 1.035.
    assert raw['IN04'] == pytest.approx(1.035)
    assert capped['IN04'] == pytest.approx(1.0)
    # BASE is now held below 1.0, so it is unaffected by the cap.
    assert capped['BASE'] == pytest.approx(raw['BASE'])
