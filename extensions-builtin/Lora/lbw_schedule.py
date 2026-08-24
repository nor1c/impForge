"""Sampling-step scheduling for native SDXL LoRA patches.

Composition (framing, pose, limb layout) is settled during the first few
high-sigma steps, while identity, face and fine detail are built afterwards.
A character LoRA that is active from step 0 competes with the prompt over the
pose, and the usual workarounds -- lowering strength or pushing the deep-block
weights down -- weaken identity by as much as they free up the composition,
because in SDXL the deep blocks (IN07/IN08/M00) carry identity and layout
together.

Delaying the LoRA in time rather than attenuating it in weight avoids that
trade-off: the composition phase runs without the LoRA, then the LoRA applies
at full strength with the remaining steps to establish identity.

Two backend details shape this implementation:

* LoRA patches are baked into the model weights by
  ``ModelPatcher.patch_weight_to_device`` instead of being applied per forward
  pass, so toggling a LoRA mid-sampling means restoring the affected weights
  from the patcher backup and re-baking them. Only keys whose patch strengths
  actually change are touched.
* ``ModelPatcher.clone()`` returns a fresh instance, so per-patcher attributes
  do not survive the clones taken by Kohya HRFix and APG during sampling.
  Clones do share the underlying ``model`` object and the ``backup`` dict, so
  the schedule is stored on the model. Clones shallow-copy the per-key patch
  lists though, so every patcher a schedule has been seen on is tracked and
  mutated together.

Scheduling applies to UNet patches only. Text-encoder conditioning is computed
once before sampling starts, so a step window cannot change it; the text
encoder therefore keeps its full requested strength.

Step windows are declined under lowvram, where weights are patched per forward
pass from a captured patch list rather than baked once.
"""

from __future__ import annotations

import logging
import time

import torch

import lbw_engine

logger = logging.getLogger(__name__)

_ATTR = '_lbw_step_schedule'

# Re-baking spends a full weight recomputation per affected key. Report it once
# per transition when it is slow enough to look like a stall.
_SLOW_REBAKE_SECONDS = 0.75


class ScheduledPatches:
    """Patch entries that are only active during part of the sampling run."""

    def __init__(self):
        # key -> list of (entry_index, window, original_strength)
        self.entries = {}
        self.labels = []
        # Patchers whose patch lists hold these entries. ModelPatcher.clone()
        # shallow-copies the per-key lists, so a clone taken during sampling
        # keeps its own list objects. Mutating only the patcher we collected
        # from would leave a clone's LowVramPatch reading stale strengths.
        self.patchers = []

    def add(self, key, index, window, strength):
        self.entries.setdefault(key, []).append((index, window, strength))

    def track(self, patcher):
        if patcher is not None and not any(existing is patcher for existing in self.patchers):
            self.patchers.append(patcher)

    def __bool__(self):
        return bool(self.entries)


def _patches_of(patcher):
    patches = getattr(patcher, 'patches', None)
    if patches is None and hasattr(patcher, 'patcher'):
        patches = getattr(patcher.patcher, 'patches', None)
    return patches


def _is_lowvram(patcher):
    """Whether the model is loaded with per-forward weight patching.

    Under lowvram, ``LowVramPatch`` reads ``patches`` during the forward pass
    instead of the weights being baked once. It captures the patch list of the
    patcher that was loaded, which is not necessarily the one a schedule is
    collected on, so muting entries is not reliably observed.
    """
    model = getattr(patcher, 'model', None)
    return bool(getattr(model, 'lowvram_patch_counter', 0))


def _schedule_host(patcher):
    """The object shared across clones that stores the schedule."""
    return getattr(patcher, 'model', None)


def get_schedule(patcher):
    host = _schedule_host(patcher)
    if host is None:
        return None
    return getattr(host, _ATTR, None)


def clear(patcher):
    host = _schedule_host(patcher)
    if host is not None and hasattr(host, _ATTR):
        delattr(host, _ATTR)


def collect(patcher, before, window, label=None):
    """Record the UNet patch entries added for a step-limited LoRA."""
    if window is None or patcher is None:
        return None
    patches = _patches_of(patcher)
    host = _schedule_host(patcher)
    if patches is None or host is None:
        return None

    if _is_lowvram(patcher):
        logger.warning(
            '[LBW] %s requested a step window but the model is loaded in lowvram mode, '
            'where weights are patched per forward pass and the window cannot be applied. '
            'The LoRA runs at full strength for every step.',
            label or '<unknown>',
        )
        return None

    schedule = getattr(host, _ATTR, None)
    if schedule is None:
        schedule = ScheduledPatches()
    count = 0
    for key, index in lbw_engine.diff_added_keys(patches, before or {}):
        schedule.add(key, index, window, patches[key][index][0])
        count += 1
    if not count:
        return getattr(host, _ATTR, None)
    if label:
        schedule.labels.append(label)
    schedule.track(patcher)
    setattr(host, _ATTR, schedule)
    logger.info(
        '[LBW] %s scheduled for steps %s across %d patch key(s)',
        label or '<unknown>',
        lbw_engine.format_window(window) or 'all',
        count,
    )
    return schedule


def _patch_lists(schedule, patcher):
    """Every patch dict the scheduled entries live in, without duplicates."""
    schedule.track(patcher)
    seen = []
    for tracked in schedule.patchers:
        patches = _patches_of(tracked)
        if patches is not None and not any(patches is existing for existing in seen):
            seen.append(patches)
    return seen


def apply_step(patcher, step, total_steps=None):
    """Match scheduled patch strengths to the given sampling step.

    Returns the set of patch keys whose strengths changed.
    """
    schedule = get_schedule(patcher)
    if not schedule:
        return set()
    patch_lists = _patch_lists(schedule, patcher)
    if not patch_lists:
        return set()

    changed = set()
    for key, records in schedule.entries.items():
        for patches in patch_lists:
            entries = patches.get(key)
            if entries is None:
                continue
            for index, window, strength in records:
                if index >= len(entries):
                    continue
                active = lbw_engine.window_is_active(window, step, total_steps)
                wanted = strength if active else 0.0
                entry = entries[index]
                if entry[0] == wanted:
                    continue
                entries[index] = (wanted,) + tuple(entry[1:])
                changed.add(key)
    return changed


def rebake(patcher, keys):
    """Re-bake weights for keys whose patch strengths changed."""
    if not keys or patcher is None:
        return
    backup = getattr(patcher, 'backup', None)
    if backup is None:
        return

    import ldm_patched.modules.utils as utils

    device = getattr(getattr(patcher, 'model', None), 'device', None)
    started = time.perf_counter()
    baked = 0
    with torch.no_grad():
        for key in keys:
            entry = backup.get(key)
            if entry is None:
                # Never baked, so there is nothing to restore before repatching.
                continue
            if entry.inplace_update:
                utils.copy_to_param(patcher.model, key, entry.weight)
            else:
                utils.set_attr_param(patcher.model, key, entry.weight)
            del backup[key]
            patcher.patch_weight_to_device(key, device_to=device)
            baked += 1

    elapsed = time.perf_counter() - started
    if baked and elapsed >= _SLOW_REBAKE_SECONDS:
        logger.info(
            '[LBW] re-baked %d weight(s) for a step window transition in %.1fs; '
            'this pause happens once per window boundary.',
            baked, elapsed,
        )


def update(patcher, step, total_steps=None):
    """Apply the schedule for one sampling step, re-baking what changed."""
    changed = apply_step(patcher, step, total_steps)
    if changed:
        rebake(patcher, changed)
    return changed


def restore(patcher):
    """Return every scheduled entry to its unscheduled strength.

    Sampling can finish with entries muted (any ``stop=`` window, or a
    ``start=`` window on a run that was interrupted early). The patched UNet is
    cached and reused for the next generation, so the baked weights are put back
    into their unscheduled state here instead of relying on the next run's first
    step to notice.
    """
    schedule = get_schedule(patcher)
    if not schedule:
        return set()
    patch_lists = _patch_lists(schedule, patcher)
    if not patch_lists:
        return set()

    changed = set()
    for key, records in schedule.entries.items():
        for patches in patch_lists:
            entries = patches.get(key)
            if entries is None:
                continue
            for index, _window, strength in records:
                if index >= len(entries):
                    continue
                entry = entries[index]
                if entry[0] != strength:
                    entries[index] = (strength,) + tuple(entry[1:])
                    changed.add(key)
    rebake(patcher, changed)
    return changed
