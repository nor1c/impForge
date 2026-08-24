from __future__ import annotations
import gradio as gr
import logging
import os
import re

import lora_patches
import functools
import network
import lbw_engine

import torch
from typing import Union

from modules import shared, sd_models, errors, scripts
from ldm_patched.modules.utils import load_torch_file
from ldm_patched.modules.sd import load_lora_for_models

last_lora_summary = []
last_stack_peaks = ''
"""Peak effective weight per LoRA, kept so it can be reported again.

load_networks() is skipped entirely when a generation reuses the previous
LoRA configuration, so anything logged only from inside it disappears from the
second generation onwards. Holding the summary here lets the caller surface it
either way.
"""


def _record_stack_peaks(entries, ceiling):
    global last_stack_peaks
    parts = []
    for name, unet_strength, te_strength, ratios in entries:
        weights = lbw_engine.effective_weights(unet_strength, te_strength, ratios, ceiling)
        block = max(weights, key=weights.get)
        parts.append(f'{name}:{weights[block]:.2f}@{block}')
    last_stack_peaks = ', '.join(parts)
    return last_stack_peaks


@functools.lru_cache(maxsize=5)
def load_lora_state_dict(filename, mtime_ns, file_size):
    return load_torch_file(filename, safe_load=True)


def convert_diffusers_name_to_compvis(key, is_sd2):
    pass


def assign_network_names_to_compvis_modules(sd_model):
    pass


class BundledTIHash(str):
    def __init__(self, hash_str):
        self.hash = hash_str

    def __str__(self):
        return self.hash if shared.opts.lora_bundled_ti_to_infotext else ''


def load_network(name, network_on_disk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    return net


def purge_networks_from_memory():
    pass


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None, lbw_ratios=None):
    global last_lora_summary, last_stack_peaks

    current_sd = sd_models.model_data.get_sd_model()
    if current_sd is None:
        return

    te_multipliers = te_multipliers or [1.0] * len(names)
    unet_multipliers = unet_multipliers or list(te_multipliers)
    dyn_dims = dyn_dims or [None] * len(names)
    lbw_ratios = lbw_ratios or [None] * len(names)
    arrays = (te_multipliers, unet_multipliers, dyn_dims, lbw_ratios)
    if any(len(values) != len(names) for values in arrays):
        raise ValueError('LoRA names, multipliers, dynamic dimensions, and block weights must have matching lengths.')

    unavailable_networks = []
    for name in names:
        if name.lower() in forbidden_network_aliases and available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    def resolve_networks():
        return [
            available_networks.get(name) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name)
            for name in names
        ]

    networks_on_disk = resolve_networks()
    if any(item is None for item in networks_on_disk):
        list_available_networks()
        networks_on_disk = resolve_networks()
    missing = [name for name, item in zip(names, networks_on_disk) if item is None]
    if missing:
        raise ValueError(f"LoRA files were not found: {', '.join(missing)}")

    resolved = []
    for network_on_disk, raw in zip(networks_on_disk, lbw_ratios):
        source = 'flat'
        ratios = None
        if raw is not None:
            if isinstance(raw, (tuple, list)):
                ratios = lbw_engine.parse_lbw_spec(','.join(str(value) for value in raw))
            else:
                ratios = lbw_engine.parse_lbw_spec(raw)
            if ratios is None:
                raise ValueError(f'Invalid block-weight specification for {network_on_disk.name}.')
            source = 'explicit'
        else:
            ratios = lbw_engine.infer_ratios_from_path(network_on_disk.filename)
            if ratios is not None:
                source = 'path'
        resolved.append((ratios, source))

    targets = []
    for item, name, unet, te, (ratios, source) in zip(
        networks_on_disk, names, unet_multipliers, te_multipliers, resolved
    ):
        file_stat = os.stat(item.filename)
        targets.append((item.filename, file_stat.st_mtime_ns, file_stat.st_size, unet, te, ratios, source, name))
    target_hash = str([
        (filename, mtime_ns, file_size, unet, te, ratios, source)
        for filename, mtime_ns, file_size, unet, te, ratios, source, name in targets
    ])
    if not targets:
        loaded_networks.clear()
        last_lora_summary = []
        last_stack_peaks = ''
        current_sd.forge_objects.unet = current_sd.forge_objects_original.unet
        current_sd.forge_objects.clip = current_sd.forge_objects_original.clip
        current_sd.forge_objects_after_applying_lora = current_sd.forge_objects_original.shallow_copy()
        current_sd.current_lora_hash = target_hash
        return
    if current_sd.current_lora_hash == target_hash:
        return

    pending_networks = []
    for network_on_disk, name in zip(networks_on_disk, names):
        net = load_network(name, network_on_disk)
        net.mentioned_name = name
        network_on_disk.read_hash()
        pending_networks.append(net)

    working_unet = current_sd.forge_objects_original.unet
    working_clip = current_sd.forge_objects_original.clip
    ceiling = lbw_engine.EFFECTIVE_CEILING if getattr(shared.opts, 'lora_clamp_effective_weight', True) else None
    summaries = []
    try:
        for filename, mtime_ns, file_size, strength_model, strength_clip, ratios, source, name in targets:
            lora_sd = load_lora_state_dict(filename, mtime_ns, file_size)
            before_unet = lbw_engine.snapshot_patches(working_unet.patches) if working_unet is not None else {}
            before_clip = {}
            if working_clip is not None:
                clip_patches = getattr(working_clip, 'patches', None)
                if clip_patches is None and hasattr(working_clip, 'patcher'):
                    clip_patches = getattr(working_clip.patcher, 'patches', None)
                if clip_patches is not None:
                    before_clip = lbw_engine.snapshot_patches(clip_patches)

            working_unet, working_clip = load_lora_for_models(
                working_unet, working_clip, lora_sd, strength_model, strength_clip, filename=filename
            )
            stats = lbw_engine.apply_block_weights(
                working_unet, working_clip, before_unet, before_clip, ratios,
                lora_name=filename, ceiling=ceiling,
            ) if ratios is not None else lbw_engine.new_stats()
            preset = next((key for key, value in lbw_engine.SDXL_PRESETS.items() if value == ratios), 'custom' if ratios else 'flat')
            summaries.append(
                f'{name}={preset}/{source},TE:{strength_clip:g},UNet:{strength_model:g},'
                f"scaled:{stats['scaled']},folded:{stats['folded']},passthrough:{stats['passthrough']},unknown:{stats['unknown']}"
            )
    except Exception:
        loaded_networks.clear()
        last_lora_summary = []
        last_stack_peaks = ''
        current_sd.current_lora_hash = str([])
        current_sd.forge_objects.unet = current_sd.forge_objects_original.unet
        current_sd.forge_objects.clip = current_sd.forge_objects_original.clip
        current_sd.forge_objects_after_applying_lora = current_sd.forge_objects_original.shallow_copy()
        raise

    loaded_networks.clear()
    loaded_networks.extend(pending_networks)
    lbw_engine.log_stack_load([
        (name, strength_model, strength_clip, ratios)
        for _f, _m, _s, strength_model, strength_clip, ratios, _src, name in targets
    ], ceiling=ceiling)
    _record_stack_peaks([
        (name, strength_model, strength_clip, ratios)
        for _f, _m, _s, strength_model, strength_clip, ratios, _src, name in targets
    ], ceiling)
    current_sd.forge_objects.unet = working_unet
    current_sd.forge_objects.clip = working_clip
    current_sd.forge_objects_after_applying_lora = current_sd.forge_objects.shallow_copy()
    current_sd.current_lora_hash = target_hash
    last_lora_summary = summaries

def allowed_layer_without_weight(layer):
    if isinstance(layer, torch.nn.LayerNorm) and not layer.elementwise_affine:
        return True

    return False


def store_weights_backup(weight):
    if weight is None:
        return None

    return weight.to(devices.cpu, copy=True)


def restore_weights_backup(obj, field, weight):
    if weight is None:
        setattr(obj, field, None)
        return

    getattr(obj, field).copy_(weight)


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    pass


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    pass


def network_forward(org_module, input, original_forward):
    pass


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    pass


def network_Linear_forward(self, input):
    pass


def network_Linear_load_state_dict(self, *args, **kwargs):
    pass


def network_Conv2d_forward(self, input):
    pass


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    pass


def network_GroupNorm_forward(self, input):
    pass


def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    pass


def network_LayerNorm_forward(self, input):
    pass


def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    pass


def network_MultiheadAttention_forward(self, *args, **kwargs):
    pass


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    pass


def process_network_files(names: list[str] | None = None):
    candidates = list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    for filename in candidates:
        if os.path.isdir(filename):
            continue
        name = os.path.splitext(os.path.basename(filename))[0]
        # if names is provided, only load networks with names in the list
        if names and name not in names:
            continue
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


def update_available_networks_by_names(names: list[str]):
    process_network_files(names)


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    process_network_files()


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


originals: lora_patches.LoraPatches = None

extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
