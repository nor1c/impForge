from modules import extra_networks, shared, errors
import networks


class ExtraNetworkLora(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lora')

        self.errors = {}
        """mapping of network names to the number of errors the network had during operation"""

    remove_symbols = str.maketrans('', '', ":,")

    def activate(self, p, params_list):
        additional = shared.opts.sd_lora

        self.errors.clear()

        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        def reject_activation(message):
            networks.load_networks([])
            raise ValueError(message)

        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        lbw_ratios = []
        step_windows = []
        for params in params_list:
            if not params.positional:
                reject_activation('A LoRA tag must include a LoRA name.')

            name = params.positional[0]
            names.append(name)

            try:
                te_multiplier = float(params.positional[1]) if len(params.positional) > 1 else 1.0
                te_multiplier = float(params.named.get('te', te_multiplier))
                unet_multiplier = float(params.positional[2]) if len(params.positional) > 2 else te_multiplier
                unet_multiplier = float(params.named.get('unet', unet_multiplier))
                dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
                dyn_dim = int(params.named['dyn']) if 'dyn' in params.named else dyn_dim
            except (TypeError, ValueError) as error:
                reject_activation(f'LoRA {name!r} has an invalid numeric multiplier: {error}')

            lbw_spec = params.named.get('lbw') or params.named.get('w')
            role_spec = params.named.get('role') or params.named.get('type')
            if lbw_spec is not None and networks.lbw_engine.parse_lbw_spec(lbw_spec) is None:
                reject_activation(f'LoRA {name!r} has an invalid lbw value {lbw_spec!r}. Use a preset or exactly 12 or 13 finite numbers.')
            if lbw_spec is None and role_spec is not None:
                if networks.lbw_engine.parse_role(role_spec) is None:
                    valid_roles = 'char, char_strong, char_max, style, style_pure'
                    reject_activation(f'LoRA {name!r} has an unknown role {role_spec!r}. Valid roles: {valid_roles}.')
                lbw_spec = role_spec

            try:
                step_window = networks.lbw_engine.parse_step_window(
                    params.named.get('start'),
                    params.named.get('stop'),
                    params.named.get('step'),
                )
            except ValueError as error:
                reject_activation(f'LoRA {name!r} has an invalid step window: {error}')

            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            dyn_dims.append(dyn_dim)
            lbw_ratios.append(lbw_spec)
            step_windows.append(step_window)

        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims, lbw_ratios=lbw_ratios, step_windows=step_windows)

        if getattr(networks, 'last_lora_summary', None):
            p.extra_generation_params['LoRA role split'] = '; '.join(networks.last_lora_summary)

        # The applied weight is strength x preset ratio, so it is rarely the
        # number written in the prompt. Record what actually ran.
        if getattr(networks, 'last_stack_peaks', ''):
            p.extra_generation_params['LoRA effective peak'] = networks.last_stack_peaks

        if shared.opts.lora_add_hashes_to_infotext:
            if not getattr(p, "is_hr_pass", False) or not hasattr(p, "lora_hashes"):
                p.lora_hashes = {}

            for item in networks.loaded_networks:
                if item.network_on_disk.shorthash and item.mentioned_name:
                    p.lora_hashes[item.mentioned_name.translate(self.remove_symbols)] = item.network_on_disk.shorthash

            if p.lora_hashes:
                p.extra_generation_params["Lora hashes"] = ', '.join(f'{k}: {v}' for k, v in p.lora_hashes.items())

    def deactivate(self, p):
        # A stop= window (or an interrupted start= run) can leave scheduled
        # patches muted. The patched UNet is cached for the next generation, so
        # put the baked weights back into their unscheduled state here.
        forge_objects = getattr(getattr(p, 'sd_model', None), 'forge_objects', None)
        unet = getattr(forge_objects, 'unet', None)
        if unet is not None:
            try:
                networks.lbw_schedule.restore(unet)
            except Exception:
                errors.report('Failed to restore scheduled LoRA weights', exc_info=True)

        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))

            self.errors.clear()
