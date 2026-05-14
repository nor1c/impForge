import copy
import torch

from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.sampler_helpers import convert_cond
from ldm_patched.modules.samplers import encode_model_conds
from ldm_patched.modules.args_parser import args


class UnetPatcher(ModelPatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlnet_linked_list = None
        self.extra_preserved_memory_during_sampling = 0
        self.extra_model_patchers_during_sampling = []
        self.extra_concat_condition = None
        self.compiled = False
        self.compile_signature = None
        self._compile_original_model = None
        self._compile_reuse_logged = set()
        self._compile_failed_signatures = set()

    def clone(self):
        n = UnetPatcher(self.model, self.load_device, self.offload_device, self.size)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.controlnet_linked_list = self.controlnet_linked_list
        n.extra_preserved_memory_during_sampling = self.extra_preserved_memory_during_sampling
        n.extra_model_patchers_during_sampling = self.extra_model_patchers_during_sampling.copy()
        n.extra_concat_condition = self.extra_concat_condition
        n.patches_uuid = self.patches_uuid
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.compiled = self.compiled
        n.compile_signature = self.compile_signature
        n._compile_original_model = self._compile_original_model
        n._compile_reuse_logged = self._compile_reuse_logged.copy()
        n._compile_failed_signatures = self._compile_failed_signatures.copy()
        n.parent = self
        return n

    def add_extra_preserved_memory_during_sampling(self, memory_in_bytes: int):
        # Use this to ask Forge to preserve a certain amount of memory during sampling.
        # If GPU VRAM is 8 GB, and memory_in_bytes is 2GB, i.e., memory_in_bytes = 2 * 1024 * 1024 * 1024
        # Then the sampling will always use less than 6GB memory by dynamically offload modules to CPU RAM.
        # You can estimate this using model_management.module_size(any_pytorch_model) to get size of any pytorch models.
        self.extra_preserved_memory_during_sampling += memory_in_bytes
        return

    def add_extra_model_patcher_during_sampling(self, model_patcher: ModelPatcher):
        # Use this to ask Forge to move extra model patchers to GPU during sampling.
        # This method will manage GPU memory perfectly.
        self.extra_model_patchers_during_sampling.append(model_patcher)
        return

    def add_extra_torch_module_during_sampling(self, m: torch.nn.Module, cast_to_unet_dtype: bool = True):
        # Use this method to bind an extra torch.nn.Module to this UNet during sampling.
        # This model `m` will be delegated to Forge memory management system.
        # `m` will be loaded to GPU everytime when sampling starts.
        # `m` will be unloaded if necessary.
        # `m` will influence Forge's judgement about use GPU memory or
        # capacity and decide whether to use module offload to make user's batch size larger.
        # Use cast_to_unet_dtype if you want `m` to have same dtype with unet during sampling.

        if cast_to_unet_dtype:
            m.to(self.model.diffusion_model.dtype)

        patcher = ModelPatcher(model=m, load_device=self.load_device, offload_device=self.offload_device)

        self.add_extra_model_patcher_during_sampling(patcher)
        return patcher
    
    def _compile_settings(self, backend):
        force_disable_cudagraphs = getattr(args, "cuda_malloc", False)
        has_custom_options = any([
            args.torch_compile_epilogue_fusion,
            args.torch_compile_max_autotune,
            args.torch_compile_fallback_random,
            args.torch_compile_shape_padding,
            args.torch_compile_cudagraphs,
            args.torch_compile_trace,
            args.torch_compile_graph_diagram
        ])

        if backend == "cudagraphs":
            if force_disable_cudagraphs:
                print("torch.compile backend cudagraphs is incompatible with --cuda-malloc; using inductor without cudagraphs instead.")
                backend = "inductor"
            else:
                return {
                    "backend": backend,
                    "fullgraph": True,
                    "dynamic": False,
                }

        if force_disable_cudagraphs and args.torch_compile_mode == "reduce-overhead":
            print("torch.compile reduce-overhead uses CUDA graphs, which are incompatible with --cuda-malloc; disabling Inductor cudagraphs for this compile.")

        if force_disable_cudagraphs:
            return {
                "backend": backend,
                "fullgraph": False,
                "dynamic": False,
                "options": {
                    "triton.cudagraphs": False,
                    "triton.cudagraph_trees": False,
                },
            }

        compile_settings = {
            "backend": backend,
            "fullgraph": False,
            "dynamic": False,
        }

        if has_custom_options:
            options = {}
            if args.torch_compile_epilogue_fusion:
                options["epilogue_fusion"] = True
            if args.torch_compile_max_autotune:
                options["max_autotune"] = True
            if args.torch_compile_fallback_random:
                options["fallback_random"] = True
            if args.torch_compile_shape_padding:
                options["shape_padding"] = True
            if args.torch_compile_cudagraphs:
                options["triton.cudagraphs"] = True
            if args.torch_compile_trace:
                options["trace.enabled"] = True
            if args.torch_compile_graph_diagram:
                options["trace.graph_diagram"] = True
            compile_settings["options"] = options
        else:
            compile_settings["mode"] = args.torch_compile_mode

        return compile_settings

    def _get_diffusion_model(self):
        if hasattr(self.model, 'diffusion_model'):
            return self.model.diffusion_model
        return self.model

    def _set_diffusion_model(self, model):
        if hasattr(self.model, 'diffusion_model'):
            self.model.diffusion_model = model
        else:
            self.model = model

    def compile_model(self, backend="inductor", signature=None, quiet_reuse=False):
        """Compile the UNet after model/LoRA patches are selected."""
        if not hasattr(torch, 'compile'):
            print("torch.compile not available - requires PyTorch 2.0 or newer")
            return False

        if self.compiled and self.compile_signature == signature:
            reuse_key = repr(signature)
            if not quiet_reuse and reuse_key not in self._compile_reuse_logged:
                print(f"Reusing compiled UNet for signature: {signature}")
                self._compile_reuse_logged.add(reuse_key)
            return True

        failure_key = repr(signature)
        if failure_key in self._compile_failed_signatures:
            return False
        
        try:
            torch_version = torch.__version__.split('.')
            if int(torch_version[0]) < 2:
                print(f"torch.compile requires PyTorch 2.0 or newer. Current version: {torch.__version__}")
                return False

            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
            dynamo.config.verbose = True
            dynamo.config.cache_size_limit = 32

            real_model = self._compile_original_model or self._get_diffusion_model()
            real_model = getattr(real_model, "_orig_mod", real_model)
            self._compile_original_model = real_model
            compile_settings = self._compile_settings(backend)

            print(f"Compiling UNet using torch.compile with settings: {compile_settings}")
            if signature is not None:
                print(f"Compile signature: {signature}")

            # Store settings for later recompilation if needed
            real_model.compile_settings = compile_settings
            
            try:
                compiled_model = torch.compile(real_model, **compile_settings)
                self._set_diffusion_model(compiled_model)
                self.compiled = True
                self.compile_signature = signature
                print("UNet compilation successful")
                return True
            except Exception as e:
                print(f"Warning: torch.compile failed with error: {str(e)}")
                print("Falling back to uncompiled model")
                self._set_diffusion_model(real_model)
                self.compiled = False
                self.compile_signature = None
                self._compile_failed_signatures.add(failure_key)
                return False
        except Exception as e:
            print(f"Error during model compilation: {str(e)}")
            self._compile_failed_signatures.add(repr(signature))
            return False

    def compile_model_if_needed(self, signature=None):
        if not args.torch_compile:
            return False
        return self.compile_model(backend=args.torch_compile_backend, signature=signature)

    def restore_uncompiled_model(self):
        if self._compile_original_model is None:
            return False
        self._set_diffusion_model(self._compile_original_model)
        self.compiled = False
        self.compile_signature = None
        return True

    def mark_compile_signature_failed(self, signature=None):
        self._compile_failed_signatures.add(repr(signature))
        return self.restore_uncompiled_model()

    def set_cuda_graph_context(self, context):
        diffusion_model = self._get_diffusion_model()
        if hasattr(diffusion_model, "set_cache_context"):
            diffusion_model.set_cache_context(context)
            return True
        return False

    def add_patched_controlnet(self, cnet):
        cnet.set_previous_controlnet(self.controlnet_linked_list)
        self.controlnet_linked_list = cnet
        return

    def list_controlnets(self):
        results = []
        pointer = self.controlnet_linked_list
        while pointer is not None:
            results.append(pointer)
            pointer = pointer.previous_controlnet
        return results

    def append_model_option(self, k, v, ensure_uniqueness=False):
        if k not in self.model_options:
            self.model_options[k] = []

        if ensure_uniqueness and v in self.model_options[k]:
            return

        self.model_options[k].append(v)
        return

    def append_transformer_option(self, k, v, ensure_uniqueness=False):
        if 'transformer_options' not in self.model_options:
            self.model_options['transformer_options'] = {}

        to = self.model_options['transformer_options']

        if k not in to:
            to[k] = []

        if ensure_uniqueness and v in to[k]:
            return

        to[k].append(v)
        return

    def set_transformer_option(self, k, v):
        if 'transformer_options' not in self.model_options:
            self.model_options['transformer_options'] = {}

        self.model_options['transformer_options'][k] = v
        return

    def add_conditioning_modifier(self, modifier, ensure_uniqueness=False):
        self.append_model_option('conditioning_modifiers', modifier, ensure_uniqueness)
        return

    def add_sampler_pre_cfg_function(self, modifier, ensure_uniqueness=False):
        self.append_model_option('sampler_pre_cfg_function', modifier, ensure_uniqueness)
        return

    def set_memory_peak_estimation_modifier(self, modifier):
        self.model_options['memory_peak_estimation_modifier'] = modifier
        return

    def add_alphas_cumprod_modifier(self, modifier, ensure_uniqueness=False):
        """

        For some reasons, this function only works in A1111's Script.process_batch(self, p, *args, **kwargs)

        For example, below is a worked modification:

        class ExampleScript(scripts.Script):

            def process_batch(self, p, *args, **kwargs):
                unet = p.sd_model.forge_objects.unet.clone()

                def modifier(x):
                    return x ** 0.5

                unet.add_alphas_cumprod_modifier(modifier)
                p.sd_model.forge_objects.unet = unet

                return

        This add_alphas_cumprod_modifier is the only patch option that should be used in process_batch()
        All other patch options should be called in process_before_every_sampling()

        """

        self.append_model_option('alphas_cumprod_modifiers', modifier, ensure_uniqueness)
        return

    def add_block_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option('block_modifiers', modifier, ensure_uniqueness)
        return

    def add_block_inner_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option('block_inner_modifiers', modifier, ensure_uniqueness)
        return

    def add_controlnet_conditioning_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option('controlnet_conditioning_modifiers', modifier, ensure_uniqueness)
        return

    def set_controlnet_model_function_wrapper(self, wrapper):
        self.set_transformer_option('controlnet_model_function_wrapper', wrapper)
        return

    def set_model_replace_all(self, patch, target="attn1"):
        for block_name in ['input', 'middle', 'output']:
            for number in range(16):
                for transformer_index in range(16):
                    self.set_model_patch_replace(patch, target, block_name, number, transformer_index)
        return

    def encode_conds_after_clip(self, conds, noise, prompt_type="positive"):
        return encode_model_conds(
            model_function=self.model.extra_conds,
            conds=convert_cond(conds),
            noise=noise,
            device=noise.device,
            prompt_type=prompt_type
        )

    def load_frozen_patcher(self, state_dict, strength):
        patch_dict = {}
        for k, w in state_dict.items():
            model_key, patch_type, weight_index = k.split('::')
            if model_key not in patch_dict:
                patch_dict[model_key] = {}
            if patch_type not in patch_dict[model_key]:
                patch_dict[model_key][patch_type] = [None] * 16
            patch_dict[model_key][patch_type][int(weight_index)] = w

        patch_flat = {}
        for model_key, v in patch_dict.items():
            for patch_type, weight_list in v.items():
                patch_flat[model_key] = (patch_type, weight_list)

        self.add_patches(patches=patch_flat, strength_patch=float(strength), strength_model=1.0)
        return


def copy_and_update_model_options(model_options, patch, name, block_name, number, transformer_index=None):
    model_options = model_options.copy()
    transformer_options = model_options.get("transformer_options", {}).copy()
    patches_replace = transformer_options.get("patches_replace", {}).copy()
    name_patches = patches_replace.get(name, {}).copy()
    block = (block_name, number, transformer_index) if transformer_index is not None else (block_name, number)
    name_patches[block] = patch
    patches_replace[name] = name_patches
    transformer_options["patches_replace"] = patches_replace
    model_options["transformer_options"] = transformer_options
    return model_options
