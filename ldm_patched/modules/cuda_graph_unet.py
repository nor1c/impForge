"""
Native CUDA Graph capture for the SDXL UNet forward pass.

Why this exists
---------------
Per-step Python/launch overhead on a UNet forward is ~3-8 ms on Blackwell.
Over a 25-step generation that's ~100-200 ms of pure dispatch cost (not
actual GPU compute). Capturing the UNet as a CUDA graph on the first step
and replaying it for subsequent steps drops this to near zero.

How it works
------------
1. On the first UNet call with a given (batch, channels, height, width,
   dtype, context_shape) key, we allocate persistent *static* input buffers
   on GPU and run the forward eagerly into a static output buffer. This
   also acts as a warmup pass so cudnn benchmarking settles.
2. A second eager pass (still not captured) is run in a side stream. This
   is required — CUDA Graphs reject capture on the first execution because
   the caching allocator may not have settled.
3. A third pass is run inside torch.cuda.graph(), recording every kernel
   into a graph object.
4. On all subsequent calls with the same key, we copy the live inputs into
   the static buffers, graph.replay(), then copy the static output to a
   fresh tensor for the caller.

Why this is safe-ish
--------------------
- Capture key includes tensor shapes + dtypes + whether controlnet / patches
  are active. If ANY of those change, we re-capture (not free, but correct).
- Any exception during capture → disable graph for this key permanently,
  fall back to eager.
- Skipped when control is not None (ControlNet), when input_block_patch is
  set (HRFix / regional), or when transformer_patches is non-empty — these
  all do dynamic things that can't be captured deterministically.
- Only activated when --unet-cuda-graph is passed.
"""
from __future__ import annotations
import copy
import logging
import torch

logger = logging.getLogger(__name__)


class UNetCudaGraphWrapper:
    """Wraps a UNet diffusion_model to capture and replay CUDA graphs per
    unique input signature."""

    def __init__(self, diffusion_model):
        self._inner = diffusion_model
        # Cache keyed by (shape of x, shape of context, dtype of x, has-y,
        # num_video_frames). Each value is a CapturedGraph or None if we
        # permanently bailed out for that key.
        self._graphs: dict[tuple, "CapturedGraph | None"] = {}
        self._disabled = False
        self._cache_context = None

    def set_cache_context(self, context):
        """Invalidate captured graphs when external model state changes.

        LoRA patches are applied outside the wrapped UNet module. A graph
        captured for one LoRA set must not be replayed for another.
        """
        if self._cache_context != context:
            self._graphs.clear()
            self._cache_context = context
        return self

    def clear_cache(self):
        self._graphs.clear()
        return self

    def __getattr__(self, name):
        # Delegate anything not on the wrapper to the inner model. Needed
        # because samplers sometimes reach into .dtype, .model_channels, etc.
        return getattr(self._inner, name)

    def __call__(self, x, timesteps=None, context=None, y=None, control=None,
                 transformer_options=None, **kwargs):
        transformer_options = transformer_options or {}
        # Dynamic-control detection — bail out when anything could change
        # within the forward that our graph capture can't handle.
        if self._disabled or control is not None or x.device.type != "cuda":
            return self._inner(x, timesteps=timesteps, context=context, y=y,
                                control=control,
                                transformer_options=transformer_options, **kwargs)

        if self._has_dynamic_transformer_options(transformer_options):
            return self._inner(x, timesteps=timesteps, context=context, y=y,
                                control=control,
                                transformer_options=transformer_options, **kwargs)

        if kwargs:
            # Any extra-conds we don't explicitly handle → fall back.
            return self._inner(x, timesteps=timesteps, context=context, y=y,
                                control=control,
                                transformer_options=transformer_options, **kwargs)

        key = (
            self._cache_context,
            tuple(x.shape), x.dtype, x.device.index,
            self._tensor_key(timesteps),
            self._tensor_key(context),
            self._tensor_key(y),
            tuple(transformer_options.get("cond_or_uncond", ())),
        )

        entry = self._graphs.get(key, False)
        if entry is None:
            # Previously marked permanently-bailed.
            return self._inner(x, timesteps=timesteps, context=context, y=y,
                                control=control,
                                transformer_options=transformer_options, **kwargs)

        if entry is False:
            # First-time key: try to capture.
            try:
                captured = self._capture(x, timesteps, context, y, transformer_options)
                self._graphs[key] = captured
                entry = captured
            except Exception as e:
                logger.warning("[UNet CUDA graph] capture failed for key %s: %s. Falling back to eager.", key, e)
                self._graphs[key] = None
                return self._inner(x, timesteps=timesteps, context=context, y=y,
                                    control=control,
                                    transformer_options=transformer_options, **kwargs)

        # Replay existing graph.
        try:
            return entry.replay(x, timesteps, context, y)
        except Exception as e:
            logger.warning("[UNet CUDA graph] replay failed for key %s: %s. Disabling.", key, e)
            self._graphs[key] = None
            return self._inner(x, timesteps=timesteps, context=context, y=y,
                                control=control,
                                transformer_options=transformer_options, **kwargs)

    def _capture(self, x, timesteps, context, y, transformer_options):
        """Run 2 warmup passes + 1 capture pass."""
        device = x.device
        transformer_options = copy.copy(transformer_options)
        # Allocate static input buffers matching live shapes/dtypes.
        static_x = torch.empty_like(x)
        static_t = torch.empty_like(timesteps) if timesteps is not None else None
        static_ctx = torch.empty_like(context) if context is not None else None
        static_y = torch.empty_like(y) if y is not None else None

        static_x.copy_(x)
        if static_t is not None: static_t.copy_(timesteps)
        if static_ctx is not None: static_ctx.copy_(context)
        if static_y is not None: static_y.copy_(y)

        # Warmup passes: required by CUDA Graph capture to ensure the caching
        # allocator and cudnn benchmarks have settled.
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                warm_out = self._inner(
                    static_x, timesteps=static_t, context=static_ctx, y=static_y,
                    control=None, transformer_options=transformer_options,
                )
        torch.cuda.current_stream(device).wait_stream(side_stream)
        torch.cuda.synchronize(device)

        # Capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out = self._inner(
                static_x, timesteps=static_t, context=static_ctx, y=static_y,
                control=None, transformer_options=transformer_options,
            )

        logger.info("[UNet CUDA graph] captured for shape %s dtype %s", tuple(x.shape), x.dtype)
        return CapturedGraph(graph, static_x, static_t, static_ctx, static_y, captured_out)

    @staticmethod
    def _tensor_key(tensor):
        if tensor is None:
            return None
        return (tuple(tensor.shape), tensor.dtype, tensor.device.type, tensor.device.index)

    @staticmethod
    def _has_dynamic_transformer_options(transformer_options):
        if not transformer_options:
            return False

        # Only allow the stable metadata used by plain CFG batching. Anything
        # patch-like can alter Python control flow or module state per step.
        allowed_keys = {"cond_or_uncond"}
        for key, value in transformer_options.items():
            if key in allowed_keys:
                continue
            if key in {"patches", "patches_replace"}:
                if value:
                    return True
                continue
            return True

        return False


class CapturedGraph:
    __slots__ = ("graph", "static_x", "static_t", "static_ctx", "static_y", "static_out")

    def __init__(self, graph, static_x, static_t, static_ctx, static_y, static_out):
        self.graph = graph
        self.static_x = static_x
        self.static_t = static_t
        self.static_ctx = static_ctx
        self.static_y = static_y
        self.static_out = static_out

    def replay(self, x, timesteps, context, y):
        # Copy live inputs into the static buffers the graph reads from.
        self.static_x.copy_(x, non_blocking=True)
        if self.static_t is not None and timesteps is not None:
            self.static_t.copy_(timesteps, non_blocking=True)
        if self.static_ctx is not None and context is not None:
            self.static_ctx.copy_(context, non_blocking=True)
        if self.static_y is not None and y is not None:
            self.static_y.copy_(y, non_blocking=True)

        self.graph.replay()
        # Clone the output so downstream code can safely mutate / free it
        # without corrupting the next replay's output buffer.
        return self.static_out.clone()


def wrap_if_enabled(diffusion_model):
    """If --unet-cuda-graph is set, wrap the diffusion model for graph capture.
    Otherwise return the original model unchanged."""
    try:
        from ldm_patched.modules.args_parser import args
        if not getattr(args, "unet_cuda_graph", False):
            return diffusion_model
    except Exception:
        return diffusion_model

    if not torch.cuda.is_available():
        return diffusion_model

    logger.info("[UNet CUDA graph] wrapping diffusion model for CUDA graph capture")
    return UNetCudaGraphWrapper(diffusion_model)
