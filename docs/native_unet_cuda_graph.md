# Native UNet CUDA Graph

This repo includes a lightweight native CUDA graph wrapper for the SDXL UNet. It is enabled with:

```bat
--unet-cuda-graph
```

## Purpose

The wrapper reduces repeated UNet kernel launch overhead during generation. It does not use `torch.compile`, Inductor, Triton, or extra packages.

## Current Recommended Flags

Keep this with the existing RTX 5070 Ti launch setup:

```bat
--cuda-malloc ^
--cuda-stream ^
--always-gpu ^
--force-channels-last ^
--fast fp16_accumulation ^
--vae-in-bf16 ^
--use-sage-attention ^
--unet-cuda-graph
```

## Safety Behavior

The graph wrapper is conservative:

- It captures only fixed-shape CUDA UNet calls.
- It clears captured graphs when checkpoint or LoRA signature changes.
- It falls back to normal eager UNet when ControlNet, transformer patches, extra kwargs, or graph capture/replay errors are detected.
- It does not change sampler math, CFG, steps, resolution, or image quality settings.

## Testing Notes

The first compatible generation can be slower because graph capture performs warmup work. Compare the second and third generations using the same checkpoint, LoRA set, batch size, and resolution.

Do not combine this with `--torch-compile`; that path was unstable on the current Windows/CUDA/Torch setup.
