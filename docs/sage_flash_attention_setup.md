# SageAttention and FlashAttention Setup

This repo is configured to use SageAttention and FlashAttention for faster SDXL generation on NVIDIA CUDA GPUs. The current target machine is an RTX 5070 Ti using CUDA 12.8 Torch wheels.

## What This Patch Does

The WebUI launch flag is `--use-sage-attention`.

At runtime, `ldm_patched/ldm/modules/attention.py` uses selective routing:

- SageAttention is used for large supported self-attention calls.
- FlashAttention is used as an internal fallback for large unsupported self-attention head dimensions.
- PyTorch SDPA is kept for cross-attention, small attention calls, masked attention, CPU tensors, unsupported dtypes, and error fallback.

This is intentional. Benchmarks on the RTX 5070 Ti showed Sage/Flash are faster for large self-attention, but PyTorch SDPA is faster for short cross-attention against text tokens.

## Current Launch Flags

`webui-user.bat` should include:

```bat
set COMMANDLINE_ARGS=^
  --skip-version-check ^
  --skip-python-version-check ^
  --skip-torch-cuda-test ^
  --pin-shared-memory ^
  --cuda-malloc ^
  --cuda-stream ^
  --always-gpu ^
  --force-channels-last ^
  --fast fp16_accumulation ^
  --vae-in-bf16 ^
  --no-hashing ^
  --use-sage-attention ^
  --port 44444 ^
  --ckpt-dir "F:\AI\MODELS\CKPT" ^
  --lora-dir "F:\AI\MODELS\Lora" ^
  --embeddings-dir "D:\AI\MODELS\embeddings" ^
  --vae-dir "D:\AI\MODELS\VAE"
```

Do not add these unless retesting carefully:

- `--use-flash-attention`: not needed; FlashAttention is loaded internally as Sage fallback.
- `--unet-cuda-graph`: previously broke LoRA/model loading.
- `--always-high-vram` together with `--always-gpu`: these are mutually exclusive.
- `--fast fp8_matrix_mult cublas_ops`: avoid until quality/speed is retested.

## Current Config Settings

In `config.json`, keep these values:

```json
"token_merging_ratio": 0.2,
"fp8_storage": "Disable",
"cache_fp16_weight": false
```

`fp8_storage` and `cache_fp16_weight` are disabled because they can add overhead or conflict with the attention optimization path.

## Install on a New Computer

These steps assume Windows, this WebUI repo, and Python venv at `venv\Scripts\python.exe`. Adjust paths if your checkout is elsewhere.

### 1. Create or Enter the WebUI Venv

Start WebUI once, or otherwise create the venv normally for this repo. Then use only this Python and pip:

```bat
F:\impForge\venv\Scripts\python.exe
F:\impForge\venv\Scripts\pip.exe
```

Do not use system `python` for these checks. System Python may be CPU-only.

### 2. Install CUDA Torch

For the RTX 5070 Ti setup, use Torch `2.9.0+cu128`:

```bat
"F:\impForge\venv\Scripts\pip.exe" install "torch==2.9.0+cu128" "torchvision==0.24.0+cu128" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

Verify:

```bat
"F:\impForge\venv\Scripts\python.exe" -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected output should include:

```text
2.9.0+cu128
12.8
True
NVIDIA GeForce RTX 5070 Ti
```

### 3. Install Triton for Windows

```bat
"F:\impForge\venv\Scripts\pip.exe" install triton-windows==3.5.1
```

### 4. Install SageAttention Wheel

Use a wheel matching CUDA 12.8 and Torch 2.9.0 or higher. The known working wheel name was:

```text
sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
```

Install from the folder where you downloaded the wheel:

```bat
"F:\impForge\venv\Scripts\pip.exe" install "C:\path\to\sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
```

### 5. Install FlashAttention Wheel

Use a wheel matching CUDA 12.8, Torch 2.9.0, Python 3.11, and Windows x64. The known working wheel name was:

```text
flash_attn-2.8.3+cu128torch2.9.0cxx11abiFALSE-cp311-cp311-win_amd64.whl
```

Important: install FlashAttention with `--no-deps`. Without `--no-deps`, pip may replace CUDA Torch with a CPU-only or incompatible Torch build.

```bat
"F:\impForge\venv\Scripts\pip.exe" install "C:\path\to\flash_attn-2.8.3+cu128torch2.9.0cxx11abiFALSE-cp311-cp311-win_amd64.whl" --no-deps
```

### 6. Verify Imports

```bat
"F:\impForge\venv\Scripts\python.exe" -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); import sageattention; print('sageattention ok'); import flash_attn; print('flash_attn:', getattr(flash_attn, '__version__', 'unknown'))"
```

Expected:

```text
torch: 2.9.0+cu128
cuda: 12.8
cuda available: True
sageattention ok
flash_attn: 2.8.3
```

### 7. If Torch Gets Broken

If `torch.cuda.is_available()` becomes `False`, reinstall CUDA Torch:

```bat
"F:\impForge\venv\Scripts\pip.exe" install "torch==2.9.0+cu128" "torchvision==0.24.0+cu128" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

Then reinstall FlashAttention with `--no-deps`.

## Runtime Verification

Start WebUI with `webui-user.bat`. Startup should show:

```text
Found SageAttention 1.x/2.x (sageattention package)
Found FlashAttention (flash_attn package) - will be used as SageAttention fallback when needed
Using sage attention 1.x/2.x (with flash attention fallback for unsupported head_dims)
```

After a txt2img or img2img generation, the console should print something like:

```text
[Attention backends] sage=... | flash=... | sdpa=... | total=...
[Attention fallbacks] ...
```

Healthy output:

- `sage` should be greater than zero for large SDXL self-attention.
- `flash` may be greater than zero for unsupported large self-attention head dimensions.
- `sdpa` is expected and normal for cross-attention and small attention calls.
- `sage_error` and `flash_error` should be zero or absent.

If `sage_error` or `flash_error` appears, copy the full console output and debug `ldm_patched/ldm/modules/attention.py` routing first.

## Local Synthetic Test

This quick test does not start WebUI. It checks that Sage, Flash fallback, and SDPA fallback all route correctly:

```bat
"F:\impForge\venv\Scripts\python.exe" -c "exec('''import sys, torch\nimport ldm_patched.modules.options as options\noptions.enable_args_parsing()\nsys.argv=[\"check\",\"--use-sage-attention\"]\nimport ldm_patched.ldm.modules.attention as attn\nattn.reset_attention_counters()\nq=torch.randn(1,2048,128,device=\"cuda\",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); attn.attention_sage(q,k,v,2)\nq=torch.randn(1,1024,160,device=\"cuda\",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); attn.attention_sage(q,k,v,1)\nq=torch.randn(1,1024,128,device=\"cuda\",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); attn.attention_sage(q,k,v,2)\ntorch.cuda.synchronize()\nattn.print_attention_counters()\n''')"
```

Expected summary:

```text
[Attention backends] sage=1 | flash=1 | sdpa=1 | total=3
```

## Notes

- The goal is lower real generation time, not maximum SageAttention call count.
- Cross-attention often stays on SDPA because text-token attention is short and SDPA benchmarks faster there.
- Wheel compatibility is strict. Match CUDA, Torch, Python, and platform tags.
- For this repo, focus on SDXL/IllustriousXL. Do not spend time scanning checkpoint or LoRA model files when maintaining this setup.
