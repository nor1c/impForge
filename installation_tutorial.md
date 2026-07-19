# impForge Installation Tutorial

A step-by-step guide to set up impForge from scratch on Windows. This tool is an optimized fork of reForge / Stable Diffusion WebUI Forge, built specifically for SDXL models (especially IllustriousXL) on RTX 5070 Ti with SageAttention and FlashAttention acceleration.

---

## 1. System Requirements

| Component  | Requirement                          |
| ---------- | ------------------------------------ |
| OS         | Windows 10 / 11 (64-bit)             |
| GPU        | NVIDIA RTX 5070 Ti (or any CUDA 12.8 compatible GPU) |
| VRAM       | 12 GB minimum, 16 GB recommended for SDXL |
| RAM        | 16 GB minimum, 32 GB recommended     |
| Storage    | 50 GB free (models excluded)         |
| Python     | 3.10.6 – 3.13.x (3.11 recommended)   |
| Git        | Latest stable                        |
| CUDA       | 12.8                                 |

Tools you'll need before starting:

- [Python 3.10+](https://www.python.org/downloads/) — tick *Add Python to PATH* during install
- [Git for Windows](https://git-scm.com/download/win)
- [NVIDIA GPU Driver](https://www.nvidia.com/download/) — version 572.xx or newer for CUDA 12.8 support
- [7-Zip](https://www.7-zip.org/) or similar archive tool (for model files)

---

## 2. Clone the Repository

Open **Command Prompt** or **PowerShell** and run:

```bat
git clone https://github.com/YOUR_ORG/impForge.git
cd impForge
```

If you don't have the repo URL, obtain it from the maintainer.

> **Important:** Do NOT place the repo under a path containing a leading dot (`.`) in any parent folder name. Examples:
> - **OK:** `F:\impForge`
> - **NOT OK:** `C:\Users\noric\.hidden\impForge`

---

## 3. First Launch (Auto-Setup)

impForge includes an auto-bootstrapping system. The first time you run it, it will:

- Create a Python virtual environment (`venv\`)
- Install PyTorch 2.9.0+cu128 and torchvision
- Clone required sub-repositories (generative-models, k-diffusion, BLIP, etc.)
- Install all Python dependencies from `requirements_versions.txt`
- Install extension dependencies

### Step-by-step first run

```bat
webui-user.bat
```

What happens during first run:

1. **Python version check** — warns if version is outside 3.7–3.13 range. Use `--skip-python-version-check` to bypass.
2. **Virtual environment creation** — `venv\` directory is created in the repo root.
3. **CUDA Torch installation** — `torch==2.9.0 torchvision==0.24.0+cu128` from PyTorch CUDA 12.8 wheels.
4. **Core package installation** — CLIP, OpenCLIP, xformers, and all pip dependencies.
5. **Repository cloning** — into `repositories\`:
   - `stable-diffusion-stability-ai`
   - `generative-models` (SDXL)
   - `k-diffusion`
   - `BLIP`
6. **Extension installers** — runs `install.py` for each enabled extension.

After the first run completes, the WebUI starts. Press `Ctrl+C` to stop it before proceeding with Sage/Flash attention setup.

### Verify core install

```bat
venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:

```
2.9.0+cu128
True
NVIDIA GeForce RTX 5070 Ti
```

If `torch.cuda.is_available()` returns `False`, something went wrong with the CUDA Torch install. See [Troubleshooting](#troubleshooting).

---

## 4. Install SageAttention and FlashAttention (Performance Acceleration)

These two libraries speed up SDXL generation by ~0.7-0.8 it/s on RTX 5070 Ti through GPU-optimized attention kernels.

### 4.1 Install Triton for Windows

SageAttention depends on Triton.

```bat
venv\Scripts\pip.exe install triton-windows==3.5.1
```

### 4.2 Install SageAttention

Download the wheel matching **CUDA 12.8 + Torch 2.9.0+** from the SageAttention releases page or obtain it from the repository maintainer.

Known working wheel: `sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl`

```bat
venv\Scripts\pip.exe install "C:\path\to\sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
```

### 4.3 Install FlashAttention

FlashAttention serves as an internal fallback when SageAttention encounters unsupported attention head dimensions.

Known working wheel: `flash_attn-2.8.3+cu128torch2.9.0cxx11abiFALSE-cp311-cp311-win_amd64.whl`

> **Critical:** Use `--no-deps`. Without it, pip may replace CUDA Torch with a CPU-only build.

```bat
venv\Scripts\pip.exe install "C:\path\to\flash_attn-2.8.3+cu128torch2.9.0cxx11abiFALSE-cp311-cp311-win_amd64.whl" --no-deps
```

### 4.4 Verify attention libraries

```bat
venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); import sageattention; print('sageattention ok'); import flash_attn; print('flash_attn ok')"
```

Expected output:

```
torch: 2.9.0+cu128
cuda: 12.8
cuda available: True
sageattention ok
flash_attn ok
```

### 4.5 Run synthetic attention test

This quick test validates that Sage, Flash, and SDPA attention paths all route correctly — no WebUI needed.

```bat
venv\Scripts\python.exe -c "exec('''import sys, torch
import ldm_patched.modules.options as options
options.enable_args_parsing()
sys.argv=[\"check\",\"--use-sage-attention\"]
import ldm_patched.ldm.modules.attention as attn
attn.reset_attention_counters()
q=torch.randn(1,2048,128,device=\"cuda\",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); attn.attention_sage(q,k,v,2)
q=torch.randn(1,1024,160,device=\"cuda\",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); attn.attention_sage(q,k,v,1)
q=torch.randn(1,1024,128,device=\"cuda\",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); attn.attention_sage(q,k,v,2)
torch.cuda.synchronize()
attn.print_attention_counters()
''')"
```

Expected output:

```
[Attention backends] sage=1 | flash=1 | sdpa=1 | total=3
```

---

## 5. Configure webui-user.bat

Edit `webui-user.bat` in the repo root. A **minimal** starting configuration:

```bat
@echo off

set PYTHON=
set GIT=
set VENV_DIR=

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
  --lora-dir "F:\AI\MODELS\Lora"

call webui.bat
```

### Flag reference

| Flag                          | Purpose                                                        |
| ----------------------------- | -------------------------------------------------------------- |
| `--skip-version-check`        | Skip checking for upstream updates                             |
| `--skip-python-version-check` | Allow Python versions outside 3.7–3.13 range                   |
| `--skip-torch-cuda-test`      | Skip GPU availability check at startup                         |
| `--pin-shared-memory`         | Pin shared memory for faster CPU→GPU transfers                 |
| `--cuda-malloc`               | Use CUDA memory allocator (replaces PyTorch default)           |
| `--cuda-stream`               | Use CUDA stream for async operations                           |
| `--always-gpu`                | Keep model weights on GPU at all times                         |
| `--force-channels-last`       | Memory layout optimization for faster convolutions             |
| `--fast fp16_accumulation`    | Faster FP16 accumulation in attention layers                   |
| `--vae-in-bf16`               | Run VAE in BF16 precision (lower VRAM, faster)                 |
| `--no-hashing`                | Skip SHA256 hashing of model files on first load               |
| `--use-sage-attention`        | Enable SageAttention (with FlashAttn fallback) for SDXL        |
| `--port 44444`                | WebUI listens on this port                                     |
| `--ckpt-dir`                  | Directory containing checkpoint `.safetensors` files           |
| `--lora-dir`                  | Directory containing LoRA `.safetensors` files                 |
| `--embeddings-dir`            | Directory containing textual inversion embeddings              |
| `--vae-dir`                   | Directory containing VAE models                                |

### Do NOT add these flags (tested as problematic on RTX 5070 Ti):

- `--use-flash-attention` — FlashAttn is loaded internally as Sage fallback; separate flag not needed
- `--unet-cuda-graph` — previously broke LoRA/model loading
- `--always-high-vram` together with `--always-gpu` — mutually exclusive
- `--fast fp8_matrix_mult cublas_ops` — avoid until quality/speed is retested

---

## 6. Download and Place Models

impForge is built for **SDXL** and **IllustriousXL** models.

### 6.1 Checkpoint Model (Required)

Place an SDXL/IllustriousXL `.safetensors` checkpoint file in your `--ckpt-dir` directory.

Example: `F:\AI\MODELS\CKPT\waiNsfwIllustrious16.BHnI.safetensors`

### 6.2 VAE (Optional but recommended)

Place a VAE file in your `--vae-dir` directory or alongside the checkpoint with the same base filename and `vae` in the name.

### 6.3 LoRA Files (Optional)

Place `.safetensors` LoRA files in your `--lora-dir` directory.

### 6.4 Embeddings (Optional)

Place textual inversion `.pt` or `.safetensors` files in your `--embeddings-dir` directory.

---

## 7. Launch and Verify

```bat
webui-user.bat
```

### Startup check

During startup, the console should print:

```
Found SageAttention 1.x/2.x (sageattention package)
Found FlashAttention (flash_attn package) - will be used as SageAttention fallback when needed
Using sage attention 1.x/2.x (with flash attention fallback for unsupported head_dims)
```

Open your browser at `http://localhost:44444` (or whichever port you configured).

### Generation test

1. Go to the **txt2img** tab
2. Enter a prompt (e.g., `1girl, blue sky, masterpiece, best quality`)
3. Enter a negative prompt (`lowres, bad anatomy, worst quality`)
4. Set **Width** to `1024` and **Height** to `1024` (SDXL native resolution)
5. Set **Sampling steps** to `20`
6. Click **Generate**

After generation, the console should print attention routing statistics:

```
[Attention backends] sage=... | flash=... | sdpa=... | total=...
[Attention fallbacks] ...
```

Healthy output:
- `sage` > 0 — SageAttn used for large self-attention (expected)
- `flash` ≥ 0 — FlashAttn used for unsupported head dims (normal)
- `sdpa` > 0 — PyTorch SDPA used for cross-attention and small calls (normal, by design)
- `sage_error` and `flash_error` should be **zero**

If `sage` stays at zero, verify the `--use-sage-attention` flag is present in your `webui-user.bat` and re-run the synthetic test from section 4.5.

---

## 8. Recommended Config Settings

In the WebUI **Settings** tab (or directly in `config.json`), verify:

```json
"token_merging_ratio": 0.2,
"fp8_storage": "Disable",
"cache_fp16_weight": false
```

- `token_merging_ratio` at 0.2 gives a safe speed boost without visible quality loss
- `fp8_storage` and `cache_fp16_weight` are kept disabled — they can add overhead or conflict with the Sage/Flash attention path

---

## 9. Using LoRA Role Splitting (impForge Feature)

impForge provides native SDXL LoRA role splitting. In your prompt, use `<lora:filename:strength:strength:role=NAME>`:

```
<lora:art_style:0.9:0.9:role=style>
<lora:character:0.95:0.95:role=char>
```

Available roles:

| Role          | Purpose                                                     |
| ------------- | ----------------------------------------------------------- |
| `style`       | Preserves style, reduces character/composition leakage      |
| `style_pure`  | Stricter style isolation                                     |
| `char`        | Default character profile (identity, hair, eyes, clothing)   |
| `char_strong` | Stronger character identity when details wash out            |
| `char_max`    | Maximum identity — use cautiously, may cause artifacts       |

Practical combo: `role=style` at 0.8 + `role=char` at 1.0. Bump to `char_strong` only if character details remain weak.

See `docs/sdxl_lora_role_split.md` for full documentation.

---

## 10. Batch Prompt Features

impForge includes these batch prompt-list extensions:

### Inline Negative Prompt
Per-line negative tags via `(n:...)` markers:

```
from below, low angle, (n:from below)
```
→ The `from below` tag moves to the negative prompt for this line only.

### Inline Resolution
Per-line width/height via `(r:width, height)` markers:

```
wide shot, city, (r:1024, 1380)
```
→ This line renders at 1024×1380. The marker is removed before generation.

---

## Troubleshooting

### `torch.cuda.is_available()` returns `False`

Reinstall CUDA Torch:

```bat
venv\Scripts\pip.exe install "torch==2.9.0+cu128" "torchvision==0.24.0+cu128" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

Then reinstall FlashAttention with `--no-deps`.

### `no kernel image is available for execution on the device`

This usually means a CUDA version mismatch in SageAttention or FlashAttention wheels. Verify your wheel files match:
- **CUDA:** 12.8 (`cu128`)
- **Torch:** 2.9.0+
- **Python:** 3.11 (cp311)
- **Platform:** Windows x64 (win_amd64)

### PIL import failure on startup

`webui-user.bat` has a preflight check. If it catches a broken Pillow install, it auto-reinstalls. If you see this repeatedly:

```bat
venv\Scripts\pip.exe install --force-reinstall pillow==10.4.0
```

### WebUI won't start (port already in use)

Change `--port` in `webui-user.bat` or kill the process holding the port:

```bat
netstat -ano | findstr :44444
taskkill /PID <PID> /F
```

### SageAttention errors at runtime

If `sage_error` appears in console output:
1. Verify the SageAttention wheel matches your CUDA + Torch versions
2. Run the synthetic test from section 4.5
3. Check `ldm_patched/ldm/modules/attention.py` for routing logic

### Out of memory (OOM) errors

- Set lower resolution (try `832x1216` instead of `1024x1024`)
- Remove `--always-gpu` and add `--medvram` or `--lowvram` (significant speed penalty)
- Close other GPU-heavy applications
- Generate one image at a time instead of batching

---

## Quick Reference: Full Install Summary

```bat
# 1. Clone
git clone <REPO_URL> impForge
cd impForge

# 2. First run (creates venv, installs everything)
webui-user.bat
# Press Ctrl+C when WebUI starts

# 3. Triton
venv\Scripts\pip.exe install triton-windows==3.5.1

# 4. SageAttention
venv\Scripts\pip.exe install "path\to\sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"

# 5. FlashAttention (with --no-deps!)
venv\Scripts\pip.exe install "path\to\flash_attn-2.8.3+cu128torch2.9.0cxx11abiFALSE-cp311-cp311-win_amd64.whl" --no-deps

# 6. Verify
venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); import sageattention; import flash_attn; print('all ok')"

# 7. Edit webui-user.bat → set model paths, port, flags

# 8. Place model files in configured directories

# 9. Launch
webui-user.bat
```
