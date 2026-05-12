@echo off

set PYTHON=
set GIT=
set VENV_DIR=

set COMMANDLINE_ARGS=--skip-version-check --skip-python-version-check --skip-torch-cuda-test --always-gpu --force-channels-last --allow-fp16-accumulation --pin-shared-memory --cuda-malloc --cuda-stream --listen --enable-insecure-extension-access --port 44444 --ckpt-dir "F:\AI\MODELS\CKPT" --lora-dir "F:\AI\MODELS\Lora" --embeddings-dir "D:\AI\MODELS\embeddings" --vae-dir "D:\AI\MODELS\VAE"

@REM Uncomment following code to reference an existing A1111 checkout.
@REM set A1111_HOME=Your A1111 checkout dir
@REM
@REM set VENV_DIR=%A1111_HOME%\\venv
@REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% ^
@REM  --ckpt-dir %A1111_HOME%\\models\\Stable-diffusion ^
@REM  --hypernetwork-dir %A1111_HOME%\\models\\hypernetworks ^
@REM  --embeddings-dir %A1111_HOME%\\embeddings ^
@REM  --lora-dir %A1111_HOME%\\models\\Lora

call webui.bat
