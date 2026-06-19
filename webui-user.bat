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
  --lora-dir "F:\AI\MODELS\Lora" ^
  --embeddings-dir "D:\AI\MODELS\embeddings" ^
  --vae-dir "D:\AI\MODELS\VAE"

@REM Uncomment following code to reference an existing A1111 checkout.
@REM set A1111_HOME=Your A1111 checkout dir
@REM
@REM set VENV_DIR=%A1111_HOME%\\venv
@REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% ^
@REM  --ckpt-dir %A1111_HOME%\\models\\Stable-diffusion ^
@REM  --hypernetwork-dir %A1111_HOME%\\models\\hypernetworks ^
@REM  --embeddings-dir %A1111_HOME%\\embeddings ^
@REM  --lora-dir %A1111_HOME%\\models\\Lora

:: Preflight check: PIL import
if not defined VENV_DIR set "VENV_DIR=%~dp0venv"
if exist "%VENV_DIR%\Scripts\python.exe" (
  "%VENV_DIR%\Scripts\python.exe" -c "from PIL import Image" >nul 2>&1
  if errorlevel 1 (
    echo PIL import failed - reinstalling Pillow...
    "%VENV_DIR%\Scripts\pip.exe" install --force-reinstall pillow==10.4.0
  )
)

call webui.bat
