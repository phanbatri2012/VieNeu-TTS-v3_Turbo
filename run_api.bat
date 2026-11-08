@echo off
setlocal
set "VIENEU_ROOT=%~dp0.."
set "VIENEU_BACKEND=onnx"
set "VIENEU_DEVICE=cpu"
set "VIENEU_PRECISION=fp32"
set "VIENEU_BACKBONE_DIR=%VIENEU_ROOT%\models\vieneu-v3-turbo"
set "VIENEU_ONNX_DIR=%VIENEU_ROOT%\models\vieneu-v3-turbo\onnx_update"
set "VIENEU_CODEC_DIR=%VIENEU_ROOT%\models\moss-audio-tokenizer"
set "VIENEU_HOME=%VIENEU_ROOT%\voices"
set "HF_HOME=%VIENEU_ROOT%\cache"
set "HF_HUB_CACHE=%VIENEU_ROOT%\cache\hub"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HOST=127.0.0.1"
set "PORT=8000"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0src;%~dp0"

"%VIENEU_ROOT%\.venv\Scripts\python.exe" -m apps.openai_speech
exit /b %ERRORLEVEL%
