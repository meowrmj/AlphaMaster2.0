@echo off
setlocal

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "VSLANG=1033"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if exist "%VCVARS%" goto have_vcvars
echo Missing vcvars64.bat: %VCVARS%
exit /b 1
:have_vcvars

if exist "%CUDA_HOME%\bin\nvcc.exe" goto have_nvcc
echo Missing nvcc.exe: %CUDA_HOME%\bin\nvcc.exe
exit /b 1
:have_nvcc

call "%VCVARS%"
chcp 65001 >nul
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"

if "%~1"=="" (
  where cl
  where nvcc
  .venv\Scripts\python.exe scripts\probe_native_backend.py
  exit /b %ERRORLEVEL%
)

%*
exit /b %ERRORLEVEL%
