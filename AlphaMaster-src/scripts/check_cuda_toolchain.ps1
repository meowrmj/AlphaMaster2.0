$ErrorActionPreference = "SilentlyContinue"

Write-Host "== Python/PyTorch =="
@'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_home", CUDA_HOME)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
'@ | & .\.venv\Scripts\python.exe -

Write-Host ""
Write-Host "== PATH tools =="
Write-Host "nvcc:" (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
Write-Host "cl:" (Get-Command cl.exe -ErrorAction SilentlyContinue).Source

Write-Host ""
Write-Host "== CUDA Toolkit folders =="
Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" | Select-Object FullName

Write-Host ""
Write-Host "== Visual Studio cl.exe candidates =="
Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter cl.exe | Select-Object -First 8 FullName

Write-Host ""
Write-Host "== Native backend probe =="
& .\.venv\Scripts\python.exe scripts\probe_native_backend.py
