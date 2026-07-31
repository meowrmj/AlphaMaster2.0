param(
    [switch]$Install,
    [string]$CudaVersion = "12.8"
)

$ErrorActionPreference = "Stop"

Write-Host "AlphaMaster native CUDA toolchain setup"
Write-Host "CUDA Toolkit target version: $CudaVersion"
Write-Host ""

$commands = @(
    "winget install --id Microsoft.VisualStudio.2022.BuildTools --source winget --override `"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`"",
    "winget install --id Nvidia.CUDA --version $CudaVersion --source winget"
)

if (-not $Install) {
    Write-Host "Dry run only. Nothing will be installed."
    Write-Host ""
    Write-Host "Commands to run:"
    foreach ($cmd in $commands) {
        Write-Host "  $cmd"
    }
    Write-Host ""
    Write-Host "To install, run:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\setup_cuda_native_toolchain.ps1 -Install"
    Write-Host ""
    Write-Host "After installation, open 'x64 Native Tools Command Prompt for VS 2022' or restart this terminal, then run:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\check_cuda_toolchain.ps1"
    Write-Host "  .\.venv\Scripts\python.exe scripts\probe_native_backend.py --build"
    Write-Host "  .\.venv\Scripts\python.exe scripts\test_native_elementwise.py"
    exit 0
}

foreach ($cmd in $commands) {
    Write-Host ""
    Write-Host "Running: $cmd"
    iex $cmd
}

Write-Host ""
Write-Host "Install commands finished. Restart the terminal or use the VS x64 Native Tools prompt, then run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\check_cuda_toolchain.ps1"
