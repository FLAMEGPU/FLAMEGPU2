# Powershell script for installing CUDA on appveyor instances

$env:CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/8.0/prod/network_installers/cuda_8.0.44_win10_network-exe"
$env:CUDA_REPO_PKG="cuda_8.0.44_win10_network.exe"

# Get network installer
Write-Host 'Downloading CUDA Network Installer'
Invoke-WebRequest $env:CUDA_REPO_PKG_LOCATION -OutFile $env:CUDA_REPO_PKG | Out-Null
Write-Host 'Downloading Complete'
  
# Invoke silent install of CUDA compiler and runtime with Visual Studio integration (via network installer)
Write-Host 'Installing CUDA Compiler and Runtime'
$install_result = & .\$env:CUDA_REPO_PKG -s compiler_8.0 visual_studio_integration_8.0 | Out-String
$nvcc_version = nvcc -V | Out-String 
Write-Host "$install_result $nvcc_version"
Write-Host 'Installation Complete.'

# List directory output
$rules = ls "C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V120\BuildCustomizations\" | Out-String
Write-Host "$rules"