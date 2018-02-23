# Powershell script for installing CUDA on appveyor instances

#$env:CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/8.0/prod/network_installers/cuda_8.0.44_windows_network-exe"
$env:CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/9.1/Prod/network_installers/cuda_9.1.85_win10_network"
#$env:CUDA_REPO_PKG="cuda_8.0.44_win10_network.exe"
$env:CUDA_REPO_PKG="cuda_9.1.85_win10_network.exe"

# Get network installer
Write-Host 'Downloading CUDA Network Installer'
Invoke-WebRequest $env:CUDA_REPO_PKG_LOCATION -OutFile $env:CUDA_REPO_PKG | Out-Null
Write-Host 'Downloading Complete'
  
# Invoke silent install of CUDA compiler and runtime with Visual Studio integration (via network installer)
Write-Host 'Installing CUDA Compiler and Runtime'
& .\$env:CUDA_REPO_PKG -s compiler_9.1 nvcc_9.1 Display.Driver visual_studio_integration_9.1 nvprof_9.1 memcheck_9.1 gpu-library-advisor_9.1 nvprune_9.1 cudart_9.1 cublas_9.1 cublas_dev_9.1 curand_9.1 curand_dev_9.1| Out-Null
#& .\$env:CUDA_REPO_PKG -s compiler_8.0 visual_studio_integration_8.0 command_line_tools_8.0 cudart_8.0| Out-Null
Write-Host 'Installation Complete.'


if NOT EXIST "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.1\bin\cudart64_91.dll" ( 
echo "Failed to install CUDA"
exit /B 1
)

nvcc -V
# TODO: Test the install was successful