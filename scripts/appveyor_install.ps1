# Powershell script for installing vc++ support and CUDA on appveyor instances
# @future - Use powershell variables for cuda versions etc. 

# CUDA 8
# $env:CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/8.0/prod/network_installers/cuda_8.0.44_windows_network-exe"
# $env:CUDA_REPO_PKG="cuda_8.0.44_win10_network.exe"

# CUDA 9.1
#$env:CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/9.1/Prod/network_installers/cuda_9.1.85_win10_network"
#$env:CUDA_REPO_PKG="cuda_9.1.85_win10_network.exe"

# CUDA 10.1
$env:CUDA_REPO_PKG_LOCATION="http://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe"
$env:CUDA_REPO_PKG="cuda_10.1.243_win10_network.exe"


Write-Host "APPVEYOR_BUILD_WORKER_IMAGE"
Write-Host $APPVEYOR_BUILD_WORKER_IMAGE
# Install vc++, depending on which visual studio image is being used. 
# VS 2015
if (Test-Path -Path "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" -PathType Leaf) {
    Write-Host 'VS 2015 detected?'
    cmd.exe /c "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64
    cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86_amd64
}
elseif (Test-Path -Path "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" -PathType Leaf) {
    Write-Host 'VS 2015 detected?'
    # VS 2017
    cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
}
elseif (Test-Path -Path "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" -PathType Leaf) {
    Write-Host 'VS 2015 detected?'
    # VS 2019?
    cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
}
exit 1

# Install CUDA
# Get CUDA network installer
Write-Host 'Downloading CUDA Network Installer'
Invoke-WebRequest $env:CUDA_REPO_PKG_LOCATION -OutFile $env:CUDA_REPO_PKG | Out-Null
Write-Host 'Downloading Complete'
  
# Invoke silent install of CUDA compiler and runtime with Visual Studio integration (via network installer)
Write-Host 'Installing CUDA Compiler and Runtime'

# Do not need Display.Driver

# CUDA 8.0
# & .\$env:CUDA_REPO_PKG -s compiler_8.0 visual_studio_integration_8.0 command_line_tools_8.0 cudart_8.0| Out-Null

# CUDA 9.1
# & .\$env:CUDA_REPO_PKG -s nvcc_9.1 visual_studio_integration_9.1 cudart_9.1 curand_9.1 curand_dev_9.1| Out-Null

# CUDA 10.1
& .\$env:CUDA_REPO_PKG -s nvcc_10.1 visual_studio_integration_10.1 curand_10.1 curand_dev_10.1|  Out-Null


Write-Host 'Installation Complete.'
