# Powershell script for installing vc++ support and CUDA on appveyor instances

# Select CUDA version, requires major, minor and patch to be included.
# $env:CUDA_VERSION_FULL="8.0.44"
# $env:CUDA_VERSION_FULL="9.1.85"
$env:CUDA_VERSION_FULL="10.1.243"


$cuda_version_pattern = [Regex]::new("^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)$")

$matches = $cuda_version_pattern.Matches($env:CUDA_VERSION_FULL)
Write-Host $matches
if($matches){
    Write-Host "Matched"
} else {
    Write-Host "did not match"
}

$env:CUDA_MAJOR=$matches[major]
$env:CUDA_MINOR=$matches[minor]
$env:CUDA_PATCH=$matches[patch]

Write-Host "CUDA_VER: $env:CUDA_MAJOR.$env:CUDA_MINOR.$env:CUDA_PATCH"

exit 1

# CUDA 8
# $env:CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/8.0/prod/network_installers/cuda_8.0.44_windows_network-exe"
# $env:CUDA_REPO_PKG="cuda_8.0.44_win10_network.exe"

# CUDA 9.1
#$env:CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/9.1/Prod/network_installers/cuda_9.1.85_win10_network"
#$env:CUDA_REPO_PKG="cuda_9.1.85_win10_network.exe"

# CUDA 10.1
$env:CUDA_REPO_PKG_LOCATION="http://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe"
$env:CUDA_REPO_PKG="cuda_10.1.243_win10_network.exe"

# Install vc++ for the appropriate visual studio version if this is executed on appveyor.
if (Test-Path env:APPVEYOR_BUILD_WORKER_IMAGE){
    Write-Host "Installing vc++ for $env:APPVEYOR_BUILD_WORKER_IMAGE"
    if ($env:APPVEYOR_BUILD_WORKER_IMAGE -eq "Visual Studio 2015"){
        cmd.exe /c "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64
        cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86_amd64
    }
    elseif ($env:APPVEYOR_BUILD_WORKER_IMAGE -eq "Visual Studio 2017"){
        cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
    }
    elseif ($env:APPVEYOR_BUILD_WORKER_IMAGE -eq "Visual Studio 2019"){
        cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    }
}

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
