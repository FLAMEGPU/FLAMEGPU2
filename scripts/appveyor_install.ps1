# Powershell script for installing vc++ support and CUDA on appveyor instances

# Select CUDA version, requires major, minor and patch to be included.

$env:CUDA_VERSION_FULL="8.0.44"
# $env:CUDA_VERSION_FULL="9.1.85"
# $env:CUDA_VERSION_FULL="10.1.243"

# Validate input CUDA version, extracting major minor and patch via regex
$cuda_ver_matched = $env:CUDA_VERSION_FULL -match "^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)$"
if(-not $cuda_ver_matched){
    # Error if invalid CUDA version specified.
    Write-Host "Invalid CUDA version specified, <major>.<minor>.<patch> required. '$env:CUDA_VERSION_FULL'."
    exit 1
}
$env:CUDA_MAJOR=$Matches.major
$env:CUDA_MINOR=$Matches.minor
$env:CUDA_PATCH=$Matches.patch

# Build CUDA related variables.
Write-Host "CUDA_VER: $($env:CUDA_MAJOR).$($env:CUDA_MINOR).$($env:CUDA_PATCH)"
$env:CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/$($env:CUDA_MAJOR).$($env:CUDA_MINOR)/prod/network_installers/cuda_$($env:CUDA_VERSION_FULL)_windows_network-exe"
$env:CUDA_REPO_PKG="cuda_$($env:CUDA_VERSION_FULL)_win10_network.exe"

$env:CUDA_PACKAGES=""

$env:CUDA_PACKAGES += "nvcc_$($env:CUDA_MAJOR).$($env:CUDA_MINOR)"
$env:CUDA_PACKAGES += "visual_studio_integration_$($env:CUDA_MAJOR).$($env:CUDA_MINOR)"
$env:CUDA_PACKAGES += "curand_$($env:CUDA_MAJOR).$($env:CUDA_MINOR)"
$env:CUDA_PACKAGES += "curand_dev_$($env:CUDA_MAJOR).$($env:CUDA_MINOR"


Write-Host $env:CUDA_REPO_PKG_LOCATION
Write-Host $env:CUDA_REPO_PKG
Write-Host $env:CUDA_PACKAGES
exit 1

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
Write-Host "Downloading CUDA Network Installer for $($env:CUDA_VERSION_FULL)"
Invoke-WebRequest $env:CUDA_REPO_PKG_LOCATION -OutFile $env:CUDA_REPO_PKG | Out-Null
Write-Host "Downloading Complete"
  
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
