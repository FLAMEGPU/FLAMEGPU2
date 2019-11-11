# Powershell script for installing vc++ support and CUDA on appveyor instances

## -------------------
## Select CUDA version
## -------------------
# $CUDA_VERSION_FULL =  "8.0.44"  # CUDA 8.0 GA 1
# $CUDA_VERSION_FULL =  "8.0.61"  # CUDA 8.0 GA 2
# $CUDA_VERSION_FULL =  "9.0.176" # CUDA 9.0
# $CUDA_VERSION_FULL =  "9.1.85"  # CUDA 9.1
# $CUDA_VERSION_FULL =  "9.2.148" # CUDA 9.2
# $CUDA_VERSION_FULL = "10.0.130" # CUDA 10.0
# $CUDA_VERSION_FULL = "10.1.105" # CUDA 1.1
# $CUDA_VERSION_FULL = "10.1.168" # CUDA 10.1 update1
$CUDA_VERSION_FULL = "10.1.243" # CUDA 10.1 update2


## -----------------
## Prepare Variables
## -----------------
# Validate CUDA version, extracting components via regex
$cuda_ver_matched = $CUDA_VERSION_FULL -match "^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)$"
if(-not $cuda_ver_matched){
    Write-Host "Invalid CUDA version specified, <major>.<minor>.<patch> required. '$CUDA_VERSION_FULL'."
    exit 1
}
$CUDA_MAJOR=$Matches.major
$CUDA_MINOR=$Matches.minor
$CUDA_PATCH=$Matches.patch

# Build CUDA related variables.
$CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/$($CUDA_MAJOR).$($CUDA_MINOR)/prod/network_installers/cuda_$($CUDA_VERSION_FULL)_windows_network-exe"
$CUDA_REPO_PKG="cuda_$($CUDA_VERSION_FULL)_win10_network.exe"

# Build list of required cuda packages to be installed. See https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#install-cuda-software for pacakge details. 

# CUDA < 9.1 had a differnt package name for the compiler.
$NVCC_PACKAGE_NAME="nvcc"
if ([int]$CUDA_MAJOR -le 8 -Or ([int]$CUDA_MAJOR -eq 9 -And [int]$CUDA_MINOR -eq 0)){
    $NVCC_PACKAGE_NAME="compiler"
}
# Build string containing list of pacakges. Do not need Display.Driver
$CUDA_PACKAGES = ""
$CUDA_PACKAGES += "$($NVCC_PACKAGE_NAME)_$($CUDA_MAJOR).$($CUDA_MINOR) "
$CUDA_PACKAGES += "visual_studio_integration_$($CUDA_MAJOR).$($CUDA_MINOR) "
# $CUDA_PACKAGES += "curand_$($CUDA_MAJOR).$($CUDA_MINOR) "
$CUDA_PACKAGES += "curand_dev_$($CUDA_MAJOR).$($CUDA_MINOR) "

Write-Host $CUDA_REPO_PKG_LOCATION
Write-Host $CUDA_REPO_PKG
Write-Host $CUDA_PACKAGES
exit 1

## ------------
## Install vc++
## ------------
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

## ------------
## Install CUDA
## ------------

# Get CUDA network installer
Write-Host "Downloading CUDA Network Installer for $($CUDA_VERSION_FULL)"
Invoke-WebRequest $CUDA_REPO_PKG_LOCATION -OutFile $CUDA_REPO_PKG | Out-Null
Write-Host "Downloading Complete"
  
# Invoke silent install of CUDA (via network installer)
Write-Host "Installing CUDA $($CUDA_VERSION_FULL) Compiler and Runtime"
& .\$CUDA_REPO_PKG -s $CUDA_PACKAGES | Out-Null

Write-Host "Installation Complete."
