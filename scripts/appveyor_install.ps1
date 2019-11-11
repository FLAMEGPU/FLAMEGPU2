# Powershell script for installing vc++ support and CUDA on appveyor instances

## -------------------
## Select CUDA version
## To make this part of an appveyor matrix, move this to appveyor.yaml and read from $env:CUDA_VERSION_FULL (or equivalent)
## -------------------

# $CUDA_VERSION_FULL =  "8.0.44"  # CUDA 8.0 GA 1
# $CUDA_VERSION_FULL =  "8.0.61"  # CUDA 8.0 GA 2
# $CUDA_VERSION_FULL =  "9.0.176" # CUDA 9.0
# $CUDA_VERSION_FULL =  "9.1.85"  # CUDA 9.1
$CUDA_VERSION_FULL =  "9.2.148" # CUDA 9.2
# $CUDA_VERSION_FULL = "10.0.130" # CUDA 10.0
# $CUDA_VERSION_FULL = "10.1.105" # CUDA 10.1
# $CUDA_VERSION_FULL = "10.1.168" # CUDA 10.1 update1
# $CUDA_VERSION_FULL = "10.1.243" # CUDA 10.1 update2


$CUDA_KNOWN_URLS = @{
    "8.0.44" = "http://developer.nvidia.com/compute/cuda/8.0/Prod/network_installers/cuda_8.0.44_win10_network-exe";
    "8.0.61" = "http://developer.nvidia.com/compute/cuda/8.0/Prod2/network_installers/cuda_8.0.61_win10_network-exe";
    "9.0.176" = "http://developer.nvidia.com/compute/cuda/9.0/Prod/network_installers/cuda_9.0.176_win10_network-exe";
    "9.1.85" = "http://developer.nvidia.com/compute/cuda/9.1/Prod/network_installers/cuda_9.1.85_win10_network";
    "9.2.148" = "http://developer.nvidia.com/compute/cuda/9.2/Prod2/network_installers2/cuda_9.2.148_win10_network";
    "10.0.130" = "http://developer.nvidia.com/compute/cuda/10.0/Prod/network_installers/cuda_10.0.130_win10_network";
    "10.1.105" = "http://developer.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.105_win10_network.exe";
    "10.1.168" = "http://developer.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.168_win10_network.exe";
    "10.1.243" = "http://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe";
}

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


# If the specified version is in the known addresses, use that one. 
$CUDA_REPO_PKG_REMOTE=""
if($CUDA_KNOWN_URLS.containsKey($CUDA_VERSION_FULL)){
    $CUDA_REPO_PKG_REMOTE=$CUDA_KNOWN_URLS[$CUDA_VERSION_FULL]
} else{
    # Guess what the url is given the most recent pattern (at the time of writing, 10.1)
    Write-Host "note: URL for CUDA ${$CUDA_VERSION_FULL} not known, estimating."
    $CUDA_REPO_PKG_REMOTE="http://developer.download.nvidia.com/compute/cuda/$($CUDA_MAJOR).$($CUDA_MINOR)/Prod/network_installers/cuda_$($CUDA_VERSION_FULL)_win10_network.exe"
}
$CUDA_REPO_PKG_LOCAL="cuda_$($CUDA_VERSION_FULL)_win10_network.exe"

# Build list of required cuda packages to be installed. See https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#install-cuda-software for pacakge details. 

# CUDA < 9.1 had a differnt package name for the compiler.
$NVCC_PACKAGE_NAME="nvcc"
if ([int]$CUDA_MAJOR -le 8 -Or ([int]$CUDA_MAJOR -eq 9 -And [int]$CUDA_MINOR -eq 0)){
    $NVCC_PACKAGE_NAME="compiler"
}
# Build string containing list of pacakges. Do not need Display.Driver
$CUDA_PACKAGES  = "$($NVCC_PACKAGE_NAME)_$($CUDA_MAJOR).$($CUDA_MINOR)"
$CUDA_PACKAGES += " visual_studio_integration_$($CUDA_MAJOR).$($CUDA_MINOR)"
$CUDA_PACKAGES += " curand_dev_$($CUDA_MAJOR).$($CUDA_MINOR)"


## ------------
## Install vc++
## ------------
if (Test-Path env:APPVEYOR_BUILD_WORKER_IMAGE){
    Write-Host "Installing vc++ for $env:APPVEYOR_BUILD_WORKER_IMAGE"
    if ($env:APPVEYOR_BUILD_WORKER_IMAGE -eq "Visual Studio 2015"){
        Write-Host "Vs2015"
        cmd.exe /c "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64
        cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86_amd64
    }
    elseif ($env:APPVEYOR_BUILD_WORKER_IMAGE -eq "Visual Studio 2017"){
        Write-Host "Vs2017"
        cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
    }
    elseif ($env:APPVEYOR_BUILD_WORKER_IMAGE -eq "Visual Studio 2019"){
        Write-Host "Vs2019"
        cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    }
}


## ------------
## Install CUDA
## ------------

if ($CUDA_PACKAGES -eq "nvcc_10.1 visual_studio_integration_10.1 curand_10.1 curand_dev_10.1"){
    Write-Host "CUDA_PACKAGES MAtch"
} else {
    Write-Host "CUDA Packages does not match"
}

# Get CUDA network installer
Write-Host "Downloading CUDA Network Installer for $($CUDA_VERSION_FULL) from: $($CUDA_REPO_PKG_REMOTE)"
Invoke-WebRequest $CUDA_REPO_PKG_REMOTE -OutFile $CUDA_REPO_PKG_LOCAL | Out-Null
if(Test-Path -Path $CUDA_REPO_PKG_LOCAL){
    Write-Host "Downloading Complete"
} else {
    Write-Host "Error: Failed to download $($CUDA_REPO_PKG_LOCAL) from $($CUDA_REPO_PKG_REMOTE)"
    exit 1
}

# Invoke silent install of CUDA (via network installer)
Write-Host "Installing CUDA $($CUDA_VERSION_FULL) Compiler and Runtime"
Start-Process -Wait -FilePath .\"$($CUDA_REPO_PKG_LOCAL)" -ArgumentList "-s $($CUDA_PACKAGES)"

# Check the return status of the CUDA installer.
if ($? -eq $false) {
    Write-Host "Error: CUDA installer reported error. $($LASTEXITCODE)"
    exit 1 
}

# Check for NVCC in the expected location
#$nvcc_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v$($CUDA_MAJOR).$($CUDA_MINOR)/bin/nvcc.exe"
#if(Test-Path -Path $nvcc_path){
#    Start-Process -Wait -FilePath "$nvcc_path" -ArgumentList "--version"
#} else {
#    Write-Host "Error: nvcc not-found in expected location."
#    exit 1
#}

Write-Host "Installation Complete!"
