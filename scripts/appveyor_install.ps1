# Powershell script for installing CUDA on appveyor instances

$env:CUDA_REPO_PKG_LOCATION="https://developer.nvidia.com/compute/cuda/8.0/prod/network_installers/cuda_8.0.44_win10_network-exe"
$env:CUDA_REPO_PKG="cuda_8.0.44_win10_network.exe"

# Get network installer
Write-Host 'Downloading CUDA Network Installer'
&dl_job = Invoke-WebRequest $env:CUDA_REPO_PKG_LOCATION -OutFile $env:CUDA_REPO_PKG
Wait-Job -Job $dl_job
Write-Host 'Downloading Complete'
  
# Invoke silent install of CUDA compiler and runtime (via network installer)
Write-Host 'Installing CUDA Compiler and Runtime'
& "$env:CUDA_REPO_PKG.exe -s compiler_8.0 | Out-Null"
Write-Host 'Installation Complete.'
<# try {
    Write-Host 'Installing CUDA Compiler and Runtime'
    $cuda_install_proc = Start-Process -FilePath "$env:CUDA_REPO_PKG" -ArgumentList "/s compiler_8.0" -Wait -PassThru
    $proc1.waitForExit()
    Write-Host 'Installation Complete.'
} catch [exception] {
    write-host '$_ is' $_
    write-host '$_.GetType().FullName is' $_.GetType().FullName
    write-host '$_.Exception is' $_.Exception
    write-host '$_.Exception.GetType().FullName is' $_.Exception.GetType().FullName
    write-host '$_.Exception.Message is' $_.Exception.Message
}
#>
