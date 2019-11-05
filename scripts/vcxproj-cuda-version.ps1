# Powershell script to update CUDA version in all visual studio projects.
# Script takles an argument specifying the new cuda version to use. Default to 9.2
param([string]$new_cuda_version="9.2")

# Get the path of this script
$script_path = split-path -parent $MyInvocation.MyCommand.Definition
# Get the path of the root project directory
$flamegpu2_root = split-path -parent $script_path

# Get all the files in the relevant directory
$all_files = get-childitem $flamegpu2_root -recurse
# Filter out all the vcxproj files
$proj_files = $all_files | where {$_.extension -eq ".vcxproj"} | % { $_.FullName }
# For each vcxproj file
foreach ($file in $proj_files){
    # Replace occurances of CUDA D.D (where D is a digit) with CUDA {new_verison} in place
    (Get-Content -Path $file) | ForEach-Object {$_ -Replace "(CUDA )([0-9]+\.[0-9])", "CUDA $new_cuda_version"} | Set-Content -Encoding UTF8 -Path $file
}
