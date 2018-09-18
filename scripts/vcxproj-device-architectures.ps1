# Powershell script to update CUDA version in all visual studio projects.
# Script takes an argument containing the space/comma/semicolon separate list of compute capabilities 
param([string]$SMS="35;37;50;52;60;61;70")

# Get the path of this script
$script_path = split-path -parent $MyInvocation.MyCommand.Definition
# Get the path of the root project directory
$flamegpu2_root = split-path -parent $script_path

# Define SM separators
$sm_separators = ";", ",", " "
# Split the list and sort numerically
$sm_list = $SMS.Split($sm_separators,[System.StringSplitOptions]::RemoveEmptyEntries)| foreach-object { [int] $_ } | Sort-Object

# Prepare the gencode argument string
$gencodes = ""
foreach($sm in $sm_list){
    $gencodes = $gencodes + "compute_$sm,sm_$sm;"
}
# Use the highest SM to enable JIT compilation of future architectures
$last_sm = $sm_list[-1]
$gencodes = $gencodes + "compute_$last_sm,compute_$last_sm;"

# Prepare the code generation string
$code_generation_string = "<CodeGeneration>$gencodes</CodeGeneration>"

# Get all the files in the relevant directory
$all_files = get-childitem $flamegpu2_root -recurse
# Filter out all the vcxproj files
$proj_files = $all_files | where {$_.extension -eq ".vcxproj"} | % { $_.FullName }
# For each vcxproj file
foreach ($file in $proj_files){
    # Replace the CodeGeneration tag with the new one.
    (Get-Content -Path $file) | ForEach-Object {$_ -Replace "(<CodeGeneration>)([a-zA-Z0-9_,;]+)(</CodeGeneration>)", $code_generation_string} | Set-Content -Encoding UTF8 -Path $file
}


 # <CodeGeneration>compute_30,sm_30;compute_35,sm_35</CodeGeneration>
