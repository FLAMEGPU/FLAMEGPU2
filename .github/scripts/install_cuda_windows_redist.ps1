# Assumes vswhere.exe, curl and 7z.exe are on the path

## -------------------
## Constants
## -------------------

# The root server with redist packages and manifests
$REDIST_ROOT = "https://developer.download.nvidia.com/compute/cuda/redist"
$PLATFORM = "windows-x86_64"

# So far, the mainifests all follow the same pattern, so no need for a list of known cuda versions (yet)

# Cuda pacakges to install, separate name conversions may be required
$CUDA_PACKAGES_IN = @(
    "visual_studio_integration";
    "cuda_nvcc";
    "cuda_nvrtc";
    "cuda_cudart";
    "cuda_cccl";
    "libcurand";
    # cuda 12+
    "libnvjitlink";
    # cuda 13+ packages
    "cuda_crt";
    "libnvptxcompiler";
    "libnvvm";
)

# Get the cuda version from the environment as env:cuda.
$CUDA_VERSION_FULL = $env:cuda
# Make sure CUDA_VERSION_FULL is set and valid, otherwise error.

# Validate CUDA version, extracting components via regex
$cuda_ver_matched = $CUDA_VERSION_FULL -match "^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)$"
if(-not $cuda_ver_matched){
    Write-Output "Invalid CUDA version specified, <major>.<minor>.<patch> required. '$CUDA_VERSION_FULL'."
    exit 1
}
$CUDA_MAJOR=$Matches.major
$CUDA_MINOR=$Matches.minor
$CUDA_PATCH=$Matches.patch

# If cuda version is le the first redist with a full manifest (11.4.2), error
# 11.4.2 is the first version with a releasee date and cuda pacakges
if([version]$CUDA_VERSION_FULL -lt [version]"11.4.2") {
    Write-Output "Error: cuda version $($CUDA_VERSION_FULL) is below the minimum 11.4.2 available via $($REDIST_ROOT). Aborting."
    # Abort the script.
    exit 1
}

# adjust packages based on which cuda versions they are available in
$CUDA_PACKAGES = @()
Foreach ($package in $CUDA_PACKAGES_IN) {
   if($package -eq "libnvjitlink" -and [version]$CUDA_VERSION_FULL -lt [version]"12.0") {
        # nvjitlink is a from CUDA 12.0, otherwise it should be skipped.
        continue
    } elseif($package -eq "cuda_crt" -and [version]$CUDA_VERSION_FULL -lt [version]"13.0") {
        # crt is a from CUDA 13.0, otherwise it should be skipped.
        continue
    } elseif($package -eq "libnvptxcompiler" -and [version]$CUDA_VERSION_FULL -lt [version]"13.0") {
        # nvptxcompiler is a from CUDA 13.0, otherwise it should be skipped.
        continue
    } elseif($package -eq "libnvvm" -and [version]$CUDA_VERSION_FULL -lt [version]"13.0") {
        # nvvm is a from CUDA 13.0, otherwise it should be skipped.
        continue
    }
    $CUDA_PACKAGES += "$($package)"
}
echo "$($CUDA_PACKAGES)"


# Comptue the manifest uri/url
$CUDA_REDIST_MANIFEST_URL="$REDIST_ROOT/redistrib_$($CUDA_VERSION_FULL).json"

# Build the cuda toolkit directory
$CUDA_TOOLKIT_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$($CUDA_MAJOR).$($CUDA_MINOR)"

# Find where visual studio is and where the build customiszation files should go.
$vswherePath = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs_path = & $vswherePath -latest -property installationPath
$vs_version = & $vswherePath -latest -property installationVersion
# Gross and semi-hardcoded to be 160/170
$vs_version_path = "v$(-join ($vs_version.Split('.'))[0])0"
$build_customizations_path = Join-Path -Path $vs_path -ChildPath "MSBuild\Microsoft\VC\$vs_version_path\BuildCustomizations"

# Suppress Invoke-WebRequest progress bars for faster download
$ProgressPreference = 'SilentlyContinue'

# Download the manifest for the requrested cuda version
echo "manifest: ${CUDA_REDIST_MANIFEST_URL}"
$manifest = Invoke-RestMethod -Uri $CUDA_REDIST_MANIFEST_URL
# echo "$($manifest.release_label)"

# @todo: use a temorary directory

# For each package, if it is found: download, unzip, copy/install and clean up
$json_packages = ""
Foreach ($package in $CUDA_PACKAGES) {
    # Ensure the package is in the manifest, else emit a warning and skip the package
    if($manifest.$package -eq $null) {
        echo "Warning: package $package not found in manifest"
        continue
    }
    # Ensure the platform is available for the package, else emit a warning and skip the package
    if ($manifest.$package.$PLATFORM -eq $null) {
        echo "Warning: platform $PLATFORM not found for pacakge $package in manifest"
        continue
    }
    # Ensure the a relative url is is available, else emit a warning and skip the package
    if ($manifest.$package.$PLATFORM.relative_path -eq $null) {
        echo "Warning: $package.$PLATFORM.relative_path is missing, unable to download"
        continue
    }
    # Download the package, retrying if the download or etraction fails, or the SHA did not match, up to a maximum number of times
    $attempt = 0
    $max_retry = 3
    $sleep_seconds = 10
    $download_extract_success = $false
    $package_url = "$($REDIST_ROOT)/$($manifest.$package.$PLATFORM.relative_path)"
    $package_zip = Split-Path -Path $package_url -Leaf
    $package_dir = [System.IO.Path]::GetFileNameWithoutExtension($package_zip)
    $expected_sha256 = "$($manifest.$package.$PLATFORM.sha256)"
    do {
        $attempt++
        $retry_needed = $false
        try {
            echo "Downloading $package from $package_url"
            Invoke-WebRequest -Uri $package_url -OutFile $package_zip -ErrorAction Stop
            echo "Checking zip sha256 matches ${expected_sha256}"
            $zip_sha256 = (Get-Filehash -Algorithm SHA256 -Path $package_zip -ErrorAction Stop).Hash
            if ($zip_sha256 -ieq $expected_sha256) {
                Write-Host "SHA256 match for $package"
                echo "Extracting $package_zip"
                Expand-Archive -Path $package_zip -DestinationPath . -Force -ErrorAction Stop
                # Flag to exit the do while
                $download_extract_success = $true
            } else {
                Write-Warning "SHA256 for $package does not match. $zip_sha256 != $expected_sha256"
                $retry_needed = $true
            }
        } catch {
            Write-Warning "Failed to download or extract ${package} (attempt $attempt): $($_.Exception.Message)"
            $retry_needed = $true
        }
        # Cleanup and sleep if retrying
        if ($retry_needed -and $attempt -lt $max_retry) {
            if (Test-Path -Path $package_zip) {
                Remove-Item -Path $package_zip -Force
            }
            Start-Sleep -Seconds $sleep_seconds
        }
    } while (-not $download_extract_success -and $attempt -lt $max_retry)
    # If the do while exitied wihtout success ful exctraction, give up.
    if (-not $download_extract_success) {
        Write-Error "Failed to download and extract $package after $max_retry attempts."
        exit 1
    }

    # 'install' the package (I.e. copy to the expected location). some packages need special handling
    if ($package -eq "visual_studio_integration") {
        # Install build customisations, assuming consistnet packaging from nvidia
        echo "Installing build customisations to $build_customizations_path"
        New-Item -Path "$build_customizations_path" -ItemType Directory -Force
        $source = "$($package_dir)\visual_studio_integration\MSBuildExtensions"
        echo "$source"
        cp -Path "$source\*" -Destination "$build_customizations_path" -Recurse
    } else {
        echo "Installing $package in $CUDA_TOOLKIT_DIR"
        New-Item -Path "$CUDA_TOOLKIT_DIR" -ItemType Directory -Force
        # Just extract into the expected cuda installation directory
        cp -Path "$package_dir\*" -Destination "$CUDA_TOOLKIT_DIR\" -Recurse -Force
    }
    # Delete the extracted files
    Remove-Item -Path $package_dir -Recurse -Force
    # Delete the downloaded zip
    Remove-Item -Path $package_zip -Recurse -Force
}

# Set environment variables

# Store the CUDA_PATH in the environment for the current session, to be forwarded in the action.
$CUDA_PATH = "$CUDA_TOOLKIT_DIR"
$CUDA_PATH_VX_Y = "CUDA_PATH_V$($CUDA_MAJOR)_$($CUDA_MINOR)"
# Set environmental variables in this session
$env:CUDA_PATH = "$($CUDA_PATH)"
$env:CUDA_PATH_VX_Y = "$($CUDA_PATH_VX_Y)"
Write-Output "CUDA_PATH $($CUDA_PATH)"
Write-Output "CUDA_PATH_VX_Y $($CUDA_PATH_VX_Y)"

# PATH needs updating elsewhere, anything in here won't persist.
# Append $CUDA_PATH/bin to path.

# If executing on github actions, emit the appropriate echo statements to update environment variables
if (Test-Path "env:GITHUB_ACTIONS") {
    # Set paths for subsequent steps, using $env:CUDA_PATH
    echo "Adding CUDA to CUDA_PATH, CUDA_PATH_X_Y and PATH"
    echo "CUDA_PATH=$env:CUDA_PATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    echo "$env:CUDA_PATH_VX_Y=$env:CUDA_PATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    echo "$env:CUDA_PATH/bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}
