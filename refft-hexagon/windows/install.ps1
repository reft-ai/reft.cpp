<#
.SYNOPSIS
    RefineFuture.AI - refft-hexagon installer (Windows).

.DESCRIPTION
    Fetches the latest calendar-versioned release (tag format: vYYYY.MM.DD.NN)
    from https://github.com/refinefuture-ai/refft.cpp/releases, picks the asset
    matching this machine, downloads it, and installs it. If the package ships
    its own installer script it is executed; otherwise the package is unpacked
    into the install prefix.

.EXAMPLE
    .\install.ps1

.EXAMPLE
    irm https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/windows/install.ps1 | iex

.EXAMPLE
    & ([scriptblock]::Create((irm <url>/install.ps1))) -Prefix 'C:\refft' -AddToPath
#>

[CmdletBinding()]
param(
    # Install a specific tag instead of the latest release.
    [string]$Tag,

    # Install prefix (default: %LOCALAPPDATA%\Programs\refft-hexagon).
    [string]$Prefix,

    # Hexagon architecture variant, e.g. v73 / v81. Default: the highest available.
    [string]$HexagonVersion = $env:REFFT_HEXAGON_VERSION,

    # Override CPU architecture detection (arm64 / x86_64).
    [string]$Arch,

    # Directory to keep the downloaded asset in.
    [string]$DownloadDir,

    # Keep the downloaded archive after installing.
    [switch]$Keep,

    # List matching releases and assets, then exit.
    [switch]$List,

    # GitHub token (or set $env:GITHUB_TOKEN) to raise the API rate limit.
    [string]$Token = $env:GITHUB_TOKEN,

    # Append the installed bin directory to the user PATH.
    [switch]$AddToPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo         = 'refinefuture-ai/refft.cpp'
$ApiBase      = "https://api.github.com/repos/$Repo"
$ReleasesUrl  = "https://github.com/$Repo/releases"
$PkgPrefix    = 'refft-hexagon'
$PlatformName = 'windows'
$TagRegex     = '^v\d{4}\.\d{2}\.\d{2}\.\d{2}$'

# ---------------------------------------------------------------- utilities --

function Write-Step($Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Info($Message) { Write-Host "    $Message" }
function Write-Warn($Message) { Write-Host "warning: $Message" -ForegroundColor Yellow }
function Stop-WithError($Message) { Write-Host "error: $Message" -ForegroundColor Red; exit 1 }

function Get-AuthHeaders {
    $headers = @{ 'Accept' = 'application/vnd.github+json'; 'User-Agent' = 'refft-hexagon-installer' }
    if ($Token) { $headers['Authorization'] = "Bearer $Token" }
    return $headers
}

# --------------------------------------------------------------- release api --

function Get-ReleaseList {
    $all = @()
    for ($page = 1; $page -le 3; $page++) {
        $batch = @(Invoke-RestMethod -Uri "$ApiBase/releases?per_page=100&page=$page" `
                                     -Headers (Get-AuthHeaders) -Method Get)
        if ($batch.Count -eq 0) { break }
        $all += $batch
        if ($batch.Count -lt 100) { break }
    }
    return $all
}

function Get-TagSortKey($TagName) {
    # vYYYY.MM.DD.NN -> a comparable integer YYYYMMDDNN
    # (year, then month, then day, then the NN-th release of that day).
    $parts = $TagName.TrimStart('v').Split('.')
    return [int64]$parts[0] * 100000000L + [int64]$parts[1] * 1000000L +
           [int64]$parts[2] * 10000L + [int64]$parts[3]
}

function Get-CalendarReleases {
    Get-ReleaseList |
        Where-Object { $_.tag_name -match $TagRegex } |
        Sort-Object -Property @{ Expression = { Get-TagSortKey $_.tag_name } } -Descending
}

function Get-ReleaseByTag($TagName) {
    try {
        return Invoke-RestMethod -Uri "$ApiBase/releases/tags/$TagName" `
                                 -Headers (Get-AuthHeaders) -Method Get
    } catch {
        Stop-WithError "release not found: $TagName"
    }
}

# ------------------------------------------------------------ platform match --

function Get-HostArch {
    if ($Arch) { return $Arch.ToLower() }
    try {
        switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
            'Arm64' { return 'arm64' }
            'X64'   { return 'x86_64' }
            'X86'   { return 'x86' }
        }
    } catch { }
    switch ($env:PROCESSOR_ARCHITECTURE) {
        'ARM64' { return 'arm64' }
        'AMD64' { return 'x86_64' }
        default { return 'x86_64' }
    }
}

function Get-ArchAliases($Value) {
    switch ($Value) {
        'arm64'  { return @('arm64', 'aarch64') }
        'x86_64' { return @('x86_64', 'amd64', 'x64') }
        default  { return @($Value) }
    }
}

function Get-AssetRank($Name) {
    # Higher is better: the hexagon variant dominates, then the archive kind.
    $hexRank = 0
    if ($Name -match '-v(\d+)') { $hexRank = [int]$Matches[1] }
    $kind = switch -Wildcard ($Name) {
        '*.zip'    { 4; break }
        '*.tar.xz' { 3; break }
        '*.tar.gz' { 2; break }
        '*.tgz'    { 2; break }
        default    { 0 }
    }
    return $hexRank * 10 + $kind
}

function Select-PlatformAsset($Assets, $HostArch) {
    $aliases = Get-ArchAliases $HostArch
    $candidates = $Assets | Where-Object {
        $name = $_.name
        if (-not $name.StartsWith($PkgPrefix)) { return $false }
        $hit = $false
        foreach ($a in $aliases) { if ($name -like "*windows-$a*") { $hit = $true } }
        if (-not $hit) { return $false }
        if ($HexagonVersion) {
            $want = $HexagonVersion.TrimStart('v', 'V')
            if ($name -notlike "*-v$want*") { return $false }
        }
        return $true
    }
    $candidates = @($candidates)
    if ($candidates.Count -eq 0) { return $null }
    return $candidates | Sort-Object -Property @{ Expression = { Get-AssetRank $_.name } } -Descending |
           Select-Object -First 1
}

# ------------------------------------------------------------------- install --

function Save-Asset($Url, $Destination) {
    $previous = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is far faster without it
    try {
        $headers = @{ 'User-Agent' = 'refft-hexagon-installer' }
        if ($Token) { $headers['Authorization'] = "Bearer $Token" }
        Invoke-WebRequest -Uri $Url -OutFile $Destination -Headers $headers -UseBasicParsing
    } finally {
        $ProgressPreference = $previous
    }
}

function Test-Sha256($Path, $Expected) {
    if (-not $Expected) {
        Write-Warn 'no checksum published for this asset, skipping verification'
        return
    }
    $actual = (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $Expected.ToLower()) {
        Stop-WithError "checksum mismatch (expected $Expected, got $actual)"
    }
    Write-Info 'sha256 verified'
}

function Expand-Package($ArchivePath, $Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    switch -Wildcard ($ArchivePath) {
        '*.zip' {
            Expand-Archive -Path $ArchivePath -DestinationPath $Destination -Force
            break
        }
        { $_ -like '*.tar.xz' -or $_ -like '*.tar.gz' -or $_ -like '*.tgz' -or $_ -like '*.tar' } {
            $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
            if (-not $tar) {
                Stop-WithError "tar.exe is required to extract $ArchivePath (Windows 10 1803+ ships it)"
            }
            & $tar.Source -xf $ArchivePath -C $Destination
            if ($LASTEXITCODE -ne 0) { Stop-WithError "failed to extract $ArchivePath" }
            break
        }
        default { Stop-WithError "unsupported archive format: $ArchivePath" }
    }
}

function Get-PackageRoot($Directory) {
    # Archives usually hold a single top-level folder; treat it as the root.
    $entries = @(Get-ChildItem -Force -Path $Directory)
    if ($entries.Count -eq 1 -and $entries[0].PSIsContainer) { return $entries[0].FullName }
    return $Directory
}

function Find-BundledInstaller($Root) {
    foreach ($candidate in @('install.ps1', 'setup.ps1', 'install.cmd', 'install.bat', 'setup.cmd')) {
        $path = Join-Path $Root $candidate
        if (Test-Path -LiteralPath $path -PathType Leaf) { return $path }
    }
    $nested = Join-Path $Root 'scripts\install.ps1'
    if (Test-Path -LiteralPath $nested -PathType Leaf) { return $nested }
    return $null
}

function Invoke-BundledInstaller($InstallerPath, $Root, $InstallPrefix, $ReleaseTag) {
    $env:REFFT_INSTALL_PREFIX = $InstallPrefix
    $env:REFFT_RELEASE_TAG    = $ReleaseTag
    $env:REFFT_PLATFORM       = $PlatformName
    Push-Location $Root
    try {
        if ($InstallerPath -like '*.ps1') {
            & powershell -NoProfile -ExecutionPolicy Bypass -File $InstallerPath
        } else {
            & cmd.exe /c $InstallerPath
        }
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
            Stop-WithError 'the bundled installer failed'
        }
    } finally {
        Pop-Location
    }
}

function Install-Tree($Root, $InstallPrefix) {
    New-Item -ItemType Directory -Force -Path $InstallPrefix | Out-Null
    Copy-Item -Path (Join-Path $Root '*') -Destination $InstallPrefix -Recurse -Force
}

function Add-UserPath($Directory) {
    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($current -split ';' -contains $Directory) {
        Write-Info "already on PATH: $Directory"
        return
    }
    $updated = if ([string]::IsNullOrEmpty($current)) { $Directory } else { "$current;$Directory" }
    [Environment]::SetEnvironmentVariable('Path', $updated, 'User')
    $env:Path = "$env:Path;$Directory"
    Write-Info "added to the user PATH: $Directory"
}

# ---------------------------------------------------------------------- main --

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$hostArch = Get-HostArch
Write-Step 'RefineFuture.AI refft-hexagon installer'
Write-Info "platform : $PlatformName/$hostArch"
Write-Info "source   : $ReleasesUrl"

if ($List) {
    Write-Step 'Releases matching vYYYY.MM.DD.NN (newest first)'
    $releases = @(Get-CalendarReleases)
    if (-not $releases) { Stop-WithError "no release matching vYYYY.MM.DD.NN was found at $ReleasesUrl" }
    $releases | ForEach-Object { Write-Info $_.tag_name }
    Write-Step "Assets of $($releases[0].tag_name)"
    $releases[0].assets | ForEach-Object { Write-Info $_.name }
    exit 0
}

if ($Tag) {
    $release = Get-ReleaseByTag $Tag
} else {
    Write-Step 'Resolving the latest release (tag pattern vYYYY.MM.DD.NN)'
    $release = @(Get-CalendarReleases) | Select-Object -First 1
    if (-not $release) { Stop-WithError "no release matching vYYYY.MM.DD.NN was found at $ReleasesUrl" }
}
Write-Info "release  : $($release.tag_name)"

$assets = @($release.assets)
if (-not $assets) { Stop-WithError "release $($release.tag_name) has no downloadable assets" }

$asset = Select-PlatformAsset $assets $hostArch
if (-not $asset) {
    $suffix = if ($HexagonVersion) { " (hexagon $HexagonVersion)" } else { '' }
    Write-Warn "no asset in $($release.tag_name) matches $PlatformName/$hostArch$suffix"
    Write-Info 'available assets:'
    $assets | ForEach-Object { Write-Info "  $($_.name)" }
    Stop-WithError 'nothing to install'
}
Write-Info "asset    : $($asset.name)"

$expectedSha = $null
if ($asset.PSObject.Properties.Name -contains 'digest' -and $asset.digest) {
    $expectedSha = ($asset.digest -replace '^sha256:', '')
}

if (-not $Prefix) {
    $Prefix = Join-Path $env:LOCALAPPDATA "Programs\$PkgPrefix"
}

$workDir = Join-Path ([System.IO.Path]::GetTempPath()) ("refft-hexagon-" + [System.Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

try {
    if ($DownloadDir) {
        New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
        $archivePath = Join-Path $DownloadDir $asset.name
    } else {
        $archivePath = Join-Path $workDir $asset.name
    }

    Write-Step "Downloading $($asset.name)"
    Save-Asset $asset.browser_download_url $archivePath
    Test-Sha256 $archivePath $expectedSha

    Write-Step 'Extracting package'
    $extractDir = Join-Path $workDir 'extract'
    Expand-Package $archivePath $extractDir
    $packageRoot = Get-PackageRoot $extractDir

    $installer = Find-BundledInstaller $packageRoot
    if ($installer) {
        Write-Step "Running bundled installer: $(Split-Path -Leaf $installer)"
        Invoke-BundledInstaller $installer $packageRoot $Prefix $release.tag_name
        $installedAt = "$Prefix (bundled installer)"
    } else {
        Write-Step "No bundled installer found, extracting to $Prefix"
        Install-Tree $packageRoot $Prefix
        $installedAt = $Prefix
    }

    $binDir = if (Test-Path (Join-Path $Prefix 'bin')) { Join-Path $Prefix 'bin' } else { $Prefix }
    if ($AddToPath) { Add-UserPath $binDir }

    if ($Keep -and -not $DownloadDir) {
        Copy-Item -Path $archivePath -Destination (Join-Path (Get-Location) $asset.name) -Force
        $archivePath = Join-Path (Get-Location) $asset.name
    }
    if ($Keep -or $DownloadDir) { Write-Info "archive kept at: $archivePath" }

    Write-Host ''
    Write-Step 'Installation complete'
    Write-Info "release   : $($release.tag_name)"
    Write-Info "package   : $($asset.name)"
    Write-Info "installed : $installedAt"
    if (-not $AddToPath) {
        Write-Info "add to PATH: `$env:Path += ';$binDir'"
    }
} finally {
    if (Test-Path $workDir) { Remove-Item -Recurse -Force $workDir -ErrorAction SilentlyContinue }
}
