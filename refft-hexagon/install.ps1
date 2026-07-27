<#
.SYNOPSIS
    RefineFuture.AI - refft-hexagon universal installer (PowerShell entry point).

.DESCRIPTION
    Detects the platform and hands over to the matching installer:
        Windows                       -> windows\install.ps1
        Android device reachable via adb (from a Windows host) -> android package pushed by windows\install.ps1
    On Linux/Android hosts use install.sh instead.

.EXAMPLE
    irm https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/install.ps1 | iex

.EXAMPLE
    & ([scriptblock]::Create((irm <url>/refft-hexagon/install.ps1))) -Prefix 'C:\refft'
#>

param(
    # Everything is forwarded verbatim to the platform installer.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Forward
)

$ErrorActionPreference = 'Stop'

$Repo    = 'refinefuture-ai/refft.cpp'
$Ref     = if ($env:REFFT_INSTALL_REF) { $env:REFFT_INSTALL_REF } else { 'main' }
$RawBase = "https://raw.githubusercontent.com/$Repo/$Ref/refft-hexagon"

function Write-Step($Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Stop-WithError($Message) { Write-Host "error: $Message" -ForegroundColor Red; exit 1 }

$isWindowsHost = $true
if (Get-Variable -Name IsWindows -ErrorAction SilentlyContinue) { $isWindowsHost = $IsWindows }
if (-not $isWindowsHost) {
    Stop-WithError "this entry point targets Windows; on Linux/Android run:`n    curl -fsSL $RawBase/install.sh | sh"
}

Write-Step 'Detected platform: windows'

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$localScript = $null
if ($PSScriptRoot) {
    $candidate = Join-Path $PSScriptRoot 'windows\install.ps1'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { $localScript = $candidate }
}

if ($localScript) {
    Write-Step "Using local installer: $localScript"
    & $localScript @Forward
    exit $LASTEXITCODE
}

$url = "$RawBase/windows/install.ps1"
Write-Step "Fetching $url"
try {
    $content = Invoke-RestMethod -Uri $url -Headers @{ 'User-Agent' = 'refft-hexagon-installer' }
} catch {
    Stop-WithError "failed to download $url"
}
if (-not $content) { Stop-WithError "downloaded installer is empty: $url" }

& ([scriptblock]::Create($content)) @Forward
