# growforge Windows installer build.
#
# Builds the release executable, then compiles tools\installer.iss into
# target\installer\growforge-setup-<version>.exe. Nothing is published and
# nothing outside the repository tree is written: releasing the setup is a
# separate, deliberate step.
#
# Run from anywhere:
#
#   powershell -ExecutionPolicy Bypass -File D:\3D_Construct\tools\build_installer.ps1
#
# Every path below is absolute and rooted in this script's own location, and
# every one of them is checked before it is used - the same rule tools\sweep.ps1
# is held to, for the same reason: an empty or relative path silently becomes
# the process working directory, which under a scheduler is System32.
#
# Windows PowerShell 5.1 compatible: no '&&', no ternaries, no null-coalescing.

$ErrorActionPreference = 'Stop'

function Fail($message) {
    Write-Host "installer build failed: $message"
    exit 1
}

# The repository is this script's parent's parent. Confirmed by the manifest,
# so a copy of the script somewhere else cannot quietly build something else.
$tree = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $tree 'Cargo.toml'
if (-not (Test-Path -LiteralPath $manifest)) {
    Fail "no Cargo.toml under $tree - this script must stay in the repository's tools folder"
}

$script = Join-Path $PSScriptRoot 'installer.iss'
if (-not (Test-Path -LiteralPath $script)) { Fail "no installer script at $script" }

$exe = Join-Path $tree 'target\release\growforge.exe'
$outputDir = Join-Path $tree 'target\installer'

# A build over a live one corrupts both. Same refusal as the sweep, for the
# same reason.
if (Get-Process cargo, rustc -ErrorAction SilentlyContinue) {
    Fail 'cargo or rustc is already running in this tree'
}

$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if ($null -eq $cargo) { Fail 'cargo is not on PATH' }

# ISCC is not put on PATH by any of its installers, so it is looked for where
# each of them leaves it. Every candidate root is checked for emptiness first:
# Join-Path on an unset variable yields a relative path, and a relative path is
# how a script ends up running something it did not mean to.
$isccCandidates = @()
foreach ($root in @($env:LOCALAPPDATA, $env:ProgramFiles, ${env:ProgramFiles(x86)})) {
    if (-not [string]::IsNullOrWhiteSpace($root)) {
        $isccCandidates += (Join-Path $root 'Programs\Inno Setup 6\ISCC.exe')
        $isccCandidates += (Join-Path $root 'Inno Setup 6\ISCC.exe')
    }
}
$iscc = $null
foreach ($candidate in $isccCandidates) {
    if (Test-Path -LiteralPath $candidate) { $iscc = $candidate; break }
}
if ($null -eq $iscc) {
    Fail 'Inno Setup 6 is not installed (winget install -e --id JRSoftware.InnoSetup)'
}

Write-Host "growforge installer build"
Write-Host "  tree      $tree"
Write-Host "  compiler  $iscc"

Write-Host 'building the release executable ...'
& $cargo.Source build --release --manifest-path $manifest
if ($LASTEXITCODE -ne 0) { Fail "cargo build --release exited $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $exe)) { Fail "the build left no executable at $exe" }

# The version is read off the executable that was just built rather than out of
# the manifest: it is the same value - build.rs stamps it there from
# CARGO_PKG_VERSION - but it is the one a user's copy of the file will show,
# and it cannot disagree with the binary the setup is about to wrap.
$version = (Get-Item -LiteralPath $exe).VersionInfo.ProductVersion
if ([string]::IsNullOrWhiteSpace($version)) {
    Fail "$exe carries no version resource - is build.rs still in the tree?"
}
$version = $version.Trim()
if ($version -notmatch '^\d+\.\d+\.\d+') {
    Fail "the executable's version resource reads '$version', which is not a version"
}
Write-Host "  version   $version"

Write-Host 'compiling the setup ...'
& $iscc "/DAppVersion=$version" $script /Q
if ($LASTEXITCODE -ne 0) { Fail "ISCC exited $LASTEXITCODE" }

$setup = Join-Path $outputDir ("growforge-setup-{0}.exe" -f $version)
if (-not (Test-Path -LiteralPath $setup)) { Fail "ISCC reported success but $setup is not there" }

$megabytes = [math]::Round((Get-Item -LiteralPath $setup).Length / 1MB, 2)
Write-Host "wrote $setup ($megabytes MB)"
exit 0
