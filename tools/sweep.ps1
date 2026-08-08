# growforge build-artifact sweep.
# Deletes caches that regrow on the next build and nothing else. Safe to
# run at any time EXCEPT during an active cargo build in this tree.
# Scheduled weekly (Task Scheduler, "growforge sweep"); also runnable by
# hand. What it never touches: source, .git, configs, Documents.
#
# Every deletion below roots in a validated absolute path. Nothing may
# ever pass a relative or empty path to Remove-Item: an empty string
# makes Get-ChildItem fall back to the process working directory, which
# under Task Scheduler is System32.

$ErrorActionPreference = 'SilentlyContinue'
$tree = 'D:\3D_Construct'
$log = Join-Path $PSScriptRoot 'sweep.log'

function Size-GB($path) {
    if (-not (Test-Path $path)) { return 0 }
    $s = (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum
    [math]::Round($s / 1GB, 2)
}

# Refuse to run while cargo/rustc are alive - a sweep mid-build corrupts
# the build.
if (Get-Process cargo, rustc -ErrorAction SilentlyContinue) {
    "$(Get-Date -Format s) skipped: cargo/rustc running" | Add-Content $log
    exit 0
}

$before = Size-GB "$tree\target"

# Always: incremental caches (the compounding bulk) and generated docs.
Remove-Item "$tree\target\debug\incremental" -Recurse -Force
Remove-Item "$tree\target\release\incremental" -Recurse -Force
Remove-Item "$tree\target\doc" -Recurse -Force

# Guard rail: if the whole target still exceeds 40 GB of accumulated
# stale artifacts, drop the debug profile entirely - the next test build
# recreates it in minutes.
if ((Size-GB "$tree\target") -gt 40) {
    Remove-Item "$tree\target\debug" -Recurse -Force
}

# Session scratchpads older than 14 days (probe crates, logs, captures).
# The base path is validated before ANY enumeration: if LOCALAPPDATA is
# unset, Join-Path silently yields "" and Get-ChildItem "" walks the
# working directory instead - the one way this script could ever reach
# outside its own targets. It must be provably rooted or skipped.
if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    "$(Get-Date -Format s) skipped scratchpads: LOCALAPPDATA unset" | Add-Content $log
} else {
    $claudeTmp = Join-Path $env:LOCALAPPDATA 'Temp\claude'
    if (-not [string]::IsNullOrWhiteSpace($claudeTmp) -and (Test-Path $claudeTmp)) {
        Get-ChildItem $claudeTmp -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue } |
            Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-14) } |
            Remove-Item -Recurse -Force
    }
}

$after = Size-GB "$tree\target"
"$(Get-Date -Format s) swept: target $before GB -> $after GB" | Add-Content $log
