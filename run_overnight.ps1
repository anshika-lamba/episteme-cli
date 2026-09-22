<#
.SYNOPSIS
  Sequential overnight grid. One provider at a time. Safe to re-run after a reboot.

.DESCRIPTION
  Default (no switches) is the preregistered 996-trial plan. It does not start
  Cohere, and it does not start 7 families x 1000 trials.

    groq    allam-2-7b           --seeds 9     (324 trials)
    mistral ministral-8b-latest  --seeds 9     (324 trials)
    gemini  gemma-3-12b-it       --seeds 7     (252 trials)

  7 families x 1000 trials is about 7000 trials / 56000 calls. Free tiers will
  not finish that in one night (Groq allam-2-7b is 1000 requests/day; a Cohere
  trial key is 1000 calls/month). That path runs only when you pass
  -Seven -TrialsEach 1000 -AllowOverBudget, and Cohere only with -IncludeCohere
  as well. The script prints the quota warning before the first call.

  Re-run the same command after a reboot, a quota wall, or a crash. Finished
  trial_ids are skipped. Trials that died on a quota wall or on transport
  exhaustion ("gave up after") are re-queued. A torn last line is truncated.

.EXAMPLE
  .\run_overnight.ps1

.EXAMPLE
  .\run_overnight.ps1 -DryRun

.EXAMPLE
  .\run_overnight.ps1 -Seven -TrialsEach 1000 -IncludeCohere -AllowOverBudget
#>
param(
    [int]$TrialsEach = 0,
    [switch]$Seven,
    [switch]$IncludeCohere,
    [switch]$AllowOverBudget,
    [switch]$DryRun,
    [int]$TransportRetries = 8,
    [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "results") | Out-Null
$LogPath = Join-Path $RepoRoot "results\overnight.log"

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $Message
    Add-Content -Path $LogPath -Value $line -Encoding utf8
    Write-Host $line
}

# 36 cells per seed. Prereg = 0 means "not in the locked budget".
# Cohere is the 7th family and stays off unless -IncludeCohere.
$Catalog = @(
    @{ Name = "groq-allam";   Provider = "groq";    Model = "allam-2-7b";             Prereg = 9; Cohere = $false; Subset = "" },
    @{ Name = "groq-llama8";  Provider = "groq";    Model = "llama-3.1-8b-instant";   Prereg = 0; Cohere = $false; Subset = "" },
    @{ Name = "groq-llama70"; Provider = "groq";    Model = "llama-3.3-70b-versatile"; Prereg = 0; Cohere = $false; Subset = "" },
    @{ Name = "mistral";      Provider = "mistral"; Model = "ministral-8b-latest";    Prereg = 9; Cohere = $false; Subset = "" },
    @{ Name = "gemma-12b";    Provider = "gemini";  Model = "gemma-3-12b-it";         Prereg = 7; Cohere = $false; Subset = "" },
    @{ Name = "gemma-27b";    Provider = "gemini";  Model = "gemma-3-27b-it";         Prereg = 0; Cohere = $false; Subset = "" },
    @{ Name = "cohere";       Provider = "cohere";  Model = "command-r7b-12-2024";    Prereg = 6; Cohere = $true;  Subset = "priority" }
)

$Selected = New-Object System.Collections.Generic.List[object]
foreach ($job in $Catalog) {
    $include = $false
    if ($job.Cohere) {
        $include = [bool]$IncludeCohere
    } elseif ($Seven) {
        $include = $true
    } elseif ($job.Prereg -gt 0) {
        $include = $true
    }
    if (-not $include) { continue }
    if ($TrialsEach -gt 0) {
        $seeds = [int][Math]::Ceiling($TrialsEach / 36.0)
        $subset = ""
    } else {
        $seeds = [int]$job.Prereg
        $subset = [string]$job.Subset
    }
    if ($seeds -le 0) {
        Write-Log ("skip {0}: no seed count in the locked budget. Pass -TrialsEach N -AllowOverBudget to size it." -f $job.Name)
        continue
    }
    $Selected.Add(@{ Job = $job; Seeds = $seeds; Subset = $subset })
}

if ($Selected.Count -eq 0) {
    Write-Log "nothing selected"
    exit 0
}

$planned = 0
foreach ($item in $Selected) { $planned += ([int]$item.Seeds * 36) }

Write-Log ("plan: {0} job(s), {1} trials, one provider at a time, transport-retries={2}" -f $Selected.Count, $planned, $TransportRetries)
foreach ($item in $Selected) {
    $job = $item.Job
    $extra = ""
    if ($item.Subset) { $extra = " --subset $($item.Subset)" }
    Write-Log ("  {0,-14} {1,-28} seeds={2}{3}" -f $job.Provider, $job.Model, $item.Seeds, $extra)
}

if ($planned -gt 996 -and -not $AllowOverBudget) {
    Write-Log "REFUSING to start: $planned trials exceeds the preregistered 996."
    Write-Log "7 families x 1000 trials is about 7000 trials / 56000 calls and will not finish on free tiers (Groq allam-2-7b is 1000 RPD; Cohere trial keys are 1000 calls/month)."
    Write-Log "Re-run with -AllowOverBudget if you mean it. Nothing was called."
    exit 2
}
if ($IncludeCohere) {
    Write-Log "WARNING: Cohere is in this plan. A trial key is a hard 1000 calls/month across every endpoint. This spends the ledger you were holding in reserve."
}
if ($AllowOverBudget) {
    Write-Log "WARNING: -AllowOverBudget is set. Daily/monthly caps will stop a provider; re-run this script to resume the rest. Do not expect 7 x 1000 to finish tonight."
}

$bashCandidates = @(
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files\Git\usr\bin\bash.exe"
)
$bash = $bashCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($bash) {
    Write-Log "sandbox shell candidate: $bash"
} else {
    Write-Log "WARN: neither C:\Program Files\Git\bin\bash.exe nor C:\Program Files\Git\usr\bin\bash.exe exists. run_grid.py will also check bash next to git.exe and WSL, then refuse to start if none of them run a command."
}
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Log "FAIL: '$Python' was not found on PATH."
    exit 1
}

$quotaHit = @{}
$index = 0
foreach ($item in $Selected) {
    $index += 1
    $job = $item.Job
    if ($quotaHit.ContainsKey($job.Provider)) {
        Write-Log ("skip {0}/{1}: an earlier {0} job already hit a quota wall. Re-run tomorrow; finished trials are kept." -f $job.Provider, $job.Model)
        continue
    }
    $pyArgs = @(
        "run_grid.py",
        "--provider", $job.Provider,
        "--model", $job.Model,
        "--seeds", "$($item.Seeds)",
        "--transport-retries", "$TransportRetries"
    )
    if ($item.Subset) {
        $pyArgs += @("--subset", $item.Subset)
    }
    if ($DryRun) {
        $pyArgs += "--dry-run"
    }
    Write-Log ("[{0}/{1}] {2} {3}" -f $index, $Selected.Count, $Python, ($pyArgs -join " "))
    & $Python @pyArgs
    $rc = $LASTEXITCODE
    if ($null -eq $rc) {
        Write-Log "stopping: $Python did not return an exit code"
        exit 1
    }
    Write-Log ("[{0}/{1}] exit {2}  {3}/{4}" -f $index, $Selected.Count, $rc, $job.Provider, $job.Model)
    if ($rc -eq 1) {
        Write-Log "stopping: preflight failed (sandbox). Later providers would fail the same way."
        exit 1
    }
    if ($rc -eq 2) {
        Write-Log ("skipping the rest of {0} this run (missing key or bad config)." -f $job.Provider)
        $quotaHit[$job.Provider] = $true
        continue
    }
    if ($rc -eq 3) {
        $quotaHit[$job.Provider] = $true
        Write-Log ("{0} stopped on a quota wall. Next provider will still run. Re-run this script to resume." -f $job.Provider)
    } elseif ($rc -eq 4) {
        Write-Log ("{0}/{1} hit the transport circuit breaker (3 trials exhausted retries). Next job still runs. Re-run to resume." -f $job.Provider, $job.Model)
    } elseif ($rc -ne 0) {
        Write-Log "stopping: unexpected exit $rc"
        exit $rc
    }
    if (-not $DryRun -and $index -lt $Selected.Count) {
        Write-Log "stagger 30s before the next provider (one provider at a time)"
        Start-Sleep -Seconds 30
    }
}

Write-Log "overnight script finished. Re-run the same command to resume anything still missing."
exit 0
