# Thin wrapper: activate venv + run orchestrator. Pass-through args.
# --batch            : evaluate pending jobs via career-ops batch runner (the agent CLI is
#                      BATCH_CLI in .env, resolved by pipeline/agent_cli.py — which owns the default)
# --skip-pdf         : skip PDF generation (report + tracker only)
# --min-score <N>    : skip tracker for jobs scoring below N (0 = off)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Parse flags — index loop required to consume --min-score's value argument
$runBatch = $false
$skipPdf = $false
$minScore = $null
$orchestrateArgs = @()
$i = 0
while ($i -lt $args.Count) {
    switch ($args[$i]) {
        "--batch"     { $runBatch = $true }
        "--skip-pdf"  { $skipPdf = $true }
        "--min-score" { $i++; $minScore = $args[$i] }
        default       { $orchestrateArgs += $args[$i] }
    }
    $i++
}

# Before the first-run profile gate and the scrape, not after them: career-ops'
# runner rejects a CLI it does not know only once the whole run has been spent
# (#187), and a fresh copy would sit through the setup interview first. It needs
# only the flags parsed above — the CLI comes from BATCH_CLI and .env, neither
# of which the profile gate writes. The rule is the registry's
# (pipeline/agent_cli.py); this only asks it, and it says why.
if ($runBatch) {
    Push-Location $root
    try { & "$root\.venv\Scripts\python.exe" -m pipeline.agent_cli --check-batch } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path "$root\career-ops\config\profile.yml")) {
    Write-Host "Profile not found — running first-time setup..."
    node "$root\setup-profile.mjs"
}

& "$root\.venv\Scripts\python.exe" "$root\orchestrate.py" @orchestrateArgs
# A native command's exit code is not an error to $ErrorActionPreference, and
# `pwsh -File` exits 0 for a script that runs off its end, so without this a
# failed run — a crash, or --evaluate-batch's unclean tracker merge — went on
# to --batch and could end green, where run.sh's `set -e` stops.
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($runBatch) {
    # The registry resolves BATCH_CLI (reading .env, which the Setup wizard
    # writes) and owns the default — no second copy of it here.
    # From $root: `pipeline` is not pip-installed, so `-m` finds it by cwd.
    # The model too: OLLAMA_MODEL (the older, --batch-only name) when set, else
    # AGENT_MODEL / the registry's default for this CLI (`--resolved-model`,
    # which may print nothing — then the CLI starts on its own default and no
    # --model is passed).
    Push-Location $root
    try {
        $batchCli = (& "$root\.venv\Scripts\python.exe" -m pipeline.agent_cli --resolved | Select-Object -Last 1)
        $batchModel = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } `
                      else { (& "$root\.venv\Scripts\python.exe" -m pipeline.agent_cli --resolved-model | Select-Object -Last 1) }
    } finally { Pop-Location }
    if (-not $batchCli) {
        Write-Host "run.ps1: could not resolve the agent CLI (python -m pipeline.agent_cli --resolved printed nothing)."
        exit 1
    }
    $batchCli = "$batchCli".Trim()
    $batchModel = "$batchModel".Trim()
    $batchRunner = "$root\career-ops\batch\batch-runner.sh"
    $batchArgs = @("--cli", $batchCli)
    $displayStr = $batchCli
    if ($batchModel) { $batchArgs += @("--model", $batchModel); $displayStr += " / $batchModel" }
    if ($skipPdf) { $batchArgs += "--skip-pdf" }
    if ($minScore) { $batchArgs += @("--min-score", $minScore) }
    Write-Host ""
    Write-Host "==> Running batch evaluation ($displayStr)..."
    # Prefer Git for Windows bash over WSL bash (WSL requires a VM via HCS).
    $gitBash = @(
        "$env:ProgramFiles\Git\bin\bash.exe",
        "${env:ProgramFiles(x86)}\Git\bin\bash.exe",
        "$env:LocalAppData\Programs\Git\bin\bash.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    $bashExe = if ($gitBash) { $gitBash } `
               elseif ((Get-Command bash -ErrorAction SilentlyContinue) -and
                       (Get-Command bash).Source -notmatch 'System32') { "bash" } `
               else { $null }
    if ($bashExe) {
        & $bashExe $batchRunner @batchArgs
        # A failed runner ends the script here, as run.sh's `set -e` ends it.
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        # The runner merged its own additions as its last step. A row the agent
        # CLI wrote with an unreadable score was refused there — archived before
        # career-ops#5862, kept in the queue since; this repairs and re-merges
        # either kind, and exits 1 unless the merge was clean, which ends this
        # script in failure as run.sh's `set -e` ends it. See
        # pipeline/merge_additions.py.
        & "$root\.venv\Scripts\python.exe" -m pipeline.merge_additions
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Write-Host "ERROR: Git for Windows not found. Install from https://git-scm.com/download/win"
        exit 1
    }
}

