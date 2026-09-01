# §4-NTT Sprint NTT-bench — driver script.
#
# Pushes the freshly-built sp_ntt_bench_toks binary to /data/local/tmp/,
# then loops 6 cells (2 models × 3 configs) with appropriate env vars per
# cell. Captures verbatim adb stdout to ntt_bench_toks_run.txt and pulls
# the accumulated JSONL report to ntt_bench_toks_report.jsonl.
#
# Per PLAN-NTT-bench.md §"Env var caching": math-core's g_ntt_attn reads
# SP_ENGINE_NTT_ATTN once at process start. We invoke the harness once per
# cell so each cell gets a fresh g_ntt_attn read.
#
# Prerequisites (already in place per M.1 / NTT.5c smoke runs):
#   - Executive model (qwen3_rt.sp-{model,tokenizer}) at /data/local/tmp/
#   - Memory model (qwen25-coder-0.5b-memory.sp-{model,tokenizer}) at /data/local/tmp/
#   - libsp_compute_skel.so at /data/local/tmp/
#
# Usage:
#   .\ntt_bench_toks_run.ps1 [-PromptLen 16] [-DecodeN 32] [-Reps 3]

param(
    [int]$PromptLen = 16,
    [int]$DecodeN = 32,
    [int]$Reps = 3,
    [string]$BinaryPath = "..\..\..\target\aarch64-linux-android\release\sp_ntt_bench_toks"
)

$ErrorActionPreference = "Continue"  # keep going across cells even if one errors

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogPath = Join-Path $ScriptDir "ntt_bench_toks_run.txt"
$ReportJsonl = "/data/local/tmp/ntt_bench_toks_report.jsonl"
$LocalReport = Join-Path $ScriptDir "ntt_bench_toks_report.jsonl"
$LocalJson   = Join-Path $ScriptDir "ntt_bench_toks_report.json"

$ExecModel  = "/data/local/tmp/qwen3_rt.sp-model"
$ExecTok    = "/data/local/tmp/qwen3_rt.sp-tokenizer"
$MemoModel  = "/data/local/tmp/qwen25-coder-0.5b-memory.sp-model"
$MemoTok    = "/data/local/tmp/qwen25-coder-0.5b-memory.sp-tokenizer"

# Reset log + on-device report
"" | Out-File -FilePath $LogPath -Encoding utf8

function LogTee([string]$msg) {
    Write-Host $msg
    $msg | Out-File -FilePath $LogPath -Encoding utf8 -Append
}

LogTee "[bench-driver] === NTT-bench run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
LogTee "[bench-driver] PromptLen=$PromptLen DecodeN=$DecodeN Reps=$Reps"

LogTee "[bench-driver] adb device check..."
$adbDevs = adb devices 2>&1 | Out-String
LogTee $adbDevs

LogTee "[bench-driver] confirming pre-staged artifacts on device..."
$artifactCheck = adb shell "ls -la $ExecModel $ExecTok $MemoModel $MemoTok /data/local/tmp/libsp_compute_skel.so" 2>&1 | Out-String
LogTee $artifactCheck

# Push the harness binary
$BinaryFull = Resolve-Path $BinaryPath -ErrorAction SilentlyContinue
if (-not $BinaryFull) {
    LogTee "[bench-driver] ERROR: binary not found at $BinaryPath"
    LogTee "[bench-driver] Build first via: cd tools\sp_daemon ; .\build-android.bat --bin sp_ntt_bench_toks --release"
    exit 1
}
LogTee "[bench-driver] pushing binary $BinaryFull -> /data/local/tmp/sp_ntt_bench_toks ..."
adb push $BinaryFull /data/local/tmp/sp_ntt_bench_toks 2>&1 | Out-Null
adb shell chmod +x /data/local/tmp/sp_ntt_bench_toks 2>&1 | Out-Null

# Reset the on-device JSONL report
LogTee "[bench-driver] resetting on-device report at $ReportJsonl ..."
adb shell "rm -f $ReportJsonl" 2>&1 | Out-Null

# Cell matrix
$cells = @(
    @{ N=1; Model="Executive"; Config="fp32";     Env="";                                                        ModelPath=$ExecModel; TokPath=$ExecTok },
    @{ N=2; Model="Executive"; Config="host_ntt"; Env="SP_ENGINE_NTT_ATTN=1";                                    ModelPath=$ExecModel; TokPath=$ExecTok },
    @{ N=3; Model="Executive"; Config="hex_ntt";  Env="SP_ENGINE_NTT_ATTN=1 SP_ENGINE_NTT_ATTN_HEX=1";           ModelPath=$ExecModel; TokPath=$ExecTok },
    @{ N=4; Model="Memory";    Config="fp32";     Env="";                                                        ModelPath=$MemoModel; TokPath=$MemoTok },
    @{ N=5; Model="Memory";    Config="host_ntt"; Env="SP_ENGINE_NTT_ATTN=1";                                    ModelPath=$MemoModel; TokPath=$MemoTok },
    @{ N=6; Model="Memory";    Config="hex_ntt";  Env="SP_ENGINE_NTT_ATTN=1 SP_ENGINE_NTT_ATTN_HEX=1";           ModelPath=$MemoModel; TokPath=$MemoTok }
)

foreach ($cell in $cells) {
    LogTee ""
    LogTee "[bench-driver] ============================================================"
    LogTee "[bench-driver] cell $($cell.N): model=$($cell.Model) config=$($cell.Config)"
    LogTee "[bench-driver]   env: $($cell.Env)"
    LogTee "[bench-driver]   model_path=$($cell.ModelPath)"
    LogTee "[bench-driver] ============================================================"

    $envPrefix = "ADSP_LIBRARY_PATH=`"/data/local/tmp;`""
    if ($cell.Env -ne "") {
        $envPrefix = "$envPrefix $($cell.Env)"
    }

    $cmd = @"
$envPrefix /data/local/tmp/sp_ntt_bench_toks `
  --cell $($cell.N) `
  --model-path $($cell.ModelPath) `
  --tok-path $($cell.TokPath) `
  --model-label $($cell.Model) `
  --config-label $($cell.Config) `
  --report-jsonl $ReportJsonl `
  --prompt-len $PromptLen `
  --decode-n $DecodeN `
  --reps $Reps
"@

    LogTee "[bench-driver] invoking..."
    # Capture both stdout + stderr from the adb shell invocation.
    $out = adb shell $cmd 2>&1 | Out-String
    LogTee $out
}

# Pull the accumulated JSONL
LogTee "[bench-driver] ============================================================"
LogTee "[bench-driver] pulling report from device..."
adb pull $ReportJsonl $LocalReport 2>&1 | Out-Null
if (Test-Path $LocalReport) {
    LogTee "[bench-driver] report pulled: $LocalReport"
    LogTee "[bench-driver] --- report contents ---"
    Get-Content $LocalReport | ForEach-Object { LogTee $_ }

    # Synthesize a single combined JSON with all cells in an array.
    $cellsArr = @()
    Get-Content $LocalReport | Where-Object { $_.Trim().Length -gt 0 } | ForEach-Object {
        try {
            $obj = $_ | ConvertFrom-Json
            $cellsArr += $obj
        } catch {
            LogTee "[bench-driver] WARN: failed to parse line as JSON: $_"
        }
    }
    $combined = @{
        sprint = "NTT-bench"
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        prompt_len = $PromptLen
        decode_n = $DecodeN
        reps = $Reps
        cells = $cellsArr
    }
    $combined | ConvertTo-Json -Depth 12 | Out-File -FilePath $LocalJson -Encoding utf8
    LogTee "[bench-driver] combined JSON: $LocalJson ($($cellsArr.Count) cells)"
} else {
    LogTee "[bench-driver] ERROR: report file not pulled"
    exit 2
}

LogTee "[bench-driver] === NTT-bench run finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
