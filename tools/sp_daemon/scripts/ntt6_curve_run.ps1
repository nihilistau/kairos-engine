# NTT.6 — long-context tiled benchmark driver.
#
# Extension of ntt_bench_toks_run.ps1 for measuring tok/s vs ctx (prompt_len).
# Per cell, the binary is invoked once with the requested -Ctx as --prompt-len.
# Cell matrix per ctx:
#   c1: Gemma3-1B fp32
#   c2: Gemma3-1B host_ntt
#   c3: Memory     fp32
#   c4: Memory     host_ntt
#   c5: Memory     hex_ntt        (only at small ctx; structurally catastrophic at long ctx — see PLAN-NTT-6.md)
#
# Per PLAN-NTT-6.md Decision D + E:
#   - This harness covers fp32 + host_ntt + Memory hex_ntt (cDSP NTT-attention overlay).
#   - It does NOT cover Gemma hex-vrmpy (HX.3b daemon path) — separate methodology required.
#
# Output files (per ctx):
#   ntt6_curve_ctxN_run.txt        verbatim adb stdout
#   ntt6_curve_ctxN_report.jsonl   per-cell JSON lines from harness
#   ntt6_curve_ctxN_report.json    aggregated array
#
# Usage:
#   .\ntt6_curve_run.ps1 -Ctx 512 -Reps 3 [-IncludeHex] [-OnlyModel Memory|Gemma3]

param(
    [int]$Ctx = 512,
    [int]$Reps = 3,
    [int]$DecodeN = 4,                       # decode is invariant — keep tiny to save wall-clock
    [switch]$IncludeHex = $false,            # cell 5 (Memory hex_ntt) — only safe at small ctx
    [string]$OnlyModel = "",                 # "" = both; "Memory" or "Gemma3" filters
    [string]$BinaryPath = "/data/local/tmp/sp_ntt_bench_toks",  # use on-device binary (no rebuild)
    [string]$AdbExe = "D:\Files\Android\pt-latest\platform-tools\adb.exe",
    [string]$Serial = "R5CT22445JA"
)

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path (Split-Path -Parent (Split-Path -Parent $ScriptDir)) "sp_compute_skel\data\ntt6_cells"
if (-not (Test-Path $DataDir)) { New-Item -ItemType Directory -Force -Path $DataDir | Out-Null }

$LogPath     = Join-Path $DataDir "ntt6_curve_ctx${Ctx}_run.txt"
$LocalReport = Join-Path $DataDir "ntt6_curve_ctx${Ctx}_report.jsonl"
$LocalJson   = Join-Path $DataDir "ntt6_curve_ctx${Ctx}_report.json"
$ReportJsonl = "/data/local/tmp/ntt6_curve_ctx${Ctx}_report.jsonl"

$GemmaModel  = "/data/local/tmp/gemma3-1b.sp-model"
$GemmaTok    = "/data/local/tmp/gemma3-1b.sp-tokenizer"
$MemoModel   = "/data/local/tmp/qwen25-coder-0.5b-memory.sp-model"
$MemoTok     = "/data/local/tmp/qwen25-coder-0.5b-memory.sp-tokenizer"

"" | Out-File -FilePath $LogPath -Encoding utf8

function LogTee([string]$msg) {
    Write-Host $msg
    $msg | Out-File -FilePath $LogPath -Encoding utf8 -Append
}

LogTee "[ntt6-driver] === NTT.6 curve run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
LogTee "[ntt6-driver] Ctx=$Ctx Reps=$Reps DecodeN=$DecodeN IncludeHex=$IncludeHex OnlyModel='$OnlyModel'"
LogTee "[ntt6-driver] Binary on device: $BinaryPath"

# Verify adb + binary + skel
LogTee "[ntt6-driver] adb devices:"
$adbDevs = & $AdbExe -s $Serial shell "ls -la $BinaryPath /data/local/tmp/libsp_compute_skel.so /data/local/tmp/sp22u/libsp_hex_skel.so 2>&1" | Out-String
LogTee $adbDevs

# Reset on-device report
& $AdbExe -s $Serial shell "rm -f $ReportJsonl" 2>&1 | Out-Null

# Cell matrix (build from filters)
$AllCells = @(
    @{ N=1; Model="Gemma3"; Config="fp32";     Env="";                                                  ModelPath=$GemmaModel; TokPath=$GemmaTok },
    @{ N=2; Model="Gemma3"; Config="host_ntt"; Env="SP_ENGINE_NTT_ATTN=1";                              ModelPath=$GemmaModel; TokPath=$GemmaTok },
    @{ N=3; Model="Memory"; Config="fp32";     Env="";                                                  ModelPath=$MemoModel;  TokPath=$MemoTok  },
    @{ N=4; Model="Memory"; Config="host_ntt"; Env="SP_ENGINE_NTT_ATTN=1";                              ModelPath=$MemoModel;  TokPath=$MemoTok  },
    @{ N=5; Model="Memory"; Config="hex_ntt";  Env="SP_ENGINE_NTT_ATTN=1 SP_ENGINE_NTT_ATTN_HEX=1";     ModelPath=$MemoModel;  TokPath=$MemoTok  }
)

$cells = @()
foreach ($c in $AllCells) {
    if ($OnlyModel -ne "" -and $c.Model -ne $OnlyModel) { continue }
    if ($c.Config -eq "hex_ntt" -and -not $IncludeHex) { continue }
    $cells += $c
}

LogTee "[ntt6-driver] running $($cells.Count) cells:"
foreach ($c in $cells) { LogTee "  cell $($c.N) $($c.Model)/$($c.Config) env='$($c.Env)'" }

foreach ($cell in $cells) {
    LogTee ""
    LogTee "[ntt6-driver] ============================================================"
    LogTee "[ntt6-driver] cell $($cell.N) ctx=$Ctx model=$($cell.Model) config=$($cell.Config)"
    LogTee "[ntt6-driver]   env: $($cell.Env)"
    LogTee "[ntt6-driver] ============================================================"

    $envPrefix = "ADSP_LIBRARY_PATH=`"/data/local/tmp;`""
    if ($cell.Env -ne "") {
        $envPrefix = "$envPrefix $($cell.Env)"
    }

    # Single-line command — PowerShell backtick continuation doesn't survive being passed to
    # `adb shell` (adb concatenates as words for /system/bin/sh, which sees the continuation
    # bytes as garbage and the next-line flags as separate commands).
    $cmd = "$envPrefix $BinaryPath --cell $($cell.N) --model-path $($cell.ModelPath) --tok-path $($cell.TokPath) --model-label $($cell.Model) --config-label $($cell.Config) --report-jsonl $ReportJsonl --prompt-len $Ctx --decode-n $DecodeN --reps $Reps"

    $cellStart = Get-Date
    LogTee "[ntt6-driver] invoking at $($cellStart.ToString('HH:mm:ss')) ..."
    $out = & $AdbExe -s $Serial shell $cmd 2>&1 | Out-String
    LogTee $out
    $cellElapsed = (Get-Date) - $cellStart
    LogTee "[ntt6-driver] cell $($cell.N) elapsed: $([int]$cellElapsed.TotalSeconds) s"
}

# Pull the report
LogTee "[ntt6-driver] ============================================================"
LogTee "[ntt6-driver] pulling report from device..."
& $AdbExe -s $Serial pull $ReportJsonl $LocalReport 2>&1 | Out-Null
if (Test-Path $LocalReport) {
    LogTee "[ntt6-driver] report pulled: $LocalReport"
    LogTee "[ntt6-driver] --- report contents ---"
    Get-Content $LocalReport | ForEach-Object { LogTee $_ }

    $cellsArr = @()
    Get-Content $LocalReport | Where-Object { $_.Trim().Length -gt 0 } | ForEach-Object {
        try {
            $obj = $_ | ConvertFrom-Json
            $cellsArr += $obj
        } catch {
            LogTee "[ntt6-driver] WARN: failed to parse line as JSON: $_"
        }
    }
    $combined = @{
        sprint = "NTT.6-curve"
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        ctx = $Ctx
        reps = $Reps
        decode_n = $DecodeN
        cells = $cellsArr
    }
    $combined | ConvertTo-Json -Depth 12 | Out-File -FilePath $LocalJson -Encoding utf8
    LogTee "[ntt6-driver] combined JSON: $LocalJson ($($cellsArr.Count) cells)"
} else {
    LogTee "[ntt6-driver] ERROR: report file not pulled"
    exit 2
}

LogTee "[ntt6-driver] === NTT.6 curve run finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
