#!/system/bin/sh
# Sprint WIRE-HEX-FINISH — on-device per-SSE-event timing helper.
# Usage: timed_chat.sh PROMPT_TOKENS_JSON MAX_TOKENS
#
# Example: sh /data/local/tmp/timed_chat.sh '[2,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]' 32
#
# Drives /v1/chat with the given prompt_tokens + max_tokens, timestamps each
# SSE delta arrival with millisecond resolution (date +%s%3N), and emits:
#   FIRST_DELTA_MS_FROM_START <ms>      — prefill wall-clock (time to first decoded token)
#   DELTA_<N>_MS_FROM_FIRST <ms> | <sse-line>   — per-token deltas + payload
#   DONE_MS_FROM_START <ms>             — total wall-clock (request to [DONE])
#   STEADY_DECODE_MS <ms>               — DONE_MS - FIRST_DELTA_MS (decode-only wall)
#   N_TOKENS <count>                    — number of delta events received
#
# prefill_tok/s = prompt_len / (FIRST_DELTA_MS / 1000)
# decode_tok/s  = (N_TOKENS - 1) / (STEADY_DECODE_MS / 1000)
# (N_TOKENS - 1 because the first delta is the argmax-of-prefill, not a decode_step output.)
prompt="$1"
maxtok="$2"
start_ms=$(date +%s%3N)
first_ms=""
last_ms=""
n_tokens=0
curl -s -N -X POST -H "Content-Type: application/json" \
  -d "{\"prompt_tokens\":${prompt},\"max_tokens\":${maxtok}}" \
  http://127.0.0.1:8087/v1/chat 2>/dev/null | while IFS= read -r line; do
    now_ms=$(date +%s%3N)
    case "$line" in
      data:*)
        if [ -z "$first_ms" ]; then
          first_ms="$now_ms"
          echo "FIRST_DELTA_MS_FROM_START $(($first_ms - $start_ms))"
        fi
        case "$line" in
          *DONE*) echo "DONE_MS_FROM_START $(($now_ms - $start_ms))"; echo "STEADY_DECODE_MS $(($now_ms - $first_ms))"; echo "N_TOKENS $n_tokens" ;;
          *) n_tokens=$((n_tokens+1)); echo "DELTA_${n_tokens}_MS_FROM_FIRST $(($now_ms - $first_ms)) | $line" ;;
        esac
        ;;
    esac
done
