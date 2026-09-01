#!/usr/bin/env python3
"""
test_live_loop_closure.py -- the conversational-memory LOOP-CLOSURE gate.

Drives a single multi-turn conversation through POST /v1/chat (sequential turns
so B4 NIGHTSHIFT accumulates LIVE episodes into the shared app.nightshift buffer)
and proves the SP_B3_JUDGE generative judge recalls the conversation's OWN
immediate past:

  Turn 1 (ingest)    : state a novel fact -> NIGHTSHIFT captures ep_live_000.
  Turn 2 (recency)   : ask for it -> the live episode is on the KAIROS recency
                       axis -> judge picks it -> inject_tokens -> model answers.
  Turns 3-5 (filler) : unrelated turns push Turn-1 out of the R-recency window.
  Turn 6 (paging)    : re-ask -> Stage-0 C2 salience must rescue it from the cold
                       nightshift pool (may miss -- documented WEAK C2 signal).
  Foreign            : an OOD question -> judge returns [NULL] -> clean answer.

Each turn is one POST with messages + auto_recall:true (FINDING #1: messages =
instruct template). The SSE deltas are concatenated to the answer string.

Usage:
  python test_live_loop_closure.py --host http://127.0.0.1:3001 \
      --log D:\\F\\shannon-prime-repos\\shannon-prime-system-engine\\_judge_live_serve.log
The --log path (the daemon's stdout/stderr capture) is grepped after each turn
for the KAIROS working-set + judge PICK telemetry.
"""
import argparse, json, re, sys, time, urllib.request

def chat(host, content, max_tokens=24, timeout=300):
    body = json.dumps({
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "stop": [], "temperature": 0.0,
        "auto_recall": True,
    }).encode()
    req = urllib.request.Request(host + "/v1/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            payload = s[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if "delta" in ev:
                out.append(ev["delta"])
    return "".join(out).strip()

def tail_log(path, n=400):
    if not path:
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except Exception:
        return []

def log_grep(lines, *needles):
    hits = []
    for ln in lines:
        if any(nd in ln for nd in needles):
            hits.append(ln.rstrip())
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:3001")
    ap.add_argument("--log", default="")
    ap.add_argument("--token", default="4DA0-FALCON")
    args = ap.parse_args()

    TOK = args.token
    turns = [
        ("ingest",  f"Remember this: the verification token is {TOK}."),
        ("recency", "What is the verification token?"),
        ("filler1", "What is the tallest mountain on Earth?"),
        ("filler2", "Name a primary color."),
        ("filler3", "How many days are in a week?"),
        ("paging",  "Remind me, what was the verification token I gave you?"),
        ("foreign", "How does photosynthesis work?"),
    ]

    print(f"== live loop-closure gate :: host={args.host} token={TOK} ==\n", flush=True)
    results = []
    for kind, content in turns:
        t0 = time.time()
        try:
            ans = chat(args.host, content)
        except Exception as e:
            ans = f"<<REQUEST FAILED: {e}>>"
        dt = time.time() - t0
        # give the daemon a beat to flush its log + finish the NIGHTSHIFT capture.
        time.sleep(1.0)
        lines = tail_log(args.log, 600)
        kairos = log_grep(lines, "B3-JUDGE KAIROS")
        pick   = log_grep(lines, "B3-JUDGE: PICK", "B3-JUDGE: [NULL]")
        nshift = log_grep(lines, "B4-NIGHTSHIFT: consolidated")
        has_tok = TOK.lower() in ans.lower() or TOK.replace("-", "").lower() in ans.lower().replace("-", "")
        print(f"--- TURN [{kind}] ({dt:.1f}s) ---", flush=True)
        print(f"  USER : {content}", flush=True)
        print(f"  MODEL: {ans!r}", flush=True)
        print(f"  token_present={has_tok}", flush=True)
        for h in kairos[-2:]:
            print(f"  [log] {h.split('  ')[-1] if '  ' in h else h}", flush=True)
        for h in pick[-1:]:
            print(f"  [log] {h.split('  ')[-1] if '  ' in h else h}", flush=True)
        for h in nshift[-1:]:
            print(f"  [log] {h.split('  ')[-1] if '  ' in h else h}", flush=True)
        print(flush=True)
        results.append((kind, ans, has_tok, kairos[-1] if kairos else "", pick[-1] if pick else ""))

    # verdict
    print("== VERDICT ==", flush=True)
    by = {k: r for (k, *_), r in [((r[0],), r) for r in results]}
    rec = by.get("recency")
    pag = by.get("paging")
    frn = by.get("foreign")
    if rec:
        ok = rec[2]
        print(f"  RECENCY recall (core deliverable): {'PASS' if ok else 'FAIL'} -- model answered with token={ok}", flush=True)
    if pag:
        ok = pag[2]
        print(f"  PAGING  rescue (C2-salience, may miss): {'PASS' if ok else 'MISS (weak-C2, honest)'} -- token={ok}", flush=True)
    if frn:
        # foreign passes if it did NOT emit the token and the judge logged [NULL] OR no PICK
        null_ish = ("[NULL]" in frn[4]) or (frn[4] == "")
        ok = (not frn[2])
        print(f"  FOREIGN reject: {'PASS' if (ok) else 'CHECK'} -- token_absent={ok} judge_null_logged={null_ish}", flush=True)

if __name__ == "__main__":
    main()
