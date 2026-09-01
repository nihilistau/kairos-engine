"""g_c2_semantic.py — G-SWARM-C2-SEMANTIC: does a C2-style SimHash preserve L5's similarity?

The L4 overlay only earns its network surface if C2-Hamming near-neighbor recall tracks the
proven L5-cosine signal (G-REP-LAYER-L5: 88.5% paraphrase). This measures, on the faithful
corpus: paraphrase-query L5 embed -> nearest episode by (a) L5 cosine [ground truth] vs (b)
256-bit SimHash Hamming [the C2 overlay]. If SimHash retains ~the cosine recall, discovery is
viable; if it collapses, it's an honest negative and the mesh stays exact-fetch only.

Usage: python g_c2_semantic.py <faithful_corpus_dir>
"""
import struct, glob, os, sys
import numpy as np
L5, HD, G_NH = 5, 512, 16
CORP = sys.argv[1] if len(sys.argv) > 1 else "_faithful_corpus"

def l5_embed(path):
    b = open(path, "rb").read()
    ng, qd = struct.unpack("<II", b[:8])
    a = np.frombuffer(b[8:], "<f4").astype(np.float64).reshape(ng, G_NH, HD)
    v = a[L5].mean(0)                      # global layer 5, mean over heads (== write_ep_l5)
    return v / (np.linalg.norm(v) + 1e-30)

# episodes: ep.l5 keys (already L2-normed), in fact order
n = len(glob.glob(os.path.join(CORP, "eps", "fct_*", "ep.l5")))
E = np.array([np.frombuffer(open(os.path.join(CORP, "eps", f"fct_{i:03d}", "ep.l5"), "rb").read(), "<f4").astype(np.float64) for i in range(n)])
E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-30)
# paraphrase queries, sorted by chat id == fact order
qs = sorted(glob.glob(os.path.join(CORP, "qdump_para", "q_*.bin")), key=lambda p: int(os.path.basename(p)[2:-4]))
Q = np.array([l5_embed(p) for p in qs[:n]])
print(f"episodes={len(E)} para-queries={len(Q)} dim={HD}")

def recall_at(score, k):  # higher score = more similar; diagonal is the correct match
    order = np.argsort(-score, axis=1)
    return np.mean([i in order[i, :k] for i in range(len(score))])

# (a) ground truth: L5 cosine
cos = Q @ E.T
c1, c5 = recall_at(cos, 1), recall_at(cos, 5)

# (b) the C2 overlay: 256-bit SimHash (sign of a ±1 Rademacher projection == C2 construction)
rng = np.random.default_rng(0xC2C2)
def simhash_recall(bits):
    R = rng.choice([-1.0, 1.0], size=(bits, HD))
    Eb = (E @ R.T) >= 0
    Qb = (Q @ R.T) >= 0
    agr = bits - (Qb[:, None, :] != Eb[None, :, :]).sum(2)   # agreement (higher=better)
    return recall_at(agr, 1), recall_at(agr, 5)
s1_256, s5_256 = simhash_recall(256)

print(f"L5-cosine   recall@1={c1:.3f}  recall@5={c5:.3f}   [ground truth]")
print(f"SimHash-256 recall@1={s1_256:.3f}  recall@5={s5_256:.3f}   [the C2 overlay]")
for rb in (512, 1024, 2048):
    r1, r5 = simhash_recall(rb)
    print(f"SimHash-{rb:<4} recall@1={r1:.3f}  recall@5={r5:.3f}")

# verdict: the overlay is viable if 256-bit SimHash retains >=90% of the cosine recall@1
retain = s1_256 / c1 if c1 else 0.0
print(f"\nretention (SimHash256 recall@1 / cosine recall@1) = {retain:.2f}")
print("==== G-SWARM-C2-SEMANTIC:", "GREEN" if retain >= 0.90 else "HONEST-NEGATIVE",
      f"(C2-256 retains {retain:.0%} of L5-cosine near-neighbor recall) ====")
