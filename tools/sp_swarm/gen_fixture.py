"""gen_fixture.py — emit a cross-language provenance fixture for the Rust parity gate.

Signs a REAL MEM-OKF object with pynacl (libsodium) over the exact addr||body payload, using a
DETERMINISTIC seed, and writes tests/fixtures/prov_fixture.json. The Rust test (ed25519-dalek +
sha2) must reproduce the address, reconstruct the identical payload, and verify this signature —
proving the Rust port is byte-for-byte wire-compatible with the proven Python prototype.

Run from tools/sp_swarm:  python gen_fixture.py <path-to-lattice-repo>
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
LAT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "shannon-prime-lattice")
sys.path.insert(0, os.path.join(LAT, "tools"))
import okf_mem, swarm_provenance as prov
import nacl.signing, nacl.encoding

MEMOKF = os.path.join(LAT, "memory-okf")
# pick the first CONTENT-addressed object (re-hashable) for a clean parity anchor
full = os.path.join(MEMOKF, "full")
pick = None
for fn in sorted(os.listdir(full)):
    if not fn.endswith(".md"):
        continue
    addr = fn[:-3]
    _fm, body = okf_mem.parse_fm(okf_mem.read(os.path.join(full, fn)))
    if okf_mem.addr_of(body) == addr:
        pick = (addr, body); break
assert pick, "no content-addressed object found"
addr, body = pick

seed = bytes(range(32))                       # deterministic fixture identity
sk = nacl.signing.SigningKey(seed)
vk_hex = sk.verify_key.encode(encoder=nacl.encoding.HexEncoder).decode()
payload = prov.signing_payload(addr, body)    # (addr+"\n") + norm(body)
sig_hex = sk.sign(payload).signature.hex()

fix = {
    "node_id": "node-fixture",
    "pub_hex": vk_hex,
    "addr": addr,
    "body": body,
    "payload_hex": payload.hex(),
    "sig_hex": sig_hex,
}
out = os.path.join(HERE, "tests", "fixtures")
os.makedirs(out, exist_ok=True)
json.dump(fix, open(os.path.join(out, "prov_fixture.json"), "w", encoding="utf-8"), indent=2)
print(f"wrote prov_fixture.json: addr={addr} pub={vk_hex[:16]}.. sig={sig_hex[:16]}.. body_len={len(body)}")
