//! G-SWARM-RUST-PARITY — proves the Rust sp-swarm core is byte-for-byte interoperable with the
//! proven Python prototype (a pynacl-signed fixture), plus the Rust-native provenance rejects.
//!
//! (1) address parity: Rust sha2 reproduces Python's sha256(norm(body))[..16].
//! (2) payload parity: Rust signing_payload == the exact bytes Python signed.
//! (3) crypto parity: ed25519-dalek verifies the libsodium/pynacl signature.
//! (4) tamper: a flipped payload fails verification.
//! (5) roster rejects: unsigned / untrusted-signer / sig-invalid via verify_provenance.
//! (6) (optional) real-store integrity: every content object under $SP_SWARM_MEMOKF re-hashes.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::collections::{BTreeMap, HashMap};

fn fixture() -> serde_json::Value {
    let p = format!("{}/tests/fixtures/prov_fixture.json", env!("CARGO_MANIFEST_DIR"));
    let s = std::fs::read_to_string(&p).expect("run gen_fixture.py first to create prov_fixture.json");
    serde_json::from_str(&s).unwrap()
}
fn vk_from_hex(h: &str) -> VerifyingKey {
    let b: [u8; 32] = hex::decode(h).unwrap().try_into().unwrap();
    VerifyingKey::from_bytes(&b).unwrap()
}

#[test]
fn address_parity() {
    let f = fixture();
    let (addr, body) = (f["addr"].as_str().unwrap(), f["body"].as_str().unwrap());
    assert_eq!(sp_swarm::addr_of(body), addr, "Rust sha256/norm must reproduce Python address");
}

#[test]
fn payload_parity() {
    let f = fixture();
    let (addr, body) = (f["addr"].as_str().unwrap(), f["body"].as_str().unwrap());
    let got = hex::encode(sp_swarm::signing_payload(addr, body));
    assert_eq!(got, f["payload_hex"].as_str().unwrap(), "signing payload bytes must match Python");
}

#[test]
fn crypto_parity_verifies_pynacl_signature() {
    let f = fixture();
    let (addr, body) = (f["addr"].as_str().unwrap(), f["body"].as_str().unwrap());
    let vk = vk_from_hex(f["pub_hex"].as_str().unwrap());
    let sig = Signature::from_slice(&hex::decode(f["sig_hex"].as_str().unwrap()).unwrap()).unwrap();
    // ed25519-dalek must verify a signature produced by libsodium/pynacl over our payload.
    vk.verify(&sp_swarm::signing_payload(addr, body), &sig)
        .expect("dalek must verify the pynacl signature (cross-lang crypto parity)");
}

#[test]
fn tampered_payload_fails() {
    let f = fixture();
    let (addr, body) = (f["addr"].as_str().unwrap(), f["body"].as_str().unwrap());
    let vk = vk_from_hex(f["pub_hex"].as_str().unwrap());
    let sig = Signature::from_slice(&hex::decode(f["sig_hex"].as_str().unwrap()).unwrap()).unwrap();
    let mut p = sp_swarm::signing_payload(addr, body);
    let n = p.len(); p[n - 1] ^= 0x01; // flip a byte
    assert!(vk.verify(&p, &sig).is_err(), "tampered payload must NOT verify");
}

#[test]
fn roster_rejects() {
    let f = fixture();
    let (addr, body) = (f["addr"].as_str().unwrap(), f["body"].as_str().unwrap());
    let node = f["node_id"].as_str().unwrap();
    let sig_hex = f["sig_hex"].as_str().unwrap();
    let mut roster: HashMap<String, VerifyingKey> = HashMap::new();
    roster.insert(node.to_string(), vk_from_hex(f["pub_hex"].as_str().unwrap()));

    // valid signed frontmatter -> Ok
    let mut fm = BTreeMap::new();
    fm.insert("mem_signer".to_string(), node.to_string());
    fm.insert("mem_sig".to_string(), sig_hex.to_string());
    assert!(sp_swarm::verify_provenance(addr, &fm, body, &roster).is_ok());

    // unsigned -> Unsigned
    assert_eq!(sp_swarm::verify_provenance(addr, &BTreeMap::new(), body, &roster),
               Err(sp_swarm::Reject::Unsigned));

    // untrusted signer (not in roster) -> UntrustedSigner
    let mut fm_u = fm.clone(); fm_u.insert("mem_signer".to_string(), "node-UNKNOWN".to_string());
    assert_eq!(sp_swarm::verify_provenance(addr, &fm_u, body, &roster),
               Err(sp_swarm::Reject::UntrustedSigner));

    // forged signature -> SigInvalid
    let mut fm_f = fm.clone(); fm_f.insert("mem_sig".to_string(), "00".repeat(64));
    assert_eq!(sp_swarm::verify_provenance(addr, &fm_f, body, &roster),
               Err(sp_swarm::Reject::SigInvalid));
}

#[test]
fn real_store_content_objects_rehash() {
    let root = match std::env::var("SP_SWARM_MEMOKF") { Ok(v) => v, Err(_) => return }; // opt-in
    let full = std::path::Path::new(&root).join("full");
    let (mut content, mut c2) = (0u32, 0u32);
    let mut bad: Vec<String> = Vec::new();
    for e in std::fs::read_dir(&full).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let addr = match name.strip_suffix(".md") { Some(a) => a.to_string(), None => continue };
        let text = std::fs::read_to_string(e.path()).unwrap();
        let (fm, body) = sp_swarm::parse_fm(&text);
        match sp_swarm::classify(&addr, &fm, &body) {
            sp_swarm::Class::Content => content += 1,
            sp_swarm::Class::C2 => c2 += 1,
            sp_swarm::Class::Bad => bad.push(addr),
        }
    }
    eprintln!("real store: content={content} c2={c2} invalid={} {:?}", bad.len(), &bad[..bad.len().min(6)]);
    assert!(content > 0, "expected content-addressed objects");
    assert!(bad.is_empty(), "Rust classified {} objects Bad that Python accepts (norm/parse divergence): {:?}",
            bad.len(), &bad[..bad.len().min(6)]);
}
