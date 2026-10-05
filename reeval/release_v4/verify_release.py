"""
verify_release.py -- check an extracted v4 overlay against RELEASE_MANIFEST_V4.json.
Run from the repository root after unzipping:  python reeval/release_v4/verify_release.py
Checks: every released file present with the recorded sha256; every prerequisite baseline
file present with the sha256 of the base commit (i.e. not modified). Exit 1 on any mismatch.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAN = os.path.join(ROOT, "reeval", "release_v4", "RELEASE_MANIFEST_V4.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    m = json.load(open(MAN))
    bad = []
    for rel, s in m["files"].items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            bad.append(f"missing {rel}")
        elif sha(p) != s:
            bad.append(f"sha mismatch {rel}")
    for rel, s in m["baseline_prerequisites"].items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            bad.append(f"missing baseline {rel}")
        elif sha(p) != s:
            bad.append(f"baseline modified {rel}")
    print(f"release {m['version']} on base {m['base_commit'][:12]}: "
          f"{len(m['files'])} files, {len(m['baseline_prerequisites'])} baseline prerequisites")
    if bad:
        print("FAIL:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("PASS: all files match the manifest; baseline files unchanged")


if __name__ == "__main__":
    main()
