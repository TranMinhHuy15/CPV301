"""
RQ1b -- minimal fixes that the public RiskProp code needs before it can run.

The public repository (github.com/xingyueye5/RiskProp, commit 579376f) does not
import as released:
  1. taa/models.py has a syntax error: a comma is missing after
     `loss_cls=loss_cls * loss_cls_weight` in the second (non-"constraint")
     branch of AnticipationHead.loss_by_feat. That branch is NOT used by the
     RiskProp config (label_with="constraint"), but the syntax error stops the
     whole file from importing.
  2. taa/__init__.py does not import taa/models.py (the line is commented out),
     so AnticipationHead is never registered and the config cannot be built.
  3. taa/__init__.py imports taa/model_AdaLEA.py, which is not in the repository.

This script applies exactly these three fixes and nothing else. It refuses to
run if the files do not look as expected, and writes the diff next to itself
so the change is on record.

Usage:
    python riskprop_full/patch_authors_code.py /workspace/RiskProp
"""
import ast
import difflib
import os
import re
import sys

EXPECTED_COMMIT = "579376f"


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python patch_authors_code.py <path to RiskProp repo>")
    repo = os.path.abspath(sys.argv[1])
    models_p = os.path.join(repo, "taa", "models.py")
    init_p = os.path.join(repo, "taa", "__init__.py")
    for p in (models_p, init_p):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")

    head = os.popen(f"git -C {repo} rev-parse --short HEAD 2>/dev/null").read().strip()
    print(f"RiskProp repo at {repo}, commit {head or '?'}")
    if head and not head.startswith(EXPECTED_COMMIT):
        sys.exit(f"expected commit {EXPECTED_COMMIT}, found {head} -- check out {EXPECTED_COMMIT} first")

    diffs = []

    # ---- fix 1: missing comma in models.py ----
    src = open(models_p).read()
    pat = re.compile(r"(loss_cls=loss_cls \* loss_cls_weight)(\n[ \t]+loss_ffr=)")
    n_missing = len(pat.findall(src))
    if n_missing == 0:
        try:
            ast.parse(src)
            print("[fix 1] models.py already parses -- nothing to do")
            new_src = src
        except SyntaxError as e:
            sys.exit(f"[fix 1] unexpected syntax error in models.py: {e}")
    elif n_missing == 1:
        new_src = pat.sub(r"\1,\2", src)
        ast.parse(new_src)  # must parse now
        print("[fix 1] added the missing comma in AnticipationHead.loss_by_feat")
    else:
        sys.exit(f"[fix 1] expected 1 missing comma, found {n_missing} -- stop")
    if new_src != src:
        diffs += difflib.unified_diff(src.splitlines(True), new_src.splitlines(True),
                                      "a/taa/models.py", "b/taa/models.py")
        open(models_p, "w").write(new_src)

    # ---- fix 2 + 3: taa/__init__.py ----
    src = open(init_p).read()
    lines = src.splitlines(True)
    out = []
    changed = False
    for ln in lines:
        s = ln.strip()
        if s == "# from .models import *":
            out.append("from .models import *  # RQ1b fix 2: was commented out\n")
            changed = True
        elif s == "from .model_AdaLEA import *":
            out.append("# from .model_AdaLEA import *  # RQ1b fix 3: module not in repo\n")
            changed = True
        else:
            out.append(ln)
    new_src = "".join(out)
    if "from .models import *  # RQ1b" not in new_src and "\nfrom .models import *" not in "\n" + new_src:
        sys.exit("[fix 2] could not find the models import line in taa/__init__.py -- stop")
    if changed:
        diffs += difflib.unified_diff(src.splitlines(True), new_src.splitlines(True),
                                      "a/taa/__init__.py", "b/taa/__init__.py")
        open(init_p, "w").write(new_src)
        print("[fix 2] taa/__init__.py now imports taa.models")
        print("[fix 3] taa/__init__.py no longer imports the missing taa.model_AdaLEA")
    else:
        print("[fix 2/3] taa/__init__.py already patched -- nothing to do")

    # ---- check every taa file parses ----
    for f in sorted(os.listdir(os.path.join(repo, "taa"))):
        if f.endswith(".py"):
            ast.parse(open(os.path.join(repo, "taa", f)).read())
    print("All taa/*.py files parse.")

    if diffs:
        out_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "authors_code_fixes.diff")
        with open(out_p, "w") as f:
            f.writelines(diffs)
        print(f"Diff written to {out_p}")


if __name__ == "__main__":
    main()
