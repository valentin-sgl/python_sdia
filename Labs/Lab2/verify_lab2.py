import json
import os
import sys
from pathlib import Path

nb_path = Path(r"c:\Users\valen\Documents\Python\python_sdia\Labs\Lab2\lab2.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
ns = {}
errors = []

os.chdir(str(nb_path.parent))

for i, cell in enumerate(nb.get("cells", []), 1):
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") != "code" or not src.strip():
        continue
    if src.strip().startswith("%") or src.strip().startswith("!"):
        continue
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns, ns)
    except Exception as exc:  # pragma: no cover
        errors.append((i, type(exc).__name__, str(exc)))

print(f"EXECUTED_CODE_CELLS: {len([1 for c in nb.get('cells', []) if c.get('cell_type') == 'code' and ''.join(c.get('source', [])).strip() and not ''.join(c.get('source', [])).strip().startswith('%') and not ''.join(c.get('source', [])).strip().startswith('!')])}")
if errors:
    print("ERRORS:")
    for item in errors:
        print(item)
    raise SystemExit(1)

try:
    import tests_exo3
except Exception as exc:
    print(f"IMPORT_TESTS_ERROR: {type(exc).__name__}: {exc}")
    raise

tests_exo3.test_gradient2D(ns["gradient2D"])
tests_exo3.test_tv(ns["tv"])
tests_exo3.test_gradient2D_adjoint(ns["gradient2D"], ns["gradient2D_adjoint"])
print("ALL_TESTS_PASSED")
