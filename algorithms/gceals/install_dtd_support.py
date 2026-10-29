"""One-time, reversible G-CEALS entry-point patch for local DTD features.

Place this file alongside gceals.py and dtd_loader.py in the utils directory.
Run: python install_dtd_support.py

This script validates its edit markers before changing anything and writes
``gceals.py.bak_dtd``. It deliberately leaves the existing OpenML path intact.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GCEALS = ROOT / "gceals.py"
LOADER = ROOT / "utils" / "dtd_loader.py"
BACKUP = ROOT / "gceals.py.bak_dtd"

ARG_LINES = """parser.add_argument('--features', type=str, default=None,
                    help='Local DTD feature archive (.npz, .npy, or a directory)')
parser.add_argument('--layers', type=str, default=None,
                    help='Comma-separated ResNet activation keys; default: all')
parser.add_argument('--labels-file', type=str, default=None,
                    help='Optional labels CSV/NPY/NPZ for local features')
parser.add_argument('--ids-file', type=str, default=None,
                    help='Optional feature image IDs .npy for matching CSV labels')
parser.add_argument('--pool', choices=['avg', 'max', 'flatten'], default='avg')
parser.add_argument('--layout', choices=['nchw', 'nhwc'], default='nchw')
parser.add_argument('--assume-order', action='store_true',
                    help='Allow positional label join without feature IDs')"""

DATA_LINES = """if args.features is not None:
    from pathlib import Path
    import hashlib
    import re
    from utils.dtd_loader import load_dtd_features

    wanted = [x.strip() for x in args.layers.split(',')] if args.layers else None
    features = load_dtd_features(
        args.features, layers=wanted,
        labels_file=args.labels_file, ids_file=args.ids_file,
        pool=args.pool, layout=args.layout, assume_order=args.assume_order)
    X_actual, y_actual = features.X, features.y
    # Cache identifiers are distinct for each feature file/layer combination.
    source = Path(args.features).resolve()
    digest = hashlib.sha256(str(source).encode('utf-8')).hexdigest()[:8]
    label = source.stem + '_' + '_'.join(features.layers) + '_' + digest
    args.dataset = 'dtd_' + re.sub(r'[^A-Za-z0-9_-]+', '_', label)[:110]
    print('Loaded DTD features:', X_actual.shape,
          'classes:', len(features.class_names), 'layers:', features.layer_dimensions)
else:
    X_actual, y_actual = get_data(args.dataset)"""


def _replace_line_once(text: str, match: str, replacement: str) -> str:
    pattern = re.compile(r"(?m)^([ \t]*)" + re.escape(match) + r"[ \t]*$")
    found = list(pattern.finditer(text))
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one line '{match}', found {len(found)}. No files changed.")
    result = found[0]
    indent = result.group(1)
    formatted = "\n".join(indent + line if line else "" for line in replacement.splitlines())
    return text[:result.start()] + formatted + text[result.end():]


def main() -> None:
    if not GCEALS.is_file():
        raise SystemExit(f"Cannot find {GCEALS}. Place this script beside gceals.py.")
    if not LOADER.is_file():
        raise SystemExit(f"Cannot find {LOADER}. Place dtd_loader.py inside utils/ first.")
    original = GCEALS.read_text(encoding="utf-8")
    if "--features" in original and "load_dtd_features" in original:
        print("DTD support appears to be installed already; no changes made.")
        return
    try:
        updated = _replace_line_once(original, "args = parser.parse_args()", ARG_LINES + "\nargs = parser.parse_args()")
        updated = _replace_line_once(updated, "X_actual, y_actual = get_data(args.dataset)", DATA_LINES)
        compile(updated, str(GCEALS), "exec")
    except (RuntimeError, SyntaxError) as exc:
        raise SystemExit(f"Patch validation failed: {exc}") from exc
    if BACKUP.exists():
        raise SystemExit(f"Backup {BACKUP} already exists; refusing to overwrite it.")
    shutil.copy2(GCEALS, BACKUP)
    GCEALS.write_text(updated, encoding="utf-8")
    print(f"Patched {GCEALS}")
    print(f"Original backed up to {BACKUP}")
    print("Use python gceals.py --help to see --features and --layers options.")


if __name__ == "__main__":
    main()
