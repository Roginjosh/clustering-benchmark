"""Load ResNet activation features from DTD for G-CEALS (no OpenML dependency).

Supports a single .npz archive with one or more named activation arrays, a
single .npy matrix, or a directory of named .npy activation arrays.

An activation may be [N, D] or [N, C, H, W] (PyTorch convention). By default,
spatial dimensions are globally average-pooled into channels before stacking.
Labels are for *evaluation*, not an input feature. The returned feature matrix
is deliberately NOT standardized: gceals.py applies StandardScaler already.

Use ``python utils/dtd_loader.py --help`` to inspect the command-line options.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_LABEL_KEYS = ("y", "labels", "label", "targets", "target", "classes", "class_labels")
_ID_KEYS = ("ids", "image_ids", "image_id", "filenames", "filename", "image_paths")
_RESERVED = set(_LABEL_KEYS) | set(_ID_KEYS) | {"layer_keys", "class_names"}


@dataclass
class DTDFeatures:
    X: np.ndarray                 # [N, total_features], float32, unscaled
    y: np.ndarray                 # [N], int64, consecutive label codes
    ids: np.ndarray | None        # [N], strings, if provided
    class_names: list[str]        # class_names[y[i]] is the original label
    layers: list[str]             # selected layer keys in requested order
    layer_dimensions: dict[str, int]


def _read_numpy(path: Path):
    """Disallow pickle to avoid executing code from untrusted feature exports."""
    try:
        return np.load(path, allow_pickle=False, mmap_mode="r" if path.suffix.lower() == ".npy" else None)
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                f"{path} contains a pickle-dependent object array. "
                "Re-export labels/IDs as Unicode string arrays (dtype=str)."
            ) from exc
        raise


def _find_first(keys: Sequence[str], candidates: Sequence[str]) -> str | None:
    return next((key for key in candidates if key in keys), None)


def _as_vector(arr, name: str) -> np.ndarray:
    result = np.asarray(arr)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got {result.shape}")
    return result


def _as_ids(raw, name: str) -> np.ndarray:
    v = _as_vector(raw, name)
    result = np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value)
                         for value in v], dtype=str)
    if len(set(result.tolist())) != len(result):
        raise ValueError(f"Duplicate image identifiers found in {name}.")
    return result


def _read_label_csv(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        label_key = _find_first(columns, ("label", "class", "target", "y", "category"))
        id_key = _find_first(columns, ("image_id", "id", "filename", "image", "path"))
        if not label_key:
            raise ValueError(f"No label column in {path}. Expected 'label', 'class', or 'target'.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows in {path}.")
    labels = np.asarray([row[label_key] for row in rows], dtype=str)
    ids = _as_ids([row[id_key] for row in rows], str(path)) if id_key else None
    return labels, ids


def _read_labels(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix.lower() == ".csv":
        return _read_label_csv(path)
    if path.suffix.lower() == ".npy":
        return _as_vector(_read_numpy(path), "labels"), None
    if path.suffix.lower() == ".npz":
        with _read_numpy(path) as data:
            k = _find_first(data.files, _LABEL_KEYS)
            if k is None:
                raise ValueError(f"No labels/y key in {path}.")
            ik = _find_first(data.files, _ID_KEYS)
            labels = _as_vector(data[k], k).copy()
            ids = _as_ids(data[ik], ik) if ik else None
        return labels, ids
    raise ValueError("Labels must be a .csv, .npy, or .npz file.")


def _pool_features(array, key: str, pool: str, layout: str) -> np.ndarray:
    a = np.asarray(array)
    if a.ndim < 2:
        raise ValueError(f"{key} must have [N,D] or [N,C,...] dimensions; got {a.shape}.")
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"Feature array {key} must be numeric, got {a.dtype}.")
    if a.ndim == 2:
        output = a
    elif pool == "avg":
        if layout == "nchw":
            output = a.mean(axis=tuple(range(2, a.ndim)), dtype=np.float64)
        else:  # NHWC: channels are the final dimension
            output = a.mean(axis=tuple(range(1, a.ndim - 1)), dtype=np.float64)
    elif pool == "max":
        axes = tuple(range(2, a.ndim)) if layout == "nchw" else tuple(range(1, a.ndim - 1))
        output = a.max(axis=axes)
    elif pool == "flatten":
        output = a.reshape(a.shape[0], -1)
    else:
        raise ValueError("pool must be 'avg', 'max', or 'flatten'.")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"Non-finite values (NaN or infinity) found in layer {key}.")
    return np.asarray(output, dtype=np.float32)


def load_dtd_features(
    source: str | Path,
    *,
    layers: Sequence[str] | None = None,
    labels_file: str | Path | None = None,
    ids_file: str | Path | None = None,
    pool: str = "avg",
    layout: str = "nchw",
    assume_order: bool = False,
) -> DTDFeatures:
    """Return an unscaled feature matrix and numeric labels for G-CEALS.

    ``source`` is a .npz, a .npy, or a directory containing ``<layer>.npy``
    files and optional ``labels.npy`` / ``ids.npy``. A .npz may contain
    named layer arrays plus ``y`` or ``labels`` and ``ids``.

    Layer arrays must share row order and sample count. To match CSV labels by
    image ID, feature IDs must be provided in the archive/directory or via
    ``ids_file``. With ``assume_order=True``, rows may be joined positionally
    even if the CSV includes image IDs (only use after verifying ordering).
    """
    source = Path(source).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    if pool not in {"avg", "max", "flatten"}:
        raise ValueError("pool must be avg, max, or flatten")
    if layout not in {"nchw", "nhwc"}:
        raise ValueError("layout must be nchw or nhwc")

    external_labels = None
    external_label_ids = None
    if labels_file is not None:
        external_labels, external_label_ids = _read_labels(Path(labels_file).expanduser())

    arrays: dict[str, np.ndarray] = {}
    labels = None
    feature_ids = None
    if source.is_dir():
        npys = {p.stem: p for p in sorted(source.glob("*.npy"))}
        for name, path in npys.items():
            if name in _LABEL_KEYS:
                if labels is None:
                    labels = _as_vector(_read_numpy(path), name)
            elif name in _ID_KEYS:
                if feature_ids is None:
                    feature_ids = _as_ids(_read_numpy(path), name)
            else:
                arrays[name] = _read_numpy(path)
    elif source.suffix.lower() == ".npz":
        with _read_numpy(source) as data:
            for name in data.files:
                if name in _LABEL_KEYS and labels is None:
                    labels = _as_vector(data[name], name).copy()
                elif name in _ID_KEYS and feature_ids is None:
                    feature_ids = _as_ids(data[name], name)
                elif name not in _RESERVED:
                    arrays[name] = np.asarray(data[name])
    elif source.suffix.lower() == ".npy":
        arrays["X"] = _read_numpy(source)
    else:
        raise ValueError("Feature source must be a .npz, .npy, or directory of .npy files.")

    if ids_file is not None:
        if feature_ids is not None:
            raise ValueError("Feature IDs are already embedded; remove ids_file.")
        id_path = Path(ids_file).expanduser()
        if id_path.suffix.lower() != ".npy":
            raise ValueError("ids_file must be a .npy Unicode/string array.")
        feature_ids = _as_ids(_read_numpy(id_path), str(id_path))

    if not arrays:
        raise ValueError(f"No feature arrays found in {source}.")
    selected = list(layers) if layers else sorted(arrays)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one layer and do not repeat layer names.")
    missing = [k for k in selected if k not in arrays]
    if missing:
        raise ValueError(f"Unknown layer(s): {missing}. Available: {sorted(arrays)}")

    matrices = []
    dimensions = {}
    n_samples = None
    for name in selected:
        m = _pool_features(arrays[name], name, pool, layout)
        if n_samples is None:
            n_samples = len(m)
        elif len(m) != n_samples:
            raise ValueError(f"Layer {name} has {len(m)} rows; expected {n_samples}.")
        matrices.append(m)
        dimensions[name] = m.shape[1]
    if n_samples is None or n_samples < 2:
        raise ValueError("At least two samples are needed.")
    if feature_ids is not None and len(feature_ids) != n_samples:
        raise ValueError("Number of feature IDs does not match feature rows.")

    if external_labels is not None:
        labels = external_labels
        if external_label_ids is not None:
            if feature_ids is None:
                if not assume_order:
                    raise ValueError(
                        "Labels CSV contains image IDs but feature arrays have none. "
                        "Provide matching feature IDs or explicitly pass assume_order=True."
                    )
            else:
                label_lookup = {key: idx for idx, key in enumerate(external_label_ids)}
                if len(label_lookup) != len(external_label_ids):
                    raise ValueError("Duplicate image IDs in labels file.")
                missing_ids = [key for key in feature_ids if key not in label_lookup]
                if missing_ids:
                    raise ValueError(f"Missing label IDs for examples: {missing_ids[:5]}")
                labels = labels[[label_lookup[key] for key in feature_ids]]

    if labels is None:
        raise ValueError("Labels missing: supply 'y'/'labels' in archive, labels.npy, or --labels-file.")
    labels = _as_vector(labels, "labels")
    if len(labels) != n_samples:
        raise ValueError(f"Found {len(labels)} labels but {n_samples} samples.")
    if any(v is None or str(v).strip().lower() in {"", "nan", "none"} for v in labels):
        raise ValueError("Labels contain missing or empty entries.")

    # Preserve numeric ordering if original labels are numeric; otherwise sort class names.
    try:
        classes, encoded = np.unique(labels, return_inverse=True)
    except TypeError:
        classes, encoded = np.unique(labels.astype(str), return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Expected at least two classes for evaluation.")
    X = np.concatenate(matrices, axis=1).astype(np.float32, copy=False)
    return DTDFeatures(X=X, y=encoded.astype(np.int64), ids=feature_ids,
                       class_names=[str(c) for c in classes],
                       layers=selected, layer_dimensions=dimensions)


def inspect_source(source: str | Path) -> None:
    """Print available keys, shapes and dtypes without performing training."""
    source = Path(source).expanduser()
    if source.is_dir():
        items = [(p.stem, _read_numpy(p)) for p in sorted(source.glob("*.npy"))]
    elif source.suffix.lower() == ".npz":
        with _read_numpy(source) as f:
            items = [(key, f[key]) for key in f.files]
    elif source.suffix.lower() == ".npy":
        items = [("X", _read_numpy(source))]
    else:
        raise ValueError("Source must be .npy, .npz, or a directory of .npy files")
    if not items:
        raise ValueError(f"No arrays in {source}")
    for name, value in items:
        print(f"{name:<24} shape={str(value.shape):<24} dtype={value.dtype}")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect and prepare DTD ResNet features for G-CEALS")
    p.add_argument("--source", required=True, help="Features .npz, .npy, or a directory of layer .npy files")
    p.add_argument("--inspect", action="store_true", help="Print keys/shapes without loading training data")
    p.add_argument("--layers", help="Comma-separated layer names in desired order; default: all")
    p.add_argument("--labels-file", help="Optional labels .csv, .npy, or .npz")
    p.add_argument("--ids-file", help="Optional feature IDs .npy for matching a labels CSV")
    p.add_argument("--pool", choices=("avg", "max", "flatten"), default="avg")
    p.add_argument("--layout", choices=("nchw", "nhwc"), default="nchw")
    p.add_argument("--assume-order", action="store_true", help="Trust CSV row ordering without feature IDs")
    p.add_argument("--save", help="Optional prepared .npz path (X, y, ids, layer_keys, class_names)")
    args = p.parse_args()
    if args.inspect:
        inspect_source(args.source)
        return
    layer_names = [x.strip() for x in args.layers.split(",")] if args.layers else None
    result = load_dtd_features(args.source, layers=layer_names, labels_file=args.labels_file,
                               ids_file=args.ids_file, pool=args.pool, layout=args.layout,
                               assume_order=args.assume_order)
    print(f"X: {result.X.shape}, {result.X.dtype}; y: {result.y.shape}, classes: {len(result.class_names)}")
    print("Layers (after pooling):", result.layer_dimensions)
    print("Class counts:", dict(zip(*np.unique(result.y, return_counts=True))))
    print("Image IDs:", "present" if result.ids is not None else "absent (row order assumed)")
    if args.save:
        output = Path(args.save).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(X=result.X, y=result.y, layer_keys=np.asarray(result.layers, dtype=str),
                      class_names=np.asarray(result.class_names, dtype=str))
        if result.ids is not None:
            kwargs["ids"] = result.ids
        np.savez_compressed(output, **kwargs)
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
