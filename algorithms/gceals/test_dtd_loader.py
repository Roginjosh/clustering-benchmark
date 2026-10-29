from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import numpy as np
from dtd_loader import load_dtd_features, inspect_source

rng = np.random.default_rng(7)
with TemporaryDirectory() as tmp:
    base = Path(tmp)
    n = 24
    ids = np.array([f"image-{i:03d}.jpg" for i in range(n)])
    labels = np.array(["banded", "fibrous", "wrinkled", "waffled"] * 6)
    stages = {"stem": rng.random((n, 64, 8, 8)), "layer2": rng.random((n, 512, 4, 4)), "fc": rng.random((n, 1000))}
    np.savez(base/'features.npz', **stages, labels=labels, ids=ids)
    out = load_dtd_features(base/'features.npz', layers=['stem','fc'])
    assert out.X.shape == (n, 1064) and out.y.shape == (n,) and len(out.class_names) == 4
    assert np.allclose(out.X[:,:64], stages['stem'].mean(axis=(2,3)), atol=1e-6)
    assert (out.ids == ids).all()
    np.savez_compressed(base/'prepared.npz', X=out.X, y=out.y, ids=out.ids,
                        layer_keys=np.asarray(out.layers), class_names=np.asarray(out.class_names))
    out2 = load_dtd_features(base/'prepared.npz')
    assert out2.X.shape == out.X.shape and out2.y.shape == out.y.shape

    stage_dir = base/'layers'; stage_dir.mkdir()
    for name, m in stages.items(): np.save(stage_dir/f'{name}.npy', m)
    np.save(stage_dir/'ids.npy', ids)
    csvpath=base/'labels.csv'
    with csvpath.open('w',newline='') as f:
        wr=csv.DictWriter(f, fieldnames=['image_id','label']);wr.writeheader()
        for i in reversed(range(n)): wr.writerow({'image_id':ids[i], 'label':labels[i]})
    out3=load_dtd_features(stage_dir, layers=['layer2'], labels_file=csvpath)
    assert out3.X.shape == (n,512)
    assert [out3.class_names[i] for i in out3.y] == labels.tolist()

    # Important safety guard: prevent positional CSV joins without verified IDs
    (stage_dir/'ids.npy').unlink()
    try:
        load_dtd_features(stage_dir, layers=['layer2'], labels_file=csvpath)
    except ValueError as exc:
        assert 'image IDs' in str(exc)
    else: raise AssertionError('expected ID alignment error')

    # Missing/invalid values should trigger a meaningful error.
    with np.load(base/'features.npz', allow_pickle=False) as f:
        pass
    bad=np.ones((n,10));bad[0,2]=np.nan
    np.savez(base/'bad.npz', layer=bad, labels=labels)
    try: load_dtd_features(base/'bad.npz')
    except ValueError as exc: assert 'Non-finite' in str(exc)
    else: raise AssertionError('expected nan guard')
    print('All DTD loader tests passed')
