"""Adapter for the team's G-CEALS implementation."""

from __future__ import annotations

import contextlib
import io
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def run_gceals(
    X,
    y,
    *,
    n_clusters,
    random_state=42,
    device="cpu",
    latent_dim=10,
    batch_size=256,
    pretrain_epochs=100,
    finetune_epochs=100,
    learning_rate=1e-5,
    gamma=0.1,
    stop_w_factor=0.1,
):
    """Train G-CEALS from scratch and return cluster predictions."""

    import torch

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    # The original implementation imports modules from its own directory.
    gceals_dir = (
        Path(__file__).resolve().parents[2]
        / "algorithms"
        / "gceals"
    )

    sys.path.insert(0, str(gceals_dir))

    try:
        from gceals import pretrain, train
    finally:
        sys.path.pop(0)

    args = SimpleNamespace(
        dataset="benchmark",
        name=f"benchmark_seed_{random_state}",
        device=device,
        latent_dim=latent_dim,
        n_clusters=n_clusters,
        batch_size=batch_size,
        batch_count=int(np.ceil(len(X) / batch_size)),
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        l_rate=learning_rate,
        gamma=gamma,
        update_interval=1,
        stop_w_factor=stop_w_factor,
        timed=True,
        redo_pretrain=True,
        plot_interval=100,
    )
    Path("debug").mkdir(parents=True, exist_ok=True)
    data_tensor = torch.as_tensor(
        np.asarray(X, dtype=np.float32),
        device=device,
    )

    # Keep training diagnostics out of the worker's JSON output.
    # Redirect them to stderr so failures remain inspectable.
    with contextlib.redirect_stdout(sys.stderr):
        ae_model = pretrain(data_tensor, args)
        model = train(ae_model, data_tensor, y, args)

    model.eval()
    with torch.no_grad():
        full_p, _, _ = model(data_tensor)

    pred = full_p.argmax(dim=1).cpu().numpy()

    return pred
