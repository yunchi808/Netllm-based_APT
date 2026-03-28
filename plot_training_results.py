#!/usr/bin/env python3
"""
Visualize APT training outputs: step-wise loss from train_losses.txt
and optional per-epoch mean from console.log.

Usage:
  python plot_training_results.py --model-dir "path/to/.../rank_-1_w_20_..."
  python plot_training_results.py --loss-file path/to/train_losses.txt --out plot.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

_APT_ROOT = Path(__file__).resolve().parent
if str(_APT_ROOT) not in sys.path:
    sys.path.insert(0, str(_APT_ROOT))
try:
    from config import cfg as _apt_cfg  # noqa: E402

    _FT_PLMS_HINT = str(_apt_cfg.ft_plms_dir)
except Exception:
    _FT_PLMS_HINT = ""

DEFAULT_RESULT_DIR: str | None = None  # Set in code or use --result-dir / --model-dir in terminal


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x.copy()
    pad = window // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def load_losses(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 0:
        return np.array([float(data)])
    return data.flatten()


def parse_epoch_means_from_console(console_path: Path) -> list[tuple[int, float, float | None]]:
    """
    Parse lines like:
      'training/train_loss_mean': 0.123,
      'training/train_loss_std': 0.045,
    grouped by preceding 'Epoch #N' in the same block.
    Returns list of (epoch_idx, mean, std_or_none).
    """
    if not console_path.is_file():
        return []
    text = console_path.read_text(encoding="utf-8", errors="replace")
    # Split by epoch headers
    blocks = re.split(r"={10,}\s*Epoch\s*#(\d+)\s*={10,}", text)
    # blocks[0] is preamble, then pairs (epoch_num, block_content)
    out: list[tuple[int, float, float | None]] = []
    i = 1
    while i + 1 < len(blocks):
        try:
            ep = int(blocks[i])
        except ValueError:
            i += 2
            continue
        chunk = blocks[i + 1]
        m_mean = re.search(r"['\"]training/train_loss_mean['\"]\s*:\s*([0-9.eE+-]+)", chunk)
        m_std = re.search(r"['\"]training/train_loss_std['\"]\s*:\s*([0-9.eE+-]+)", chunk)
        if m_mean:
            mean_v = float(m_mean.group(1))
            std_v = float(m_std.group(1)) if m_std else None
            out.append((ep, mean_v, std_v))
        i += 2
    out.sort(key=lambda t: t[0])
    return out


def plot_results(
    losses: np.ndarray,
    epoch_stats: list[tuple[int, float, float | None]],
    steps_per_epoch: int | None,
    out_path: Path,
    ma_window: int,
    title: str,
    dpi: int,
    show_window: bool = True,
) -> None:
    import matplotlib

    if show_window:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(len(losses))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    fig.suptitle(title, fontsize=12)

    # --- Panel 1: step loss ---
    ax0 = axes[0]
    ax0.plot(steps, losses, color="#4C72B0", alpha=0.35, linewidth=0.6, label="train loss (per step)")
    if ma_window > 1 and len(losses) >= ma_window:
        ma = moving_average(losses, ma_window)
        ax0.plot(steps, ma, color="#C44E52", linewidth=1.2, label=f"moving avg (w={ma_window})")
    ax0.set_ylabel("Cross-entropy loss")
    ax0.set_xlabel("Training step (batch index within full run)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Step-wise training loss")

    # --- Panel 2: epoch mean or downsampled curve ---
    ax1 = axes[1]
    if epoch_stats:
        eps = [e for e, _, _ in epoch_stats]
        means = [m for _, m, _ in epoch_stats]
        stds = [s for _, _, s in epoch_stats]
        x = np.array(eps, dtype=float)
        ax1.plot(x, means, "o-", color="#55A868", markersize=4, label="epoch mean loss")
        if all(s is not None for s in stds):
            yerr = np.array([s if s is not None else 0.0 for s in stds])
            ax1.fill_between(x, np.array(means) - yerr, np.array(means) + yerr, alpha=0.25, color="#55A868")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean loss")
        ax1.set_title("Per-epoch mean (from console.log)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)
    elif steps_per_epoch and steps_per_epoch > 0:
        n_epochs = len(losses) // steps_per_epoch
        if n_epochs > 0:
            ep_means = []
            for e in range(n_epochs):
                sl = slice(e * steps_per_epoch, (e + 1) * steps_per_epoch)
                ep_means.append(float(np.mean(losses[sl])))
            ax1.plot(range(n_epochs), ep_means, "o-", color="#8172B2", markersize=4, label=f"epoch mean (steps/ep={steps_per_epoch})")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Mean loss")
            ax1.set_title("Per-epoch mean (uniform split)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper right", fontsize=8)
    else:
        # Downsample for readability
        n = max(1, len(losses) // 500)
        idx = np.arange(0, len(losses), n)
        ax1.plot(idx, losses[idx], color="#CCB974", linewidth=1.0, label=f"subsample every {n} steps")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss (subsampled; pass --steps-per-epoch or keep console.log for epoch plot)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    if show_window:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_RESULT_DIR,
        help="Training run directory (contains train_losses.txt). "
        + (f"Typical parent: {_FT_PLMS_HINT}" if _FT_PLMS_HINT else "See config.ft_plms_dir in config.py"),
    )
    p.add_argument(
        "--result-dir",
        dest="model_dir",
        type=str,
        default=DEFAULT_RESULT_DIR,
        help="Alias of --model-dir",
    )
    p.add_argument("--loss-file", type=str, default=None, help="Path to train_losses.txt")
    p.add_argument("--out", type=str, default=None, help="Output PNG path")
    p.add_argument("--ma-window", type=int, default=100, help="Moving average window for step curve")
    p.add_argument("--steps-per-epoch", type=int, default=None, help="If set, second panel uses uniform epoch split")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no-show", action="store_true", help="Do not show popup window; only save PNG (for headless)")
    args = p.parse_args()

    if args.loss_file:
        loss_path = Path(args.loss_file)
    elif args.model_dir:
        loss_path = Path(args.model_dir) / "train_losses.txt"
    else:
        msg = "Provide --model-dir/--result-dir or --loss-file (or set DEFAULT_RESULT_DIR in the script)."
        if _FT_PLMS_HINT:
            msg += f" Fine-tuned runs live under: {_FT_PLMS_HINT}"
        p.error(msg)

    if not loss_path.is_file():
        raise SystemExit(f"Loss file not found: {loss_path}")

    losses = load_losses(loss_path)
    model_dir = loss_path.parent
    console_path = model_dir / "console.log"
    epoch_stats = parse_epoch_means_from_console(console_path)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = model_dir / "training_curves.png"

    title = f"APT training — {model_dir.name}\nsteps={len(losses)}, final_loss={losses[-1]:.6f}, min={losses.min():.6f}"

    plot_results(
        losses=losses,
        epoch_stats=epoch_stats,
        steps_per_epoch=args.steps_per_epoch,
        out_path=out_path,
        ma_window=args.ma_window,
        title=title,
        dpi=args.dpi,
        show_window=not args.no_show,
    )

    print(f"Loaded {len(losses)} step losses from {loss_path}")
    if epoch_stats:
        print(f"Parsed {len(epoch_stats)} epoch summaries from {console_path}")


if __name__ == "__main__":
    main()
