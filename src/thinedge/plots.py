"""Reliability diagrams for the baseline report. Needs matplotlib (the ``model`` extra)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"with_regime": "with Impact Player flag", "no_regime": "without regime flag"}
COLOURS = {"with_regime": "#1b7f5c", "no_regime": "#c2562e"}


def rolling_figure(report: dict, path: Path) -> Path:
    """Left: validation Brier by year relative to the simplest candidate. Right: final-exam Brier."""
    by = report["validation"]["by_candidate"]
    base = "no_flag/expanding"
    chosen = report["selection"]["chosen"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    for name, entry in by.items():
        years = sorted(entry["brier_by_year"], key=int)
        diff = [entry["brier_by_year"][y] - by[base]["brier_by_year"][y] for y in years]
        left.plot([int(y) for y in years], diff, marker="o", lw=2.2 if name == chosen else 1.2,
                  label=name + ("  (chosen)" if name == chosen else ""))
    left.axhline(0, color="#888888", lw=1, ls="--")
    left.set_title(f"Validation: Brier minus {base} (below zero is better)")
    left.set_xlabel("year predicted")
    left.grid(alpha=0.25)
    left.legend(frameon=False, fontsize=8)

    by_subset = report["final_exam"].get("by_subset", {})
    subset = "fresh" if "fresh" in by_subset else "all"
    exam = by_subset[subset]["by_candidate"] if by_subset else report["final_exam"]["by_candidate"]
    names = list(exam)
    briers = [exam[n]["all"]["brier"] for n in names]
    err = [[b - exam[n]["all"]["brier_ci"][0] for n, b in zip(names, briers)],
           [exam[n]["all"]["brier_ci"][1] - b for n, b in zip(names, briers)]]
    colours = ["#1b7f5c" if n == chosen else "#9aa5b1" for n in names]
    right.barh(names, briers, xerr=err, color=colours, capsize=3)
    right.set_xlim(min(briers) - 0.02, max(briers) + 0.02)
    right.set_title(f"Final exam, {subset} matches {report['test_years'][0]}-{report['test_years'][-1]}: pooled Brier (lower is better)",
                    fontsize=10)
    right.set_xlabel("Brier score, 95% match-level interval")
    right.invert_yaxis()
    right.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


ARM_LABELS = {
    "fixed": "fixed (trained once)",
    "par_oracle": "par fit on the test year (diagnostic)",
    "par_refresh": "par curve refreshed yearly",
    "full_retrain": "fully retrained yearly",
}
ARM_COLOURS = {"fixed": "#c2562e", "par_oracle": "#7b4fa3", "par_refresh": "#2b6cb0", "full_retrain": "#1b7f5c"}


def walkforward_figure(report: dict, path: Path) -> Path:
    """IPL reliability curves in the test years where a refresh is possible, one line per arm."""
    years = [y for y in report["test_years"] if y > report["first_test_year"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, (("first_innings", "First innings"), ("chase", "Chase"))):
        ax.plot([0, 1], [0, 1], color="#888888", lw=1, ls="--", label="perfect calibration")
        for arm, curves in report["calibration_ipl_refreshable"].items():
            bins = curves[key]
            total = sum(b["n"] for b in bins)
            xs, ys = [b["mean_pred"] for b in bins], [b["mean_obs"] for b in bins]
            ax.plot(xs, ys, color=ARM_COLOURS[arm], lw=1.5, label=ARM_LABELS[arm])
            ax.scatter(xs, ys, s=[30 + 600 * b["n"] / total for b in bins], color=ARM_COLOURS[arm], alpha=0.55)
        gap = report["arms"]["fixed"]["ipl_refreshable"][key]["gap"]
        ax.set_title(f"{title} (fixed model gap {gap:+.3f})")
        ax.set_xlabel("predicted win probability")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed win frequency")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle(f"Walk-forward vs fixed baseline: IPL matches in {', '.join(map(str, years))}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def calibration_figure(report: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, (("first_innings", "First innings"), ("chase", "Chase"))):
        ax.plot([0, 1], [0, 1], color="#888888", lw=1, ls="--", label="perfect calibration")
        for name, variant in report["variants"].items():
            bins = variant["calibration"][key]
            total = sum(b["n"] for b in bins)
            ax.plot(
                [b["mean_pred"] for b in bins],
                [b["mean_obs"] for b in bins],
                color=COLOURS[name],
                lw=1.5,
                label=LABELS[name],
            )
            ax.scatter(
                [b["mean_pred"] for b in bins],
                [b["mean_obs"] for b in bins],
                s=[30 + 600 * b["n"] / total for b in bins],
                color=COLOURS[name],
                alpha=0.6,
            )
        metrics = report["variants"]["with_regime"]["metrics"][key]
        ax.set_title(f"{title} (Brier skill {metrics['skill']:.3f}, ECE {metrics['ece']:.3f})")
        ax.set_xlabel("predicted win probability")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed win frequency")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle(
        f"Baseline calibration on matches from {report['train_end']} onward "
        f"({report['split']['test_matches']} matches; marker size = share of balls)"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
