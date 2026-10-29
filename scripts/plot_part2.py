"""Build Part 2 report figures from the two completed formal JSONL logs.

This local reporting helper does not import a trainer or contact a backend.
Requires Matplotlib. Inputs and the provided experiment configs are read only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    {"name": "p2a_number_search_reinforce", "label": "REINFORCE", "group_size": 1,
     "loss_fn": "importance_sampling", "color": "#2166ac", "linestyle": "-"},
    {"name": "p2b_number_search_grpo", "label": "GRPO", "group_size": 8,
     "loss_fn": "ppo", "color": "#d95f02", "linestyle": "--"},
)
DIAGNOSTICS = (
    "train/grad_norm", "train/importance_weight_mean",
    "train/importance_weight_max", "train/importance_weight_std",
    "train/frac_degenerate_groups", "train/entropy_proxy",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_run(spec: dict) -> dict:
    name = spec["name"]
    log_path = ROOT / "runs" / f"{name}.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    train = [r for r in records if any(k.startswith("train/") for k in r)]
    evaluation = [r for r in records if any(k.startswith("eval/") for k in r)]
    require([r["step"] for r in train] == list(range(1, 51)), f"{name}: incomplete train log")
    require([r["step"] for r in evaluation] == [0, 10, 20, 30, 40, 50],
            f"{name}: incomplete evaluation log")
    require(len(records) == 56, f"{name}: unexpected extra records")
    for record in records:
        require(all(not isinstance(v, float) or math.isfinite(v) for v in record.values()),
                f"{name}: non-finite metric at step {record['step']}")
    for record in evaluation:
        require(record["eval/num_tasks"] == 50 and record["eval/episodes_per_task"] == 4,
                f"{name}: evaluation settings differ from the formal config")
    train_pass = f"train/pass@{spec['group_size']}"
    required_train = {"train/success_rate", train_pass, "train/num_groups",
                      "train/num_sequences", "train/num_action_tokens",
                      "train/frac_degenerate_groups"}
    skipped = []
    updates = []
    for record in train:
        require(required_train <= record.keys(), f"{name}: missing train metrics")
        require(record["train/num_groups"] == 8, f"{name}: incorrect task-group count")
        if record.get("train/skipped"):
            skipped.append(record["step"])
            require(record["train/num_sequences"] == record["train/num_action_tokens"] == 0,
                    f"{name}: a skipped step unexpectedly has training sequences")
            require(all(key not in record for key in DIAGNOSTICS
                        if key != "train/frac_degenerate_groups"),
                    f"{name}: a skipped step unexpectedly has update diagnostics")
        else:
            updates.append(record["step"])
            require(set(DIAGNOSTICS) <= record.keys(), f"{name}: missing update diagnostics")
    manifest = json.loads((ROOT / "runs" / f"{name}.checkpoints.json").read_text())
    expected = ["baseline", "step_000000"] + [f"step_{step:06d}" for step in updates]
    require([s["name"] for s in manifest["snapshots"]] == expected,
            f"{name}: sampler snapshots disagree with successful updates")
    require(manifest["completed_steps"] == len(updates), f"{name}: update count mismatch")
    require(manifest["final_sampler_path"] == manifest["snapshots"][-1]["path"],
            f"{name}: final sampler mismatch")
    summary = {
        "run_name": name, "algorithm": spec["label"], "group_size": spec["group_size"],
        "loss_fn": spec["loss_fn"], "training_iterations": len(train),
        "optimizer_updates": len(updates), "skipped_iterations": len(skipped),
        "skipped_steps": skipped,
        "sampled_training_episodes": sum(r["train/num_groups"] for r in train) * spec["group_size"],
        "training_sequences": sum(r["train/num_sequences"] for r in train),
        "training_action_tokens": sum(r["train/num_action_tokens"] for r in train),
        "baseline_eval_success_rate": evaluation[0]["eval/success_rate"],
        "step20_eval_success_rate": evaluation[2]["eval/success_rate"],
        "final_eval_success_rate": evaluation[-1]["eval/success_rate"],
        "baseline_eval_pass_at_4": evaluation[0]["eval/pass@4"],
        "final_eval_pass_at_4": evaluation[-1]["eval/pass@4"],
        "after_baseline_seconds": evaluation[-1]["time"] - evaluation[0]["time"],
        "abandoned_train_episodes": sum(r["train/abandoned_episodes"] for r in train),
        "abandoned_eval_episodes": sum(r["eval/abandoned_episodes"] for r in evaluation),
        "final_sampler_path": manifest["final_sampler_path"],
        "log_path": str(log_path.relative_to(ROOT)),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "config_path": f"configs/{name}.yaml",
    }
    for key in DIAGNOSTICS:
        values = [r[key] for r in train if key in r]
        summary[key] = {"min": min(values), "max": max(values), "observations": len(values)}
    return {"spec": spec, "train": train, "eval": evaluation, "summary": summary}


def series(run: dict, phase: str, key: str) -> tuple[list[int], list[float]]:
    # Keep every iteration in x: NaN breaks lines at missing skipped-step data.
    records = run[phase]
    return [r["step"] for r in records], [r.get(key, math.nan) for r in records]


def configure_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=9)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.grid(alpha=0.22, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot_series(ax, run: dict, phase: str, key: str, *, label: str | None = None,
                marker: str | None = None) -> None:
    x, y = series(run, phase, key)
    spec = run["spec"]
    ax.plot(x, y, color=spec["color"], linestyle=spec["linestyle"],
            linewidth=1.3, marker=marker, markersize=3.4,
            label=label or spec["label"], alpha=0.95)


def save_figure(fig, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=260,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def reward_figure(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    panels = (
        (axes[0, 0], "(a) train/success_rate", "train", "train/success_rate"),
        (axes[0, 1], "(b) eval/success_rate", "eval", "eval/success_rate"),
        (axes[1, 0], "(c) Training pass@k (different k)", "train", None),
        (axes[1, 1], "(d) eval/pass@4", "eval", "eval/pass@4"),
    )
    for ax, title, phase, common_key in panels:
        configure_axis(ax, title, "Fraction")
        ax.set_ylim(-0.025, 1.055)
        for run in runs:
            key = common_key or f"train/pass@{run['spec']['group_size']}"
            label = (f"{run['spec']['label']}: {key}" if "pass@" in key
                     else run["spec"]["label"])
            plot_series(ax, run, phase, key, label=label,
                        marker="o" if phase == "eval" else None)
        ax.legend(loc="lower right", fontsize=7.4, framealpha=0.92)
    fig.suptitle("Part 2: reward quality", fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.009, "Raw single-run metrics; step 0 is the untrained evaluation baseline.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=2.0, w_pad=1.5)
    save_figure(fig, output, "part2_rewards")


def diagnostics_figure(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.0))
    labels = ("Gradient norm", "Importance ratio", "Importance ratio", "Standard deviation",
              "Fraction of groups", "Mean sampled -log p")
    for index, (ax, key, ylabel) in enumerate(zip(axes.flat, DIAGNOSTICS, labels)):
        configure_axis(ax, f"({chr(97 + index)}) {key}", ylabel)
        for run in runs:
            if key == "train/frac_degenerate_groups" and run["spec"]["group_size"] == 1:
                continue  # REINFORCE has no group baseline: its logged zero is N/A.
            plot_series(ax, run, "train", key, marker="o")
        if key == "train/importance_weight_mean":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        elif key == "train/importance_weight_max":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.set_ylim(0.98, 2.04)
        elif key == "train/frac_degenerate_groups":
            ax.set_ylim(-0.025, 1.055)
            ax.text(0.035, 0.95, "REINFORCE: N/A\n(no group baseline)", transform=ax.transAxes,
                    fontsize=8, va="top", color=runs[0]["spec"]["color"],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        else:
            ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=7.6, framealpha=0.9)
    fig.suptitle("Part 2: optimization, policy drift, and data health", fontsize=12.5,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.008, "Gaps in GRPO update diagnostics are skipped iterations; no values are filled with zero.",
             ha="center", fontsize=7.8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97), h_pad=1.8, w_pad=1.5)
    save_figure(fig, output, "part2_diagnostics")


def write_summaries(runs: list[dict], report_dir: Path) -> None:
    summaries = [run["summary"] for run in runs]
    payload = {
        "source": "Completed formal runs only; probe and archived failed startup excluded.",
        "timing_definition": "Iteration-0 evaluation record to iteration-50 evaluation record; "
                             "excludes startup and baseline evaluation.",
        "missing_values": "Skipped-update diagnostics remain missing, not zero.",
        "runs": summaries,
    }
    (report_dir / "part2_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ("run_name", "algorithm", "group_size", "loss_fn", "training_iterations",
              "optimizer_updates", "skipped_iterations", "sampled_training_episodes",
              "training_sequences", "training_action_tokens", "baseline_eval_success_rate",
              "step20_eval_success_rate", "final_eval_success_rate", "baseline_eval_pass_at_4",
              "final_eval_pass_at_4", "after_baseline_seconds")
    with (report_dir / "part2_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "report")
    args = parser.parse_args()
    report_dir = args.out.resolve()
    output = report_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    runs = [load_run(spec) for spec in RUNS]
    reward_figure(runs, output)
    diagnostics_figure(runs, output)
    write_summaries(runs, report_dir)
    print(f"Wrote Part 2 figures (PNG/PDF) and summaries to {report_dir}")


if __name__ == "__main__":
    main()
