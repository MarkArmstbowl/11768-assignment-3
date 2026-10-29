"""Build Part 4.1 figures and Part 4.1/4.2 summaries from formal logs.

Local reporting only. Part 4.2 reuses the existing Part 3 comparison figures;
no earlier plots, reports, raw runs, configs or training code are modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from plot_part2 import DIAGNOSTICS, configure_axis, load_run as load_sync, plot_series, require, save_figure
from plot_part3 import SPECS as TIMING_SPECS, load_run as load_timing


ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    {"name": "p2a_number_search_reinforce", "short": "p2a", "label": "REINFORCE G=1",
     "group_size": 1, "loss_fn": "importance_sampling", "color": "#2166ac", "linestyle": "-"},
    {"name": "p41_number_search_grpo_g2", "short": "p41", "label": "GRPO G=2",
     "group_size": 2, "loss_fn": "ppo", "color": "#7570b3", "linestyle": "-."},
    {"name": "p2b_number_search_grpo", "short": "p2b", "label": "GRPO G=8",
     "group_size": 8, "loss_fn": "ppo", "color": "#d95f02", "linestyle": "--"},
)


def enrich_summary(run: dict) -> None:
    records, summary, spec = run["train"], run["summary"], run["spec"]
    require({key for r in records + run["eval"] for key in r
             if key.startswith(("train/pass@", "eval/pass@"))}
            == {f"train/pass@{spec['group_size']}", "eval/pass@4"},
            f"{spec['name']}: unplotted pass metric")
    summary["degenerate_fraction"] = (None if spec["group_size"] == 1
                                       else statistics.mean(r["train/frac_degenerate_groups"] for r in records))
    summary["num_degenerate_groups"] = (None if spec["group_size"] == 1
                                        else sum(r["train/num_degenerate_groups"] for r in records))
    summary["successful_training_episodes"] = round(sum(
        r["train/mean_reward"] * r["train/num_groups"] * spec["group_size"] for r in records))
    for key in ("train/adv_std", "train/mean_abs_advantage"):
        values = [r[key] for r in records if key in r]
        summary[key] = {"min": min(values), "max": max(values),
                        "mean_over_recorded_iterations": statistics.mean(values), "observations": len(values)}
    summary["windows"] = {}
    for length in (10, 20, 50):
        window = records[:length]
        groups = sum(r["train/num_groups"] for r in window)
        if spec["group_size"] == 1:
            degenerate = all_success = all_failure = mixed = None
        else:
            degenerate = sum(r["train/num_degenerate_groups"] for r in window)
            all_failure = round(sum(r["train/num_groups"] * (1 - r[f"train/pass@{spec['group_size']}"])
                                    for r in window))
            all_success = degenerate - all_failure
            mixed = groups - degenerate
            require(min(all_failure, all_success, mixed) >= 0, "Invalid binary group accounting")
        summary["windows"][str(length)] = {
            "num_groups": groups, "degenerate_groups": degenerate,
            "degenerate_fraction": None if degenerate is None else degenerate / groups,
            "mixed_groups": mixed, "all_success_groups": all_success, "all_failure_groups": all_failure,
            "mean_train_success": statistics.mean(r["train/success_rate"] for r in window),
            "retained_sequences": sum(r["train/num_sequences"] for r in window),
            "mean_retained_sequences": statistics.mean(r["train/num_sequences"] for r in window),
        }


def shared_legend(fig, ax, top: float = 0.95) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, top),
               ncol=3, fontsize=8, frameon=False)


def reward_figure(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    panels = (("train", "train/success_rate"), ("eval", "eval/success_rate"),
              ("train", None), ("eval", "eval/pass@4"))
    for index, (ax, (phase, key)) in enumerate(zip(axes.flat, panels)):
        title = key or "Training pass@k (different k)"
        configure_axis(ax, f"({chr(97 + index)}) {title}", "Fraction")
        ax.set_ylim(-0.025, 1.055)
        for run in runs:
            metric = key or f"train/pass@{run['spec']['group_size']}"
            label = f"{run['spec']['short']}: {metric}" if "pass@" in metric else None
            plot_series(ax, run, phase, metric, label=label, marker="o" if phase == "eval" else None)
        ax.legend(loc="lower right", fontsize=7.3, framealpha=0.92)
    fig.suptitle("Part 4.1: group size and reward quality", fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.009, "Raw formal runs; training pass@1, @2 and @8 are distinct quantities.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=2, w_pad=1.5)
    save_figure(fig, output, "part4_1_rewards")


def diagnostics_figure(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.0))
    ylabels = ("Gradient norm", "Importance ratio", "Importance ratio", "Standard deviation",
               "Fraction of groups", "Mean sampled -log p")
    for index, (ax, key, ylabel) in enumerate(zip(axes.flat, DIAGNOSTICS, ylabels)):
        configure_axis(ax, f"({chr(97 + index)}) {key}", ylabel)
        for run in runs:
            if key == "train/frac_degenerate_groups" and run["spec"]["group_size"] == 1:
                continue
            plot_series(ax, run, "train", key, marker="o")
        if key == "train/importance_weight_mean":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        elif key == "train/importance_weight_max":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.set_ylim(bottom=0.98)
        elif key == "train/frac_degenerate_groups":
            ax.set_ylim(-0.025, 1.055)
            ax.text(0.03, 0.95, "REINFORCE: N/A\n(no group baseline)", transform=ax.transAxes,
                    va="top", fontsize=8, color=runs[0]["spec"]["color"],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle("Part 4.1: optimization, drift, and data health", fontsize=13, fontweight="bold", y=0.99)
    shared_legend(fig, axes[0, 0])
    fig.text(0.5, 0.008, "Skipped update diagnostics stay missing; REINFORCE degeneracy is N/A.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.93), h_pad=1.8, w_pad=1.5)
    save_figure(fig, output, "part4_1_diagnostics")


def data_figure(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    panels = (("train/num_sequences", "Retained sequences"),
              ("train/num_action_tokens", "Action targets"),
              ("train/adv_std", "Standard deviation"),
              ("train/mean_abs_advantage", "Mean absolute advantage"))
    for index, (ax, (key, ylabel)) in enumerate(zip(axes.flat, panels)):
        configure_axis(ax, f"({chr(97 + index)}) {key}", ylabel)
        for run in runs:
            plot_series(ax, run, "train", key, marker="o" if "adv" in key else None)
        ax.set_ylim(bottom=0)
    fig.suptitle("Part 4.1: retained data and advantage statistics", fontsize=12.5,
                 fontweight="bold", y=0.99)
    shared_legend(fig, axes[0, 0])
    fig.text(0.5, 0.009, "Advantage moments use retained action tokens; G=1 also retains zero-advantage episodes.",
             ha="center", fontsize=7.8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.93), h_pad=2, w_pad=1.5)
    save_figure(fig, output, "part4_1_data")


def write_csv(path: Path, summaries: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
    runs = [load_sync(spec) for spec in SPECS]
    for run in runs:
        enrich_summary(run)
    reward_figure(runs, output)
    diagnostics_figure(runs, output)
    data_figure(runs, output)
    summaries = [run["summary"] for run in runs]
    (report_dir / "part4_1_summary.json").write_text(json.dumps({
        "source": "Formal p2a, p41, p2b only; probes excluded.",
        "interpretation": "REINFORCE degeneracy is N/A. Retained sequences may have zero advantage. "
                          "Logged advantage moments describe retained action tokens, not unconditional episodes.",
        "runs": summaries,
    }, indent=2) + "\n")
    write_csv(report_dir / "part4_1_summary.csv", summaries,
              ("run_name", "group_size", "loss_fn", "baseline_eval_success_rate", "final_eval_success_rate",
               "final_eval_pass_at_4", "optimizer_updates", "skipped_iterations", "sampled_training_episodes",
               "training_sequences", "training_action_tokens", "degenerate_fraction", "num_degenerate_groups"))

    timing_runs = [load_timing(spec) for spec in TIMING_SPECS]
    sync, asynchronous = [run["summary"] for run in timing_runs]
    reused = {}
    for stem in ("part3_rewards", "part3_diagnostics", "part3_timing"):
        path = ROOT / "report" / "figures" / f"{stem}.png"
        reused[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (report_dir / "part4_2_summary.json").write_text(json.dumps({
        "source": "Formal p2b and p3; same comparison and figure files as Part 3, no new training.",
        "timing_definition": "All 50 train/step_time_s intervals, including skipped iterations; "
                             "excludes cold start, baseline evaluation, and final evaluation drain.",
        "speedup_sync_over_async": sync["mean_step_seconds"] / asynchronous["mean_step_seconds"],
        "step_time_reduction_fraction": 1 - asynchronous["mean_step_seconds"] / sync["mean_step_seconds"],
        "runs": [sync, asynchronous], "reused_figure_sha256": reused,
    }, indent=2) + "\n")
    write_csv(report_dir / "part4_2_summary.csv", [sync, asynchronous],
              ("run_name", "mean_step_seconds", "median_step_seconds", "total_step_seconds",
               "time_weighted_learner_wait_fraction", "optimizer_updates", "skipped_iterations",
               "retained_training_sequences", "final_eval_success_rate", "final_eval_pass_at_4",
               "groups_consumed", "groups_dropped_stale", "stale_fraction_of_dequeued_groups"))
    print(f"Wrote Part 4.1 figures and Part 4.1/4.2 summaries to {report_dir}")


if __name__ == "__main__":
    main()
