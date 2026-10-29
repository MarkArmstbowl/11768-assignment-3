"""Plot the two formal Part 4.3 runs and write reproducible local summaries.

Reads saved logs/configs/checkpoint manifests only; no trainer or backend calls.
Earlier report sections and experimental outputs are never overwritten.
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
import yaml

from plot_part2 import configure_axis, plot_series, require, save_figure
from plot_part3 import DIAGNOSTICS, load_run


ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    {"name": "p43_number_search_async_stale4_iw", "label": "IW (PPO)", "short": "IW",
     "async": True, "loss_fn": "ppo", "loss_flag": 1,
     "color": "#1b9e77", "linestyle": "-"},
    {"name": "p43_number_search_async_stale4_noiw", "label": "No-IW (CE)", "short": "No-IW",
     "async": True, "loss_fn": "cross_entropy", "loss_flag": 0,
     "color": "#7570b3", "linestyle": "--"},
)


def enrich(run: dict) -> dict:
    """Check the selected loss and add interpretation-specific statistics."""
    spec, records, summary = run["spec"], run["train"], run["summary"]
    config_path = ROOT / "configs" / f"{spec['name']}.yaml"
    config = yaml.safe_load(config_path.read_text())
    cfg = config["train"]
    require(cfg["loss_fn"] == spec["loss_fn"] and cfg["run_name"] == spec["name"],
            f"{spec['name']}: unexpected loss/run name")
    require(cfg["max_staleness"] == 4 and cfg["group_size"] == 8
            and cfg["groups_per_batch"] == 8 and cfg["num_steps"] == 50
            and not cfg["resume"], f"{spec['name']}: expected fresh formal stale4 configuration")
    require(all(r["train/loss_fn_is_iw"] == spec["loss_flag"] for r in records),
            f"{spec['name']}: loss flag differs from configured objective")
    require(all(r.get("train/abandoned_episodes", 0) == 0 for r in records)
            and summary["abandoned_episodes"] == 0,
            f"{spec['name']}: inspect abandoned episodes before reporting")
    for r in records:
        if r.get("train/skipped"):
            require(r["train/mean_reward"] == 1 and r["train/num_degenerate_groups"] == 8,
                    f"{spec['name']}: skipped batch requires separate explanation")

    summary["loss_fn"] = spec["loss_fn"]
    summary["max_staleness"] = cfg["max_staleness"]
    summary["config_path"] = str(config_path.relative_to(ROOT))
    summary["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    summary["num_degenerate_groups"] = sum(r["train/num_degenerate_groups"] for r in records)
    summary["degenerate_fraction"] = summary["num_degenerate_groups"] / summary["groups_consumed"]
    summary["all_failure_groups"] = round(sum(
        r["train/num_groups"] * (1 - r["train/pass@8"]) for r in records))
    summary["all_success_groups"] = summary["num_degenerate_groups"] - summary["all_failure_groups"]
    summary["final_policy_version"] = records[-1]["train/policy_version"]
    for key in ("train/mean_abs_advantage", "train/adv_std"):
        values = [r[key] for r in records if key in r]
        summary[key] = {"min": min(values), "max": max(values), "first": values[0],
                        "last": values[-1], "mean_over_recorded_iterations": statistics.mean(values),
                        "observations": len(values)}
    summary["evaluations"] = [{k: r[k] for k in
                              ("step", "eval/success_rate", "eval/pass@4", "eval/policy_version")}
                             for r in run["eval"]]
    return {"model": config["model"],
            "train": {k: v for k, v in cfg.items() if k not in ("loss_fn", "run_name")}}


def legend(fig, ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=2, fontsize=8.5, frameon=False)


def rewards(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    for index, (ax, phase, metric) in enumerate(zip(
            axes.flat, ("train", "eval", "train", "eval"),
            ("train/success_rate", "eval/success_rate", "train/pass@8", "eval/pass@4"))):
        configure_axis(ax, f"({chr(97 + index)}) {metric}", "Fraction")
        ax.set_ylim(-0.025, 1.055)
        for run in runs:
            label = f"{run['spec']['short']}: {metric}" if "pass@" in metric else None
            plot_series(ax, run, phase, metric, label=label,
                        marker="o" if phase == "eval" else None)
        ax.legend(loc="lower right", fontsize=7.6, framealpha=0.92)
    fig.suptitle("Part 4.3: importance weighting at staleness limit 4",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.009, "Formal runs; train/pass@8 and eval/pass@4 are distinct quantities.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=2, w_pad=1.5)
    save_figure(fig, output, "part4_3_rewards")


def diagnostics(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.0))
    ylabels = ("Gradient norm", "Importance ratio", "Importance ratio", "Standard deviation",
               "Fraction of groups", "Mean sampled -log p")
    for index, (ax, key, ylabel) in enumerate(zip(axes.flat, DIAGNOSTICS, ylabels)):
        configure_axis(ax, f"({chr(97 + index)}) {key}", ylabel)
        for run in runs:
            plot_series(ax, run, "train", key, marker="o")
        if key == "train/importance_weight_mean":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        elif key == "train/importance_weight_max":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.set_ylim(bottom=0.98)
        elif key == "train/frac_degenerate_groups":
            ax.set_ylim(-0.025, 1.055)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle("Part 4.3: optimization, drift, and data health",
                 fontsize=13, fontweight="bold", y=0.99)
    legend(fig, axes[0, 0])
    fig.text(0.5, 0.008, "No-IW ratios are diagnostic only; skipped-update diagnostics stay missing.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.93), h_pad=1.8, w_pad=1.5)
    save_figure(fig, output, "part4_3_diagnostics")


def drift_and_data(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    panels = (("train/importance_weight_min", "Importance ratio"),
              ("train/num_sequences", "Retained sequences"),
              ("train/policy_version", "Successful optimizer updates"),
              ("train/mean_abs_advantage", "Mean absolute advantage"))
    for index, (ax, (key, ylabel)) in enumerate(zip(axes.flat, panels)):
        configure_axis(ax, f"({chr(97 + index)}) {key}", ylabel)
        for run in runs:
            plot_series(ax, run, "train", key, marker="o")
        ax.set_ylim(bottom=0)
        if key == "train/importance_weight_min":
            ax.set_ylim(top=1.03)
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
    fig.suptitle("Part 4.3: lower ratios and retained training signal",
                 fontsize=12.5, fontweight="bold", y=0.99)
    legend(fig, axes[0, 0])
    fig.text(0.5, 0.009, "Versions count updates, not iterations; advantage moments use retained action tokens.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.93), h_pad=2, w_pad=1.5)
    save_figure(fig, output, "part4_3_drift_data")


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
    runs = [load_run(spec) for spec in SPECS]
    common_configs = [enrich(run) for run in runs]
    require(common_configs[0] == common_configs[1], "Configs differ beyond loss_fn and run_name")
    rewards(runs, output)
    diagnostics(runs, output)
    drift_and_data(runs, output)
    summaries = [run["summary"] for run in runs]
    (report_dir / "part4_3_summary.json").write_text(json.dumps({
        "source": "Two completed formal p43 runs only; no probes or resumed segments.",
        "comparison": "Provided configs differ only in loss_fn and run_name. No-IW still logs "
                      "importance ratios diagnostically; they do not enter its CE objective.",
        "interpretation": "Each config run once; different realized baselines, task/sample order, "
                          "retained data and update counts limit causal conclusions.",
        "runs": summaries,
    }, indent=2) + "\n")
    fields = ("run_name", "loss_fn", "max_staleness", "baseline_eval_success_rate",
              "final_eval_success_rate", "final_eval_pass_at_4", "optimizer_updates",
              "skipped_iterations", "retained_training_sequences", "training_action_tokens",
              "degenerate_fraction", "all_success_groups", "all_failure_groups",
              "groups_dropped_stale", "final_policy_version", "mean_step_seconds")
    with (report_dir / "part4_3_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote Part 4.3 figures and summaries to {report_dir}")


if __name__ == "__main__":
    main()
