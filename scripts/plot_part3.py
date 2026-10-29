"""Build Part 3 figures and summaries from completed formal p2b/p3 logs.

Uses Matplotlib only for local reporting. Does not import trainers or contact
any backend. Part 2 reports, configs, and raw run artifacts are read only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from plot_part2 import configure_axis, plot_series, require, save_figure


ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    {"name": "p2b_number_search_grpo", "label": "Sync (p2b)", "short": "p2b",
     "color": "#d95f02", "linestyle": "--", "async": False},
    {"name": "p3_number_search_async", "label": "Async (p3)", "short": "p3",
     "color": "#1b9e77", "linestyle": "-", "async": True},
)
DIAGNOSTICS = (
    "train/grad_norm", "train/importance_weight_mean", "train/importance_weight_max",
    "train/importance_weight_std", "train/frac_degenerate_groups", "train/entropy_proxy",
)
UPDATE_KEYS = set(DIAGNOSTICS) - {"train/frac_degenerate_groups"} | {"train/importance_weight_min"}
EXAMPLE_PATHS = (
    "step_000005/train_number_search.train.65_rollout0.jsonl",
    "step_000025/train_number_search.train.272_rollout0.jsonl",
    "step_000045/train_number_search.train.428_rollout0.jsonl",
)


def load_run(spec: dict) -> dict:
    name = spec["name"]
    path = ROOT / "runs" / f"{name}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train = [r for r in records if any(k.startswith("train/") for k in r)]
    evaluation = sorted([r for r in records if any(k.startswith("eval/") for k in r)],
                        key=lambda r: r["step"])
    summaries = [r for r in records if any(k.startswith("summary/") for k in r)]
    require([r["step"] for r in train] == list(range(1, 51)), f"{name}: incomplete train log")
    require([r["step"] for r in evaluation] == [0, 10, 20, 30, 40, 50],
            f"{name}: incomplete evaluation log")
    require(len(records) == 56 + int(spec["async"]), f"{name}: extra or missing records")
    require(len(summaries) == int(spec["async"]), f"{name}: incorrect summary count")
    require({key for r in records for key in r if key.startswith(("train/pass@", "eval/pass@"))}
            == {"train/pass@8", "eval/pass@4"}, f"{name}: unplotted pass metric")
    for record in records:
        require(all(not isinstance(v, float) or math.isfinite(v) for v in record.values()),
                f"{name}: nonfinite metric")
    for record in evaluation:
        require(record["eval/num_tasks"] == 50 and record["eval/episodes_per_task"] == 4,
                f"{name}: unexpected evaluation configuration")

    updates, skipped, version_at_step = [], [], {0: 0}
    for record in train:
        require(record["train/num_groups"] == 8, f"{name}: unexpected group count")
        if record.get("train/skipped"):
            skipped.append(record["step"])
            require(record["train/num_sequences"] == record["train/num_action_tokens"] == 0,
                    f"{name}: skipped iteration contains training sequences")
            require(not UPDATE_KEYS.intersection(record), f"{name}: fabricated skipped diagnostics")
        else:
            updates.append(record["step"])
            require(UPDATE_KEYS <= record.keys(), f"{name}: missing update diagnostics")
        version_at_step[record["step"]] = len(updates)
        if spec["async"]:
            require(record["train/policy_version"] == len(updates), f"{name}: policy-version mismatch")
    manifest = json.loads((ROOT / "runs" / f"{name}.checkpoints.json").read_text())
    names = (["baseline", "v0"] + [f"v{i}" for i in range(1, len(updates) + 1)]
             if spec["async"] else ["baseline", "step_000000"] + [f"step_{i:06d}" for i in updates])
    require([s["name"] for s in manifest["snapshots"]] == names, f"{name}: snapshot names disagree")
    require(manifest["completed_steps"] == len(updates), f"{name}: update-count mismatch")
    require(manifest["final_sampler_path"] == manifest["snapshots"][-1]["path"],
            f"{name}: final sampler mismatch")
    if spec["async"]:
        totals = summaries[0]
        require(totals["summary/train_steps"] == 50, f"{name}: incomplete summary")
        require(totals["summary/groups_consumed"] == sum(r["train/num_groups"] for r in train),
                f"{name}: consumed-group mismatch")
        for record in evaluation:
            require(record["eval/policy_version"] == version_at_step[record["step"]],
                    f"{name}: evaluation version mismatch")

    step_times = [r["train/step_time_s"] for r in train]
    dropped = summaries[0]["summary/groups_dropped_stale"] if spec["async"] else 0
    groups = sum(r["train/num_groups"] for r in train)
    summary = {
        "run_name": name, "training_iterations": len(train), "optimizer_updates": len(updates),
        "skipped_steps": skipped, "skipped_iterations": len(skipped),
        "groups_consumed": groups, "groups_dropped_stale": dropped,
        "stale_fraction_of_dequeued_groups": dropped / (groups + dropped),
        "episodes_in_accepted_groups": groups * 8,
        "retained_training_sequences": sum(r["train/num_sequences"] for r in train),
        "training_action_tokens": sum(r["train/num_action_tokens"] for r in train),
        "baseline_eval_success_rate": evaluation[0]["eval/success_rate"],
        "final_eval_success_rate": evaluation[-1]["eval/success_rate"],
        "final_eval_pass_at_4": evaluation[-1]["eval/pass@4"],
        "mean_step_seconds": statistics.mean(step_times),
        "median_step_seconds": statistics.median(step_times), "total_step_seconds": sum(step_times),
        "total_update_seconds": sum(r["train/update_time_s"] for r in train),
        "total_learner_wait_seconds": sum(r["train/learner_wait_s"] for r in train),
        "time_weighted_learner_wait_fraction": sum(r["train/learner_wait_s"] for r in train) / sum(step_times),
        "after_baseline_through_final_eval_seconds": evaluation[-1]["time"] - evaluation[0]["time"],
        "final_eval_tail_seconds": evaluation[-1]["time"] - train[-1]["time"],
        "abandoned_episodes": sum(r.get("train/abandoned_episodes", 0)
                                   + r.get("eval/abandoned_episodes", 0) for r in records),
        "final_sampler_path": manifest["final_sampler_path"],
        "log_path": str(path.relative_to(ROOT)),
        "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for key in DIAGNOSTICS + ("train/importance_weight_min",):
        values = [r[key] for r in train if key in r]
        summary[key] = {"min": min(values), "max": max(values), "first": values[0],
                        "last": values[-1], "observations": len(values)}
    return {"spec": spec, "train": train, "eval": evaluation, "summary": summary}


def load_examples() -> list[dict]:
    examples = []
    for relative in EXAMPLE_PATHS:
        path = ROOT / "runs" / "p3_number_search_async_trajectories" / relative
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        header = rows[0]
        require(header["phase"] == "train", "A behavioral example is not a training rollout")
        low, high, target = (header["info"][key] for key in ("low", "high", "target"))
        midpoints = excluded = 0
        guesses = []
        for row in rows[1:]:
            call = row["tool_call"]
            require(call and call["name"] == "guess" and len(call["args"]) == 1,
                    "Selected example is not a valid single-guess rollout")
            guess = call["args"][0]
            midpoints += guess in {(low + high) // 2, (low + high + 1) // 2}
            excluded += not low <= guess <= high
            feedback = "low" if guess < target else "high" if guess > target else "correct"
            guesses.append({"guess": guess, "feedback": feedback, "env": row["env"]})
            if guess < target:
                low = max(low, guess + 1)
            elif guess > target:
                high = min(high, guess - 1)
        examples.append({"path": str(path.relative_to(ROOT)), "step": header["step"],
                         "task_id": header["task_id"], "policy_version": header["policy_version"],
                         "info": header["info"], "max_steps": header["max_steps"],
                         "reward": header["reward"], "num_steps": header["num_steps"],
                         "midpoint_choices": midpoints, "excluded_range_choices": excluded,
                         "guesses": guesses, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return examples


def reward_figure(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    for index, (ax, phase, key) in enumerate(zip(axes.flat,
            ("train", "eval", "train", "eval"),
            ("train/success_rate", "eval/success_rate", "train/pass@8", "eval/pass@4"))):
        configure_axis(ax, f"({chr(97 + index)}) {key}", "Fraction")
        ax.set_ylim(-0.025, 1.055)
        for run in runs:
            label = f"{run['spec']['short']}: {key}" if "pass@" in key else None
            plot_series(ax, run, phase, key, label=label, marker="o" if phase == "eval" else None)
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    fig.suptitle("Part 3: synchronous and asynchronous reward quality", fontsize=12.5,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.009, "Unsmoothed formal runs; iteration 0 is the evaluation baseline.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=2.0, w_pad=1.5)
    save_figure(fig, output, "part3_rewards")


def diagnostics_figure(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.0))
    labels = ("Gradient norm", "Importance ratio", "Importance ratio", "Standard deviation",
              "Fraction of groups", "Mean sampled -log p")
    for index, (ax, key, ylabel) in enumerate(zip(axes.flat, DIAGNOSTICS, labels)):
        configure_axis(ax, f"({chr(97 + index)}) {key}", ylabel)
        for run in runs:
            plot_series(ax, run, "train", key, marker="o")
        if key == "train/importance_weight_mean":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        elif key == "train/importance_weight_max":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.set_ylim(bottom=0.98)  # Autoscale the top: p3 reaches 2.2193.
        elif key == "train/frac_degenerate_groups":
            ax.set_ylim(-0.025, 1.055)
        else:
            ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=7.6, framealpha=0.9)
    fig.suptitle("Part 3: optimization, importance ratios, and data health", fontsize=12.5,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.008, "Missing update diagnostics at skipped iterations remain gaps, not zeros.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97), h_pad=1.8, w_pad=1.5)
    save_figure(fig, output, "part3_diagnostics")


def timing_figure(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    panels = (("(a) train/importance_weight_min", "Importance ratio"),
              ("(b) train/step_time_s", "Seconds"),
              ("(c) Cumulative learner-iteration time", "Minutes"),
              ("(d) train/learner_wait_frac", "Fraction"))
    keys = ("train/importance_weight_min", "train/step_time_s", None, "train/learner_wait_frac")
    for ax, (title, ylabel), key in zip(axes.flat, panels, keys):
        configure_axis(ax, title, ylabel)
        for run in runs:
            if key is None:
                cumulative = [0]
                for record in run["train"]:
                    cumulative.append(cumulative[-1] + record["train/step_time_s"] / 60)
                ax.plot(range(51), cumulative, color=run["spec"]["color"],
                        linestyle=run["spec"]["linestyle"], label=run["spec"]["label"], linewidth=1.4)
            else:
                plot_series(ax, run, "train", key, marker="o" if "weight" in key else None)
        ax.set_ylim(bottom=0)
        if key in ("train/importance_weight_min", "train/learner_wait_frac"):
            ax.set_ylim(-0.025, 1.055)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.92)
    fig.suptitle("Part 3: lower importance ratios and system timing", fontsize=12.5,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.009, "Timing includes all 50 iterations; cumulative time excludes final evaluation drain.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=2.0, w_pad=1.5)
    save_figure(fig, output, "part3_timing")


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
    reward_figure(runs, output)
    diagnostics_figure(runs, output)
    timing_figure(runs, output)
    sync, asynchronous = [run["summary"] for run in runs]
    payload = {
        "source": "Completed formal p2b and p3 only; probes excluded.",
        "timing_definition": "All 50 train/step_time_s values, including skipped iterations; "
                             "excludes cold start, baseline evaluation, and final evaluation drain.",
        "speedup_sync_over_async": sync["mean_step_seconds"] / asynchronous["mean_step_seconds"],
        "step_time_reduction_fraction": 1 - asynchronous["mean_step_seconds"] / sync["mean_step_seconds"],
        "runs": [run["summary"] for run in runs], "trajectory_examples": load_examples(),
    }
    (report_dir / "part3_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ("run_name", "training_iterations", "optimizer_updates", "skipped_iterations",
              "groups_consumed", "groups_dropped_stale", "retained_training_sequences",
              "training_action_tokens", "baseline_eval_success_rate", "final_eval_success_rate",
              "final_eval_pass_at_4", "mean_step_seconds", "median_step_seconds", "total_step_seconds",
              "time_weighted_learner_wait_fraction", "after_baseline_through_final_eval_seconds")
    with (report_dir / "part3_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(run["summary"] for run in runs)
    print(f"Wrote Part 3 figures (PNG/PDF) and summaries to {report_dir}")


if __name__ == "__main__":
    main()
