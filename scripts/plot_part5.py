"""Build Part 5 reporting artifacts from existing outputs; never run a model.

Requires Matplotlib, as do the earlier reporting helpers. All experiment
configs, logs, transcripts and checkpoints are read only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from plot_part2 import configure_axis, plot_series, require, save_figure


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "p5_textcraft_grpo"
DIAGNOSTICS = (
    "train/grad_norm", "train/importance_weight_mean", "train/importance_weight_max",
    "train/importance_weight_std", "train/frac_degenerate_groups", "train/entropy_proxy",
)
EVALUATIONS = (
    ("Fine-tuned", "medium", "ft_medium", "a3", 213),
    ("Fine-tuned", "hard", "ft_hard", "a3", 136),
    ("Base", "medium", "base_medium", "Qwen/Qwen3.5-4B", 213),
    ("Base", "hard", "base_hard", "Qwen/Qwen3.5-4B", 136),
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_training() -> dict:
    path = ROOT / "runs" / f"{RUN_NAME}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train = [row for row in rows if "train/success_rate" in row]
    evaluation = sorted([row for row in rows if "eval/success_rate" in row],
                        key=lambda row: row["step"])
    summaries = [row for row in rows if "summary/train_steps" in row]
    require(len(rows) == 57, "p5: unexpected record count")
    require([row["step"] for row in train] == list(range(1, 51)), "p5: incomplete training")
    require([row["step"] for row in evaluation] == [0, 10, 20, 30, 40, 50],
            "p5: incomplete evaluation")
    require(len(summaries) == 1 and summaries[0]["summary/train_steps"] == 50,
            "p5: missing final summary")
    require({key for row in rows for key in row
             if key.startswith(("train/pass@", "eval/pass@"))}
            == {"train/pass@8", "eval/pass@4"}, "p5: unexpected pass metric")
    for row in rows:
        require(all(not isinstance(value, float) or math.isfinite(value)
                    for value in row.values()), "p5: nonfinite metric")
    for row in train:
        require(row["train/num_groups"] == 16 and not row.get("train/skipped"),
                "p5: unexpected groups or skipped update")
        require(row["train/policy_version"] == row["step"], "p5: incorrect policy version")
        require(set(DIAGNOSTICS) <= row.keys(), "p5: missing diagnostic")
    for row in evaluation:
        require(row["eval/num_tasks"] == 50 and row["eval/episodes_per_task"] == 4,
                "p5: periodic evaluation settings differ")
        require(row["eval/policy_version"] == row["step"], "p5: incorrect evaluation version")
    manifest = json.loads((ROOT / "runs" / f"{RUN_NAME}.checkpoints.json").read_text())
    require(manifest["completed_steps"] == 50 and manifest["resumes"] == 0,
            "p5: unexpected completion/resume count")
    require([item["name"] for item in manifest["snapshots"]]
            == ["baseline", "v0"] + [f"v{i}" for i in range(1, 51)], "p5: snapshots differ")
    require(manifest["final_sampler_path"] == manifest["snapshots"][-1]["path"],
            "p5: incorrect final sampler")
    require(manifest.get("hf_model_id"), "p5: missing public export ID")
    step_seconds = sum(row["train/step_time_s"] for row in train)
    summary = {
        "run_name": RUN_NAME, "training_iterations": 50, "optimizer_updates": 50,
        "groups_consumed": summaries[0]["summary/groups_consumed"],
        "groups_dropped_stale": summaries[0]["summary/groups_dropped_stale"],
        "retained_training_sequences": sum(row["train/num_sequences"] for row in train),
        "training_action_tokens": sum(row["train/num_action_tokens"] for row in train),
        "mean_step_seconds": statistics.mean(row["train/step_time_s"] for row in train),
        "total_step_seconds": step_seconds,
        "time_weighted_learner_wait_fraction":
            sum(row["train/learner_wait_s"] for row in train) / step_seconds,
        "after_baseline_through_final_eval_seconds": evaluation[-1]["time"] - evaluation[0]["time"],
        "final_eval_tail_seconds": evaluation[-1]["time"] - train[-1]["time"],
        "abandoned_train_episodes": sum(row["train/abandoned_episodes"] for row in train),
        "abandoned_periodic_eval_episodes": sum(row["eval/abandoned_episodes"] for row in evaluation),
        "periodic_evaluations": evaluation,
        "final_sampler_path": manifest["final_sampler_path"],
        "hf_model_id": manifest["hf_model_id"],
        "log_path": str(path.relative_to(ROOT)), "log_sha256": sha256(path),
    }
    require(summary["groups_consumed"] == 800, "p5: consumed-group count differs")
    for key in DIAGNOSTICS + ("train/importance_weight_min", "train/success_rate", "train/pass@8",
                              "train/step_time_s", "train/episode_time_mean_s"):
        values = [row[key] for row in train]
        summary[key] = {"first": values[0], "last": values[-1], "min": min(values),
                        "max": max(values), "mean": statistics.mean(values), "observations": len(values)}
    return {"spec": {"name": RUN_NAME, "label": "p5", "color": "#2166ac", "linestyle": "-"},
            "train": train, "eval": evaluation, "summary": summary}


def load_standalone() -> list[dict]:
    results = []
    decoder = json.JSONDecoder()
    for label, split, stem, model, tasks in EVALUATIONS:
        path = ROOT / "runs" / f"p5_eval_{stem}.console.log"
        text = path.read_text()
        candidates = []
        for match in re.finditer(r"^\{", text, flags=re.MULTILINE):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except ValueError:
                continue
            if isinstance(value, dict) and "success_rate" in value:
                candidates.append(value)
        require(len(candidates) == 1, f"{stem}: expected exactly one final result")
        value = candidates[0]
        require(value["model"] == model and value["num_tasks"] == tasks
                and value["episodes_per_task"] == 4, f"{stem}: incorrect evaluation settings")
        episodes = tasks * 4
        successes = round(value["success_rate"] * episodes)
        passed_tasks = round(value["pass@4"] * tasks)
        require(math.isclose(value["success_rate"], successes / episodes)
                and math.isclose(value["pass@4"], passed_tasks / tasks), f"{stem}: nonintegral scores")
        warnings = [line for line in text.splitlines() if "[rollout] WARNING:" in line]
        require(len(warnings) == value["abandoned_episodes"], f"{stem}: abandoned count differs")
        failures = Counter()
        for line in warnings:
            if "TypeError:" in line:
                failures["parser_TypeError"] += 1
            elif "HTTPStatusError:" in line:
                match = re.search(r"'(\d{3}) ", line)
                require(match is not None, f"{stem}: unclassified HTTP error")
                failures[f"HTTP_{match[1]}"] += 1
            else:
                match = re.search(r"failure \((\w+):", line)
                require(match is not None, f"{stem}: unclassified warning")
                failures[match[1]] += 1
        results.append({"model_label": label, "split": split, "num_episodes": episodes,
                        "successful_episodes": successes, "passed_tasks": passed_tasks,
                        **value, "failure_categories": dict(sorted(failures.items())),
                        "console_path": str(path.relative_to(ROOT)), "console_sha256": sha256(path)})
    return results


def hard_budget_audit() -> dict:
    """Static lower bound, independent of any agent/model or environment run."""
    book = json.loads((ROOT / "data/textcraft_recipes.json").read_text())
    recipes, base_items = book["recipes"], set(book["base_items"])
    require(all(len(options) == 1 for options in recipes.values()), "Recipes are not unique")
    result = {}
    for filename in ("textcraft_train.jsonl", "textcraft_val.jsonl", "textcraft_val_hard.jsonl"):
        tasks = [json.loads(line) for line in (ROOT / "data" / filename).read_text().splitlines()]
        minimum_crafts, impossible = [], 0
        for task in tasks:
            info = task["info"]
            require(set(info["initial_inventory"]) <= base_items, "Initially crafted item breaks bound")
            needed: set[str] = set()

            def visit(item: str) -> None:
                if item in base_items or item in needed:
                    return
                require(item in recipes, f"Unknown target/ingredient: {item}")
                needed.add(item)
                for ingredient, count in recipes[item][0]["ingredients"].items():
                    require(count > 0, "Nonpositive ingredient breaks bound")
                    visit(ingredient)

            for item, count in info["target_items"].items():
                require(count > 0, "Nonpositive target breaks bound")
                visit(item)
            minimum_crafts.append(len(needed))
            impossible += len(needed) + 1 > task["max_steps"]
        result[filename] = {"tasks": len(tasks), "min_required_craft_calls": min(minimum_crafts),
                            "max_required_craft_calls": max(minimum_crafts),
                            "mean_required_craft_calls": statistics.mean(minimum_crafts),
                            "craft_plus_finish_exceeds_budget": impossible}
    return result


def figures(run: dict, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    for index, (ax, phase, key) in enumerate(zip(axes.flat, ("train", "eval", "train", "eval"),
            ("train/success_rate", "eval/success_rate", "train/pass@8", "eval/pass@4"))):
        configure_axis(ax, f"({chr(97 + index)}) {key}", "Fraction")
        ax.set_ylim(-0.025, 1.055)
        plot_series(ax, run, phase, key, label=key, marker="o" if phase == "eval" else None)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Part 5: TextCraft-Synth reward quality", fontsize=12.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.009, "Periodic evaluation: first 50 medium tasks, k=4; not the full-split comparison.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=2.0, w_pad=1.5)
    save_figure(fig, output, "part5_rewards")

    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.0))
    labels = ("Gradient norm", "Importance ratio", "Importance ratio", "Standard deviation",
              "Fraction of groups", "Mean sampled -log p")
    for index, (ax, key, ylabel) in enumerate(zip(axes.flat, DIAGNOSTICS, labels)):
        configure_axis(ax, f"({chr(97 + index)}) {key}", ylabel)
        plot_series(ax, run, "train", key, label="p5", marker="o")
        if key == "train/importance_weight_mean":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        elif key == "train/importance_weight_max":
            ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
            ax.set_ylim(bottom=0.98)
        elif key == "train/frac_degenerate_groups":
            ax.set_ylim(-0.025, 1.055)
        else:
            ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Part 5: optimization, policy drift, and data health", fontsize=12.5,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.008, "All 50 iterations updated; raw action-token ratios are measured before clipping.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97), h_pad=1.8, w_pad=1.5)
    save_figure(fig, output, "part5_diagnostics")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    keys = ("train/importance_weight_min", "train/step_time_s", "train/learner_wait_frac", None)
    titles = ("(a) train/importance_weight_min", "(b) train/step_time_s",
              "(c) train/learner_wait_frac", "(d) eval/avg_steps (50-task subset)")
    for ax, key, title in zip(axes.flat, keys, titles):
        configure_axis(ax, title, "Seconds" if key == "train/step_time_s" else
                       "Environment actions" if key is None else "Fraction / ratio")
        plot_series(ax, run, "eval" if key is None else "train", key or "eval/avg_steps",
                    label=key or "eval/avg_steps", marker="o" if key is None or "weight" in key else None)
        ax.set_ylim(bottom=0)
        if key in ("train/importance_weight_min", "train/learner_wait_frac"):
            ax.set_ylim(-0.025, 1.055)
        ax.legend(loc="best", fontsize=7.5)
    fig.suptitle("Part 5: lower importance ratios and system timing", fontsize=12.5,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.009, "Unsmoothed data; learner intervals exclude the final evaluation drain.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=2.0, w_pad=1.5)
    save_figure(fig, output, "part5_timing")


def main() -> None:
    run = load_training()
    standalone = load_standalone()
    report = ROOT / "report"
    output = report / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures(run, output)
    payload = {"source": "Original completed formal hero run and four complete standalone outputs.",
               "failure_protocol": "All abandoned episodes retained as failures in unchanged denominators.",
               "training": run["summary"], "standalone_evaluations": standalone,
               "static_budget_audit": hard_budget_audit()}
    (report / "part5_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (report / "part5_summary.csv").open("w", newline="") as stream:
        fields = ("model_label", "split", "num_tasks", "num_episodes", "successful_episodes",
                  "passed_tasks", "success_rate", "pass@4", "avg_steps", "abandoned_episodes")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(standalone)
    print("Wrote three Part 5 figures (PNG/PDF) and JSON/CSV summaries.")


if __name__ == "__main__":
    main()
