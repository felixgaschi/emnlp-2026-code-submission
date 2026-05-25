#!/usr/bin/env python3
"""Compute per-language win rates between two merged result directories."""

import argparse
import csv
from pathlib import Path


DEFAULT_TASK_METRICS = {
    "xnli": "accuracy",
    "wikiann": "f1",
    "xtreme_r.udpos": "accuracy",
}


def parse_task_metric(values):
    task_metrics = {}
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(
                f"Expected TASK:METRIC, got {value!r}"
            )
        task, metric = value.split(":", 1)
        task_metrics[task] = metric
    return task_metrics


def read_rows(path):
    with path.open(newline="") as f:
        return {row["seed"]: row for row in csv.DictReader(f)}


def metric_columns(row, metric):
    suffix = f"_{metric}"
    cols = []
    for col in row:
        if not col.startswith("final_eval_") or not col.endswith(suffix):
            continue
        lang = col[len("final_eval_") : -len(suffix)]
        if lang not in {"avg", "same"}:
            cols.append(col)
    return cols


def as_float(row, col):
    value = row.get(col, "")
    if value == "":
        return None
    return float(value)


def summarize(task, seed, metric, pairs, left_name, right_name):
    left_wins = sum(1 for left, right in pairs if left > right)
    right_wins = sum(1 for left, right in pairs if right > left)
    ties = sum(1 for left, right in pairs if left == right)
    n = len(pairs)
    deltas = [left - right for left, right in pairs]

    return {
        "task": task,
        "seed": seed,
        "metric": metric,
        "n_common_langs_or_pairs": n,
        f"{left_name}_wins": left_wins,
        f"{right_name}_wins": right_wins,
        "ties": ties,
        f"{left_name}_win_rate": left_wins / n if n else "",
        f"{right_name}_win_rate": right_wins / n if n else "",
        f"mean_delta_{left_name}_minus_{right_name}": (
            sum(deltas) / len(deltas) if deltas else ""
        ),
        f"{left_name}_common_mean": (
            sum(left for left, _ in pairs) / n if n else ""
        ),
        f"{right_name}_common_mean": (
            sum(right for _, right in pairs) / n if n else ""
        ),
        f"{left_name}_same": "",
        f"{right_name}_same": "",
        f"same_delta_{left_name}_minus_{right_name}": "",
    }


def result_filename(task, steps, batch_size):
    return (
        "xlm-roberta-large__mix_opus100_nllb__before_noaligner__"
        f"{task}__{steps}__{batch_size}.csv"
    )


def compute_win_rates(args):
    task_metrics = dict(DEFAULT_TASK_METRICS)
    if args.task_metric:
        task_metrics.update(parse_task_metric(args.task_metric))

    rows = []
    overall_pairs = []
    for task, metric in task_metrics.items():
        filename = result_filename(task, args.steps, args.batch_size)
        left_rows = read_rows(args.left / filename)
        right_rows = read_rows(args.right / filename)
        common_seeds = sorted(set(left_rows) & set(right_rows), key=int)

        task_pairs = []
        for seed in common_seeds:
            left_row = left_rows[seed]
            right_row = right_rows[seed]
            common_cols = sorted(
                set(metric_columns(left_row, metric))
                & set(metric_columns(right_row, metric))
            )
            pairs = []
            for col in common_cols:
                left_value = as_float(left_row, col)
                right_value = as_float(right_row, col)
                if left_value is not None and right_value is not None:
                    pairs.append((left_value, right_value))

            row = summarize(task, seed, metric, pairs, args.left_name, args.right_name)
            same_col = f"final_eval_same_{metric}"
            left_same = as_float(left_row, same_col) if same_col in left_row else None
            right_same = as_float(right_row, same_col) if same_col in right_row else None
            row[f"{args.left_name}_same"] = left_same if left_same is not None else ""
            row[f"{args.right_name}_same"] = (
                right_same if right_same is not None else ""
            )
            row[f"same_delta_{args.left_name}_minus_{args.right_name}"] = (
                left_same - right_same
                if left_same is not None and right_same is not None
                else ""
            )
            rows.append(row)
            task_pairs.extend(pairs)

        rows.append(summarize(task, "all", metric, task_pairs, args.left_name, args.right_name))
        overall_pairs.extend(task_pairs)

    rows.append(
        summarize("all_tasks", "all", "mixed", overall_pairs, args.left_name, args.right_name)
    )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-language win rates between merged result CSVs."
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--steps", type=int, default=32000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--task-metric",
        action="append",
        help="Override or add a task metric as TASK:METRIC. Can be repeated.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = compute_win_rates(args)
    if not rows:
        raise SystemExit("No comparable rows found.")

    if args.output is None:
        args.output = args.left / f"win_rates_vs_{args.right_name}.csv"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(args.output)
    for row in rows:
        left_rate = float(row[f"{args.left_name}_win_rate"])
        right_rate = float(row[f"{args.right_name}_win_rate"])
        delta = float(row[f"mean_delta_{args.left_name}_minus_{args.right_name}"])
        print(
            f"{row['task']} seed {row['seed']}: "
            f"{args.left_name} {row[f'{args.left_name}_wins']}/"
            f"{row['n_common_langs_or_pairs']} ({left_rate:.3f}), "
            f"{args.right_name} {row[f'{args.right_name}_wins']}/"
            f"{row['n_common_langs_or_pairs']} ({right_rate:.3f}), "
            f"delta_mean={delta:.4f}"
        )


if __name__ == "__main__":
    main()
