#!/usr/bin/env python3
"""Inspect DuReaderQG jsonl data for the T5 QA project."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


DEFAULT_TRAIN = Path("/Users/yunye/Documents/培训/丁师兄/实战/data/DuReaderQG/train.json")
DEFAULT_DEV = Path("/Users/yunye/Documents/培训/丁师兄/实战/data/DuReaderQG/dev.json")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def describe_lengths(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(mean(ordered), 2),
        "median": median(ordered),
        "p90": ordered[int(len(ordered) * 0.9)],
        "p95": ordered[int(len(ordered) * 0.95)],
    }


def profile_split(name: str, rows: list[dict]) -> dict:
    required = ("context", "question", "answer", "id")
    field_counter = Counter()
    missing_counter = Counter()
    context_lengths: list[int] = []
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    answer_in_context = 0

    for row in rows:
        field_counter.update(row.keys())
        for field in required:
            if field not in row or row[field] in ("", None):
                missing_counter[field] += 1

        context = str(row.get("context", ""))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))

        context_lengths.append(len(context))
        question_lengths.append(len(question))
        answer_lengths.append(len(answer))
        if answer and answer in context:
            answer_in_context += 1

    return {
        "split": name,
        "num_rows": len(rows),
        "fields": dict(field_counter),
        "missing_required_fields": dict(missing_counter),
        "context_char_length": describe_lengths(context_lengths),
        "question_char_length": describe_lengths(question_lengths),
        "answer_char_length": describe_lengths(answer_lengths),
        "answer_in_context_count": answer_in_context,
        "answer_in_context_ratio": round(answer_in_context / len(rows), 4) if rows else 0,
    }


def build_model_input(row: dict) -> str:
    return f"question: {row['question']} context: {row['context']}"


def write_samples(output_path: Path, train_rows: list[dict], dev_rows: list[dict], n: int) -> None:
    selected = [("train", row) for row in train_rows[:n]]
    selected += [("dev", row) for row in dev_rows[:n]]
    with output_path.open("w", encoding="utf-8") as f:
        for split, row in selected:
            item = {
                "split": split,
                "id": row.get("id"),
                "model_input": build_model_input(row),
                "target_answer": row.get("answer"),
                "question": row.get("question"),
                "context": row.get("context"),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)

    profile = {
        "task": "Chinese generative question answering",
        "model_input_template": "question: <question> context: <context>",
        "target_template": "<answer>",
        "train_path": str(args.train),
        "dev_path": str(args.dev),
        "splits": [
            profile_split("train", train_rows),
            profile_split("dev", dev_rows),
        ],
    }

    profile_path = args.output_dir / "data_profile.json"
    sample_path = args.output_dir / "sample_cases.jsonl"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    write_samples(sample_path, train_rows, dev_rows, args.sample_count)

    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print(f"\nWrote: {profile_path}")
    print(f"Wrote: {sample_path}")


if __name__ == "__main__":
    main()
