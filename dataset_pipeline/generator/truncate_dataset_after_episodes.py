"""
Drop the first K training episodes from dataset.csv using Attack steps recorded in outcome.txt.

If outcome.txt was appended across multiple runs, use --tail-episodes to take only the last N
episodes so the step sum matches the current dataset (default 6500).
"""
import argparse
import csv
import re
import sys
from pathlib import Path


def parse_attack_steps(outcome_path: Path) -> list[int]:
    text = outcome_path.read_text(encoding="utf-8", errors="replace")
    return [int(m.group(1)) for m in re.finditer(r"Attack steps:(\d+)", text)]


def count_valid_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for row in reader if len(row) == 7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Truncate dataset.csv after first K episodes.")
    parser.add_argument("--csv", type=Path, default=Path("dataset.csv"), help="Input CSV (default: ./dataset.csv)")
    parser.add_argument("--outcome", type=Path, required=True, help="outcome.txt from the same run")
    parser.add_argument("--skip-episodes", type=int, default=1500, help="Number of leading episodes to remove")
    parser.add_argument("--tail-episodes", type=int, default=6500, help="Use only the last N entries in outcome.txt")
    parser.add_argument("-o", "--output", type=Path, default=Path("dataset_after1500episodes.csv"), help="Output CSV path")
    args = parser.parse_args()

    steps = parse_attack_steps(args.outcome)
    if len(steps) < args.tail_episodes:
        print(
            f"error: outcome has {len(steps)} episodes, need at least --tail-episodes {args.tail_episodes}",
            file=sys.stderr,
        )
        sys.exit(1)

    tail = steps[-args.tail_episodes :]
    if args.skip_episodes > len(tail):
        print("error: --skip-episodes larger than --tail-episodes", file=sys.stderr)
        sys.exit(1)

    skip_rows = sum(tail[: args.skip_episodes])
    tail_sum = sum(tail)

    n_valid = count_valid_rows(args.csv)
    if n_valid != tail_sum:
        print(
            f"warning: dataset valid rows ({n_valid}) != sum of last {args.tail_episodes} outcome steps ({tail_sum}). "
            "Check --tail-episodes or that csv matches this outcome.",
            file=sys.stderr,
        )

    written = 0
    skipped = 0
    with args.csv.open(newline="", encoding="utf-8", errors="replace") as fin, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader, None)
        if not header or len(header) != 7:
            print("error: expected 7-column header in csv", file=sys.stderr)
            sys.exit(1)
        writer.writerow(header)

        for row in reader:
            if len(row) != 7:
                continue
            if skipped < skip_rows:
                skipped += 1
                continue
            writer.writerow(row)
            written += 1

    print(
        f"Skipped {skipped} transitions (first {args.skip_episodes} episodes in last {args.tail_episodes} outcome block). "
        f"Wrote {written} rows to {args.output.resolve()}."
    )


if __name__ == "__main__":
    main()
