import csv
import glob
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GENRES = ("horror", "sci-fi", "action")
CSV_PATTERN = re.compile(r"^(genre|baseline)_val_losses_seed(\d+)\.csv$")
METRICS = ("val_total", "val_start", "val_duration")


def _load_results():
    results = defaultdict(dict)
    for path in glob.glob("*_val_losses_seed*.csv"):
        match = CSV_PATTERN.match(path.rsplit("/", 1)[-1])
        if not match:
            continue
        arm, seed = match.group(1), int(match.group(2))
        with open(path, newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        for genre in GENRES:
            genre_rows = [
                {key: (value if key in ("genre",) else float(value))
                 for key, value in row.items()}
                for row in rows
                if row["genre"] == genre
            ]
            if genre_rows:
                results[seed][arm, genre] = {
                    "final": max(genre_rows, key=lambda row: row["epoch"]),
                    "best": min(genre_rows, key=lambda row: row["val_total"]),
                }
    return results


def _print_results(results):
    for genre in GENRES:
        print(f"\n{genre}")
        for seed in sorted(results):
            if not all((arm, genre) in results[seed] for arm in ("baseline", "genre")):
                continue
            print(f"seed {seed}:")
            for point in ("final", "best"):
                print(f"  {point}:")
                for metric in METRICS:
                    baseline = results[seed]["baseline", genre][point][metric]
                    genre_value = results[seed]["genre", genre][point][metric]
                    print(
                        f"    {metric}: baseline={baseline:.6f}, "
                        f"genre={genre_value:.6f}, "
                        f"difference={genre_value - baseline:.6f}"
                    )

        for point in ("final", "best"):
            print(f"  {point} mean +/- std:")
            for metric in METRICS:
                for arm in ("baseline", "genre"):
                    values = [
                        results[seed][arm, genre][point][metric]
                        for seed in results
                        if (arm, genre) in results[seed]
                    ]
                    if values:
                        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
                        print(
                            f"    {arm} {metric}: "
                            f"{statistics.mean(values):.6f} +/- {deviation:.6f}"
                        )
                differences = [
                    results[seed]["genre", genre][point][metric]
                    - results[seed]["baseline", genre][point][metric]
                    for seed in results
                    if ("baseline", genre) in results[seed]
                    and ("genre", genre) in results[seed]
                ]
                if differences:
                    deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
                    print(
                        f"    difference {metric}: "
                        f"{statistics.mean(differences):.6f} +/- {deviation:.6f}"
                    )


def _save_chart(results):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    positions = range(len(GENRES))
    width = 0.35
    for axis, metric, title in zip(
        axes,
        ("val_start", "val_duration"),
        ("Validation start MSE", "Validation duration MSE"),
    ):
        values_by_arm = {}
        for arm in ("baseline", "genre"):
            values_by_arm[arm] = []
            for genre in GENRES:
                values = [
                    results[seed][arm, genre]["final"][metric]
                    for seed in results
                    if (arm, genre) in results[seed]
                ]
                values_by_arm[arm].append(statistics.mean(values) if values else float("nan"))
        axis.bar([p - width / 2 for p in positions], values_by_arm["baseline"], width, label="baseline")
        axis.bar([p + width / 2 for p in positions], values_by_arm["genre"], width, label="genre")
        axis.set_xticks(list(positions), GENRES)
        axis.set_title(title)
        axis.set_ylabel("MSE")
        axis.legend()
    figure.tight_layout()
    figure.savefig("ablation_comparison.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    results = _load_results()
    _print_results(results)
    _save_chart(results)
