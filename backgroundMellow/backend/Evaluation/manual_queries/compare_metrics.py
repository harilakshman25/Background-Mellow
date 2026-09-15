"""Compare locally saved manual-query metric JSON files."""

import csv
import json
import sys
from pathlib import Path


def load_records(folder: Path) -> list[dict]:
    records = []
    for path in sorted(folder.glob("*-metrics.json")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        automated = data.get("automatedMetrics", {})
        human = data.get("humanScores", {})
        records.append(
            {
                "genre": data.get("genre", path.stem.removesuffix("-metrics")),
                "clap_score": automated.get("clapScore", ""),
                "spectral_richness": automated.get("spectralRichness", ""),
                "noise_floor_db": automated.get("noiseFloor", ""),
                "audio_onsets": automated.get("audioOnsets", ""),
                "human_final_score": data.get("finalScore", ""),
                "sync_accuracy": human.get("syncAccuracy", ""),
                "semantic_fit": human.get("semanticFit", ""),
                "acoustic_quality": human.get("acousticQuality", ""),
                "narrative_flow": human.get("narrativeFlow", ""),
                "cinematic_impact": human.get("cinematicImpact", ""),
                "metrics_file": path.name,
                "audio_file": f"{data.get('genre', path.stem.removesuffix('-metrics'))}.wav",
            }
        )
    return records


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "corridor_scene"
    records = load_records(folder)
    if not records:
        raise SystemExit(f"No *-metrics.json files found in {folder}")

    output_path = folder / "comparison.csv"
    fieldnames = list(records[0])
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {output_path}")
    print("\nGenre comparison:")
    for record in records:
        print(
            f"{record['genre']}: human={record['human_final_score']}/10, "
            f"CLAP={record['clap_score']}, onsets={record['audio_onsets']}"
        )


if __name__ == "__main__":
    main()