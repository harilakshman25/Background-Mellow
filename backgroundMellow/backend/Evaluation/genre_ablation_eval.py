"""Evaluate whether genre changes cue selection and rendered audio.

This is intentionally separate from the API endpoint. It calls the same cue
decision function used by the endpoint, saves manifests, and optionally renders
the resulting cues through the existing specialist/superimposition pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_EVAL_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = _EVAL_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from Evaluation.evaluator import AudioEvaluator
from Tools.decide_audio import decide_audio_cues
from helper.audio_conversions import dict_to_cue
from helper.lib import init_models
from helper.parallel_audio_generation import parallel_audio_generation
from superimposition_model.superimposition_model import SuperimpositionModel

logger = logging.getLogger(__name__)


GENRE_ANCHORS = {
    "horror": "a tense, oppressive horror movie soundscape with unsettling ambience, ominous drones, eerie textures, and frightening suspense",
    "action": "an intense action movie soundscape with driving percussion, urgent rhythmic energy, explosive impacts, and fast kinetic momentum",
    "sci-fi": "a futuristic science fiction movie soundscape with synthetic textures, resonant technology, deep electronic pulses, and mysterious space-age ambience",
    "romance": "a warm romantic movie soundscape with tender harmonies, intimate ambience, gentle melodic emotion, and soft affectionate atmosphere",
    "none": "a neutral cinematic soundscape with balanced ambience, natural scene sounds, restrained music, and no strong genre-specific mood",
}

TEST_STORIES = [
    "A person enters a quiet building, walks across a room, and notices a light turning on in the distance.",
    "Two people wait at a station while a vehicle approaches and the surrounding crowd slowly moves away.",
    "A driver follows an unfamiliar road, passes several buildings, and stops near a bridge before sunset.",
    "A group gathers in a kitchen, prepares a meal, and pauses when someone knocks at the front door.",
    "A traveler opens an old map, crosses a field, and looks toward a town visible beyond the trees.",
]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _write_manifest(path: Path, story_id: str, story: str, genre: str, cues: list[Any], total_duration_ms: int) -> None:
    payload = {
        "story_id": story_id,
        "story": story,
        "requested_genre": genre,
        "total_duration_ms": total_duration_ms,
        "cues": [_jsonable(cue) for cue in cues],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_manifest_stats(manifest_records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "story_id",
        "genre",
        "cue_count",
        "audio_type_distribution",
        "mean_weight_db",
        "mean_duration_ms",
        "unique_audio_classes",
        "manifest_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in manifest_records:
            cues = record["cues"]
            types = Counter(str(cue.get("audio_type", "UNKNOWN")) for cue in cues)
            weights = [float(cue["weight_db"]) for cue in cues if cue.get("weight_db") is not None]
            durations = [float(cue["duration_ms"]) for cue in cues if cue.get("duration_ms") is not None]
            classes = sorted(
                {str(cue["audio_class"]) for cue in cues if cue.get("audio_class")}
            )
            writer.writerow(
                {
                    "story_id": record["story_id"],
                    "genre": record["genre"],
                    "cue_count": len(cues),
                    "audio_type_distribution": json.dumps(dict(sorted(types.items()))),
                    "mean_weight_db": _mean(weights),
                    "mean_duration_ms": _mean(durations),
                    "unique_audio_classes": json.dumps(classes, ensure_ascii=False),
                    "manifest_path": record["manifest_path"],
                }
            )


def _load_saved_manifests(manifest_dir: Path) -> list[dict[str, Any]]:
    records = []
    for manifest_path in sorted(manifest_dir.glob("*.json")):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        records.append(
            {
                "story_id": record["story_id"],
                "genre": record["requested_genre"],
                "story": record["story"],
                "total_duration_ms": record["total_duration_ms"],
                "cues": record.get("cues", []),
                "manifest_path": str(manifest_path),
            }
        )
    return records


def _load_anchor_embeddings(evaluator: AudioEvaluator) -> dict[str, torch.Tensor]:
    texts = [GENRE_ANCHORS[genre] for genre in GENRE_ANCHORS]
    with torch.no_grad():
        embeddings = evaluator.clap_model.get_text_embedding(
            texts, use_tensor=True
        )
    return {
        genre: embeddings[index]
        for index, genre in enumerate(GENRE_ANCHORS)
    }


def _audio_genre_scores(
    evaluator: AudioEvaluator,
    audio_path: Path,
    anchor_embeddings: dict[str, torch.Tensor],
) -> dict[str, float]:
    with torch.no_grad():
        audio_embedding = evaluator.clap_model.get_audio_embedding_from_filelist(
            x=[str(audio_path)], use_tensor=True
        )
        return {
            genre: float(
                torch.nn.functional.cosine_similarity(
                    audio_embedding, embedding.unsqueeze(0)
                ).item()
            )
            for genre, embedding in anchor_embeddings.items()
        }


def _write_confusion_outputs(
    rows: list[dict[str, Any]], output_dir: Path, genres: list[str]
) -> None:
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["requested_genre"]][row["predicted_genre"]] += 1

    matrix_path = output_dir / "confusion_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["requested_genre", *genres])
        for requested in genres:
            total = sum(counts[requested].values())
            writer.writerow(
                [
                    requested,
                    *[
                        (counts[requested][predicted] / total if total else 0.0)
                        for predicted in genres
                    ],
                ]
            )

    matrix = np.array(
        [
            [
                counts[requested][predicted] / max(sum(counts[requested].values()), 1)
                for predicted in genres
            ]
            for requested in genres
        ]
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(genres)), genres, rotation=30, ha="right")
    axis.set_yticks(range(len(genres)), genres)
    axis.set_xlabel("Predicted highest-similarity genre")
    axis.set_ylabel("Requested genre")
    axis.set_title("Genre CLAP confusion matrix")
    for row_index in range(len(genres)):
        for column_index in range(len(genres)):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Rate")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _print_summary(rows: list[dict[str, Any]], genres: list[str]) -> None:
    if not rows:
        print("No rendered audio results were available; confusion matrix was not populated.")
        return
    diagonal = sum(row["requested_genre"] == row["predicted_genre"] for row in rows)
    print(f"Overall diagonal accuracy: {diagonal / len(rows):.2%} ({diagonal}/{len(rows)})")

    pair_counts = Counter()
    for row in rows:
        requested = row["requested_genre"]
        predicted = row["predicted_genre"]
        if requested != predicted:
            pair_counts[tuple(sorted((requested, predicted)))] += 1
    if pair_counts:
        print("Most confused genre pairs:")
        for pair, count in pair_counts.most_common(5):
            print(f"  {pair[0]} <-> {pair[1]}: {count}")
    else:
        print("Most confused genre pairs: none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(_EVAL_DIR / "Results" / "genre_ablation"))
    parser.add_argument("--speed-wps", type=float, default=2.0)
    parser.add_argument("--skip-render", action="store_true", help="Only write manifests and manifest_stats.csv.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_dir = Path(args.output_dir).resolve()
    manifest_dir = output_dir / "manifests"
    audio_dir = output_dir / "audio"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    genres = list(GENRE_ANCHORS)
    manifest_records = []
    for story_index, story in enumerate(TEST_STORIES, start=1):
        story_id = f"story_{story_index:02d}"
        for genre in genres:
            logger.info("Deciding cues for %s / %s", story_id, genre)
            requested_genre = None if genre == "none" else genre
            try:
                cues, total_duration_ms = decide_audio_cues(
                    story,
                    args.speed_wps,
                    narrator_enabled=True,
                    movie_bgms_enabled=False,
                    genre=requested_genre,
                )
            finally:
                time.sleep(1.5)
            manifest_path = manifest_dir / f"{story_id}__{genre}.json"
            _write_manifest(manifest_path, story_id, story, genre, cues, total_duration_ms)
            manifest_records.append(
                {
                    "story_id": story_id,
                    "genre": genre,
                    "story": story,
                    "total_duration_ms": total_duration_ms,
                    "cues": [_jsonable(cue) for cue in cues],
                    "manifest_path": str(manifest_path),
                }
            )

    saved_records = _load_saved_manifests(manifest_dir)
    _write_manifest_stats(saved_records, output_dir / "manifest_stats.csv")
    print(f"Wrote {len(saved_records)} manifests and {output_dir / 'manifest_stats.csv'}")
    if args.skip_render:
        return

    logger.info("Initializing specialist models for rendering")
    init_models()
    superimposer = SuperimpositionModel()
    evaluator = AudioEvaluator()
    anchor_embeddings = _load_anchor_embeddings(evaluator)
    rendered_rows = []

    for record in saved_records:
        try:
            if not record["cues"]:
                logger.warning(
                    "Skipping render for %s / %s: manifest contains zero cues",
                    record["story_id"],
                    record["genre"],
                )
                continue
            cues = [dict_to_cue(cue) for cue in record["cues"]]
            wrapped_cues = parallel_audio_generation(cues)
            final_audio = superimposer.superimpose_audio_cues_with_audio_base64(
                record["story"], wrapped_cues, record["total_duration_ms"]
            )
            audio_path = audio_dir / f"{record['story_id']}__{record['genre']}.wav"
            final_audio.export(audio_path, format="wav")
            scores = _audio_genre_scores(evaluator, audio_path, anchor_embeddings)
            predicted_genre = max(scores, key=scores.get)
            rendered_rows.append(
                {
                    "story_id": record["story_id"],
                    "requested_genre": record["genre"],
                    "predicted_genre": predicted_genre,
                    "audio_path": str(audio_path),
                    "scores": scores,
                }
            )
        except Exception:
            logger.exception("Rendering failed for %s / %s", record["story_id"], record["genre"])

    (output_dir / "audio_genre_scores.json").write_text(
        json.dumps(rendered_rows, indent=2), encoding="utf-8"
    )
    _write_confusion_outputs(rendered_rows, output_dir, genres)
    _print_summary(rendered_rows, genres)


if __name__ == "__main__":
    main()