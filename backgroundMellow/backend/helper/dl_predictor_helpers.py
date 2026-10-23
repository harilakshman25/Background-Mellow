import os
import random
import sys
import argparse
import csv
import math
import shutil
# Add project root to path
# import sys
# import os

# # Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# # Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root added to sys.path:", project_root)
    
# Cinemaudio-studio root (for tango_new when using Tango2)
cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
if cinema_studio_root not in sys.path:
    sys.path.append(cinema_studio_root)   
import json
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from model.dl_based_alignment_predictor import CinematicMixPredictor, GENRE_VOCAB
from helper.audio_conversions import audio_to_base64
from helper.lib import ParlerTTSModel
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from helper.dl_conversions import normalize_targets
from typing import List

DATASET_PATH = "../data/yt_videos/generated_train_dataset.pkl"
logger = logging.getLogger(__name__)
YOUTUBE_DATASET_PATH = "../data/yt_videos/yt_dataset_with_genre.jsonl"
MODEL_PATH = "model/dl_based_alignment_predictor.pth"
GENRE_MODEL_PATH = "model/dl_based_alignment_predictor_genre.pth"
BASELINE_MODEL_PATH = "model/dl_based_alignment_predictor_baseline.pth"
ORIGINAL_MODEL_PATH = "model/dl_based_alignment_predictor_original.pth"
GENRE_VAL_LOSS_PATH = "genre_val_losses.csv"
BASELINE_VAL_LOSS_PATH = "baseline_val_losses.csv"
EMBEDDER_PATH = "model/embedder.pth"

# Save the dataset to a file for reuse to avoid recalculation each time
import pickle

def create_dataset():

    descriptions = [
        "A male speaker with a neutral tone delivers his words clearly and confidently in a casual, everyday setting.",
        "A female speaker with a high-pitched voice is delivering her speech at a really fast speed in a noisy environment.",
        "A male speaker with a low-pitched voice is delivering his speech at a really slow speed in a quiet environment.",
        "A female speaker with a low-pitched voice is delivering her speech at a really fast speed in a noisy environment."
    ]
    datapath = YOUTUBE_DATASET_PATH
    dataset = []
    idx = 0
    with open(datapath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            story_prompt = data["story_prompt"]

            random_description = random.choice(descriptions)

            narrator_audio_segment = ParlerTTSModel.generate(prompt=story_prompt, description=random_description)
            narrator_audio_base64 = audio_to_base64(narrator_audio_segment)
            
            narrator_end_time = narrator_audio_segment.duration_seconds

            audio_classes = []
            clip_duration_sec = float(data.get("clip_duration_sec", 10.0))
            for audio_cues in data["cues"]:
                
                
                
                audio_classes.append({
                    "audio_class": audio_cues["audio_class"],
                    "weight_db": audio_cues["weight_db"],
                    "start_time_ms": audio_cues["starting_time"]/clip_duration_sec * narrator_end_time,
                    "duration_ms": audio_cues["duration"]/clip_duration_sec * narrator_end_time,
                })
                logger.info(f"entry {idx+1} audio_class: {audio_cues['audio_class']} start_time_ms: {audio_cues['starting_time']/clip_duration_sec * narrator_end_time} duration_ms: {audio_cues['duration']/clip_duration_sec * narrator_end_time}")
            
            
            dataset.append((
                story_prompt,
                audio_classes,
                narrator_audio_base64,
                clip_duration_sec,
                data["genre"],
            ))

            # Save dataset after each entry processed
            with open(DATASET_PATH, "wb") as dataset_file:
                pickle.dump(dataset, dataset_file)

            # Log the progress to indicate which entry has been processed
            logger.info(f"Processed and saved entry {idx+1}")
            idx += 1

    return dataset


def load_dataset():
    with open(DATASET_PATH, "rb") as dataset_file:
        return pickle.load(dataset_file)


def make_genre_stratified_split(dataset, val_fraction=0.15, min_count_for_split=30):
    """Split whole stories by genre, keeping small genres in the training set."""
    stories_by_genre = {}
    for item in dataset:
        genre = item[4] if len(item) >= 5 else "general"
        story_prompt = item[0]
        stories_by_genre.setdefault(genre, {}).setdefault(story_prompt, []).append(item)

    train_dataset = []
    val_dataset = []
    all_genres = sorted(stories_by_genre)
    for genre in all_genres:
        stories = list(stories_by_genre[genre].values())
        random.shuffle(stories)
        if len(stories) >= min_count_for_split:
            val_count = min(max(1, math.ceil(len(stories) * val_fraction)), len(stories) - 1)
            val_stories = stories[:val_count]
            train_stories = stories[val_count:]
        else:
            train_stories = stories
            val_stories = []
        train_dataset.extend(item for story in train_stories for item in story)
        val_dataset.extend(item for story in val_stories for item in story)
        print(f"{genre}: train_count={len(train_stories)}, val_count={len(val_stories)}")

    return train_dataset, val_dataset


def _collate_timeline_batch(batch):
    """Collate timeline samples with padding and optional genre IDs."""
    story_embs = torch.stack([b[0] for b in batch])
    class_embs = torch.stack([b[1] for b in batch])
    timeline_seqs = [b[2] for b in batch]
    targets = torch.stack([b[3] for b in batch])
    genre_ids = torch.stack([
        b[4] if len(b) > 4 else torch.tensor(GENRE_VOCAB["general"], dtype=torch.long)
        for b in batch
    ])

    max_T = max(seq.shape[0] for seq in timeline_seqs)
    embed_plus_3 = timeline_seqs[0].shape[1]
    device = timeline_seqs[0].device
    dtype = timeline_seqs[0].dtype

    padded = torch.zeros(len(batch), max_T, embed_plus_3, device=device, dtype=dtype)
    key_padding_mask = torch.ones(len(batch), max_T, dtype=torch.bool, device=device)
    for i, seq in enumerate(timeline_seqs):
        T_i = seq.shape[0]
        padded[i, :T_i] = seq
        key_padding_mask[i, :T_i] = False

    return story_embs, class_embs, padded, key_padding_mask, genre_ids, targets


class TimelineDataset(Dataset):
    """Dataset of (story_emb, class_emb, timeline_seq, target) for full-timeline alignment."""

    def __init__(self, samples: List[tuple]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _build_samples(dataset, model, force_genre_general=False):
    samples = []
    num_stories = len(dataset)

    for story_idx, item in enumerate(dataset):
        if len(item) == 4:
            story_prompt, audio_classes, narrator_audio_base64, scene_duration = item
            genre = "general"
        elif len(item) >= 5:
            story_prompt, audio_classes, narrator_audio_base64, scene_duration, genre = item[:5]
        else:
            story_prompt, audio_classes, narrator_audio_base64 = item
            scene_duration = 10.0
            genre = "general"
            logger.debug("[train_model] Entry has no clip_duration_sec; using default %.1fs", scene_duration)

        logger.info(
            "[train_model] Building examples for story %d/%d (%d cues), scene_duration=%.1fs",
            story_idx + 1, num_stories, len(audio_classes), scene_duration,
        )
        whisper_json = model.make_whisper_embedding(story_prompt, narrator_audio_base64)
        timeline_seq = model.get_whisper_sequence_tensor(whisper_json or [], scene_duration)

        story_emb = model.embedder.encode(
            story_prompt, convert_to_tensor=True, device=str(model.device)
        )
        genre_id = GENRE_VOCAB["general"] if force_genre_general else GENRE_VOCAB.get(
            genre, GENRE_VOCAB["general"]
        )
        genre_id = torch.tensor(genre_id, dtype=torch.long, device=model.device)

        for cue in audio_classes:
            audio_class = cue["audio_class"]
            class_emb = model.embedder.encode(
                audio_class, convert_to_tensor=True, device=str(model.device)
            )
            s_norm, w_norm, d_norm = normalize_targets(
                cue["start_time_ms"], cue["weight_db"], cue["duration_ms"],
            )
            y = torch.tensor([s_norm, w_norm, d_norm], dtype=torch.float32, device=model.device)
            samples.append((story_emb, class_emb, timeline_seq, y, genre_id))
    return samples


def _evaluate_by_genre(model, val_loader, genre):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            story_emb, class_emb, timeline_seq, key_padding_mask, genre_ids, targets = batch
            outputs = model(
                story_emb.to(model.device),
                class_emb.to(model.device),
                timeline_seq.to(model.device),
                key_padding_mask.to(model.device),
                genre_ids.to(model.device),
            )
            per_sample_loss = ((outputs - targets.to(model.device)) ** 2).sum(dim=1)
            losses.extend(per_sample_loss.tolist())
    return sum(losses) / len(losses) if losses else float("nan")


def _train_with_validation(model, train_loader, val_loaders, epochs, learning_rate, csv_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    losses = []
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "genre", "val_loss", "sample_count"])
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            for batch in train_loader:
                story_emb, class_emb, timeline_seq, key_padding_mask, genre_ids, targets = batch
                story_emb = story_emb.to(model.device)
                class_emb = class_emb.to(model.device)
                timeline_seq = timeline_seq.to(model.device)
                key_padding_mask = key_padding_mask.to(model.device)
                genre_ids = genre_ids.to(model.device)
                targets = targets.to(model.device)
                optimizer.zero_grad()
                outputs = model(story_emb, class_emb, timeline_seq, key_padding_mask, genre_ids)
                loss = torch.nn.functional.mse_loss(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            epoch_loss /= max(batch_count, 1)
            losses.append(epoch_loss)
            val_losses = {}
            for genre, loader in val_loaders.items():
                val_losses[genre] = _evaluate_by_genre(model, loader, genre)
                writer.writerow([
                    epoch + 1,
                    genre,
                    val_losses[genre],
                    len(loader.dataset),
                ])
            scheduler.step(epoch_loss)
            logger.info("Epoch %d/%d: train_loss=%.6f val_loss=%s", epoch + 1, epochs, epoch_loss, val_losses)
    return losses


def _backup_original_checkpoint():
    if not os.path.exists(ORIGINAL_MODEL_PATH) and os.path.exists(MODEL_PATH):
        shutil.copyfile(MODEL_PATH, ORIGINAL_MODEL_PATH)
        logger.info("Backed up original checkpoint to %s", ORIGINAL_MODEL_PATH)


def train_model(epochs=50, learning_rate=5e-4, force_genre_general=False, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    _backup_original_checkpoint()
    dataset = load_dataset()
    logger.info("[train_model] Loaded dataset with %d entries", len(dataset))
    train_entries, val_entries = make_genre_stratified_split(dataset)
    model = CinematicMixPredictor(load_checkpoint=False)
    train_samples = _build_samples(train_entries, model, force_genre_general)

    train_dataset = TimelineDataset(train_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=_collate_timeline_batch,
    )
    val_loaders = {}
    for genre in sorted({item[4] for item in val_entries}):
        genre_entries = [item for item in val_entries if item[4] == genre]
        genre_samples = _build_samples(genre_entries, model, force_genre_general)
        val_loaders[genre] = DataLoader(
            TimelineDataset(genre_samples),
            batch_size=8,
            shuffle=False,
            collate_fn=_collate_timeline_batch,
        )
    output_path = BASELINE_MODEL_PATH if force_genre_general else GENRE_MODEL_PATH
    csv_path = BASELINE_VAL_LOSS_PATH if force_genre_general else GENRE_VAL_LOSS_PATH
    losses = _train_with_validation(model, train_loader, val_loaders, epochs, learning_rate, csv_path)
    model.save_model(output_path)
    logger.info("Saved model to %s", output_path)
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-genre-general", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train_model(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        force_genre_general=args.force_genre_general,
        seed=args.seed,
    )
    