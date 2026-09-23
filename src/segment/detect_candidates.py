from pathlib import Path
import argparse

import librosa
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "recordings.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "segment_candidates.csv"


# Configuration


TARGET_SR = 32000

SEGMENT_DURATION = 3.0

N_FFT = 2048
HOP_LENGTH = 512

MIN_FREQ = 150
MAX_FREQ = 12000

STEP_SEC = 0.25

TOP_K = 3

MIN_START_DISTANCE = 1.5


# Audio


def load_audio(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
    )

    return y, sr


# Acoustic activity


def calculate_activity(y, sr):

    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
    )

    magnitude = np.abs(stft)

    power = magnitude**2

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=N_FFT,
    )

    band_mask = (frequencies >= MIN_FREQ) & (frequencies <= MAX_FREQ)

    band_power = power[band_mask, :]

    band_db = librosa.power_to_db(
        band_power,
        ref=np.max,
    )

    # Estimate background separately
    # for every frequency bin.
    background = np.percentile(
        band_db,
        20,
        axis=1,
        keepdims=True,
    )

    # Amount of energy above background.
    excess = band_db - background

    excess = np.maximum(
        excess,
        0,
    )

    # One activity value for each STFT frame.
    frame_activity = np.mean(
        excess,
        axis=0,
    )

    return frame_activity


# Score 3-second windows


def score_windows(
    frame_activity,
    sr,
    audio_duration,
):

    frames_per_segment = int(SEGMENT_DURATION * sr / HOP_LENGTH)

    step_frames = max(
        1,
        int(STEP_SEC * sr / HOP_LENGTH),
    )

    max_start_frame = len(frame_activity) - frames_per_segment

    if max_start_frame < 0:
        return []

    activity_threshold = np.percentile(
        frame_activity,
        60,
    )

    windows = []

    for start_frame in range(
        0,
        max_start_frame + 1,
        step_frames,
    ):

        end_frame = start_frame + frames_per_segment

        activity = frame_activity[start_frame:end_frame]

        if len(activity) == 0:
            continue

        mean_activity = float(np.mean(activity))

        occupancy = float(np.mean(activity > activity_threshold))

        median_activity = float(np.median(activity))

        peak_activity = float(np.max(activity))

        # Very short loud impulses should
        # not automatically win.
        spike_ratio = peak_activity / (median_activity + 1e-8)

        spike_penalty = max(
            1.0,
            spike_ratio / 8.0,
        )

        score = mean_activity * (0.5 + occupancy) / spike_penalty

        start_sec = start_frame * HOP_LENGTH / sr

        end_sec = start_sec + SEGMENT_DURATION

        if end_sec <= audio_duration:

            windows.append(
                {
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "score": float(score),
                    "occupancy": float(occupancy),
                    "mean_activity": float(mean_activity),
                }
            )

    return windows


# Select Top-K


def choose_top_candidates(
    windows,
    top_k=TOP_K,
):

    sorted_windows = sorted(
        windows,
        key=lambda x: x["score"],
        reverse=True,
    )

    selected = []

    # Pass 1:
    # Prefer windows sufficiently far apart.

    for candidate in sorted_windows:

        too_close = False

        for existing in selected:

            distance = abs(candidate["start_sec"] - existing["start_sec"])

            if distance < MIN_START_DISTANCE:
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

        if len(selected) >= top_k:
            return selected

    # Pass 2:
    # Short recordings may not provide enough separated
    # windows. Fill using the next best overlapping windows.

    for candidate in sorted_windows:

        already_selected = any(
            abs(candidate["start_sec"] - existing["start_sec"]) < 1e-6
            for existing in selected
        )

        if already_selected:
            continue

        selected.append(candidate)

        if len(selected) >= top_k:
            break

    return selected


# Public function
# Used later by BOTH reference and unseen test recordings.


def detect_candidates(
    audio_path,
    top_k=TOP_K,
):

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = load_audio(audio_path)

    duration = len(y) / sr

    # Very short input.
    if duration < SEGMENT_DURATION:

        return [
            {
                "rank": 1,
                "start_sec": 0.0,
                "end_sec": duration,
                "score": 0.0,
                "occupancy": 0.0,
            }
        ]

    activity = calculate_activity(
        y,
        sr,
    )

    windows = score_windows(
        activity,
        sr,
        duration,
    )

    candidates = choose_top_candidates(
        windows,
        top_k=top_k,
    )

    results = []

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):

        results.append(
            {
                "rank": rank,
                "start_sec": round(
                    candidate["start_sec"],
                    3,
                ),
                "end_sec": round(
                    candidate["end_sec"],
                    3,
                ),
                "score": candidate["score"],
                "occupancy": candidate["occupancy"],
            }
        )

    return results


# Reference dataset


def process_reference_dataset():

    metadata = pd.read_csv(METADATA_PATH)

    results = []

    total = len(metadata)

    for index, row in metadata.iterrows():

        recording_id = row["recording_id"]

        audio_path = PROJECT_ROOT / Path(row["relative_path"])

        print(f"[{index + 1}/{total}] " f"{recording_id}")

        try:

            candidates = detect_candidates(audio_path)

            for candidate in candidates:

                results.append(
                    {
                        "recording_id": recording_id,
                        # Stored for later analysis only.
                        # Species is NOT used by detector.
                        "species": row["species"],
                        **candidate,
                    }
                )

        except Exception as error:

            print(f"ERROR: {error}")

    result_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()

    print("REFERENCE SEGMENTATION FINISHED")

    print(f"Recordings: " f"{result_df['recording_id'].nunique()}")

    print(f"Candidates: " f"{len(result_df)}")

    print(f"Output: " f"{OUTPUT_PATH}")


# Unseen single recording


def process_single_file(audio_path):

    candidates = detect_candidates(audio_path)

    print()

    print("AUTOMATIC SEGMENTATION")

    print(f"File: {audio_path}")

    print()

    for candidate in candidates:

        print(
            f"Rank {candidate['rank']}: "
            f"{candidate['start_sec']:.3f}"
            f" - "
            f"{candidate['end_sec']:.3f} s "
            f"| score="
            f"{candidate['score']:.3f} "
            f"| occupancy="
            f"{candidate['occupancy']:.3f}"
        )


# CLI


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help=("Optional audio file. " "If omitted, process the " "reference dataset."),
    )

    args = parser.parse_args()

    if args.file:

        process_single_file(args.file)

    else:

        process_reference_dataset()


if __name__ == "__main__":
    main()
