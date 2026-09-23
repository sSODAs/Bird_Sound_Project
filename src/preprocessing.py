from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "recordings.csv"

CANDIDATES_PATH = PROJECT_ROOT / "data" / "metadata" / "segment_candidates.csv"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "reference"

OUTPUT_METADATA = PROJECT_ROOT / "data" / "metadata" / "processed_segments.csv"


TARGET_SR = 32000
SEGMENT_DURATION = 3.0
TARGET_SAMPLES = int(TARGET_SR * SEGMENT_DURATION)


def normalize_peak(y):

    peak = np.max(np.abs(y))

    if peak > 0:
        y = y / peak

    return y.astype(np.float32)


def force_exact_length(y):

    # Too long -> trim.
    if len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]

    # Too short -> zero padding.
    elif len(y) < TARGET_SAMPLES:

        padding = TARGET_SAMPLES - len(y)

        y = np.pad(
            y,
            (0, padding),
            mode="constant",
        )

    return y


def extract_segment(
    audio_path,
    start_sec,
):

    # librosa performs:
    # - decoding
    # - mono conversion
    # - resampling to 32 kHz
    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
    )

    start_sample = int(round(start_sec * TARGET_SR))

    end_sample = start_sample + TARGET_SAMPLES

    segment = y[start_sample:end_sample]

    segment = force_exact_length(segment)

    segment = normalize_peak(segment)

    return segment, sr


def main():

    metadata = pd.read_csv(METADATA_PATH)

    candidates = pd.read_csv(CANDIDATES_PATH)

    # Connect candidate rows to source audio paths.
    merged = candidates.merge(
        metadata[
            [
                "recording_id",
                "relative_path",
                "scientific_name",
            ]
        ],
        on="recording_id",
        how="left",
    )

    output_rows = []

    total = len(merged)

    print("REFERENCE PREPROCESSING")

    print(f"Segments to process: {total}")

    for index, row in merged.iterrows():

        recording_id = row["recording_id"]

        species = row["species"]

        rank = int(row["rank"])

        start_sec = float(row["start_sec"])

        end_sec = float(row["end_sec"])

        audio_path = PROJECT_ROOT / Path(row["relative_path"])

        print(f"[{index + 1}/{total}] " f"{recording_id} " f"rank={rank}")

        if not audio_path.exists():

            print(f"  ERROR: file not found: " f"{audio_path}")

            continue

        try:

            segment, sr = extract_segment(
                audio_path,
                start_sec,
            )

            species_dir = OUTPUT_ROOT / species

            species_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_name = f"{recording_id}" f"_rank{rank}.wav"

            output_path = species_dir / output_name

            sf.write(
                output_path,
                segment,
                sr,
                subtype="PCM_16",
            )

            relative_output = output_path.relative_to(PROJECT_ROOT)

            output_rows.append(
                {
                    "recording_id": recording_id,
                    "species": species,
                    "scientific_name": row["scientific_name"],
                    "rank": rank,
                    "segment_start_sec": start_sec,
                    "segment_end_sec": end_sec,
                    "sample_rate": sr,
                    "num_samples": len(segment),
                    "duration_sec": len(segment) / sr,
                    "processed_path": str(relative_output),
                }
            )

        except Exception as error:

            print(f"  ERROR: {error}")

    output_df = pd.DataFrame(output_rows)

    OUTPUT_METADATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_METADATA,
        index=False,
        encoding="utf-8-sig",
    )

    print()

    print("PREPROCESSING FINISHED")

    print(f"Processed segments: " f"{len(output_df)}")

    print(f"Recordings: " f"{output_df['recording_id'].nunique()}")

    print(f"Sample rates: " f"{sorted(output_df['sample_rate'].unique())}")

    print(f"Sample counts: " f"{sorted(output_df['num_samples'].unique())}")

    print(f"Output metadata: " f"{OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
