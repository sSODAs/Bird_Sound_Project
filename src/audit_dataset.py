from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "recordings.csv"


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():

    metadata = pd.read_csv(METADATA_PATH)

    print_section("DATASET SUMMARY")

    print(f"Total recordings: {len(metadata)}")
    print(f"Total species: {metadata['species'].nunique()}")

    print("\nRecordings per species:")

    print(metadata["species"].value_counts().sort_index())


    print_section("FILE STATUS")

    print(metadata["status"].value_counts(dropna=False))

    print_section("SAMPLE RATE DISTRIBUTION")

    sample_rate_counts = metadata["sample_rate"].value_counts().sort_index()

    print(sample_rate_counts)

    print_section("CHANNEL DISTRIBUTION")

    channel_counts = metadata["channels"].value_counts().sort_index()

    print(channel_counts)

    print_section("DURATION STATISTICS")

    print(metadata["duration_sec"].describe().round(2))

    print_section("DURATION BY SPECIES")

    duration_by_species = (
        metadata.groupby("species")["duration_sec"]
        .agg(
            count="count",
            min="min",
            mean="mean",
            median="median",
            max="max",
        )
        .round(2)
    )

    print(duration_by_species)

    TARGET_CLIP_DURATION = 3.0

    print_section(f"FILES SHORTER THAN {TARGET_CLIP_DURATION} SECONDS")

    short_files = metadata[metadata["duration_sec"] < TARGET_CLIP_DURATION]

    if short_files.empty:
        print("None")
    else:
        print(
            short_files[
                [
                    "recording_id",
                    "species",
                    "file_name",
                    "duration_sec",
                ]
            ].to_string(index=False)
        )

    print_section("RECORDINGS SHORTER THAN 6 SECONDS")

    very_short = metadata[metadata["duration_sec"] < 6.0]

    if very_short.empty:
        print("None")
    else:
        print(
            very_short[
                [
                    "recording_id",
                    "species",
                    "duration_sec",
                ]
            ]
            .sort_values("duration_sec")
            .to_string(index=False)
        )

    print_section("SAMPLE RATE BY SPECIES")

    sample_rate_by_species = pd.crosstab(
        metadata["species"],
        metadata["sample_rate"],
    )

    print(sample_rate_by_species)

    print_section("AUDIT RESULT")

    has_errors = (metadata["status"] != "ok").any()

    if has_errors:
        print("WARNING: Some audio files could not be read.")
    else:
        print("PASS: All audio files can be read.")

    if len(metadata) == 75:
        print("PASS: Dataset contains 75 recordings.")
    else:
        print(f"WARNING: Expected 75 recordings, " f"found {len(metadata)}.")

    if metadata["species"].value_counts().eq(15).all():
        print("PASS: Every species has 15 recordings.")
    else:
        print("WARNING: Species counts are not balanced.")


if __name__ == "__main__":
    main()
