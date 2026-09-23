from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "processed_segments.csv"

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

PREDICTIONS_PATH = RESULTS_DIR / "solution_a_predictions.csv"

SUMMARY_PATH = RESULTS_DIR / "solution_a_summary.csv"

CONFUSION_PATH = FIGURES_DIR / "solution_a_confusion_matrix.png"

TARGET_SR = 32000
TARGET_SAMPLES = 96000

MIN_FREQ = 150
MAX_FREQ = 12000

NUM_BANDS = 128

EPSILON = 1e-12


# FFT feature extraction


def extract_fft_feature(audio_path):

    y, sr = sf.read(
        audio_path,
        dtype="float32",
    )

    # Safety: convert stereo to mono
    # if a processed file is unexpectedly stereo.
    if y.ndim > 1:
        y = np.mean(
            y,
            axis=1,
        )

    if sr != TARGET_SR:
        raise ValueError(f"Expected {TARGET_SR} Hz, " f"but found {sr} Hz")

    # Guarantee exactly 96,000 samples.
    if len(y) > TARGET_SAMPLES:

        y = y[:TARGET_SAMPLES]

    elif len(y) < TARGET_SAMPLES:

        y = np.pad(
            y,
            (
                0,
                TARGET_SAMPLES - len(y),
            ),
        )

    # Hann window

    window = np.hanning(len(y))
    windowed = y * window

    # One-sided FFT

    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)

    frequencies = np.fft.rfftfreq(
        len(windowed),
        d=1.0 / sr,
    )

    # Keep useful frequency range

    mask = (frequencies >= MIN_FREQ) & (frequencies <= MAX_FREQ)

    frequencies = frequencies[mask]

    magnitude = magnitude[mask]

    # Power spectrum -> dB

    power = magnitude**2

    power = np.maximum(
        power,
        EPSILON,
    )

    power_db = 10.0 * np.log10(power)

    # Divide spectrum into frequency bands

    band_edges = np.linspace(
        MIN_FREQ,
        MAX_FREQ,
        NUM_BANDS + 1,
    )

    feature = np.zeros(
        NUM_BANDS,
        dtype=np.float32,
    )

    for index in range(NUM_BANDS):

        low = band_edges[index]

        high = band_edges[index + 1]

        if index == (NUM_BANDS - 1):

            band_mask = (frequencies >= low) & (frequencies <= high)

        else:

            band_mask = (frequencies >= low) & (frequencies < high)

        values = power_db[band_mask]

        if len(values) > 0:

            feature[index] = np.mean(values)

    # Standardize spectral shape
    # Removes overall level so that similarity depends more
    # on spectral shape than absolute volume.

    feature = feature - np.mean(feature)

    std = np.std(feature)

    if std > EPSILON:

        feature = feature / std

    # L2 normalization
    # After this step, dot product = cosine similarity.

    norm = np.linalg.norm(feature)

    if norm > EPSILON:

        feature = feature / norm

    return feature.astype(np.float32)


# Build reference feature bank


def build_feature_bank(metadata):

    features = []

    feature_times_ms = []

    total = len(metadata)

    print()
    print("Extracting FFT features...")

    for index, row in metadata.iterrows():

        audio_path = PROJECT_ROOT / Path(row["processed_path"])

        start_time = perf_counter()

        feature = extract_fft_feature(audio_path)

        elapsed_ms = (perf_counter() - start_time) * 1000.0

        features.append(feature)

        feature_times_ms.append(elapsed_ms)

        if (index + 1) % 25 == 0 or index + 1 == total:

            print(f"  {index + 1}" f"/{total}")

    feature_matrix = np.vstack(features)

    return (
        feature_matrix,
        np.array(feature_times_ms),
    )


# Leave-One-Recording-Out classification


def classify_leave_one_recording_out(
    metadata,
    features,
    feature_times_ms,
):

    species_list = sorted(metadata["species"].unique().tolist())

    recording_ids = metadata["recording_id"].drop_duplicates().tolist()

    results = []

    # Number of best reference recordings
    # used to represent each species.
    TOP_K_RECORDINGS = 3

    print()
    print("Running Leave-One-Recording-Out...")

    total = len(recording_ids)

    for number, recording_id in enumerate(
        recording_ids,
        start=1,
    ):

        # Query = all 3 windows from one recording
        query_mask = metadata["recording_id"].values == recording_id

        query_features = features[query_mask]

        query_metadata = metadata[query_mask]

        true_species = query_metadata["species"].iloc[0]

        query_feature_ms = float(np.sum(feature_times_ms[query_mask]))

        # Reference = every OTHER recording
        reference_mask = ~query_mask

        reference_metadata = metadata[reference_mask].reset_index(drop=True)

        reference_features = features[reference_mask]

        reference_recording_ids = (
            reference_metadata["recording_id"].drop_duplicates().tolist()
        )

        # Match query recording against each
        # reference recording individually

        start_time = perf_counter()

        recording_scores = []

        for reference_id in reference_recording_ids:

            ref_mask = reference_metadata["recording_id"].values == reference_id

            ref_features = reference_features[ref_mask]

            ref_species = reference_metadata[ref_mask]["species"].iloc[0]

            # Query has 3 windows.
            # Reference recording has 3 windows.
            #
            # Similarity matrix:
            #
            #           Ref1   Ref2   Ref3
            # Query1
            # Query2
            # Query3
            #
            # Because vectors are L2 normalized,
            # dot product = cosine similarity.

            similarity_matrix = query_features @ ref_features.T

            # For each query window,
            # find the most similar window
            # from this reference recording.
            best_per_query = np.max(
                similarity_matrix,
                axis=1,
            )

            # Recording-level score.
            recording_score = float(np.mean(best_per_query))

            recording_scores.append(
                {
                    "recording_id": reference_id,
                    "species": ref_species,
                    "score": recording_score,
                }
            )

        recording_scores_df = pd.DataFrame(recording_scores)

        # Species score
        #
        # Instead of averaging ALL recordings from a species,
        # use only its Top-K closest real recordings.
        #
        # This prevents a very consistent class from
        # dominating simply because its class mean is strong.

        species_scores = {}

        for species in species_list:

            species_reference_scores = (
                recording_scores_df[recording_scores_df["species"] == species]["score"]
                .sort_values(ascending=False)
                .head(TOP_K_RECORDINGS)
            )

            species_score = float(species_reference_scores.mean())

            species_scores[species] = species_score

        predicted_species = max(
            species_scores,
            key=species_scores.get,
        )

        matching_ms = (perf_counter() - start_time) * 1000.0

        total_query_ms = query_feature_ms + matching_ms

        # Save result
        result = {
            "recording_id": recording_id,
            "true_species": true_species,
            "predicted_species": predicted_species,
            "correct": (predicted_species == true_species),
            "feature_time_ms": query_feature_ms,
            "matching_time_ms": matching_ms,
            "total_query_time_ms": total_query_ms,
        }

        for species in species_list:

            result[f"score_{species}"] = species_scores[species]

        results.append(result)

        status = "OK" if predicted_species == true_species else "WRONG"

        print(
            f"[{number}/{total}] "
            f"{recording_id} "
            f"| true={true_species} "
            f"| pred={predicted_species} "
            f"| {status}"
        )

    return pd.DataFrame(results)


# Confusion matrix


def save_confusion_matrix(
    predictions,
):

    labels = sorted(predictions["true_species"].unique().tolist())

    cm = confusion_matrix(
        predictions["true_species"],
        predictions["predicted_species"],
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title("Solution A - FFT Frequency Matching")

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CONFUSION_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# Main


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(METADATA_PATH)

    print("SOLUTION A")

    print("FFT FREQUENCY MATCHING")

    print(f"Segments: " f"{len(metadata)}")

    print(f"Recordings: " f"{metadata['recording_id'].nunique()}")

    print(f"Species: " f"{metadata['species'].nunique()}")

    features, feature_times_ms = build_feature_bank(metadata)

    print()
    print(
        "Feature matrix:",
        features.shape,
    )

    predictions = classify_leave_one_recording_out(
        metadata,
        features,
        feature_times_ms,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    accuracy = accuracy_score(
        predictions["true_species"],
        predictions["predicted_species"],
    )

    mean_feature_ms = float(predictions["feature_time_ms"].mean())

    mean_matching_ms = float(predictions["matching_time_ms"].mean())

    mean_total_ms = float(predictions["total_query_time_ms"].mean())

    summary = pd.DataFrame(
        [
            {
                "metric": "recordings",
                "value": len(predictions),
            },
            {
                "metric": "accuracy",
                "value": accuracy,
            },
            {
                "metric": "mean_feature_time_ms",
                "value": mean_feature_ms,
            },
            {
                "metric": "mean_matching_time_ms",
                "value": mean_matching_ms,
            },
            {
                "metric": "mean_total_query_time_ms",
                "value": mean_total_ms,
            },
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_confusion_matrix(predictions)

    print()

    print("SOLUTION A RESULTS")

    print(f"LOO Accuracy: " f"{accuracy * 100:.2f}%")

    print(f"Mean feature time: " f"{mean_feature_ms:.3f} ms")

    print(f"Mean matching time: " f"{mean_matching_ms:.3f} ms")

    print(f"Mean total query time: " f"{mean_total_ms:.3f} ms")

    print()
    print(f"Predictions: " f"{PREDICTIONS_PATH}")

    print(f"Summary: " f"{SUMMARY_PATH}")

    print(f"Confusion matrix: " f"{CONFUSION_PATH}")


if __name__ == "__main__":
    main()
