from pathlib import Path
from time import perf_counter

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "processed_segments.csv"

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

PREDICTIONS_PATH = RESULTS_DIR / "solution_c_predictions.csv"

SUMMARY_PATH = RESULTS_DIR / "solution_c_summary.csv"

CONFUSION_PATH = FIGURES_DIR / "solution_c_confusion_matrix.png"


# ============================================================
# Configuration
# ============================================================

TARGET_SR = 32000
TARGET_SAMPLES = 96000

N_FFT = 1024
HOP_LENGTH = 256

N_MFCC = 13

# Compress the 3-second MFCC sequence
# to 48 time steps before DTW.
#
# 3 / 48 ≈ 0.0625 second per step
NUM_TIME_STEPS = 48

TOP_K_RECORDINGS = 3

EPSILON = 1e-8


# ============================================================
# Audio loading
# ============================================================


def load_audio(audio_path):

    y, sr = sf.read(
        audio_path,
        dtype="float32",
    )

    if y.ndim > 1:

        y = np.mean(
            y,
            axis=1,
        )

    if sr != TARGET_SR:

        raise ValueError(f"Expected {TARGET_SR} Hz, " f"but found {sr} Hz")

    if len(y) > TARGET_SAMPLES:

        y = y[:TARGET_SAMPLES]

    elif len(y) < TARGET_SAMPLES:

        y = np.pad(
            y,
            (
                0,
                TARGET_SAMPLES - len(y),
            ),
            mode="constant",
        )

    return y, sr


# ============================================================
# Reduce MFCC sequence to fixed number of time steps
# ============================================================


def pool_time_axis(
    feature_map,
    target_steps,
):

    # feature_map:
    #
    # rows    = MFCC coefficients
    # columns = time frames

    time_groups = np.array_split(
        np.arange(feature_map.shape[1]),
        target_steps,
    )

    pooled = np.zeros(
        (
            feature_map.shape[0],
            target_steps,
        ),
        dtype=np.float32,
    )

    for index, group in enumerate(time_groups):

        if len(group) == 0:
            continue

        pooled[:, index] = np.mean(
            feature_map[:, group],
            axis=1,
        )

    return pooled


# ============================================================
# MFCC extraction
# ============================================================


def extract_mfcc_sequence(
    audio_path,
):

    y, sr = load_audio(audio_path)

    # --------------------------------------------------------
    # 1. MFCC
    #
    # librosa internally uses a Mel-frequency representation
    # and converts it to cepstral coefficients.
    #
    # Shape:
    # 13 coefficients × time frames
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window="hann",
        center=False,
    )

    # --------------------------------------------------------
    # 2. Standardize each MFCC coefficient
    #
    # Reduces recording-level differences and focuses more
    # on the changing spectral pattern.
    # --------------------------------------------------------

    mean = np.mean(
        mfcc,
        axis=1,
        keepdims=True,
    )

    std = np.std(
        mfcc,
        axis=1,
        keepdims=True,
    )

    mfcc = mfcc - mean

    mfcc = mfcc / (std + EPSILON)

    # --------------------------------------------------------
    # 3. Temporal pooling
    #
    # Hundreds of MFCC frames
    #         ↓
    # 48 time steps
    #
    # We keep sequence order.
    # --------------------------------------------------------

    mfcc = pool_time_axis(
        mfcc,
        NUM_TIME_STEPS,
    )

    return mfcc.astype(np.float32)


# ============================================================
# DTW distance
# ============================================================


def dtw_distance(
    sequence_a,
    sequence_b,
):

    # sequence shape:
    #
    # N_MFCC × NUM_TIME_STEPS
    #
    # DTW finds the best nonlinear temporal alignment.

    cost_matrix, warping_path = librosa.sequence.dtw(
        X=sequence_a,
        Y=sequence_b,
        metric="euclidean",
        backtrack=True,
    )

    total_cost = float(cost_matrix[-1, -1])

    path_length = max(
        len(warping_path),
        1,
    )

    # Normalize by warping-path length.
    #
    # Lower = more similar.
    normalized_cost = total_cost / path_length

    return normalized_cost


# ============================================================
# Build feature bank
# ============================================================


def build_feature_bank(
    metadata,
):

    sequences = []
    feature_times_ms = []

    total = len(metadata)

    print()
    print("Extracting MFCC sequences...")

    for index, row in metadata.iterrows():

        audio_path = PROJECT_ROOT / Path(row["processed_path"])

        start_time = perf_counter()

        sequence = extract_mfcc_sequence(audio_path)

        elapsed_ms = (perf_counter() - start_time) * 1000.0

        sequences.append(sequence)

        feature_times_ms.append(elapsed_ms)

        if (index + 1) % 25 == 0 or index + 1 == total:

            print(f"  {index + 1}/{total}")

    return (
        sequences,
        np.array(feature_times_ms),
    )


# ============================================================
# Leave-One-Recording-Out
# ============================================================


def classify_leave_one_recording_out(
    metadata,
    sequences,
    feature_times_ms,
):

    species_list = sorted(metadata["species"].unique().tolist())

    recording_ids = metadata["recording_id"].drop_duplicates().tolist()

    results = []

    print()
    print("Running Leave-One-Recording-Out...")

    total = len(recording_ids)

    recording_array = metadata["recording_id"].values

    for number, recording_id in enumerate(
        recording_ids,
        start=1,
    ):

        # ====================================================
        # Query indices
        # ====================================================

        query_indices = np.where(recording_array == recording_id)[0]

        query_sequences = [sequences[index] for index in query_indices]

        true_species = metadata.iloc[query_indices[0]]["species"]

        query_feature_ms = float(np.sum(feature_times_ms[query_indices]))

        # ====================================================
        # All other recordings are references
        # ====================================================

        reference_recording_ids = [
            ref_id for ref_id in recording_ids if ref_id != recording_id
        ]

        start_matching = perf_counter()

        recording_scores = []

        # ====================================================
        # Query recording vs each reference recording
        # ====================================================

        for reference_id in reference_recording_ids:

            ref_indices = np.where(recording_array == reference_id)[0]

            ref_sequences = [sequences[index] for index in ref_indices]

            ref_species = metadata.iloc[ref_indices[0]]["species"]

            # -----------------------------------------------
            # Normally:
            #
            # 3 query segments
            # ×
            # 3 reference segments
            #
            # = 9 DTW comparisons
            # -----------------------------------------------

            distance_matrix = np.zeros(
                (
                    len(query_sequences),
                    len(ref_sequences),
                ),
                dtype=np.float32,
            )

            for query_index, query_sequence in enumerate(query_sequences):

                for ref_index, ref_sequence in enumerate(ref_sequences):

                    distance_matrix[query_index, ref_index] = dtw_distance(
                        query_sequence,
                        ref_sequence,
                    )

            # Each query window chooses its closest
            # reference window.
            best_per_query = np.min(
                distance_matrix,
                axis=1,
            )

            # Lower recording distance = more similar.
            recording_distance = float(np.mean(best_per_query))

            recording_scores.append(
                {
                    "recording_id": reference_id,
                    "species": ref_species,
                    "distance": recording_distance,
                }
            )

        recording_scores_df = pd.DataFrame(recording_scores)

        # ====================================================
        # Species decision
        #
        # Same principle as A/B:
        # use the three closest real recordings per species.
        #
        # Here LOWER is better because this is distance.
        # ====================================================

        species_distances = {}

        for species in species_list:

            distances = (
                recording_scores_df[recording_scores_df["species"] == species][
                    "distance"
                ]
                .sort_values(ascending=True)
                .head(TOP_K_RECORDINGS)
            )

            species_distance = float(distances.mean())

            species_distances[species] = species_distance

        predicted_species = min(
            species_distances,
            key=species_distances.get,
        )

        matching_ms = (perf_counter() - start_matching) * 1000.0

        total_query_ms = query_feature_ms + matching_ms

        # ====================================================
        # Save
        # ====================================================

        result = {
            "recording_id": recording_id,
            "true_species": true_species,
            "predicted_species": predicted_species,
            "correct": predicted_species == true_species,
            "feature_time_ms": query_feature_ms,
            "matching_time_ms": matching_ms,
            "total_query_time_ms": total_query_ms,
        }

        for species in species_list:

            result[f"distance_{species}"] = species_distances[species]

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


# ============================================================
# Confusion matrix
# ============================================================


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

    ax.set_title("Solution C - MFCC + DTW")

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


# ============================================================
# Main
# ============================================================


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

    print("========================================")

    print("SOLUTION C")

    print("MFCC + DTW")

    print("========================================")

    print(f"Segments: " f"{len(metadata)}")

    print(f"Recordings: " f"{metadata['recording_id'].nunique()}")

    print(f"Species: " f"{metadata['species'].nunique()}")

    # ========================================================
    # Feature extraction
    # ========================================================

    sequences, feature_times_ms = build_feature_bank(metadata)

    print()
    print(
        "MFCC sequence shape:",
        sequences[0].shape,
    )

    # Expected:
    #
    # 13 × 48

    # ========================================================
    # Validation
    # ========================================================

    predictions = classify_leave_one_recording_out(
        metadata,
        sequences,
        feature_times_ms,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Metrics
    # ========================================================

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

    # ========================================================
    # Output
    # ========================================================

    print()
    print("========================================")

    print("SOLUTION C RESULTS")

    print("========================================")

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
