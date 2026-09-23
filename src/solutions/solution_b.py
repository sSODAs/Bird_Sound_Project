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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "processed_segments.csv"

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

PREDICTIONS_PATH = RESULTS_DIR / "solution_b_predictions.csv"

SUMMARY_PATH = RESULTS_DIR / "solution_b_summary.csv"

CONFUSION_PATH = FIGURES_DIR / "solution_b_confusion_matrix.png"


TARGET_SR = 32000
TARGET_SAMPLES = 96000

# STFT configuration
N_FFT = 1024
HOP_LENGTH = 256

MIN_FREQ = 150
MAX_FREQ = 12000

# Final time-frequency representation
NUM_FREQ_BANDS = 64
NUM_TIME_BLOCKS = 12

# Same recording-level classifier as Solution A
TOP_K_RECORDINGS = 3

EPSILON = 1e-12

MAX_TIME_SHIFT_BLOCKS = 4


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

    # Guarantee exact input length.
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


# Pool spectrogram to fixed size
#
# Original STFT spectrogram may contain hundreds of
# frequency bins and time frames.
#
# We reduce it to:
#
# 64 frequency bands × 12 time blocks
#
# This keeps both frequency AND temporal structure.


def pool_spectrogram(
    spectrogram,
    num_freq_bands,
    num_time_blocks,
):

    freq_indices = np.array_split(
        np.arange(spectrogram.shape[0]),
        num_freq_bands,
    )

    time_indices = np.array_split(
        np.arange(spectrogram.shape[1]),
        num_time_blocks,
    )

    pooled = np.zeros(
        (
            num_freq_bands,
            num_time_blocks,
        ),
        dtype=np.float32,
    )

    for freq_index, freq_group in enumerate(freq_indices):

        for time_index, time_group in enumerate(time_indices):

            block = spectrogram[
                np.ix_(
                    freq_group,
                    time_group,
                )
            ]

            pooled[
                freq_index,
                time_index,
            ] = np.mean(block)

    return pooled


# STFT feature extraction


def extract_stft_feature(
    audio_path,
    return_map=False,
):

    y, sr = load_audio(audio_path)

    # 1. Short-Time Fourier Transform

    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window="hann",
        center=False,
    )

    magnitude = np.abs(stft)

    power = magnitude**2

    # 2. Frequency axis

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=N_FFT,
    )

    frequency_mask = (frequencies >= MIN_FREQ) & (frequencies <= MAX_FREQ)

    power = power[frequency_mask, :]

    frequencies = frequencies[frequency_mask]

    # 3. Log power spectrogram
    #
    # Convert power to dB.
    # Human/animal acoustic patterns are easier to represent
    # on a logarithmic energy scale than raw power.

    power = np.maximum(
        power,
        EPSILON,
    )

    log_power = librosa.power_to_db(
        power,
        ref=np.max,
        top_db=80.0,
    )

    # 4. Reduce spectrogram to fixed size
    #
    # 64 frequency regions
    # ×
    # 12 temporal regions
    #
    # For a 3-second signal:
    #
    # 12 blocks ≈ 0.25 seconds per time block

    pooled = pool_spectrogram(
        log_power,
        NUM_FREQ_BANDS,
        NUM_TIME_BLOCKS,
    )

    # Save the un-normalized map if needed for visualization.
    original_map = pooled.copy()

    # 5. Standardization
    #
    # Focus on time-frequency pattern rather than
    # absolute recording level.

    mean_value = np.mean(pooled)

    std_value = np.std(pooled)

    pooled = pooled - mean_value

    if std_value > EPSILON:

        pooled = pooled / std_value

    # 6. Flatten into one feature vector
    #
    # 64 × 12 = 768 features

    feature = pooled.flatten()

    # 7. L2 normalization
    #
    # After this:
    #
    # dot product = cosine similarity

    norm = np.linalg.norm(feature)

    if norm > EPSILON:

        feature = feature / norm

    feature = feature.astype(np.float32)

    if return_map:

        return (
            feature,
            original_map,
        )

    return feature


# Build feature bank


def build_feature_bank(metadata):

    features = []
    feature_times_ms = []

    total = len(metadata)

    print()
    print("Extracting STFT time-frequency features...")

    for index, row in metadata.iterrows():

        audio_path = PROJECT_ROOT / Path(row["processed_path"])

        start_time = perf_counter()

        feature = extract_stft_feature(audio_path)

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


def time_shift_similarity(
    feature_a,
    feature_b,
):

    map_a = feature_a.reshape(
        NUM_FREQ_BANDS,
        NUM_TIME_BLOCKS,
    )

    map_b = feature_b.reshape(
        NUM_FREQ_BANDS,
        NUM_TIME_BLOCKS,
    )

    best_similarity = -1.0

    for shift in range(
        -MAX_TIME_SHIFT_BLOCKS,
        MAX_TIME_SHIFT_BLOCKS + 1,
    ):

        if shift < 0:

            # B occurs earlier than A.
            a_part = map_a[:, -shift:]

            b_part = map_b[:, : NUM_TIME_BLOCKS + shift]

        elif shift > 0:

            # B occurs later than A.
            a_part = map_a[:, : NUM_TIME_BLOCKS - shift]

            b_part = map_b[:, shift:]

        else:

            a_part = map_a
            b_part = map_b

        a_vector = a_part.reshape(-1)

        b_vector = b_part.reshape(-1)

        a_norm = np.linalg.norm(a_vector)

        b_norm = np.linalg.norm(b_vector)

        if a_norm <= EPSILON or b_norm <= EPSILON:
            continue

        similarity = float(
            np.dot(
                a_vector,
                b_vector,
            )
            / (a_norm * b_norm)
        )

        if similarity > best_similarity:

            best_similarity = similarity

    return best_similarity


def classify_leave_one_recording_out(
    metadata,
    features,
    feature_times_ms,
):

    species_list = sorted(metadata["species"].unique().tolist())

    recording_ids = metadata["recording_id"].drop_duplicates().tolist()

    results = []

    print()
    print("Running Leave-One-Recording-Out...")

    total = len(recording_ids)

    for number, recording_id in enumerate(
        recording_ids,
        start=1,
    ):

        #
        # Query recording
        #

        query_mask = metadata["recording_id"].values == recording_id

        query_features = features[query_mask]

        query_metadata = metadata[query_mask]

        true_species = query_metadata["species"].iloc[0]

        query_feature_ms = float(np.sum(feature_times_ms[query_mask]))

        #
        # Reference database
        #
        # Remove the entire query recording.
        #

        reference_mask = ~query_mask

        reference_features = features[reference_mask]

        reference_metadata = metadata[reference_mask].reset_index(drop=True)

        reference_recording_ids = (
            reference_metadata["recording_id"].drop_duplicates().tolist()
        )

        #
        # Recording-level matching
        #

        start_time = perf_counter()

        recording_scores = []

        for reference_id in reference_recording_ids:

            ref_mask = reference_metadata["recording_id"].values == reference_id

            ref_features = reference_features[ref_mask]

            ref_species = reference_metadata[ref_mask]["species"].iloc[0]

            # -----------------------------------------------
            # Similarity between:
            #
            # 3 query windows
            # ×
            # 3 reference windows
            #
            # Each feature vector contains STFT
            # time-frequency information.
            # -----------------------------------------------

            similarity_matrix = np.zeros(
                (
                    len(query_features),
                    len(ref_features),
                ),
                dtype=np.float32,
            )

            for query_index, query_feature in enumerate(query_features):

                for ref_index, ref_feature in enumerate(ref_features):

                    similarity_matrix[
                        query_index,
                        ref_index,
                    ] = time_shift_similarity(
                        query_feature,
                        ref_feature,
                    )

            best_per_query = np.max(
                similarity_matrix,
                axis=1,
            )

            recording_score = float(np.mean(best_per_query))

            recording_scores.append(
                {
                    "recording_id": reference_id,
                    "species": ref_species,
                    "score": recording_score,
                }
            )

        recording_scores_df = pd.DataFrame(recording_scores)

        #
        # Species-level decision
        #
        # Same rule as Solution A:
        #
        # take Top-3 closest reference recordings
        # from each species.
        #

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

        #
        # Save prediction
        #

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

    ax.set_title("Solution B - STFT Time-Frequency Matching")

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


# Save example time-frequency map
#
# Useful later for presentation/report.


def save_example_feature_map(metadata):

    first_row = metadata.iloc[0]

    audio_path = PROJECT_ROOT / Path(first_row["processed_path"])

    _, feature_map = extract_stft_feature(
        audio_path,
        return_map=True,
    )

    output_path = FIGURES_DIR / "solution_b_example_time_frequency_map.png"

    fig, ax = plt.subplots(figsize=(10, 5))

    image = ax.imshow(
        feature_map,
        origin="lower",
        aspect="auto",
        extent=[
            0,
            3,
            MIN_FREQ,
            MAX_FREQ,
        ],
    )

    ax.set_title("Solution B - Pooled Time-Frequency Representation")

    ax.set_xlabel("Time (seconds)")

    ax.set_ylabel("Frequency (Hz)")

    fig.colorbar(
        image,
        ax=ax,
        label="Power (dB)",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


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

    print("SOLUTION B")

    print("STFT TIME-FREQUENCY MATCHING")

    print(f"Segments: " f"{len(metadata)}")

    print(f"Recordings: " f"{metadata['recording_id'].nunique()}")

    print(f"Species: " f"{metadata['species'].nunique()}")

    # Extract STFT features

    features, feature_times_ms = build_feature_bank(metadata)

    print()
    print(
        "Feature matrix:",
        features.shape,
    )

    # Expected:
    #
    # 225 × (64 × 12)
    # =
    # 225 × 768

    # Validation

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

    # Metrics

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

    # Figures

    save_confusion_matrix(predictions)

    example_path = save_example_feature_map(metadata)

    # Final output

    print()

    print("SOLUTION B RESULTS")

    print(f"LOO Accuracy: " f"{accuracy * 100:.2f}%")

    print(f"Mean feature time: " f"{mean_feature_ms:.3f} ms")

    print(f"Mean matching time: " f"{mean_matching_ms:.3f} ms")

    print(f"Mean total query time: " f"{mean_total_ms:.3f} ms")

    print()
    print(f"Predictions: " f"{PREDICTIONS_PATH}")

    print(f"Summary: " f"{SUMMARY_PATH}")

    print(f"Confusion matrix: " f"{CONFUSION_PATH}")

    print(f"Example TF map: " f"{example_path}")


if __name__ == "__main__":
    main()
