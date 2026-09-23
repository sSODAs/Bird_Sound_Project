from pathlib import Path
import argparse
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "recordings.csv"

PREVIEW_DIR = PROJECT_ROOT / "outputs" / "previews"

SEGMENT_DURATION = 3.0


# Find recording from metadata
def find_recording(recording_id: str):

    metadata = pd.read_csv(METADATA_PATH)

    recording_id = recording_id.upper()

    result = metadata[metadata["recording_id"].str.upper().eq(recording_id)]

    if result.empty:
        raise ValueError(f"Recording ID not found: {recording_id}")

    return result.iloc[0]


# Load audio
def load_audio(row):

    audio_path = PROJECT_ROOT / Path(row["relative_path"])


    print("RECORDING")

    print("Recording ID :", row["recording_id"])
    print("Species      :", row["common_name"])
    print("File         :", audio_path)

    # Load original sample rate.
    # Convert to mono only for inspection.
    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True,
    )

    decoded_duration = len(y) / sr

    print("Sample rate  :", sr, "Hz")
    print(
        "Decoded time :",
        round(decoded_duration, 3),
        "seconds",
    )

    if "duration_sec" in row:
        metadata_duration = row["duration_sec"]

        print(
            "Metadata time:",
            round(metadata_duration, 3),
            "seconds",
        )

        difference = abs(metadata_duration - decoded_duration)

        if difference > 0.5:
            print(
                "\nWARNING:" " Metadata duration and decoded " "duration are different."
            )

    return y, sr, audio_path


# Full recording visualization
def show_full_recording(
    y,
    sr,
    recording_id,
    species,
):

    time_axis = np.arange(len(y)) / sr

    # STFT is used here only for visualization.
    D = librosa.stft(
        y,
        n_fft=2048,
        hop_length=512,
        window="hann",
    )

    S_db = librosa.amplitude_to_db(
        np.abs(D),
        ref=np.max,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 7),
    )

    # Waveform
    axes[0].plot(
        time_axis,
        y,
        linewidth=0.5,
    )

    axes[0].set_title(f"{recording_id} - {species}")

    axes[0].set_xlabel("Time (seconds)")

    axes[0].set_ylabel("Amplitude")

    # Spectrogram
    image = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="hz",
        ax=axes[1],
    )

    axes[1].set_ylim(
        0,
        min(12000, sr / 2),
    )

    axes[1].set_title("Spectrogram")

    fig.colorbar(
        image,
        ax=axes[1],
        format="%+2.0f dB",
    )

    plt.tight_layout()

    print("\nClose the graph window " "when you are ready to choose a segment.")

    plt.show()


# Extract 3-second preview
def extract_segment(
    y,
    sr,
    start_sec,
):

    end_sec = start_sec + SEGMENT_DURATION

    total_duration = len(y) / sr

    if start_sec < 0:
        raise ValueError("Start time cannot be negative.")

    if end_sec > total_duration:
        raise ValueError(
            f"Segment ends at {end_sec:.2f}s, "
            f"but audio ends at "
            f"{total_duration:.2f}s."
        )

    start_sample = int(start_sec * sr)

    end_sample = int(end_sec * sr)

    segment = y[start_sample:end_sample]

    return segment


# Show selected 3-second segment
def show_segment(
    segment,
    sr,
    start_sec,
):

    duration = len(segment) / sr

    time_axis = np.arange(len(segment)) / sr

    D = librosa.stft(
        segment,
        n_fft=2048,
        hop_length=512,
        window="hann",
    )

    S_db = librosa.amplitude_to_db(
        np.abs(D),
        ref=np.max,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
    )

    axes[0].plot(
        time_axis,
        segment,
        linewidth=0.7,
    )

    axes[0].set_title(
        f"Selected segment " f"{start_sec:.2f} - " f"{start_sec + duration:.2f} s"
    )

    axes[0].set_xlabel("Time inside segment (seconds)")

    axes[0].set_ylabel("Amplitude")

    image = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="hz",
        ax=axes[1],
    )

    axes[1].set_ylim(
        0,
        min(12000, sr / 2),
    )

    axes[1].set_title("3-second Spectrogram")

    fig.colorbar(
        image,
        ax=axes[1],
        format="%+2.0f dB",
    )

    plt.tight_layout()
    plt.show()


# Save temporary preview
def save_preview(
    segment,
    sr,
    recording_id,
    start_sec,
):

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = f"{recording_id}" f"_start_{start_sec:.2f}s.wav"

    output_path = PREVIEW_DIR / filename

    sf.write(
        output_path,
        segment,
        sr,
    )

    print(
        "\nPreview:",
        output_path,
    )

    return output_path


def main():

    parser = argparse.ArgumentParser(
        description=("Inspect bird recording and " "preview a 3-second segment.")
    )

    parser.add_argument(
        "recording_id",
        help="Example: XC1004379",
    )

    args = parser.parse_args()

    # Find recording
    row = find_recording(args.recording_id)

    # Load audio
    y, sr, audio_path = load_audio(row)

    # Show complete recording
    show_full_recording(
        y,
        sr,
        row["recording_id"],
        row["common_name"],
    )

    total_duration = len(y) / sr


    # Interactive segment selection
    while True:

        print("3-SECOND SEGMENT")

        print(f"Audio duration: " f"{total_duration:.2f} s")

        print(
            f"Valid start range: "
            f"0.00 - "
            f"{total_duration - SEGMENT_DURATION:.2f} s"
        )

        user_input = input(
            "\nEnter segment start time " "in seconds " "(or q to quit): "
        )

        if user_input.lower() == "q":
            break

        try:
            start_sec = float(user_input)

            segment = extract_segment(
                y,
                sr,
                start_sec,
            )

        except ValueError as error:

            print(
                "\nInvalid value:",
                error,
            )

            continue

        # Save preview WAV
        preview_path = save_preview(
            segment,
            sr,
            row["recording_id"],
            start_sec,
        )

        # Open preview using Windows
        try:
            os.startfile(preview_path)
        except AttributeError:
            print("Open the preview WAV " "manually to listen.")

        # Show only selected segment
        show_segment(
            segment,
            sr,
            start_sec,
        )

        print("\nIf this segment is not good, " "enter another start time.")


if __name__ == "__main__":
    main()
