from pathlib import Path
import re

import pandas as pd
import soundfile as sf
import librosa

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_CSV = METADATA_DIR / "recordings.csv"

SPECIES_INFO = {
    "asian_koel": {
        "common_name": "Asian Koel",
        "scientific_name": "Eudynamys scolopaceus",
    },
    "coppersmith_barbet": {
        "common_name": "Coppersmith Barbet",
        "scientific_name": "Psilopogon haemacephalus",
    },
    "greater_coucal": {
        "common_name": "Greater Coucal",
        "scientific_name": "Centropus sinensis",
    },
    "spotted_dove": {
        "common_name": "Spotted Dove",
        "scientific_name": "Spilopelia chinensis",
    },
    "zebra_dove": {
        "common_name": "Zebra Dove",
        "scientific_name": "Geopelia striata",
    },
}


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}

def extract_xc_id(file_name: str) -> str:
    """
    Extract Xeno-canto ID from a file name.

    Examples:
        XC1081912.mp3
        XC1081912_Zebra_Dove.mp3
        Zebra Dove XC1081912.mp3

    Returns:
        XC1081912

    If no XC ID is found, returns an empty string.
    """

    match = re.search(r"XC\d+", file_name, flags=re.IGNORECASE)

    if match:
        return match.group(0).upper()

    return ""


def get_audio_info(file_path: Path):
    """
    Read basic technical information from an audio file.

    First tries soundfile because it is fast and does not need
    to load the complete audio signal.

    If that fails, librosa is used as a fallback.
    """

    try:
        info = sf.info(str(file_path))

        duration_sec = float(info.duration)
        sample_rate = int(info.samplerate)
        channels = int(info.channels)

        return duration_sec, sample_rate, channels

    except Exception:
        # Fallback for audio formats that soundfile cannot read.
        audio, sample_rate = librosa.load(
            str(file_path),
            sr=None,
            mono=False,
        )

        if audio.ndim == 1:
            channels = 1
            number_of_samples = len(audio)

        else:
            channels = audio.shape[0]
            number_of_samples = audio.shape[-1]

        duration_sec = number_of_samples / sample_rate

        return float(duration_sec), int(sample_rate), int(channels)


def build_metadata():
    """
    Scan all bird audio files inside data/raw/
    and create data/metadata/recordings.csv.
    """

    rows = []

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    for species_folder in sorted(RAW_DATA_DIR.iterdir()):

        if not species_folder.is_dir():
            continue

        species_key = species_folder.name

        if species_key not in SPECIES_INFO:
            print(
                f"[WARNING] Unknown species folder: "
                f"{species_key}"
            )
            continue

        species_info = SPECIES_INFO[species_key]

        print(f"\nScanning: {species_info['common_name']}")

        audio_files = sorted(
            file_path
            for file_path in species_folder.iterdir()
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in SUPPORTED_AUDIO_EXTENSIONS
            )
        )

        for file_path in audio_files:

            recording_id = extract_xc_id(file_path.name)

            try:
                (
                    duration_sec,
                    sample_rate,
                    channels,
                ) = get_audio_info(file_path)

                status = "ok"

            except Exception as error:
                print(
                    f"[ERROR] Could not read "
                    f"{file_path.name}: {error}"
                )

                duration_sec = None
                sample_rate = None
                channels = None
                status = "read_error"

            if recording_id:
                xc_number = recording_id.replace("XC", "")

                source_url = (
                    f"https://xeno-canto.org/{xc_number}"
                )

            else:
                source_url = ""

            relative_path = file_path.relative_to(
                PROJECT_ROOT
            ).as_posix()

            row = {
                "recording_id": recording_id,
                "species": species_key,
                "common_name": species_info["common_name"],
                "scientific_name": (
                    species_info["scientific_name"]
                ),
                "file_name": file_path.name,
                "relative_path": relative_path,

                # Dataset role
                "role": "reference",

                # Technical audio information
                "duration_sec": (
                    round(duration_sec, 3)
                    if duration_sec is not None
                    else None
                ),
                "sample_rate": sample_rate,
                "channels": channels,

                # Xeno-canto information
                # These can be filled later.
                "vocalization_type": "",
                "quality": "",
                "country": "",
                "license": "",
                "source": "xeno-canto",
                "source_url": source_url,

                # Dataset checking
                "status": status,
                "notes": "",
            }

            rows.append(row)

            print(
                f"  {file_path.name}"
                f" | {duration_sec:.2f}s"
                if duration_sec is not None
                else f"  {file_path.name} | ERROR"
            )

    dataframe = pd.DataFrame(rows)

    dataframe = dataframe.sort_values(
        by=[
            "species",
            "recording_id",
            "file_name",
        ]
    ).reset_index(drop=True)

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Metadata created successfully")

    print(f"Output: {OUTPUT_CSV}")
    print(f"Total recordings: {len(dataframe)}")

    print("\nRecordings per species:")
    print(
        dataframe["species"]
        .value_counts()
        .sort_index()
    )

    return dataframe

if __name__ == "__main__":
    build_metadata()