from pathlib import Path
import sys

from app.audio_processing import AudioProcessingError, analyze_audio


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/manual_test.py <path-to-wav>")
        return 2

    raw_audio_path = Path(sys.argv[1])
    output_directory = Path("manual-output")
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        result = analyze_audio(
            audio_id=raw_audio_path.stem,
            raw_audio_path=str(raw_audio_path),
            output_directory=str(output_directory),
        )
    except AudioProcessingError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"detectedScale={result.detectedScale}")
    print(f"keyConfidence={result.keyConfidence:.3f}")
    print(f"originalNoteCount={len(result.original_notes)}")
    print(f"adjustedNoteCount={len(result.adjusted_notes)}")
    print(f"chordCount={len(result.chords)}")
    print(f"midiPath={result.midiPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
