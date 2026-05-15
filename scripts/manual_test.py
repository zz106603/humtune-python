from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.audio_processing import AudioProcessingError, analyze_audio
from app.audio_processing import _write_debug_chord_midi_file
from app.audio_processing import _write_debug_combined_midi_file
from app.audio_processing import _write_debug_melody_midi_file


def _format_notes(notes: list) -> str:
    return " ".join(
        f"{note.pitch}@{note.startTime:.2f}+{note.duration:.2f}"
        for note in notes
    )


def _format_chords(chords: list) -> str:
    return " ".join(
        f"{chord.root}_{chord.type}@{chord.startTime:.2f}+{chord.duration:.2f}"
        for chord in chords
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/manual_test.py <path-to-audio>")
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

    raw_midi_path = _write_debug_melody_midi_file(
        audio_id=raw_audio_path.stem,
        output_directory=output_directory,
        suffix="raw",
        melody_notes=result.original_notes,
    )
    cleaned_midi_path = _write_debug_melody_midi_file(
        audio_id=raw_audio_path.stem,
        output_directory=output_directory,
        suffix="cleaned",
        melody_notes=result.cleaned_notes,
    )
    final_melody_midi_path = _write_debug_melody_midi_file(
        audio_id=raw_audio_path.stem,
        output_directory=output_directory,
        suffix="final-melody",
        melody_notes=result.quantized_notes,
    )
    chord_midi_path = _write_debug_chord_midi_file(
        audio_id=raw_audio_path.stem,
        output_directory=output_directory,
        suffix="chords",
        chords=result.chords,
    )
    combined_midi_path = _write_debug_combined_midi_file(
        audio_id=raw_audio_path.stem,
        output_directory=output_directory,
        suffix="combined",
        melody_notes=result.quantized_notes,
        chords=result.chords,
    )

    print(f"detectedScale={result.detectedScale}")
    print(f"keyConfidence={result.keyConfidence:.3f}")
    print(f"originalNoteCount={len(result.original_notes)}")
    print(f"cleanedNoteCount={len(result.cleaned_notes)}")
    print(f"adjustedNoteCount={len(result.adjusted_notes)}")
    print(f"quantizedNoteCount={len(result.quantized_notes)}")
    print(f"chordCount={len(result.chords)}")
    print(f"cleanedNotes={_format_notes(result.cleaned_notes)}")
    print(f"adjustedNotes={_format_notes(result.adjusted_notes)}")
    print(f"finalMelodyNotes={_format_notes(result.quantized_notes)}")
    print(f"chords={_format_chords(result.chords)}")
    print(f"rawMidiPath={raw_midi_path}")
    print(f"cleanedMidiPath={cleaned_midi_path}")
    print(f"finalMelodyMidiPath={final_melody_midi_path}")
    print(f"chordMidiPath={chord_midi_path}")
    print(f"combinedMidiPath={combined_midi_path}")
    print(f"midiPath={result.midiPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
