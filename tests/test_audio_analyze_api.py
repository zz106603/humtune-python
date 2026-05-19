from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import app.main as main_module
from app.audio_processing import Chord, Note
from app.main import app


client = TestClient(app)


def test_audio_analyze_success_creates_midi(tmp_path: Path) -> None:
    raw_audio_path = tmp_path / "melody.wav"
    output_directory = tmp_path / "output"
    _write_test_melody(raw_audio_path)
    raw_audio_path = raw_audio_path.resolve()
    output_directory = output_directory.resolve()

    response = client.post(
        "/internal/audio/analyze",
        json={
            "audioId": "melody",
            "rawAudioPath": str(raw_audio_path),
            "outputDirectory": str(output_directory),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "COMPLETED"
    assert "errorMessage" not in body
    assert body["detectedScale"]
    assert isinstance(body["keyConfidence"], float)
    assert body["originalNotes"]
    assert body["adjustedNotes"]
    assert body["chords"]
    assert body["midiPath"]
    assert body["previewAudioPath"]
    assert isinstance(body["processingTimeMs"], int)
    midi_path = Path(body["midiPath"])
    preview_audio_path = Path(body["previewAudioPath"])
    assert midi_path.is_file()
    assert midi_path.parent == output_directory
    assert preview_audio_path.is_file()
    assert preview_audio_path.parent == output_directory
    assert preview_audio_path.suffix == ".wav"


def test_audio_analyze_response_exposes_final_quantized_melody(
    tmp_path: Path,
    monkeypatch,
) -> None:
    midi_path = tmp_path / "output" / "melody.mid"
    preview_audio_path = tmp_path / "output" / "melody-preview.wav"
    midi_path.parent.mkdir(parents=True)
    midi_path.write_bytes(b"midi")
    preview_audio_path.write_bytes(b"preview")

    def fake_process_audio(
        audio_id: str,
        raw_audio_path: str,
        output_directory: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            detectedScale="C_MAJOR",
            keyConfidence=0.91,
            original_notes=[_note(61), _note(63)],
            adjusted_notes=[_note(60), _note(62)],
            quantized_notes=[_note(60), _note(64)],
            chords=[Chord(root="C", type="MAJOR", startTime=0.0, duration=1.2)],
            midiPath=str(midi_path),
            previewAudioPath=str(preview_audio_path),
        )

    monkeypatch.setattr(main_module, "process_audio", fake_process_audio)

    response = client.post(
        "/internal/audio/analyze",
        json={
            "audioId": "melody",
            "rawAudioPath": str(tmp_path / "melody.wav"),
            "outputDirectory": str(tmp_path / "output"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "COMPLETED"
    assert body["originalNotes"] == ["C#4", "D#4"]
    assert body["adjustedNotes"] == ["C4", "E4"]
    assert body["chords"] == ["C"]
    assert body["midiPath"] == str(midi_path)
    assert body["previewAudioPath"] == str(preview_audio_path)


def test_audio_analyze_fails_for_missing_raw_audio_path(tmp_path: Path) -> None:
    response = client.post(
        "/internal/audio/analyze",
        json={
            "audioId": "missing",
            "rawAudioPath": str(tmp_path / "missing.wav"),
            "outputDirectory": str(tmp_path / "output"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["errorMessage"]


def test_audio_analyze_fails_for_invalid_audio_file(tmp_path: Path) -> None:
    invalid_audio_path = tmp_path / "invalid.wav"
    invalid_audio_path.write_text("not an audio file")

    response = client.post(
        "/internal/audio/analyze",
        json={
            "audioId": "invalid",
            "rawAudioPath": str(invalid_audio_path),
            "outputDirectory": str(tmp_path / "output"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["errorMessage"]


def _write_test_melody(path: Path) -> None:
    sample_rate = 22050
    frequencies = [261.63, 329.63, 392.00, 523.25]
    notes = [_sine_wave(frequency, 0.55, sample_rate) for frequency in frequencies]
    pauses = [np.zeros(int(sample_rate * 0.05), dtype=np.float32) for _ in frequencies]
    samples = np.concatenate([part for pair in zip(notes, pauses) for part in pair])
    sf.write(path, samples, sample_rate)


def _sine_wave(frequency: float, duration_seconds: float, sample_rate: int) -> np.ndarray:
    times = np.linspace(
        0,
        duration_seconds,
        int(sample_rate * duration_seconds),
        endpoint=False,
    )
    return (0.4 * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def _note(midi_note: int) -> Note:
    pitch_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return Note(
        pitch=f"{pitch_names[midi_note % 12]}{(midi_note // 12) - 1}",
        midi_note=midi_note,
        startTime=0.0,
        duration=0.3,
        velocity=80,
    )
