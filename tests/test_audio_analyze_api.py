from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

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
    assert isinstance(body["processingTimeMs"], int)
    midi_path = Path(body["midiPath"])
    assert midi_path.is_file()
    assert midi_path.parent == output_directory


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
