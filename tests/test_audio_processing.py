import pytest

from app.audio_processing import MAJOR_SCALE_INTERVALS
from app.audio_processing import Note
from app.audio_processing import PitchFrame
from app.audio_processing import ScaleFit
from app.audio_processing import _adjust_notes_to_scale
from app.audio_processing import _fit_scale
from app.audio_processing import _frequency_to_midi_note
from app.audio_processing import _infer_chords
from app.audio_processing import _midi_note_to_name
from app.audio_processing import _pitch_frames_to_notes
from app.audio_processing import _quantize_notes


def test_frequency_to_midi_note_and_note_name() -> None:
    assert _frequency_to_midi_note(440.0) == 69
    assert _midi_note_to_name(60) == "C4"
    assert _midi_note_to_name(61) == "C#4"


def test_pitch_frames_to_notes_groups_consecutive_frames() -> None:
    frames = [
        PitchFrame(timestamp_seconds=0.0, frequency_hz=440.0, confidence=0.9),
        PitchFrame(timestamp_seconds=0.12, frequency_hz=441.0, confidence=0.9),
        PitchFrame(timestamp_seconds=0.30, frequency_hz=493.88, confidence=0.9),
        PitchFrame(timestamp_seconds=0.42, frequency_hz=493.88, confidence=0.9),
    ]

    notes = _pitch_frames_to_notes(frames)

    assert [note.pitch for note in notes] == ["A4", "B4"]
    assert all(note.velocity == 80 for note in notes)


def test_fit_scale_prefers_c_major_for_c_major_notes() -> None:
    notes = [
        _note(60, 0.0, 0.3),
        _note(64, 0.3, 0.3),
        _note(67, 0.6, 0.3),
    ]

    scale = _fit_scale(notes)

    assert scale.name == "C_MAJOR"
    assert scale.confidence == pytest.approx(1.0)


def test_adjust_notes_to_scale_snaps_out_of_scale_note() -> None:
    scale = ScaleFit(
        name="C_MAJOR",
        root_pitch_class=0,
        scale_pitch_classes=frozenset(MAJOR_SCALE_INTERVALS),
        confidence=1.0,
    )
    notes = [_note(61, 0.0, 0.3)]

    adjusted = _adjust_notes_to_scale(notes, scale)

    assert adjusted[0].pitch == "C4"
    assert adjusted[0].midi_note == 60


def test_quantize_notes_uses_eighth_note_grid_and_preserves_pitch() -> None:
    notes = [
        _note(60, 0.14, 0.16),
        _note(64, 0.41, 0.22),
    ]

    quantized = _quantize_notes(notes)

    assert [note.pitch for note in quantized] == ["C4", "E4"]
    assert quantized[0].startTime == pytest.approx(0.0)
    assert quantized[0].duration == pytest.approx(0.3)
    assert quantized[1].startTime == pytest.approx(0.3)
    assert quantized[1].duration == pytest.approx(0.3)


def test_infer_chords_scores_diatonic_triads() -> None:
    scale = ScaleFit(
        name="C_MAJOR",
        root_pitch_class=0,
        scale_pitch_classes=frozenset(MAJOR_SCALE_INTERVALS),
        confidence=1.0,
    )
    notes = [
        _note(60, 0.0, 0.3),
        _note(64, 0.3, 0.3),
        _note(67, 0.6, 0.3),
    ]

    chords = _infer_chords(notes, scale)

    assert len(chords) == 1
    assert chords[0].root == "C"
    assert chords[0].type == "MAJOR"


def _note(midi_note: int, start_time: float, duration: float) -> Note:
    return Note(
        pitch=_midi_note_to_name(midi_note),
        midi_note=midi_note,
        startTime=start_time,
        duration=duration,
        velocity=80,
    )
