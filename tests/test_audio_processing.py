import pytest
import mido

from app.audio_processing import MIDI_CHORD_CHANNEL
from app.audio_processing import MIDI_MELODY_CHANNEL
from app.audio_processing import MIDI_RELEASE_GAP_TICKS
from app.audio_processing import MIDI_TICKS_PER_BEAT
from app.audio_processing import MAJOR_SCALE_INTERVALS
from app.audio_processing import Chord
from app.audio_processing import Note
from app.audio_processing import PitchFrame
from app.audio_processing import ScaleFit
from app.audio_processing import _adjust_notes_to_scale
from app.audio_processing import _append_chord_events
from app.audio_processing import _append_note_events
from app.audio_processing import _cleanup_notes
from app.audio_processing import _fit_scale
from app.audio_processing import _frequency_to_midi_note
from app.audio_processing import _infer_chords
from app.audio_processing import _midi_note_to_name
from app.audio_processing import _pitch_frames_to_notes
from app.audio_processing import _quantize_notes
from app.audio_processing import _quantize_notes_with_metrics


def test_frequency_to_midi_note_and_note_name() -> None:
    assert _frequency_to_midi_note(440.0) == 69
    assert _midi_note_to_name(60) == "C4"
    assert _midi_note_to_name(61) == "C#4"


def test_pitch_frames_to_notes_groups_consecutive_frames() -> None:
    frames = [
        PitchFrame(timestamp_seconds=0.0, frequency_hz=440.0, confidence=0.9),
        PitchFrame(timestamp_seconds=0.03, frequency_hz=441.0, confidence=0.9),
        PitchFrame(timestamp_seconds=0.06, frequency_hz=440.0, confidence=0.9),
        PitchFrame(timestamp_seconds=0.18, frequency_hz=493.88, confidence=0.9),
        PitchFrame(timestamp_seconds=0.21, frequency_hz=493.88, confidence=0.9),
        PitchFrame(timestamp_seconds=0.24, frequency_hz=493.88, confidence=0.9),
    ]

    notes = _pitch_frames_to_notes(frames)

    assert [note.pitch for note in notes] == ["A4", "B4"]
    assert all(note.velocity == 80 for note in notes)


def test_pitch_frames_to_notes_preserves_repeated_pitch_across_gap() -> None:
    frames = [
        PitchFrame(timestamp_seconds=0.0, frequency_hz=261.63, confidence=0.9),
        PitchFrame(timestamp_seconds=0.03, frequency_hz=261.63, confidence=0.9),
        PitchFrame(timestamp_seconds=0.06, frequency_hz=261.63, confidence=0.9),
        PitchFrame(timestamp_seconds=0.18, frequency_hz=261.63, confidence=0.9),
        PitchFrame(timestamp_seconds=0.21, frequency_hz=261.63, confidence=0.9),
        PitchFrame(timestamp_seconds=0.24, frequency_hz=261.63, confidence=0.9),
    ]

    notes = _pitch_frames_to_notes(frames)

    assert [note.pitch for note in notes] == ["C4", "C4"]


def test_pitch_frames_to_notes_splits_adjacent_semitone_motion() -> None:
    frames = [
        PitchFrame(timestamp_seconds=0.0, frequency_hz=261.63, confidence=0.9),
        PitchFrame(timestamp_seconds=0.03, frequency_hz=261.63, confidence=0.9),
        PitchFrame(timestamp_seconds=0.06, frequency_hz=261.63, confidence=0.9),
        PitchFrame(timestamp_seconds=0.09, frequency_hz=277.18, confidence=0.9),
        PitchFrame(timestamp_seconds=0.12, frequency_hz=277.18, confidence=0.9),
        PitchFrame(timestamp_seconds=0.15, frequency_hz=277.18, confidence=0.9),
    ]

    notes = _pitch_frames_to_notes(frames)

    assert [note.pitch for note in notes] == ["C4", "C#4"]


def test_pitch_frames_to_notes_suppresses_short_semitone_jitter_return() -> None:
    frames = [
        PitchFrame(timestamp_seconds=0.0, frequency_hz=369.99, confidence=0.9),
        PitchFrame(timestamp_seconds=0.03, frequency_hz=369.99, confidence=0.9),
        PitchFrame(timestamp_seconds=0.06, frequency_hz=349.23, confidence=0.9),
        PitchFrame(timestamp_seconds=0.09, frequency_hz=369.99, confidence=0.9),
        PitchFrame(timestamp_seconds=0.12, frequency_hz=369.99, confidence=0.9),
    ]

    notes = _pitch_frames_to_notes(frames)

    assert [note.pitch for note in notes] == ["F#4"]


def test_pitch_frames_to_notes_splits_long_same_pitch_segment() -> None:
    frames = [
        PitchFrame(timestamp_seconds=index * 0.03, frequency_hz=392.00, confidence=0.9)
        for index in range(31)
    ]

    notes = _pitch_frames_to_notes(frames)

    assert [note.pitch for note in notes] == ["G4", "G4", "G4"]


def test_pitch_frames_to_notes_does_not_split_normal_sustained_note() -> None:
    frames = [
        PitchFrame(timestamp_seconds=index * 0.03, frequency_hz=392.00, confidence=0.9)
        for index in range(18)
    ]

    notes = _pitch_frames_to_notes(frames)

    assert [note.pitch for note in notes] == ["G4"]


def test_pitch_frames_to_notes_recovers_short_confident_voiced_segment() -> None:
    frames = [
        PitchFrame(timestamp_seconds=0.0, frequency_hz=392.00, confidence=0.8),
        PitchFrame(timestamp_seconds=0.03, frequency_hz=392.00, confidence=0.8),
    ]

    notes = _pitch_frames_to_notes(frames)

    assert [note.pitch for note in notes] == ["G4"]
    assert notes[0].duration == pytest.approx(0.08)


def test_pitch_frames_to_notes_does_not_recover_single_frame_segment() -> None:
    frames = [
        PitchFrame(timestamp_seconds=0.0, frequency_hz=392.00, confidence=0.8),
    ]

    with pytest.raises(Exception):
        _pitch_frames_to_notes(frames)


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


def test_quantize_notes_uses_stable_rhythm_buckets_and_preserves_pitch() -> None:
    notes = [
        _note(60, 0.14, 0.16),
        _note(64, 0.41, 0.22),
    ]

    quantized = _quantize_notes(notes)

    assert [note.pitch for note in quantized] == ["C4", "E4"]
    assert quantized[0].startTime == pytest.approx(0.0)
    assert quantized[0].duration == pytest.approx(0.3)
    assert quantized[1].startTime == pytest.approx(0.3)
    assert quantized[1].duration == pytest.approx(0.15)


def test_quantize_notes_clamps_micro_notes_to_sixteenth_bucket() -> None:
    notes = [
        _note(60, 0.02, 0.07),
        _note(62, 0.18, 0.08),
        _note(64, 0.34, 0.09),
    ]

    quantized = _quantize_notes(notes)

    assert [note.startTime for note in quantized] == pytest.approx([0.0, 0.3, 0.45])
    assert [note.duration for note in quantized] == pytest.approx([0.15, 0.15, 0.15])


def test_quantize_notes_preserves_repeated_long_short_pattern() -> None:
    notes = [
        _note(60, 0.0, 0.33),
        _note(62, 0.34, 0.14),
        _note(64, 0.52, 0.31),
        _note(65, 0.85, 0.13),
    ]

    quantized = _quantize_notes(notes)

    assert [note.startTime for note in quantized] == pytest.approx([0.0, 0.3, 0.6, 0.9])
    assert [note.duration for note in quantized] == pytest.approx([0.3, 0.15, 0.3, 0.15])


def test_quantize_notes_preserves_count_and_shifts_start_collisions() -> None:
    notes = [
        _note(60, 0.02, 0.08),
        _note(62, 0.06, 0.08),
        _note(64, 0.18, 0.08),
    ]

    result = _quantize_notes_with_metrics(notes)

    assert result.before_note_count == 3
    assert result.after_note_count == 3
    assert result.audible_note_count == 3
    assert result.too_short_after_quantization_count == 0
    assert result.min_duration_after_quantization == pytest.approx(0.15)
    assert result.collision_count == 1
    assert result.shifted_note_count == 1
    assert [note.startTime for note in result.notes] == pytest.approx([0.0, 0.15, 0.3])


def test_quantize_notes_clamps_duration_before_next_start() -> None:
    notes = [
        _note(60, 0.0, 0.28),
        _note(62, 0.1, 0.1),
    ]

    result = _quantize_notes_with_metrics(notes)

    assert result.overlap_fix_count == 1
    assert result.audible_note_count == 2
    assert result.too_short_after_quantization_count == 0
    assert result.notes[0].startTime + result.notes[0].duration <= result.notes[1].startTime


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


def test_cleanup_notes_preserves_nearby_distinct_melody_notes() -> None:
    notes = [
        _note(60, 0.0, 0.22),
        _note(61, 0.25, 0.24),
        _note(64, 0.8, 0.3),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [60, 61, 64]


def test_cleanup_notes_smooths_short_isolated_pitch_spike() -> None:
    notes = [
        _note(60, 0.0, 0.22),
        _note(67, 0.23, 0.08),
        _note(60, 0.32, 0.24),
    ]

    cleaned = _cleanup_notes(notes)

    assert len(cleaned) == 1
    assert cleaned[0].midi_note == 60
    assert cleaned[0].duration == pytest.approx(0.56)


def test_cleanup_notes_preserves_repeated_notes_with_melody_boundary() -> None:
    notes = [
        _note(60, 0.0, 0.18),
        _note(60, 0.25, 0.18),
        _note(62, 0.5, 0.2),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [60, 60, 62]


def test_cleanup_notes_merges_same_pitch_only_across_tiny_gap() -> None:
    notes = [
        _note(60, 0.0, 0.18),
        _note(60, 0.2, 0.18),
        _note(62, 0.55, 0.2),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [60, 62]
    assert cleaned[0].duration == pytest.approx(0.38)


def test_cleanup_notes_drops_short_fragments_and_removes_overlaps() -> None:
    notes = [
        _note(60, 0.0, 0.04),
        _note(62, 0.1, 0.3),
        _note(64, 0.35, 0.3),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [62, 64]
    assert cleaned[1].startTime == pytest.approx(0.4)


def test_scale_adjustment_preserves_note_count_and_in_scale_notes() -> None:
    notes = [
        _note(60, 0.0, 0.2),
        _note(62, 0.25, 0.2),
        _note(64, 0.5, 0.2),
        _note(65, 0.75, 0.2),
    ]
    scale = ScaleFit(
        name="C_MAJOR",
        root_pitch_class=0,
        scale_pitch_classes=frozenset(MAJOR_SCALE_INTERVALS),
        confidence=1.0,
    )

    adjusted = _adjust_notes_to_scale(notes, scale)

    assert [note.midi_note for note in adjusted] == [60, 62, 64, 65]
    assert len(adjusted) == len(notes)


def test_cleanup_does_not_collapse_thirteen_event_melody_pattern() -> None:
    midi_notes = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62]
    notes = [
        _note(midi_note, index * 0.26, 0.2)
        for index, midi_note in enumerate(midi_notes)
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == midi_notes
    assert len(cleaned) == 13


def test_append_note_events_adds_deterministic_accents_and_release_gap() -> None:
    track = mido.MidiTrack()
    notes = [
        _note(60, 0.0, 0.6),
        _note(62, 0.6, 0.6),
    ]

    metrics = _append_note_events(track, notes, MIDI_MELODY_CHANNEL)

    note_on_events = [message for message in track if message.type == "note_on"]
    note_off_events = [message for message in track if message.type == "note_off"]
    assert [message.velocity for message in note_on_events] == [90, 85]
    assert note_off_events[0].time == MIDI_TICKS_PER_BEAT - MIDI_RELEASE_GAP_TICKS
    assert metrics.midi_event_count == 4
    assert metrics.zero_or_negative_delta_count == 0


def test_append_chord_events_uses_stable_sustained_block_chord() -> None:
    track = mido.MidiTrack()
    chord = Chord(root="C", type="MAJOR", startTime=0.0, duration=1.2)

    _append_chord_events(track, [chord])

    note_on_events = [message for message in track if message.type == "note_on"]
    note_off_events = [message for message in track if message.type == "note_off"]
    assert [message.note for message in note_on_events] == [48, 52, 55]
    assert [message.velocity for message in note_on_events] == [64, 64, 64]
    assert note_off_events[0].time == MIDI_TICKS_PER_BEAT * 2
    assert [message.time for message in note_off_events[1:]] == [0, 0]
    assert all(message.channel == MIDI_CHORD_CHANNEL for message in note_on_events)


def _note(midi_note: int, start_time: float, duration: float) -> Note:
    return Note(
        pitch=_midi_note_to_name(midi_note),
        midi_note=midi_note,
        startTime=start_time,
        duration=duration,
        velocity=80,
    )
