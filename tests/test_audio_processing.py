from pathlib import Path

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
from app.audio_processing import _basic_pitch_event_to_note
from app.audio_processing import _cleanup_notes
from app.audio_processing import _cleanup_notes_with_metrics
from app.audio_processing import _fit_scale
from app.audio_processing import _frequency_to_midi_note
from app.audio_processing import _infer_chords
from app.audio_processing import _write_debug_chord_midi_file
from app.audio_processing import _write_debug_combined_midi_file
from app.audio_processing import _midi_note_to_name
from app.audio_processing import _pitch_frames_to_notes
from app.audio_processing import _quantize_notes
from app.audio_processing import _quantize_notes_with_metrics


def test_frequency_to_midi_note_and_note_name() -> None:
    assert _frequency_to_midi_note(440.0) == 69
    assert _midi_note_to_name(60) == "C4"
    assert _midi_note_to_name(61) == "C#4"


def test_basic_pitch_event_to_note_maps_tuple_event() -> None:
    note = _basic_pitch_event_to_note((0.25, 0.75, 64, 0.8))

    assert note.pitch == "E4"
    assert note.midi_note == 64
    assert note.startTime == pytest.approx(0.25)
    assert note.duration == pytest.approx(0.5)
    assert note.velocity == 102


def test_basic_pitch_event_to_note_maps_dict_event() -> None:
    note = _basic_pitch_event_to_note(
        {
            "start_time_s": 1.0,
            "end_time_s": 1.4,
            "pitch_midi": 67,
            "amplitude": 0.5,
        }
    )

    assert note.pitch == "G4"
    assert note.midi_note == 67
    assert note.startTime == pytest.approx(1.0)
    assert note.duration == pytest.approx(0.4)
    assert note.velocity == 64


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


def test_fit_scale_prefers_phrase_anchor_over_relative_minor_for_nabiya_shape() -> None:
    midi_notes = [54, 51, 51, 52, 49, 49, 47, 49, 51, 52, 54, 54, 54]
    notes = [
        _note(midi_note, index * 0.3, 0.3)
        for index, midi_note in enumerate(midi_notes)
    ]

    scale = _fit_scale(notes)

    assert scale.name == "B_MAJOR"
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


def test_adjust_notes_to_scale_merges_correction_fragments() -> None:
    scale = ScaleFit(
        name="B_MAJOR",
        root_pitch_class=11,
        scale_pitch_classes=frozenset((11 + interval) % 12 for interval in MAJOR_SCALE_INTERVALS),
        confidence=1.0,
    )
    notes = [
        _note(53, 2.37, 0.12),
        _note(52, 2.52, 0.17),
        _note(49, 2.74, 0.35),
    ]

    adjusted = _adjust_notes_to_scale(notes, scale)

    assert [note.midi_note for note in adjusted] == [52, 49]
    assert adjusted[0].startTime == pytest.approx(2.37)
    assert adjusted[0].duration == pytest.approx(0.32)


def test_quantize_notes_preserves_natural_onsets_when_far_from_grid() -> None:
    notes = [
        _note(60, 0.14, 0.16),
        _note(64, 0.41, 0.22),
    ]

    quantized = _quantize_notes(notes)

    assert [note.pitch for note in quantized] == ["C4", "E4"]
    assert quantized[0].startTime == pytest.approx(0.14)
    assert quantized[0].duration == pytest.approx(0.15)
    assert quantized[1].startTime == pytest.approx(0.41)
    assert quantized[1].duration == pytest.approx(0.22)


def test_quantize_notes_clamps_micro_notes_to_sixteenth_bucket() -> None:
    notes = [
        _note(60, 0.02, 0.07),
        _note(62, 0.18, 0.08),
        _note(64, 0.34, 0.09),
    ]

    quantized = _quantize_notes(notes)

    assert [note.startTime for note in quantized] == pytest.approx([0.0, 0.15, 0.3])
    assert [note.duration for note in quantized] == pytest.approx([0.15, 0.15, 0.15])


def test_quantize_notes_preserves_repeated_long_short_pattern() -> None:
    notes = [
        _note(60, 0.0, 0.33),
        _note(62, 0.34, 0.14),
        _note(64, 0.52, 0.31),
        _note(65, 0.85, 0.13),
    ]

    quantized = _quantize_notes(notes)

    assert [note.startTime for note in quantized] == pytest.approx([0.0, 0.3, 0.52, 0.9])
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
    assert result.collision_count == 2
    assert result.shifted_note_count == 2
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


def test_quantize_notes_preserves_repeated_articulation_without_filling_rest() -> None:
    notes = [
        _note(51, 1.22, 0.34),
        _note(51, 1.59, 0.43),
        _note(52, 2.52, 0.17),
    ]

    quantized = _quantize_notes(notes)

    assert [note.midi_note for note in quantized] == [51, 51, 52]
    assert [note.startTime for note in quantized] == pytest.approx([1.2, 1.59, 2.52])
    assert [note.duration for note in quantized] == pytest.approx([0.3, 0.43, 0.15])


def test_quantize_notes_preserves_three_repeated_final_onsets() -> None:
    notes = [
        _note(54, 5.56, 0.28),
        _note(54, 5.88, 0.38),
        _note(54, 6.29, 0.34),
    ]

    quantized = _quantize_notes(notes)

    assert [note.midi_note for note in quantized] == [54, 54, 54]
    assert [note.startTime for note in quantized] == pytest.approx([5.56, 5.88, 6.3])
    assert [note.duration for note in quantized] == pytest.approx([0.3, 0.38, 0.3])


def test_quantize_notes_preserves_nabiya_cleaned_phrasing_baseline() -> None:
    notes = [
        _note(54, 0.79, 0.209),
        _note(51, 1.219, 0.341),
        _note(51, 1.59, 0.431),
        _note(52, 2.37, 0.325),
        _note(49, 2.741, 0.352),
        _note(49, 3.124, 0.371),
        _note(47, 3.903, 0.385),
        _note(49, 4.356, 0.295),
        _note(51, 4.681, 0.406),
        _note(52, 5.123, 0.29),
        _note(54, 5.564, 0.284),
        _note(54, 5.878, 0.378),
        _note(54, 6.285, 0.336),
    ]

    quantized = _quantize_notes(notes)

    assert [note.midi_note for note in quantized] == [note.midi_note for note in notes]
    assert [note.startTime for note in quantized] == pytest.approx([
        0.79,
        1.2,
        1.59,
        2.4,
        2.7,
        3.124,
        3.903,
        4.356,
        4.681,
        5.123,
        5.564,
        5.878,
        6.3,
    ])


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


def test_infer_chords_uses_window_anchor_for_simple_major_melody() -> None:
    scale = ScaleFit(
        name="B_MAJOR",
        root_pitch_class=11,
        scale_pitch_classes=frozenset((11 + interval) % 12 for interval in MAJOR_SCALE_INTERVALS),
        confidence=1.0,
    )
    notes = [
        _note(54, 0.90, 0.15),
        _note(51, 1.20, 0.30),
        _note(51, 1.50, 0.30),
        _note(52, 2.40, 0.15),
        _note(49, 2.70, 0.30),
        _note(49, 3.00, 0.30),
        _note(47, 3.90, 0.30),
        _note(49, 4.50, 0.30),
        _note(51, 4.80, 0.30),
        _note(52, 5.10, 0.30),
        _note(54, 5.70, 0.30),
        _note(54, 6.00, 0.30),
        _note(54, 6.30, 0.30),
    ]

    chords = _infer_chords(notes, scale)

    assert [(chord.root, chord.type) for chord in chords] == [
        ("B", "MAJOR"),
        ("E", "MAJOR"),
        ("B", "MAJOR"),
        ("F#", "MAJOR"),
    ]
    assert [chord.startTime for chord in chords] == pytest.approx([0.9, 2.4, 3.9, 5.7])
    assert [chord.duration for chord in chords] == pytest.approx([1.5, 1.5, 1.8, 1.5])


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


def test_cleanup_notes_merges_short_same_pitch_basic_pitch_fragments() -> None:
    notes = [
        _note(60, 0.0, 0.10),
        _note(60, 0.18, 0.09),
        _note(62, 0.5, 0.2),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [60, 62]
    assert cleaned[0].duration == pytest.approx(0.27)


def test_cleanup_notes_does_not_merge_long_repeated_basic_pitch_onsets() -> None:
    notes = [
        _note(60, 0.0, 0.18),
        _note(60, 0.25, 0.18),
        _note(62, 0.5, 0.2),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [60, 60, 62]


def test_cleanup_notes_preserves_repeated_onsets_but_merges_tail_fragment() -> None:
    notes = [
        _note(51, 1.22, 0.34),
        _note(51, 1.59, 0.23),
        _note(51, 1.85, 0.17),
        _note(53, 2.37, 0.12),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [51, 51, 53]
    assert cleaned[0].startTime == pytest.approx(1.22)
    assert cleaned[0].duration == pytest.approx(0.34)
    assert cleaned[1].startTime == pytest.approx(1.59)
    assert cleaned[1].duration == pytest.approx(0.43)


def test_cleanup_notes_preserves_three_repeated_final_onsets() -> None:
    notes = [
        _note(54, 5.56, 0.28),
        _note(54, 5.88, 0.38),
        _note(54, 6.29, 0.34),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [54, 54, 54]


def test_cleanup_notes_filters_short_chromatic_descent_transient() -> None:
    notes = [
        _note(54, 0.79, 0.18),
        _note(53, 1.00, 0.16),
        _note(51, 1.22, 0.34),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [54, 51]


def test_cleanup_notes_filters_short_upper_neighbor_transient() -> None:
    notes = [
        _note(51, 1.59, 0.43),
        _note(53, 2.37, 0.12),
        _note(52, 2.52, 0.17),
        _note(49, 2.74, 0.35),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [51, 52, 49]


def test_cleanup_notes_preserves_longer_chromatic_melody_motion() -> None:
    notes = [
        _note(60, 0.0, 0.24),
        _note(61, 0.3, 0.22),
        _note(63, 0.6, 0.24),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [60, 61, 63]


def test_cleanup_notes_drops_short_fragments_and_removes_overlaps() -> None:
    notes = [
        _note(60, 0.0, 0.04),
        _note(62, 0.1, 0.3),
        _note(64, 0.35, 0.3),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [62, 64]
    assert cleaned[0].duration == pytest.approx(0.25)
    assert cleaned[1].startTime == pytest.approx(0.35)


def test_cleanup_notes_drops_weak_short_basic_pitch_transients() -> None:
    notes = [
        Note(pitch="C4", midi_note=60, startTime=0.0, duration=0.08, velocity=30),
        _note(62, 0.2, 0.3),
    ]

    result = _cleanup_notes_with_metrics(notes)

    assert [note.midi_note for note in result.notes] == [62]
    assert result.dropped_short_note_count == 1


def test_cleanup_notes_keeps_stronger_simultaneous_basic_pitch_candidate() -> None:
    notes = [
        Note(pitch="C4", midi_note=60, startTime=0.0, duration=0.3, velocity=50),
        Note(pitch="E4", midi_note=64, startTime=0.01, duration=0.42, velocity=80),
        _note(67, 0.55, 0.2),
    ]

    cleaned = _cleanup_notes(notes)

    assert [note.midi_note for note in cleaned] == [64, 67]


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


def test_debug_chord_outputs_make_accompaniment_audible(tmp_path: Path) -> None:
    notes = [_note(60, 0.0, 0.3), _note(64, 0.3, 0.3)]
    chords = [Chord(root="C", type="MAJOR", startTime=0.0, duration=1.2)]

    chord_path = _write_debug_chord_midi_file("sample", tmp_path, "chords", chords)
    combined_path = _write_debug_combined_midi_file(
        "sample",
        tmp_path,
        "combined",
        notes,
        chords,
    )

    assert chord_path.exists()
    assert combined_path.exists()
    chord_midi = mido.MidiFile(chord_path)
    combined_midi = mido.MidiFile(combined_path)
    chord_note_ons = [
        message
        for track in chord_midi.tracks
        for message in track
        if message.type == "note_on"
    ]
    combined_channels = {
        message.channel
        for track in combined_midi.tracks
        for message in track
        if message.type == "note_on"
    }
    assert all(message.channel == MIDI_CHORD_CHANNEL for message in chord_note_ons)
    assert combined_channels == {MIDI_MELODY_CHANNEL, MIDI_CHORD_CHANNEL}


def _note(midi_note: int, start_time: float, duration: float) -> Note:
    return Note(
        pitch=_midi_note_to_name(midi_note),
        midi_note=midi_note,
        startTime=start_time,
        duration=duration,
        velocity=80,
    )
