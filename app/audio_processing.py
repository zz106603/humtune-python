from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import librosa
import mido
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 22050
MIN_DURATION_SECONDS = 0.25
PITCH_HOP_LENGTH = 512
PITCH_FRAME_LENGTH = 2048
MIN_PITCH_FRAMES = 3
MIN_VOICED_PROBABILITY = 0.5
MIN_PITCH_HZ = librosa.note_to_hz("C2")
MAX_PITCH_HZ = librosa.note_to_hz("C7")
PITCH_FRAME_GAP_BOUNDARY_SECONDS = (PITCH_HOP_LENGTH / TARGET_SAMPLE_RATE) * 2.5
PITCH_SEGMENT_TOLERANCE_SEMITONES = 0
MIN_NOTE_DURATION_SECONDS = 0.08
RECOVERABLE_SHORT_SEGMENT_MIN_SECONDS = (PITCH_HOP_LENGTH / TARGET_SAMPLE_RATE) * 2.0
RECOVERABLE_SHORT_SEGMENT_MIN_CONFIDENCE = 0.57
JITTER_SEGMENT_MAX_SECONDS = 0.12
JITTER_MAX_SEMITONES = 1
NOTE_MERGE_TOLERANCE_SEMITONES = 1
NOTE_CLEANUP_MIN_DURATION_SECONDS = 0.08
NOTE_CLEANUP_MERGE_GAP_SECONDS = 0.04
NOTE_CLEANUP_SPIKE_MAX_DURATION_SECONDS = 0.18
BASIC_PITCH_TRANSIENT_MAX_SECONDS = 0.06
BASIC_PITCH_LOW_VELOCITY_TRANSIENT_MAX_SECONDS = 0.10
BASIC_PITCH_LOW_VELOCITY_THRESHOLD = 45
BASIC_PITCH_FRAGMENT_MERGE_GAP_SECONDS = 0.10
BASIC_PITCH_FRAGMENT_MAX_SECONDS = 0.12
BASIC_PITCH_REPEATED_ONSET_MIN_GAP_SECONDS = 0.24
BASIC_PITCH_REPEATED_NOTE_MIN_SECONDS = 0.18
BASIC_PITCH_SIMULTANEOUS_ONSET_SECONDS = 0.035
BASIC_PITCH_OVERLAP_RATIO_THRESHOLD = 0.6
BASIC_PITCH_CHROMATIC_TRANSIENT_MAX_SECONDS = 0.17
BASIC_PITCH_SHORT_NEIGHBOR_TRANSIENT_MAX_SECONDS = 0.13
DEFAULT_NOTE_VELOCITY = 80
MIDI_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_SCALE_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE_INTERVALS = (0, 2, 3, 5, 7, 8, 10)
FALLBACK_TEMPO_BPM = 100
QUANTIZE_GRID_SECONDS = (60.0 / FALLBACK_TEMPO_BPM) / 2.0
QUANTIZE_ONSET_SNAP_TOLERANCE_SECONDS = 0.05
QUANTIZE_DURATION_SNAP_TOLERANCE_SECONDS = 0.05
INTERNAL_REPEAT_SPLIT_MIN_SECONDS = QUANTIZE_GRID_SECONDS * 2.5
INTERNAL_REPEAT_TARGET_SECONDS = QUANTIZE_GRID_SECONDS
RHYTHM_BUCKET_SECONDS = (
    QUANTIZE_GRID_SECONDS / 2.0,
    QUANTIZE_GRID_SECONDS,
    QUANTIZE_GRID_SECONDS * 2.0,
    QUANTIZE_GRID_SECONDS * 4.0,
)
AUDIBLE_MIN_NOTE_SECONDS = RHYTHM_BUCKET_SECONDS[0]
CHORD_WINDOW_SECONDS = 4 * (60.0 / FALLBACK_TEMPO_BPM)
CHORD_TARGET_SECTION_SECONDS = 1.6
CHORD_SHORT_MELODY_MAX_SECONDS = 8.0
CHORD_SHORT_MELODY_MAX_WINDOWS = 4
MAJOR_CHORD_TIEBREAK_DEGREES = (0, 3, 4, 5, 1, 2, 6)
MINOR_CHORD_TIEBREAK_DEGREES = (0, 3, 4, 5, 6, 2, 1)
CHORD_WINDOW_ROOT_BONUS = 0.8
CHORD_INITIAL_TONIC_BONUS = 0.5
CHORD_BOUNDARY_SNAP_TOLERANCE_SECONDS = QUANTIZE_GRID_SECONDS * 1.5
CHORD_FINAL_BOUNDARY_LOOKAHEAD_SECONDS = QUANTIZE_GRID_SECONDS * 2.0
CHORD_MIN_SECTION_SECONDS = QUANTIZE_GRID_SECONDS * 3.0
CHORD_FINAL_SUSTAIN_SECONDS = 60.0 / FALLBACK_TEMPO_BPM
MIDI_TICKS_PER_BEAT = 480
MIDI_MELODY_CHANNEL = 0
MIDI_CHORD_CHANNEL = 1
MIDI_PIANO_PROGRAM = 0
MIDI_CHORD_VELOCITY = 64
CHORD_BASE_OCTAVE_MIDI = 48
ENABLE_ACCOMPANIMENT = True
MIDI_BAR_BEATS = 4
MIDI_RELEASE_GAP_TICKS = 24
MIDI_MIN_NOTE_TICKS = 60
MIDI_HALF_BEAT_TICKS = MIDI_TICKS_PER_BEAT // 2
PREVIEW_SAMPLE_RATE = 22050
PREVIEW_BASE_AMPLITUDE = 0.18
PREVIEW_ATTACK_SECONDS = 0.005
PREVIEW_RELEASE_SECONDS = 0.03
PREVIEW_PEAK_HEADROOM = 0.95


class AudioProcessingError(Exception):
    pass


@dataclass(frozen=True)
class PitchFrame:
    timestamp_seconds: float
    frequency_hz: float
    confidence: float
    onset_strength: float = 0.0
    rms: float = 0.0


@dataclass(frozen=True)
class Note:
    pitch: str
    midi_note: int
    startTime: float
    duration: float
    velocity: int


@dataclass(frozen=True)
class ScaleFit:
    name: str
    root_pitch_class: int
    scale_pitch_classes: frozenset[int]
    confidence: float


@dataclass(frozen=True)
class Chord:
    root: str
    type: str
    startTime: float
    duration: float


@dataclass(frozen=True)
class LoadedAudio:
    audio_id: str
    raw_audio_path: Path
    output_directory: Path
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    pitch_frames: list[PitchFrame]
    original_notes: list[Note]
    cleaned_notes: list[Note]
    detectedScale: str
    keyConfidence: float
    adjusted_notes: list[Note]
    quantized_notes: list[Note]
    chords: list[Chord]
    melodyMetrics: dict[str, float]
    feedbackEvidence: dict[str, Any]
    midiPath: str
    previewAudioPath: str | None


@dataclass(frozen=True)
class NoteCleanupResult:
    notes: list[Note]
    raw_note_count: int
    cleaned_note_count: int
    dropped_short_note_count: int
    merged_note_count: int
    overlap_fix_count: int


@dataclass(frozen=True)
class QuantizeResult:
    notes: list[Note]
    before_note_count: int
    after_note_count: int
    audible_note_count: int
    too_short_after_quantization_count: int
    min_duration_after_quantization: float
    collision_count: int
    shifted_note_count: int
    overlap_fix_count: int


@dataclass(frozen=True)
class MidiEventMetrics:
    midi_event_count: int
    zero_or_negative_delta_count: int


@dataclass(frozen=True)
class MelodyQualityAnalysis:
    metrics: dict[str, float]
    evidence: dict[str, Any]


class PreviewNote(NamedTuple):
    midi_note: int
    velocity: int
    start_time: float
    duration: float


def analyze_audio(
    audio_id: str,
    raw_audio_path: str,
    output_directory: str,
) -> LoadedAudio:
    audio_path = _validate_raw_audio_path(raw_audio_path)
    output_path = _validate_output_directory_path(output_directory)
    samples, sample_rate = _load_audio(audio_path)
    duration_seconds = _validate_loaded_audio(samples, sample_rate)
    pitch_frames: list[PitchFrame] = []
    original_notes = _transcribe_notes_with_basic_pitch(audio_path)
    cleanup_result = _cleanup_notes_with_metrics(original_notes)
    _log_note_cleanup_metrics(audio_id, cleanup_result)
    cleaned_notes = cleanup_result.notes
    scale_fit = _fit_scale(cleaned_notes)
    adjusted_notes = _adjust_notes_to_scale(cleaned_notes, scale_fit)
    quantize_result = _quantize_notes_with_metrics(adjusted_notes)
    quantized_notes = quantize_result.notes
    _log_note_pipeline_counts(
        audio_id=audio_id,
        pitch_frame_count=len(pitch_frames),
        raw_note_count=len(original_notes),
        cleaned_note_count=len(cleaned_notes),
        adjusted_note_count=len(adjusted_notes),
        quantized_note_count=len(quantized_notes),
        dropped_note_count=cleanup_result.dropped_short_note_count,
        merged_note_count=cleanup_result.merged_note_count,
    )
    _log_quantization_metrics(audio_id, quantize_result)
    _log_quantized_duration_stats(audio_id, quantized_notes)
    chords = _infer_chords(quantized_notes, scale_fit)
    quality_analysis = _calculate_melody_quality_analysis(
        cleaned_notes=cleaned_notes,
        quantized_notes=quantized_notes,
        scale_fit=scale_fit,
        chords=chords,
    )
    midi_path = _write_midi_file(audio_id, output_path, quantized_notes, chords)
    preview_audio_path = _try_write_wav_preview_file(audio_id, output_path, midi_path)

    return LoadedAudio(
        audio_id=audio_id,
        raw_audio_path=audio_path,
        output_directory=output_path,
        samples=samples,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        pitch_frames=pitch_frames,
        original_notes=original_notes,
        cleaned_notes=cleaned_notes,
        detectedScale=scale_fit.name,
        keyConfidence=scale_fit.confidence,
        adjusted_notes=adjusted_notes,
        quantized_notes=quantized_notes,
        chords=chords,
        melodyMetrics=quality_analysis.metrics,
        feedbackEvidence=quality_analysis.evidence,
        midiPath=str(midi_path),
        previewAudioPath=str(preview_audio_path) if preview_audio_path else None,
    )


def _validate_raw_audio_path(raw_audio_path: str) -> Path:
    if not raw_audio_path or not raw_audio_path.strip():
        raise AudioProcessingError("rawAudioPath is required")

    audio_path = Path(raw_audio_path)
    if not audio_path.exists():
        raise AudioProcessingError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioProcessingError(f"Audio path is not a file: {audio_path}")

    return audio_path


def _validate_output_directory_path(output_directory: str) -> Path:
    if not output_directory or not output_directory.strip():
        raise AudioProcessingError("outputDirectory is required")

    return Path(output_directory)


def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    try:
        samples, sample_rate = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
        )
    except Exception as exc:
        raise AudioProcessingError(f"Unable to read audio file: {audio_path}") from exc

    return samples, sample_rate


def _transcribe_notes_with_basic_pitch(audio_path: Path) -> list[Note]:
    try:
        from basic_pitch.inference import predict
    except ModuleNotFoundError as exc:
        raise AudioProcessingError(
            "Basic Pitch is required for note transcription"
        ) from exc

    try:
        _, _, note_events = predict(str(audio_path))
    except Exception as exc:
        raise AudioProcessingError("Unable to transcribe notes with Basic Pitch") from exc

    notes = [
        _basic_pitch_event_to_note(event)
        for event in note_events
    ]
    usable_notes = [
        note
        for note in notes
        if note.duration > 0
    ]
    if not usable_notes:
        raise AudioProcessingError("Basic Pitch produced no usable note events")

    return sorted(usable_notes, key=lambda note: (note.startTime, note.midi_note))


def _basic_pitch_event_to_note(event: Any) -> Note:
    if isinstance(event, dict):
        start_time = _event_value(event, "start_time_s", "start_time", "start", "onset")
        end_time = _event_value(event, "end_time_s", "end_time", "end", "offset")
        midi_note = int(round(_event_value(event, "pitch_midi", "midi_pitch", "pitch")))
        confidence = _event_value(event, "amplitude", "confidence", "velocity", default=0.75)
    else:
        start_time = float(event[0])
        end_time = float(event[1])
        midi_note = int(round(float(event[2])))
        confidence = float(event[3]) if len(event) > 3 else 0.75

    velocity = max(1, min(127, int(round(confidence * 127))))
    return Note(
        pitch=_midi_note_to_name(midi_note),
        midi_note=midi_note,
        startTime=float(start_time),
        duration=float(end_time - start_time),
        velocity=velocity or DEFAULT_NOTE_VELOCITY,
    )


def _event_value(event: dict[str, Any], *keys: str, default: float | None = None) -> float:
    for key in keys:
        if key in event:
            return float(event[key])
    if default is not None:
        return default
    raise AudioProcessingError(f"Basic Pitch event missing one of: {', '.join(keys)}")


def _validate_loaded_audio(samples: np.ndarray, sample_rate: int) -> float:
    if samples.size == 0:
        raise AudioProcessingError("Audio file is empty")
    if not np.all(np.isfinite(samples)):
        raise AudioProcessingError("Audio file contains invalid sample values")

    duration_seconds = float(samples.size) / float(sample_rate)
    if duration_seconds < MIN_DURATION_SECONDS:
        raise AudioProcessingError("Audio file is too short")

    return duration_seconds


def _detect_pitch_frames(samples: np.ndarray, sample_rate: int) -> list[PitchFrame]:
    try:
        frequencies, voiced_flags, voiced_probabilities = librosa.pyin(
            samples,
            fmin=MIN_PITCH_HZ,
            fmax=MAX_PITCH_HZ,
            sr=sample_rate,
            frame_length=PITCH_FRAME_LENGTH,
            hop_length=PITCH_HOP_LENGTH,
        )
    except Exception as exc:
        raise AudioProcessingError("Unable to detect pitch") from exc

    if frequencies is None or voiced_flags is None:
        raise AudioProcessingError("No usable pitch frames found")

    frame_times = librosa.frames_to_time(
        np.arange(len(frequencies)),
        sr=sample_rate,
        hop_length=PITCH_HOP_LENGTH,
    )
    onset_strengths = librosa.onset.onset_strength(
        y=samples,
        sr=sample_rate,
        hop_length=PITCH_HOP_LENGTH,
    )
    rms_values = librosa.feature.rms(
        y=samples,
        frame_length=PITCH_FRAME_LENGTH,
        hop_length=PITCH_HOP_LENGTH,
    )[0]

    pitch_frames: list[PitchFrame] = []
    for frame_index, (timestamp, frequency, is_voiced, voiced_probability) in enumerate(zip(
        frame_times,
        frequencies,
        voiced_flags,
        voiced_probabilities,
    )):
        confidence = float(voiced_probability) if np.isfinite(voiced_probability) else 0.0
        if not is_voiced:
            continue
        if not np.isfinite(frequency):
            continue
        if confidence < MIN_VOICED_PROBABILITY:
            continue

        pitch_frames.append(
            PitchFrame(
                timestamp_seconds=float(timestamp),
                frequency_hz=float(frequency),
                confidence=confidence,
                onset_strength=(
                    float(onset_strengths[frame_index])
                    if frame_index < len(onset_strengths)
                    else 0.0
                ),
                rms=float(rms_values[frame_index]) if frame_index < len(rms_values) else 0.0,
            )
        )

    if len(pitch_frames) < MIN_PITCH_FRAMES:
        raise AudioProcessingError("Not enough usable pitch frames found")

    return pitch_frames


def _pitch_frames_to_notes(pitch_frames: list[PitchFrame]) -> list[Note]:
    if not pitch_frames:
        raise AudioProcessingError("No pitch frames available for note conversion")

    notes: list[Note] = []
    current_midi = _frequency_to_midi_note(pitch_frames[0].frequency_hz)
    current_segment_frames = [pitch_frames[0]]
    last_timestamp = pitch_frames[0].timestamp_seconds

    index = 1
    while index < len(pitch_frames):
        frame = pitch_frames[index]
        frame_midi = _frequency_to_midi_note(frame.frequency_hz)
        frame_gap = frame.timestamp_seconds - last_timestamp
        should_continue_note = (
            frame_gap <= PITCH_FRAME_GAP_BOUNDARY_SECONDS
            and abs(frame_midi - current_midi) <= PITCH_SEGMENT_TOLERANCE_SEMITONES
        )
        if should_continue_note:
            current_segment_frames.append(frame)
            last_timestamp = frame.timestamp_seconds
            index += 1
            continue

        jitter_frames, next_index = _collect_jitter_return_frames(
            pitch_frames,
            index,
            current_midi,
        )
        if jitter_frames:
            current_segment_frames.extend(jitter_frames)
            last_timestamp = jitter_frames[-1].timestamp_seconds
            index = next_index
            continue

        _append_note_segment(notes, current_midi, current_segment_frames)
        current_midi = frame_midi
        current_segment_frames = [frame]
        last_timestamp = frame.timestamp_seconds
        index += 1

    _append_note_segment(notes, current_midi, current_segment_frames)

    if not notes:
        raise AudioProcessingError("No valid notes found")

    return notes


def _collect_jitter_return_frames(
    pitch_frames: list[PitchFrame],
    start_index: int,
    current_midi: int,
) -> tuple[list[PitchFrame], int]:
    first_midi = _frequency_to_midi_note(pitch_frames[start_index].frequency_hz)
    if first_midi == current_midi:
        return [], start_index

    jitter_frames: list[PitchFrame] = []
    index = start_index

    while index < len(pitch_frames):
        frame = pitch_frames[index]
        frame_midi = _frequency_to_midi_note(frame.frequency_hz)
        if abs(frame_midi - current_midi) > JITTER_MAX_SEMITONES:
            break
        if frame_midi == current_midi:
            jitter_frames.append(frame)
            return jitter_frames, index + 1

        jitter_frames.append(frame)
        duration = (
            jitter_frames[-1].timestamp_seconds
            - jitter_frames[0].timestamp_seconds
            + (PITCH_HOP_LENGTH / TARGET_SAMPLE_RATE)
        )
        if duration > JITTER_SEGMENT_MAX_SECONDS:
            return [], start_index

        index += 1

    return [], start_index


def _append_note_segment(
    notes: list[Note],
    midi_note: int,
    segment_frames: list[PitchFrame],
) -> None:
    before_count = len(notes)
    for start_time, end_time in _split_repeated_note_segment(segment_frames):
        _append_note_if_long_enough(notes, midi_note, start_time, end_time)
    if len(notes) == before_count:
        _append_recoverable_short_note_segment(notes, midi_note, segment_frames)


def _append_recoverable_short_note_segment(
    notes: list[Note],
    midi_note: int,
    segment_frames: list[PitchFrame],
) -> None:
    if not _is_recoverable_short_note_segment(segment_frames):
        return

    notes.append(
        Note(
            pitch=_midi_note_to_name(midi_note),
            midi_note=midi_note,
            startTime=float(segment_frames[0].timestamp_seconds),
            duration=float(MIN_NOTE_DURATION_SECONDS),
            velocity=DEFAULT_NOTE_VELOCITY,
        )
    )


def _is_recoverable_short_note_segment(segment_frames: list[PitchFrame]) -> bool:
    if len(segment_frames) < 2:
        return False

    duration = (
        segment_frames[-1].timestamp_seconds
        - segment_frames[0].timestamp_seconds
        + (PITCH_HOP_LENGTH / TARGET_SAMPLE_RATE)
    )
    if duration < RECOVERABLE_SHORT_SEGMENT_MIN_SECONDS:
        return False
    if duration >= MIN_NOTE_DURATION_SECONDS:
        return False

    average_confidence = sum(frame.confidence for frame in segment_frames) / len(segment_frames)
    return average_confidence >= RECOVERABLE_SHORT_SEGMENT_MIN_CONFIDENCE


def _split_repeated_note_segment(segment_frames: list[PitchFrame]) -> list[tuple[float, float]]:
    if not segment_frames:
        return []

    start_time = segment_frames[0].timestamp_seconds
    end_time = segment_frames[-1].timestamp_seconds
    duration = end_time - start_time + (PITCH_HOP_LENGTH / TARGET_SAMPLE_RATE)
    if duration < INTERNAL_REPEAT_SPLIT_MIN_SECONDS:
        return [(start_time, end_time)]

    split_count = max(1, int(round(duration / INTERNAL_REPEAT_TARGET_SECONDS)))
    if split_count <= 1:
        return [(start_time, end_time)]

    split_duration = duration / split_count
    split_segments: list[tuple[float, float]] = []
    for index in range(split_count):
        split_start = start_time + (index * split_duration)
        split_end = split_start + split_duration - (PITCH_HOP_LENGTH / TARGET_SAMPLE_RATE)
        split_segments.append((split_start, max(split_start, split_end)))

    return split_segments


def _append_note_if_long_enough(
    notes: list[Note],
    midi_note: int,
    start_time: float,
    end_time: float,
) -> None:
    duration = end_time - start_time + (PITCH_HOP_LENGTH / TARGET_SAMPLE_RATE)
    if duration < MIN_NOTE_DURATION_SECONDS:
        return

    notes.append(
        Note(
            pitch=_midi_note_to_name(midi_note),
            midi_note=midi_note,
            startTime=float(start_time),
            duration=float(duration),
            velocity=DEFAULT_NOTE_VELOCITY,
        )
    )


def _cleanup_notes(notes: list[Note]) -> list[Note]:
    return _cleanup_notes_with_metrics(notes).notes


def _cleanup_notes_with_metrics(notes: list[Note]) -> NoteCleanupResult:
    if not notes:
        raise AudioProcessingError("No notes available for cleanup")

    ordered_notes = sorted(notes, key=lambda note: (note.startTime, note.midi_note))
    normalized_input_notes = _normalize_transcribed_notes(ordered_notes)
    dropped_transient_note_count = len(ordered_notes) - len(normalized_input_notes)
    smoothed_notes = _smooth_isolated_pitch_spikes(normalized_input_notes)
    merged_notes, merged_note_count = _merge_adjacent_similar_notes(smoothed_notes)
    cleaned_notes = _drop_short_notes(merged_notes)
    dropped_short_note_count = dropped_transient_note_count + len(merged_notes) - len(cleaned_notes)

    if not cleaned_notes:
        raise AudioProcessingError("No valid notes found after cleanup")

    normalized_notes, overlap_fix_count = _remove_note_overlaps(cleaned_notes)
    if not normalized_notes:
        raise AudioProcessingError("No valid notes found after cleanup")

    return NoteCleanupResult(
        notes=normalized_notes,
        raw_note_count=len(notes),
        cleaned_note_count=len(normalized_notes),
        dropped_short_note_count=dropped_short_note_count,
        merged_note_count=merged_note_count,
        overlap_fix_count=overlap_fix_count,
    )


def _normalize_transcribed_notes(notes: list[Note]) -> list[Note]:
    stable_notes = [
        note
        for note in notes
        if _is_stable_transcribed_note(note)
    ]
    melody_candidates = _suppress_overlapping_transcription_candidates(stable_notes)
    return _filter_pitch_transients(melody_candidates)


def _is_stable_transcribed_note(note: Note) -> bool:
    _validate_note_timing(note)
    if note.duration < BASIC_PITCH_TRANSIENT_MAX_SECONDS:
        return False
    if (
        note.duration < BASIC_PITCH_LOW_VELOCITY_TRANSIENT_MAX_SECONDS
        and note.velocity < BASIC_PITCH_LOW_VELOCITY_THRESHOLD
    ):
        return False
    if note.midi_note < _frequency_to_midi_note(MIN_PITCH_HZ):
        return False
    if note.midi_note > _frequency_to_midi_note(MAX_PITCH_HZ):
        return False
    return True


def _suppress_overlapping_transcription_candidates(notes: list[Note]) -> list[Note]:
    selected_notes: list[Note] = []
    for note in notes:
        if not selected_notes:
            selected_notes.append(note)
            continue

        previous_note = selected_notes[-1]
        if not _is_competing_transcription_candidate(previous_note, note):
            selected_notes.append(note)
            continue

        if _note_priority(note) > _note_priority(previous_note):
            selected_notes[-1] = note

    return selected_notes


def _is_competing_transcription_candidate(first_note: Note, second_note: Note) -> bool:
    if abs(first_note.startTime - second_note.startTime) > BASIC_PITCH_SIMULTANEOUS_ONSET_SECONDS:
        return False

    overlap = _note_overlap_seconds(first_note, second_note)
    if overlap <= 0:
        return False

    shorter_duration = min(first_note.duration, second_note.duration)
    return (overlap / shorter_duration) >= BASIC_PITCH_OVERLAP_RATIO_THRESHOLD


def _note_priority(note: Note) -> tuple[float, float, int]:
    return (note.duration, note.velocity, -abs(note.midi_note - 60))


def _note_overlap_seconds(first_note: Note, second_note: Note) -> float:
    start_time = max(first_note.startTime, second_note.startTime)
    end_time = min(
        first_note.startTime + first_note.duration,
        second_note.startTime + second_note.duration,
    )
    return max(0.0, end_time - start_time)


def _filter_pitch_transients(notes: list[Note]) -> list[Note]:
    if len(notes) < 3:
        return notes

    filtered_notes: list[Note] = []
    for index, note in enumerate(notes):
        if index == 0 or index == len(notes) - 1:
            filtered_notes.append(note)
            continue

        previous_note = notes[index - 1]
        next_note = notes[index + 1]
        if _is_pitch_transient(previous_note, note, next_note):
            continue

        filtered_notes.append(note)

    return filtered_notes


def _is_pitch_transient(previous_note: Note, note: Note, next_note: Note) -> bool:
    if note.duration > BASIC_PITCH_CHROMATIC_TRANSIENT_MAX_SECONDS:
        return False
    if note.midi_note == previous_note.midi_note or note.midi_note == next_note.midi_note:
        return False

    previous_interval = note.midi_note - previous_note.midi_note
    next_interval = next_note.midi_note - note.midi_note
    if _is_short_chromatic_neighbor(note, next_interval):
        return True

    return (
        abs(previous_interval) == 1
        and abs(next_interval) >= 2
        and previous_note.duration >= note.duration
        and next_note.duration >= note.duration
    )


def _is_short_chromatic_neighbor(note: Note, next_interval: int) -> bool:
    return (
        note.duration <= BASIC_PITCH_SHORT_NEIGHBOR_TRANSIENT_MAX_SECONDS
        and abs(next_interval) == 1
    )


def _log_note_cleanup_metrics(audio_id: str, cleanup_result: NoteCleanupResult) -> None:
    print(
        "note_cleanup_metrics "
        f"audio_id={audio_id} "
        f"raw_note_count={cleanup_result.raw_note_count} "
        f"cleaned_note_count={cleanup_result.cleaned_note_count} "
        f"dropped_short_note_count={cleanup_result.dropped_short_note_count} "
        f"merged_note_count={cleanup_result.merged_note_count} "
        f"overlap_fix_count={cleanup_result.overlap_fix_count}",
        flush=True,
    )


def _log_note_pipeline_counts(
    audio_id: str,
    pitch_frame_count: int,
    raw_note_count: int,
    cleaned_note_count: int,
    adjusted_note_count: int,
    quantized_note_count: int,
    dropped_note_count: int,
    merged_note_count: int,
) -> None:
    print(
        "note_pipeline_counts "
        f"audio_id={audio_id} "
        f"pitch_frame_count={pitch_frame_count} "
        f"raw_note_count={raw_note_count} "
        f"cleaned_note_count={cleaned_note_count} "
        f"adjusted_note_count={adjusted_note_count} "
        f"quantized_note_count={quantized_note_count} "
        f"dropped_note_count={dropped_note_count} "
        f"merged_note_count={merged_note_count}",
        flush=True,
    )


def _log_quantized_duration_stats(audio_id: str, notes: list[Note]) -> None:
    durations = [note.duration for note in notes]
    if not durations:
        return

    print(
        "quantized_duration_stats "
        f"audio_id={audio_id} "
        f"min_duration={min(durations):.3f} "
        f"max_duration={max(durations):.3f} "
        f"avg_duration={(sum(durations) / len(durations)):.3f}",
        flush=True,
    )


def _log_quantization_metrics(audio_id: str, quantize_result: QuantizeResult) -> None:
    print(
        "quantization_metrics "
        f"audio_id={audio_id} "
        f"before_note_count={quantize_result.before_note_count} "
        f"after_note_count={quantize_result.after_note_count} "
        f"audible_note_count={quantize_result.audible_note_count} "
        f"too_short_after_quantization_count={quantize_result.too_short_after_quantization_count} "
        f"min_duration_after_quantization={quantize_result.min_duration_after_quantization:.3f} "
        f"collision_count={quantize_result.collision_count} "
        f"shifted_note_count={quantize_result.shifted_note_count} "
        f"overlap_fix_count={quantize_result.overlap_fix_count}",
        flush=True,
    )


def _log_midi_event_metrics(audio_id: str, metrics: MidiEventMetrics) -> None:
    print(
        "midi_event_metrics "
        f"audio_id={audio_id} "
        f"midi_event_count={metrics.midi_event_count} "
        f"zero_or_negative_delta_count={metrics.zero_or_negative_delta_count}",
        flush=True,
    )


def _smooth_isolated_pitch_spikes(notes: list[Note]) -> list[Note]:
    if len(notes) < 3:
        return notes

    smoothed_notes = list(notes)
    for index in range(1, len(notes) - 1):
        previous_note = notes[index - 1]
        note = notes[index]
        next_note = notes[index + 1]

        if note.duration > NOTE_CLEANUP_SPIKE_MAX_DURATION_SECONDS:
            continue
        if abs(previous_note.midi_note - next_note.midi_note) > NOTE_MERGE_TOLERANCE_SEMITONES:
            continue
        if abs(note.midi_note - previous_note.midi_note) <= NOTE_MERGE_TOLERANCE_SEMITONES:
            continue
        if abs(note.midi_note - next_note.midi_note) <= NOTE_MERGE_TOLERANCE_SEMITONES:
            continue

        target_midi_note = round((previous_note.midi_note + next_note.midi_note) / 2)
        smoothed_notes[index] = _copy_note_with_midi(note, target_midi_note)

    return smoothed_notes


def _merge_adjacent_similar_notes(notes: list[Note]) -> tuple[list[Note], int]:
    merged_notes: list[Note] = []
    merged_note_count = 0

    for note in notes:
        _validate_note_timing(note)
        if not merged_notes:
            merged_notes.append(note)
            continue

        previous_note = merged_notes[-1]
        gap_seconds = note.startTime - (previous_note.startTime + previous_note.duration)
        if not _should_merge_adjacent_notes(previous_note, note, gap_seconds):
            merged_notes.append(note)
            continue

        merged_notes[-1] = _merge_notes(previous_note, note)
        merged_note_count += 1

    return merged_notes, merged_note_count


def _should_merge_adjacent_notes(
    previous_note: Note,
    note: Note,
    gap_seconds: float,
) -> bool:
    if previous_note.midi_note != note.midi_note:
        return False
    if _looks_like_repeated_articulation(previous_note, note):
        return False
    if gap_seconds < 0:
        return True
    if gap_seconds <= NOTE_CLEANUP_MERGE_GAP_SECONDS:
        return True

    return (
        gap_seconds <= BASIC_PITCH_FRAGMENT_MERGE_GAP_SECONDS
        and min(previous_note.duration, note.duration) <= BASIC_PITCH_FRAGMENT_MAX_SECONDS
    )


def _looks_like_repeated_articulation(previous_note: Note, note: Note) -> bool:
    onset_gap = note.startTime - previous_note.startTime
    if onset_gap < BASIC_PITCH_REPEATED_ONSET_MIN_GAP_SECONDS:
        return False
    return (
        previous_note.duration >= BASIC_PITCH_REPEATED_NOTE_MIN_SECONDS
        and note.duration >= BASIC_PITCH_REPEATED_NOTE_MIN_SECONDS
    )


def _merge_notes(first_note: Note, second_note: Note) -> Note:
    start_time = min(first_note.startTime, second_note.startTime)
    end_time = max(
        first_note.startTime + first_note.duration,
        second_note.startTime + second_note.duration,
    )
    total_duration = first_note.duration + second_note.duration
    weighted_midi_note = round(
        (
            (first_note.midi_note * first_note.duration)
            + (second_note.midi_note * second_note.duration)
        )
        / total_duration
    )

    return Note(
        pitch=_midi_note_to_name(weighted_midi_note),
        midi_note=weighted_midi_note,
        startTime=float(start_time),
        duration=float(end_time - start_time),
        velocity=max(first_note.velocity, second_note.velocity),
    )


def _drop_short_notes(notes: list[Note]) -> list[Note]:
    return [
        note
        for note in notes
        if note.duration >= NOTE_CLEANUP_MIN_DURATION_SECONDS
    ]


def _remove_note_overlaps(notes: list[Note]) -> tuple[list[Note], int]:
    normalized_notes: list[Note] = []
    overlap_fix_count = 0

    for note in notes:
        if not normalized_notes:
            normalized_notes.append(note)
            continue

        previous_note = normalized_notes[-1]
        previous_end_time = previous_note.startTime + previous_note.duration
        if note.startTime >= previous_end_time:
            normalized_notes.append(note)
            continue

        overlap_fix_count += 1
        if _should_drop_overlapping_note(previous_note, note):
            continue

        trimmed_previous_duration = note.startTime - previous_note.startTime
        if trimmed_previous_duration >= NOTE_CLEANUP_MIN_DURATION_SECONDS:
            normalized_notes[-1] = Note(
                pitch=previous_note.pitch,
                midi_note=previous_note.midi_note,
                startTime=previous_note.startTime,
                duration=float(trimmed_previous_duration),
                velocity=previous_note.velocity,
            )
            normalized_notes.append(note)
            continue

        normalized_notes[-1] = note

    return normalized_notes, overlap_fix_count


def _should_drop_overlapping_note(previous_note: Note, note: Note) -> bool:
    previous_end_time = previous_note.startTime + previous_note.duration
    note_end_time = note.startTime + note.duration
    if note_end_time > previous_end_time:
        return False

    overlap = _note_overlap_seconds(previous_note, note)
    if overlap <= 0:
        return False

    overlap_ratio = overlap / note.duration
    return (
        overlap_ratio >= BASIC_PITCH_OVERLAP_RATIO_THRESHOLD
        and _note_priority(previous_note) >= _note_priority(note)
    )


def _copy_note_with_midi(note: Note, midi_note: int) -> Note:
    return Note(
        pitch=_midi_note_to_name(midi_note),
        midi_note=midi_note,
        startTime=note.startTime,
        duration=note.duration,
        velocity=note.velocity,
    )


def _frequency_to_midi_note(frequency_hz: float) -> int:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise AudioProcessingError("Invalid pitch frequency")

    return int(round(float(librosa.hz_to_midi(frequency_hz))))


def _midi_note_to_name(midi_note: int) -> str:
    note_name = MIDI_NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1
    return f"{note_name}{octave}"


def _fit_scale(notes: list[Note]) -> ScaleFit:
    if not notes:
        raise AudioProcessingError("No notes available for scale fitting")

    candidates: list[tuple[float, int, int, int, ScaleFit]] = []
    note_pitch_classes = {note.midi_note % 12 for note in notes}

    for root in range(12):
        for mode_name, intervals in (
            ("MAJOR", MAJOR_SCALE_INTERVALS),
            ("MINOR", MINOR_SCALE_INTERVALS),
        ):
            scale_pitch_classes = frozenset((root + interval) % 12 for interval in intervals)
            distance_score = sum(
                _pitch_class_distance(note.midi_note % 12, scale_pitch_classes)
                for note in notes
            )
            included_count = sum(
                1 for pitch_class in note_pitch_classes if pitch_class in scale_pitch_classes
            )
            tonic_present = 1 if root in note_pitch_classes else 0
            anchor_score = _scale_anchor_score(notes, root, mode_name)
            confidence = _scale_confidence(notes, distance_score)
            scale_fit = ScaleFit(
                name=f"{MIDI_NOTE_NAMES[root]}_{mode_name}",
                root_pitch_class=root,
                scale_pitch_classes=scale_pitch_classes,
                confidence=confidence,
            )
            candidates.append((
                distance_score,
                -included_count,
                -anchor_score,
                -tonic_present,
                scale_fit,
            ))

    if not candidates:
        raise AudioProcessingError("Unable to fit scale")

    return min(candidates, key=lambda candidate: candidate[:4])[4]


def _scale_anchor_score(notes: list[Note], root: int, mode_name: str) -> int:
    first_degree = (notes[0].midi_note - root) % 12
    last_degree = (notes[-1].midi_note - root) % 12
    lowest_degree = (min(notes, key=lambda note: note.midi_note).midi_note - root) % 12
    mode_third = 4 if mode_name == "MAJOR" else 3

    return (
        _terminal_scale_degree_score(first_degree, mode_third)
        + _terminal_scale_degree_score(last_degree, mode_third)
        + (2 if lowest_degree == 0 else 0)
    )


def _terminal_scale_degree_score(degree: int, mode_third: int) -> int:
    if degree == 0:
        return 4
    if degree == 7:
        return 3
    if degree == mode_third:
        return 2
    if degree == 5:
        return 1
    return 0


def _scale_confidence(notes: list[Note], distance_score: float) -> float:
    max_distance = max(len(notes), 1) * 6.0
    confidence = 1.0 - (float(distance_score) / max_distance)
    return max(0.0, min(1.0, confidence))


def _pitch_class_distance(pitch_class: int, scale_pitch_classes: frozenset[int]) -> int:
    return min(
        min((pitch_class - scale_pitch_class) % 12, (scale_pitch_class - pitch_class) % 12)
        for scale_pitch_class in scale_pitch_classes
    )


def _adjust_notes_to_scale(notes: list[Note], scale_fit: ScaleFit) -> list[Note]:
    if not notes:
        raise AudioProcessingError("No notes available for scale adjustment")

    adjusted_notes = [
        _adjust_note_to_scale(note, scale_fit.scale_pitch_classes)
        for note in notes
    ]
    return _stabilize_scale_adjusted_notes(adjusted_notes)


def _stabilize_scale_adjusted_notes(notes: list[Note]) -> list[Note]:
    merged_notes, _ = _merge_adjacent_similar_notes(notes)
    normalized_notes, _ = _remove_note_overlaps(merged_notes)
    if not normalized_notes:
        raise AudioProcessingError("No valid notes found after scale adjustment")
    return normalized_notes


def _adjust_note_to_scale(note: Note, scale_pitch_classes: frozenset[int]) -> Note:
    if note.midi_note % 12 in scale_pitch_classes:
        return note

    adjusted_midi_note = _nearest_scale_midi_note(note.midi_note, scale_pitch_classes)
    return Note(
        pitch=_midi_note_to_name(adjusted_midi_note),
        midi_note=adjusted_midi_note,
        startTime=note.startTime,
        duration=note.duration,
        velocity=note.velocity,
    )


def _nearest_scale_midi_note(midi_note: int, scale_pitch_classes: frozenset[int]) -> int:
    search_range = range(midi_note - 6, midi_note + 7)
    candidates = [
        candidate for candidate in search_range if candidate % 12 in scale_pitch_classes
    ]
    if not candidates:
        raise AudioProcessingError("Unable to adjust note to scale")

    return min(candidates, key=lambda candidate: (abs(candidate - midi_note), candidate))


def _quantize_notes(notes: list[Note]) -> list[Note]:
    return _quantize_notes_with_metrics(notes).notes


def _quantize_notes_with_metrics(notes: list[Note]) -> QuantizeResult:
    if not notes:
        raise AudioProcessingError("No notes available for quantization")

    ordered_notes = sorted(notes, key=lambda note: (note.startTime, note.midi_note))
    quantized_starts: list[float] = []
    collision_count = 0
    shifted_note_count = 0

    for note in ordered_notes:
        _validate_note_timing(note)
        candidate_start_time = _soft_quantized_start_time(note.startTime)
        start_time = candidate_start_time
        if quantized_starts and start_time < quantized_starts[-1] + AUDIBLE_MIN_NOTE_SECONDS:
            collision_count += 1
            start_time = quantized_starts[-1] + AUDIBLE_MIN_NOTE_SECONDS

        if start_time != candidate_start_time:
            shifted_note_count += 1

        quantized_starts.append(start_time)

    quantized_notes: list[Note] = []
    overlap_fix_count = 0
    for index, note in enumerate(ordered_notes):
        duration = _quantized_note_duration(note)
        if index + 1 < len(quantized_starts):
            available_duration = quantized_starts[index + 1] - quantized_starts[index]
            if duration > available_duration:
                duration = available_duration
                overlap_fix_count += 1
        duration = max(AUDIBLE_MIN_NOTE_SECONDS, duration)

        quantized_notes.append(
            Note(
                pitch=note.pitch,
                midi_note=note.midi_note,
                startTime=float(quantized_starts[index]),
                duration=float(duration),
                velocity=note.velocity,
            )
        )

    if not quantized_notes:
        raise AudioProcessingError("Unable to quantize notes")

    too_short_after_quantization_count = sum(
        1 for note in quantized_notes if note.duration < AUDIBLE_MIN_NOTE_SECONDS
    )
    min_duration_after_quantization = min(note.duration for note in quantized_notes)

    return QuantizeResult(
        notes=quantized_notes,
        before_note_count=len(notes),
        after_note_count=len(quantized_notes),
        audible_note_count=len(quantized_notes) - too_short_after_quantization_count,
        too_short_after_quantization_count=too_short_after_quantization_count,
        min_duration_after_quantization=float(min_duration_after_quantization),
        collision_count=collision_count,
        shifted_note_count=shifted_note_count,
        overlap_fix_count=overlap_fix_count,
    )


def _quantized_note_duration(note: Note) -> float:
    rhythm_bucket = _nearest_rhythm_bucket(note.duration)
    if abs(rhythm_bucket - note.duration) <= QUANTIZE_DURATION_SNAP_TOLERANCE_SECONDS:
        return rhythm_bucket
    return note.duration


def _soft_quantized_start_time(start_time: float) -> float:
    grid_start_time = _nearest_grid_time(start_time)
    if abs(grid_start_time - start_time) <= QUANTIZE_ONSET_SNAP_TOLERANCE_SECONDS:
        return grid_start_time
    return start_time


def _nearest_rhythm_bucket(duration_seconds: float) -> float:
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise AudioProcessingError("Invalid note duration")

    return min(
        RHYTHM_BUCKET_SECONDS,
        key=lambda bucket: (abs(bucket - duration_seconds), bucket),
    )


def _nearest_grid_time(value_seconds: float) -> float:
    return round(value_seconds / QUANTIZE_GRID_SECONDS) * QUANTIZE_GRID_SECONDS


def _validate_note_timing(note: Note) -> None:
    if not np.isfinite(note.startTime) or note.startTime < 0:
        raise AudioProcessingError("Invalid note start time")
    if not np.isfinite(note.duration) or note.duration <= 0:
        raise AudioProcessingError("Invalid note duration")


def _infer_chords(notes: list[Note], scale_fit: ScaleFit) -> list[Chord]:
    if not notes:
        raise AudioProcessingError("No notes available for chord inference")

    mode = _scale_mode(scale_fit.name)
    scale_intervals = MAJOR_SCALE_INTERVALS if mode == "MAJOR" else MINOR_SCALE_INTERVALS
    candidates = _diatonic_triad_candidates(scale_fit.root_pitch_class, scale_intervals)
    if not candidates:
        raise AudioProcessingError("No chord candidates available")

    melody_start_time = min(note.startTime for note in notes)
    melody_end_time = max(note.startTime + note.duration for note in notes)
    melody_duration = melody_end_time - melody_start_time
    window_count = _chord_window_count(melody_duration)
    window_duration = melody_duration / window_count
    chords: list[Chord] = []

    for window_index in range(window_count):
        start_time = melody_start_time + (window_index * window_duration)
        end_time = (
            melody_end_time
            if window_index == window_count - 1
            else start_time + window_duration
        )
        window_notes = [
            note for note in notes if _note_overlaps_window(note, start_time, end_time)
        ]
        if not window_notes:
            continue

        candidate = _select_chord_candidate(
            window_notes,
            candidates,
            mode,
            start_time,
            end_time,
            window_index == 0,
        )
        chords.append(
            Chord(
                root=MIDI_NOTE_NAMES[candidate["root_pitch_class"]],
                type=candidate["type"],
                startTime=float(start_time),
                duration=float(end_time - start_time),
            )
        )

    if not chords:
        raise AudioProcessingError("Unable to infer chords")

    return _normalize_chord_timing(chords, notes)


def _chord_window_count(melody_duration: float) -> int:
    if melody_duration <= 0:
        return 1
    if melody_duration <= CHORD_WINDOW_SECONDS:
        return 1
    if melody_duration <= CHORD_SHORT_MELODY_MAX_SECONDS:
        return min(
            CHORD_SHORT_MELODY_MAX_WINDOWS,
            max(2, int(np.ceil(melody_duration / CHORD_TARGET_SECTION_SECONDS))),
        )
    return max(1, int(np.ceil(melody_duration / CHORD_WINDOW_SECONDS)))


def _normalize_chord_timing(chords: list[Chord], notes: list[Note]) -> list[Chord]:
    if not chords:
        return []

    ordered_chords = sorted(chords, key=lambda chord: chord.startTime)
    melody_end_time = max(note.startTime + note.duration for note in notes)
    boundaries = [ordered_chords[0].startTime]

    for chord_index in range(1, len(ordered_chords)):
        raw_boundary = ordered_chords[chord_index].startTime
        previous_boundary = boundaries[-1]
        next_boundary = (
            ordered_chords[chord_index + 1].startTime
            if chord_index + 1 < len(ordered_chords)
            else melody_end_time
        )
        boundaries.append(
            _normalize_chord_boundary(
                raw_boundary,
                previous_boundary,
                next_boundary,
                notes,
                chord_index == len(ordered_chords) - 1,
            )
        )

    final_end_time = melody_end_time + CHORD_FINAL_SUSTAIN_SECONDS
    normalized_chords: list[Chord] = []
    for chord_index, chord in enumerate(ordered_chords):
        start_time = boundaries[chord_index]
        end_time = (
            boundaries[chord_index + 1]
            if chord_index + 1 < len(boundaries)
            else final_end_time
        )
        if end_time <= start_time:
            end_time = start_time + CHORD_MIN_SECTION_SECONDS
        normalized_chords.append(
            Chord(
                root=chord.root,
                type=chord.type,
                startTime=float(start_time),
                duration=float(end_time - start_time),
            )
        )

    return normalized_chords


def _normalize_chord_boundary(
    raw_boundary: float,
    previous_boundary: float,
    next_boundary: float,
    notes: list[Note],
    prefer_final_phrase: bool = False,
) -> float:
    boundary_epsilon = 1e-6
    earliest_boundary = previous_boundary + CHORD_MIN_SECTION_SECONDS
    latest_boundary = next_boundary - CHORD_MIN_SECTION_SECONDS
    if latest_boundary + boundary_epsilon < earliest_boundary:
        return raw_boundary

    boundary_start = raw_boundary - CHORD_BOUNDARY_SNAP_TOLERANCE_SECONDS
    boundary_end = raw_boundary + (
        CHORD_FINAL_BOUNDARY_LOOKAHEAD_SECONDS
        if prefer_final_phrase
        else CHORD_BOUNDARY_SNAP_TOLERANCE_SECONDS
    )
    onset_candidates = [
        note.startTime
        for note in notes
        if (
            boundary_start - boundary_epsilon
            <= note.startTime
            <= boundary_end + boundary_epsilon
        )
        and (
            earliest_boundary - boundary_epsilon
            <= note.startTime
            <= latest_boundary + boundary_epsilon
        )
    ]
    if not onset_candidates:
        return raw_boundary

    if prefer_final_phrase:
        final_phrase_candidates = [
            onset
            for onset in _unique_times(onset_candidates)
            if onset >= raw_boundary
            and _starts_closing_repeated_phrase(notes, onset)
        ]
        if final_phrase_candidates:
            return min(final_phrase_candidates)

    return min(
        _unique_times(onset_candidates),
        key=lambda onset: (
            abs(onset - raw_boundary),
            _active_note_count_at_boundary(notes, onset),
            onset,
        ),
    )


def _unique_times(values: list[float]) -> list[float]:
    unique_values: list[float] = []
    for value in sorted(values):
        if not unique_values or not np.isclose(value, unique_values[-1], atol=1e-6):
            unique_values.append(value)
    return unique_values


def _active_note_count_at_boundary(notes: list[Note], boundary_time: float) -> int:
    return sum(
        1
        for note in notes
        if note.startTime < boundary_time < note.startTime + note.duration
    )


def _starts_closing_repeated_phrase(notes: list[Note], onset_time: float) -> bool:
    phrase_note = _note_starting_at(notes, onset_time)
    if phrase_note is None or phrase_note.duration < AUDIBLE_MIN_NOTE_SECONDS:
        return False

    repeated_notes = [
        note
        for note in notes
        if note.startTime >= phrase_note.startTime
        and note.midi_note == phrase_note.midi_note
        and note.duration >= AUDIBLE_MIN_NOTE_SECONDS
    ]
    return len(repeated_notes) >= 2


def _note_starting_at(notes: list[Note], onset_time: float) -> Note | None:
    starting_notes = [
        note
        for note in notes
        if np.isclose(note.startTime, onset_time, atol=1e-6)
    ]
    if not starting_notes:
        return None
    return max(starting_notes, key=lambda note: (note.duration, note.velocity))


def _scale_mode(scale_name: str) -> str:
    if scale_name.endswith("_MAJOR"):
        return "MAJOR"
    if scale_name.endswith("_MINOR"):
        return "MINOR"
    raise AudioProcessingError(f"Unsupported scale name: {scale_name}")


def _diatonic_triad_candidates(
    root_pitch_class: int,
    scale_intervals: tuple[int, ...],
) -> list[dict[str, object]]:
    scale_pitch_classes = [(root_pitch_class + interval) % 12 for interval in scale_intervals]
    candidates: list[dict[str, object]] = []

    for degree in range(len(scale_pitch_classes)):
        triad = frozenset(
            (
                scale_pitch_classes[degree],
                scale_pitch_classes[(degree + 2) % len(scale_pitch_classes)],
                scale_pitch_classes[(degree + 4) % len(scale_pitch_classes)],
            )
        )
        candidates.append(
            {
                "degree": degree,
                "root_pitch_class": scale_pitch_classes[degree],
                "type": _triad_type(triad, scale_pitch_classes[degree]),
                "pitch_classes": triad,
            }
        )

    return candidates


def _triad_type(chord_pitch_classes: frozenset[int], root_pitch_class: int) -> str:
    intervals = sorted((pitch_class - root_pitch_class) % 12 for pitch_class in chord_pitch_classes)
    if intervals == [0, 4, 7]:
        return "MAJOR"
    if intervals == [0, 3, 7]:
        return "MINOR"
    if intervals == [0, 3, 6]:
        return "DIMINISHED"
    return "UNKNOWN"


def _select_chord_candidate(
    notes: list[Note],
    candidates: list[dict[str, object]],
    mode: str,
    window_start_time: float,
    window_end_time: float,
    prefer_tonic: bool,
) -> dict[str, object]:
    tie_break_degrees = (
        MAJOR_CHORD_TIEBREAK_DEGREES if mode == "MAJOR" else MINOR_CHORD_TIEBREAK_DEGREES
    )
    degree_priority = {
        degree: priority for priority, degree in enumerate(tie_break_degrees)
    }

    return max(
        candidates,
        key=lambda candidate: (
            _chord_score(notes, candidate, window_start_time, window_end_time)
            + _initial_tonic_bonus(candidate, prefer_tonic),
            -degree_priority.get(candidate["degree"], len(tie_break_degrees)),
        ),
    )


def _initial_tonic_bonus(candidate: dict[str, object], prefer_tonic: bool) -> float:
    if prefer_tonic and candidate["degree"] == 0:
        return CHORD_INITIAL_TONIC_BONUS
    return 0.0


def _chord_score(
    notes: list[Note],
    candidate: dict[str, object],
    window_start_time: float,
    window_end_time: float,
) -> float:
    chord_pitch_classes = candidate["pitch_classes"]
    root_pitch_class = candidate["root_pitch_class"]
    if not isinstance(chord_pitch_classes, frozenset):
        raise AudioProcessingError("Invalid chord candidate")
    if not isinstance(root_pitch_class, int):
        raise AudioProcessingError("Invalid chord candidate")

    score = sum(
        _note_window_overlap_duration(note, window_start_time, window_end_time)
        for note in notes
        if note.midi_note % 12 in chord_pitch_classes
    )

    anchor_note = _window_anchor_note(notes, window_start_time)
    if anchor_note.midi_note % 12 == root_pitch_class:
        score += CHORD_WINDOW_ROOT_BONUS

    return score


def _window_anchor_note(notes: list[Note], window_start_time: float) -> Note:
    onset_notes = [
        note
        for note in notes
        if note.startTime >= window_start_time
    ]
    if onset_notes:
        return min(onset_notes, key=lambda note: note.startTime)
    return min(notes, key=lambda note: note.startTime)


def _note_window_overlap_duration(
    note: Note,
    window_start_time: float,
    window_end_time: float,
) -> float:
    start_time = max(note.startTime, window_start_time)
    end_time = min(note.startTime + note.duration, window_end_time)
    return max(0.0, end_time - start_time)


def _note_overlaps_window(note: Note, start_time: float, end_time: float) -> bool:
    note_start = note.startTime
    note_end = note.startTime + note.duration
    return note_start < end_time and note_end > start_time


def _calculate_melody_quality_analysis(
    cleaned_notes: list[Note],
    quantized_notes: list[Note],
    scale_fit: ScaleFit,
    chords: list[Chord],
) -> MelodyQualityAnalysis:
    if not quantized_notes:
        raise AudioProcessingError("No notes available for melody quality analysis")

    ordered_notes = sorted(quantized_notes, key=lambda note: (note.startTime, note.midi_note))
    intervals = [
        ordered_notes[index + 1].midi_note - ordered_notes[index].midi_note
        for index in range(len(ordered_notes) - 1)
    ]
    absolute_intervals = [abs(interval) for interval in intervals]
    durations = [note.duration for note in ordered_notes]
    melody_start_time = min(note.startTime for note in ordered_notes)
    melody_end_time = max(note.startTime + note.duration for note in ordered_notes)
    melody_duration = max(melody_end_time - melody_start_time, AUDIBLE_MIN_NOTE_SECONDS)

    off_grid_note_count = _off_grid_note_count(ordered_notes)
    large_interval_jumps = _large_interval_jumps(ordered_notes)
    repeated_motifs = _repeated_motifs(ordered_notes)
    chord_tone_matched_notes = _chord_tone_matched_note_count(ordered_notes, chords)
    scale_adjusted_note_count = _scale_adjusted_note_count(cleaned_notes, scale_fit)

    interval_variance = float(np.var(intervals)) if intervals else 0.0
    mean_interval = (
        sum(absolute_intervals) / len(absolute_intervals)
        if absolute_intervals
        else 0.0
    )
    mean_duration = sum(durations) / len(durations)
    duration_std = float(np.std(durations)) if len(durations) > 1 else 0.0
    rhythm_variation = duration_std / mean_duration if mean_duration > 0 else 0.0

    metrics = {
        "pitchStability": _round_metric(1.0 - min(mean_interval, 12.0) / 12.0),
        "rhythmConsistency": _round_metric(1.0 - min(rhythm_variation, 1.0)),
        "noteDensity": _round_metric(len(ordered_notes) / melody_duration),
        "intervalVariance": _round_metric(interval_variance),
        "repetitionScore": _round_metric(_repetition_score(ordered_notes, repeated_motifs)),
        "chordToneAlignment": _round_metric(chord_tone_matched_notes / len(ordered_notes)),
    }
    evidence = {
        "offGridNoteCount": off_grid_note_count,
        "largeIntervalJumps": large_interval_jumps,
        "repeatedMotifs": repeated_motifs,
        "chordToneMatchedNotes": chord_tone_matched_notes,
        "scaleAdjustedNoteCount": scale_adjusted_note_count,
    }
    return MelodyQualityAnalysis(metrics=metrics, evidence=evidence)


def _round_metric(value: float) -> float:
    return round(float(max(0.0, value)), 4)


def _off_grid_note_count(notes: list[Note]) -> int:
    return sum(
        1
        for note in notes
        if not np.isclose(note.startTime, _nearest_grid_time(note.startTime), atol=1e-6)
    )


def _large_interval_jumps(notes: list[Note]) -> list[dict[str, object]]:
    jumps: list[dict[str, object]] = []
    for index in range(len(notes) - 1):
        interval = notes[index + 1].midi_note - notes[index].midi_note
        if abs(interval) <= 7:
            continue
        jumps.append(
            {
                "fromIndex": index,
                "toIndex": index + 1,
                "fromPitch": notes[index].pitch,
                "toPitch": notes[index + 1].pitch,
                "semitones": interval,
            }
        )
    return jumps


def _repeated_motifs(notes: list[Note]) -> list[dict[str, object]]:
    motif_positions: dict[tuple[int, ...], list[int]] = {}
    for motif_length in (3, 2):
        if len(notes) < motif_length * 2:
            continue
        for index in range(len(notes) - motif_length + 1):
            motif = tuple(note.midi_note for note in notes[index:index + motif_length])
            motif_positions.setdefault(motif, []).append(index)

    repeated: list[dict[str, object]] = []
    for motif, positions in sorted(
        motif_positions.items(),
        key=lambda item: (-len(item[1]), len(item[0]), item[1][0], item[0]),
    ):
        unique_positions = _non_overlapping_positions(positions, len(motif))
        if len(unique_positions) < 2:
            continue
        repeated.append(
            {
                "pitches": [_midi_note_to_name(midi_note) for midi_note in motif],
                "startIndexes": unique_positions,
                "occurrences": len(unique_positions),
            }
        )
        if len(repeated) >= 5:
            break
    return repeated


def _non_overlapping_positions(positions: list[int], motif_length: int) -> list[int]:
    selected_positions: list[int] = []
    for position in positions:
        if selected_positions and position < selected_positions[-1] + motif_length:
            continue
        selected_positions.append(position)
    return selected_positions


def _repetition_score(notes: list[Note], repeated_motifs: list[dict[str, object]]) -> float:
    repeated_indexes: set[int] = set()
    for motif in repeated_motifs:
        pitches = motif["pitches"]
        start_indexes = motif["startIndexes"]
        if not isinstance(pitches, list) or not isinstance(start_indexes, list):
            continue
        for start_index in start_indexes:
            if not isinstance(start_index, int):
                continue
            repeated_indexes.update(range(start_index, start_index + len(pitches)))
    return len(repeated_indexes) / len(notes) if notes else 0.0


def _chord_tone_matched_note_count(notes: list[Note], chords: list[Chord]) -> int:
    matched_count = 0
    for note in notes:
        if any(
            _note_overlaps_window(note, chord.startTime, chord.startTime + chord.duration)
            and note.midi_note % 12 in _chord_pitch_classes(chord)
            for chord in chords
        ):
            matched_count += 1
    return matched_count


def _chord_pitch_classes(chord: Chord) -> frozenset[int]:
    return frozenset(midi_note % 12 for midi_note in _chord_to_midi_notes(chord))


def _scale_adjusted_note_count(cleaned_notes: list[Note], scale_fit: ScaleFit) -> int:
    return sum(
        1
        for note in cleaned_notes
        if note.midi_note % 12 not in scale_fit.scale_pitch_classes
    )


def _write_midi_file(
    audio_id: str,
    output_directory: Path,
    melody_notes: list[Note],
    chords: list[Chord],
) -> Path:
    if not audio_id or not audio_id.strip():
        raise AudioProcessingError("audioId is required for MIDI generation")
    if not melody_notes:
        raise AudioProcessingError("No melody notes available for MIDI generation")
    if not chords:
        raise AudioProcessingError("No chords available for MIDI generation")

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to create output directory: {output_directory}") from exc

    midi_path = output_directory / f"{audio_id}.mid"
    midi_file = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    tempo = mido.bpm2tempo(FALLBACK_TEMPO_BPM)

    melody_track = mido.MidiTrack()
    chord_track = mido.MidiTrack()
    midi_file.tracks.append(melody_track)
    midi_file.tracks.append(chord_track)

    melody_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    melody_track.append(
        mido.Message(
            "program_change",
            program=MIDI_PIANO_PROGRAM,
            channel=MIDI_MELODY_CHANNEL,
            time=0,
        )
    )
    chord_track.append(
        mido.Message(
            "program_change",
            program=MIDI_PIANO_PROGRAM,
            channel=MIDI_CHORD_CHANNEL,
            time=0,
        )
    )

    melody_event_metrics = _append_note_events(melody_track, melody_notes, MIDI_MELODY_CHANNEL)
    _log_midi_event_metrics(audio_id, melody_event_metrics)
    if ENABLE_ACCOMPANIMENT:
        _append_chord_events(chord_track, chords)

    try:
        midi_file.save(midi_path)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to write MIDI file: {midi_path}") from exc

    return midi_path


def _write_debug_melody_midi_file(
    audio_id: str,
    output_directory: Path,
    suffix: str,
    melody_notes: list[Note],
) -> Path:
    if not audio_id or not audio_id.strip():
        raise AudioProcessingError("audioId is required for MIDI generation")
    if not suffix or not suffix.strip():
        raise AudioProcessingError("MIDI suffix is required")
    if not melody_notes:
        raise AudioProcessingError("No melody notes available for MIDI generation")

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to create output directory: {output_directory}") from exc

    midi_path = output_directory / f"{audio_id}-{suffix}.mid"
    midi_file = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    tempo = mido.bpm2tempo(FALLBACK_TEMPO_BPM)
    melody_track = mido.MidiTrack()
    midi_file.tracks.append(melody_track)
    melody_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    melody_track.append(
        mido.Message(
            "program_change",
            program=MIDI_PIANO_PROGRAM,
            channel=MIDI_MELODY_CHANNEL,
            time=0,
        )
    )
    _append_note_events(melody_track, melody_notes, MIDI_MELODY_CHANNEL)

    try:
        midi_file.save(midi_path)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to write MIDI file: {midi_path}") from exc

    return midi_path


def _write_debug_chord_midi_file(
    audio_id: str,
    output_directory: Path,
    suffix: str,
    chords: list[Chord],
) -> Path:
    if not audio_id or not audio_id.strip():
        raise AudioProcessingError("audioId is required for MIDI generation")
    if not suffix or not suffix.strip():
        raise AudioProcessingError("MIDI suffix is required")
    if not chords:
        raise AudioProcessingError("No chords available for MIDI generation")

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to create output directory: {output_directory}") from exc

    midi_path = output_directory / f"{audio_id}-{suffix}.mid"
    midi_file = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    tempo = mido.bpm2tempo(FALLBACK_TEMPO_BPM)
    chord_track = mido.MidiTrack()
    midi_file.tracks.append(chord_track)
    chord_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    chord_track.append(
        mido.Message(
            "program_change",
            program=MIDI_PIANO_PROGRAM,
            channel=MIDI_CHORD_CHANNEL,
            time=0,
        )
    )
    _append_chord_events(chord_track, chords)

    try:
        midi_file.save(midi_path)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to write MIDI file: {midi_path}") from exc

    return midi_path


def _write_debug_combined_midi_file(
    audio_id: str,
    output_directory: Path,
    suffix: str,
    melody_notes: list[Note],
    chords: list[Chord],
) -> Path:
    if not audio_id or not audio_id.strip():
        raise AudioProcessingError("audioId is required for MIDI generation")
    if not suffix or not suffix.strip():
        raise AudioProcessingError("MIDI suffix is required")
    if not melody_notes:
        raise AudioProcessingError("No melody notes available for MIDI generation")
    if not chords:
        raise AudioProcessingError("No chords available for MIDI generation")

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to create output directory: {output_directory}") from exc

    midi_path = output_directory / f"{audio_id}-{suffix}.mid"
    midi_file = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    tempo = mido.bpm2tempo(FALLBACK_TEMPO_BPM)

    melody_track = mido.MidiTrack()
    chord_track = mido.MidiTrack()
    midi_file.tracks.append(melody_track)
    midi_file.tracks.append(chord_track)

    melody_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    melody_track.append(
        mido.Message(
            "program_change",
            program=MIDI_PIANO_PROGRAM,
            channel=MIDI_MELODY_CHANNEL,
            time=0,
        )
    )
    chord_track.append(
        mido.Message(
            "program_change",
            program=MIDI_PIANO_PROGRAM,
            channel=MIDI_CHORD_CHANNEL,
            time=0,
        )
    )

    _append_note_events(melody_track, melody_notes, MIDI_MELODY_CHANNEL)
    _append_chord_events(chord_track, chords)

    try:
        midi_file.save(midi_path)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to write MIDI file: {midi_path}") from exc

    return midi_path


def _try_write_wav_preview_file(audio_id: str, output_directory: Path, midi_path: Path) -> Path | None:
    try:
        return _write_wav_preview_file(audio_id, output_directory, midi_path)
    except Exception as exc:
        print(f"wav_preview_generation_failed midi_path={midi_path} error={exc}", flush=True)
        return None


def _write_wav_preview_file(audio_id: str, output_directory: Path, midi_path: Path) -> Path:
    preview_notes = _extract_preview_notes_from_midi(midi_path)
    if not preview_notes:
        raise AudioProcessingError("No MIDI notes available for WAV preview generation")

    preview_path = output_directory / f"{audio_id}-preview.wav"
    samples = _render_preview_notes_to_samples(preview_notes)

    try:
        sf.write(preview_path, samples, PREVIEW_SAMPLE_RATE, subtype="PCM_16")
    except Exception as exc:
        raise AudioProcessingError(f"Unable to write WAV preview file: {preview_path}") from exc

    return preview_path


def _extract_preview_notes_from_midi(midi_path: Path) -> list[PreviewNote]:
    try:
        midi_file = mido.MidiFile(midi_path)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to read MIDI file for WAV preview: {midi_path}") from exc

    tempo = mido.bpm2tempo(FALLBACK_TEMPO_BPM)
    elapsed_seconds = 0.0
    active_notes: dict[tuple[int, int], list[tuple[float, int]]] = {}
    preview_notes: list[PreviewNote] = []

    for message in mido.merge_tracks(midi_file.tracks):
        elapsed_seconds += mido.tick2second(message.time, midi_file.ticks_per_beat, tempo)

        if message.type == "set_tempo":
            tempo = message.tempo
            continue

        if message.type == "note_on" and message.velocity > 0:
            key = (message.channel, message.note)
            active_notes.setdefault(key, []).append((elapsed_seconds, message.velocity))
            continue

        is_note_end = message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        )
        if not is_note_end:
            continue

        key = (message.channel, message.note)
        starts = active_notes.get(key)
        if not starts:
            continue

        start_time, velocity = starts.pop()
        if not starts:
            del active_notes[key]

        duration = elapsed_seconds - start_time
        if duration > 0:
            preview_notes.append(
                PreviewNote(
                    midi_note=message.note,
                    velocity=velocity,
                    start_time=start_time,
                    duration=duration,
                )
            )

    return preview_notes


def _render_preview_notes_to_samples(preview_notes: list[PreviewNote]) -> np.ndarray:
    end_time = max(note.start_time + note.duration for note in preview_notes)
    total_samples = int(np.ceil((end_time + PREVIEW_RELEASE_SECONDS) * PREVIEW_SAMPLE_RATE))
    samples = np.zeros(total_samples, dtype=np.float32)

    for note in preview_notes:
        start_index = int(round(note.start_time * PREVIEW_SAMPLE_RATE))
        note_sample_count = max(1, int(round(note.duration * PREVIEW_SAMPLE_RATE)))
        release_sample_count = int(round(PREVIEW_RELEASE_SECONDS * PREVIEW_SAMPLE_RATE))
        tone_sample_count = note_sample_count + release_sample_count

        times = np.arange(tone_sample_count, dtype=np.float32) / PREVIEW_SAMPLE_RATE
        frequency = 440.0 * (2.0 ** ((note.midi_note - 69) / 12.0))
        tone = np.sin(2.0 * np.pi * frequency * times)
        envelope = _build_preview_envelope(note_sample_count, release_sample_count)
        amplitude = PREVIEW_BASE_AMPLITUDE * (note.velocity / 127.0)
        rendered_note = (tone * envelope * amplitude).astype(np.float32)

        end_index = min(samples.size, start_index + rendered_note.size)
        samples[start_index:end_index] += rendered_note[: end_index - start_index]

    peak = float(np.max(np.abs(samples)))
    if peak > PREVIEW_PEAK_HEADROOM:
        samples *= PREVIEW_PEAK_HEADROOM / peak

    return samples


def _build_preview_envelope(note_sample_count: int, release_sample_count: int) -> np.ndarray:
    envelope = np.ones(note_sample_count + release_sample_count, dtype=np.float32)
    attack_sample_count = min(
        note_sample_count,
        max(1, int(round(PREVIEW_ATTACK_SECONDS * PREVIEW_SAMPLE_RATE))),
    )
    envelope[:attack_sample_count] = np.linspace(
        0.0,
        1.0,
        attack_sample_count,
        endpoint=True,
        dtype=np.float32,
    )

    if release_sample_count > 0:
        envelope[note_sample_count:] = np.linspace(
            1.0,
            0.0,
            release_sample_count,
            endpoint=True,
            dtype=np.float32,
        )

    return envelope


def _append_note_events(
    track: mido.MidiTrack,
    notes: list[Note],
    channel: int,
) -> MidiEventMetrics:
    current_tick = 0
    zero_or_negative_delta_count = 0
    midi_event_count = 0
    ordered_notes = sorted(notes, key=lambda note: (note.startTime, note.midi_note))
    for index, note in enumerate(ordered_notes):
        _validate_note_timing(note)
        start_tick = _seconds_to_ticks(note.startTime)
        next_start_tick = (
            _seconds_to_ticks(ordered_notes[index + 1].startTime)
            if index + 1 < len(ordered_notes)
            else None
        )
        duration_ticks = _melody_duration_ticks(
            start_tick,
            _seconds_to_ticks(note.duration),
            next_start_tick,
        )
        raw_delta_start = start_tick - current_tick
        if raw_delta_start < 0 or duration_ticks <= 0:
            zero_or_negative_delta_count += 1
        delta_start = max(0, raw_delta_start)

        track.append(
            mido.Message(
                "note_on",
                note=note.midi_note,
                velocity=_melody_velocity(note.velocity, start_tick),
                channel=channel,
                time=delta_start,
            )
        )
        midi_event_count += 1
        track.append(
            mido.Message(
                "note_off",
                note=note.midi_note,
                velocity=0,
                channel=channel,
                time=duration_ticks,
            )
        )
        midi_event_count += 1
        current_tick = start_tick + duration_ticks

    return MidiEventMetrics(
        midi_event_count=midi_event_count,
        zero_or_negative_delta_count=zero_or_negative_delta_count,
    )


def _append_chord_events(track: mido.MidiTrack, chords: list[Chord]) -> None:
    current_tick = 0
    for chord in chords:
        start_tick = _seconds_to_ticks(chord.startTime)
        duration_ticks = max(MIDI_MIN_NOTE_TICKS, _seconds_to_ticks(chord.duration))
        chord_notes = _chord_to_midi_notes(chord)
        delta_start = max(0, start_tick - current_tick)

        for index, midi_note in enumerate(chord_notes):
            track.append(
                mido.Message(
                    "note_on",
                    note=midi_note,
                    velocity=MIDI_CHORD_VELOCITY,
                    channel=MIDI_CHORD_CHANNEL,
                    time=delta_start if index == 0 else 0,
                )
            )

        for index, midi_note in enumerate(chord_notes):
            track.append(
                mido.Message(
                    "note_off",
                    note=midi_note,
                    velocity=0,
                    channel=MIDI_CHORD_CHANNEL,
                    time=duration_ticks if index == 0 else 0,
                )
            )

        current_tick = start_tick + duration_ticks


def _melody_duration_ticks(
    start_tick: int,
    duration_ticks: int,
    next_start_tick: int | None,
) -> int:
    playable_duration_ticks = max(MIDI_MIN_NOTE_TICKS, duration_ticks)
    if next_start_tick is None or next_start_tick <= start_tick:
        return playable_duration_ticks

    available_ticks = max(
        MIDI_MIN_NOTE_TICKS,
        next_start_tick - start_tick - MIDI_RELEASE_GAP_TICKS,
    )
    return min(playable_duration_ticks, available_ticks)


def _melody_velocity(base_velocity: int, start_tick: int) -> int:
    bar_ticks = MIDI_TICKS_PER_BEAT * MIDI_BAR_BEATS
    beat_position = start_tick % bar_ticks
    if beat_position == 0:
        accent = 10
    elif beat_position % MIDI_TICKS_PER_BEAT == 0:
        accent = 5
    elif beat_position % MIDI_HALF_BEAT_TICKS == 0:
        accent = 1
    else:
        accent = -3

    return _clamp_midi_velocity(base_velocity + accent)


def _clamp_midi_velocity(velocity: int) -> int:
    return max(1, min(127, velocity))


def _chord_to_midi_notes(chord: Chord) -> list[int]:
    root_pitch_class = _note_name_to_pitch_class(chord.root)
    root_midi_note = CHORD_BASE_OCTAVE_MIDI + root_pitch_class

    if chord.type == "MAJOR":
        intervals = (0, 4, 7)
    elif chord.type == "MINOR":
        intervals = (0, 3, 7)
    elif chord.type == "DIMINISHED":
        intervals = (0, 3, 6)
    else:
        raise AudioProcessingError(f"Unsupported chord type: {chord.type}")

    return [root_midi_note + interval for interval in intervals]


def _note_name_to_pitch_class(note_name: str) -> int:
    try:
        return MIDI_NOTE_NAMES.index(note_name)
    except ValueError as exc:
        raise AudioProcessingError(f"Unsupported chord root: {note_name}") from exc


def _seconds_to_ticks(seconds: float) -> int:
    if not np.isfinite(seconds) or seconds < 0:
        raise AudioProcessingError("Invalid MIDI event time")

    beat_seconds = 60.0 / FALLBACK_TEMPO_BPM
    return int(round((seconds / beat_seconds) * MIDI_TICKS_PER_BEAT))
