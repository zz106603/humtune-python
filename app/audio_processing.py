from dataclasses import dataclass
from pathlib import Path

import librosa
import mido
import numpy as np


TARGET_SAMPLE_RATE = 22050
MIN_DURATION_SECONDS = 0.25
PITCH_HOP_LENGTH = 512
PITCH_FRAME_LENGTH = 2048
MIN_PITCH_FRAMES = 3
MIN_VOICED_PROBABILITY = 0.5
MIN_PITCH_HZ = librosa.note_to_hz("C2")
MAX_PITCH_HZ = librosa.note_to_hz("C7")
MIN_NOTE_DURATION_SECONDS = 0.1
NOTE_MERGE_TOLERANCE_SEMITONES = 1
DEFAULT_NOTE_VELOCITY = 80
MIDI_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_SCALE_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE_INTERVALS = (0, 2, 3, 5, 7, 8, 10)
FALLBACK_TEMPO_BPM = 100
QUANTIZE_GRID_SECONDS = (60.0 / FALLBACK_TEMPO_BPM) / 2.0
CHORD_WINDOW_SECONDS = 4 * (60.0 / FALLBACK_TEMPO_BPM)
MAJOR_CHORD_TIEBREAK_DEGREES = (0, 3, 4, 5, 1, 2, 6)
MINOR_CHORD_TIEBREAK_DEGREES = (0, 3, 4, 5, 6, 2, 1)
MIDI_TICKS_PER_BEAT = 480
MIDI_MELODY_CHANNEL = 0
MIDI_CHORD_CHANNEL = 1
MIDI_PIANO_PROGRAM = 0
MIDI_CHORD_VELOCITY = 64
CHORD_BASE_OCTAVE_MIDI = 48


class AudioProcessingError(Exception):
    pass


@dataclass(frozen=True)
class PitchFrame:
    timestamp_seconds: float
    frequency_hz: float
    confidence: float


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
    detectedScale: str
    keyConfidence: float
    adjusted_notes: list[Note]
    quantized_notes: list[Note]
    chords: list[Chord]
    midiPath: str


def analyze_audio(
    audio_id: str,
    raw_audio_path: str,
    output_directory: str,
) -> LoadedAudio:
    audio_path = _validate_raw_audio_path(raw_audio_path)
    samples, sample_rate = _load_audio(audio_path)
    duration_seconds = _validate_loaded_audio(samples, sample_rate)
    pitch_frames = _detect_pitch_frames(samples, sample_rate)
    original_notes = _pitch_frames_to_notes(pitch_frames)
    scale_fit = _fit_scale(original_notes)
    adjusted_notes = _adjust_notes_to_scale(original_notes, scale_fit)
    quantized_notes = _quantize_notes(adjusted_notes)
    chords = _infer_chords(quantized_notes, scale_fit)
    midi_path = _write_midi_file(audio_id, Path(output_directory), quantized_notes, chords)

    return LoadedAudio(
        audio_id=audio_id,
        raw_audio_path=audio_path,
        output_directory=Path(output_directory),
        samples=samples,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        pitch_frames=pitch_frames,
        original_notes=original_notes,
        detectedScale=scale_fit.name,
        keyConfidence=scale_fit.confidence,
        adjusted_notes=adjusted_notes,
        quantized_notes=quantized_notes,
        chords=chords,
        midiPath=str(midi_path),
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

    pitch_frames: list[PitchFrame] = []
    for timestamp, frequency, is_voiced, voiced_probability in zip(
        frame_times,
        frequencies,
        voiced_flags,
        voiced_probabilities,
    ):
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
    current_start = pitch_frames[0].timestamp_seconds
    last_timestamp = pitch_frames[0].timestamp_seconds

    for frame in pitch_frames[1:]:
        frame_midi = _frequency_to_midi_note(frame.frequency_hz)
        if abs(frame_midi - current_midi) <= NOTE_MERGE_TOLERANCE_SEMITONES:
            current_midi = round((current_midi + frame_midi) / 2)
            last_timestamp = frame.timestamp_seconds
            continue

        _append_note_if_long_enough(notes, current_midi, current_start, last_timestamp)
        current_midi = frame_midi
        current_start = frame.timestamp_seconds
        last_timestamp = frame.timestamp_seconds

    _append_note_if_long_enough(notes, current_midi, current_start, last_timestamp)

    if not notes:
        raise AudioProcessingError("No valid notes found")

    return notes


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

    candidates: list[tuple[float, int, int, ScaleFit]] = []
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
            confidence = _scale_confidence(notes, distance_score)
            scale_fit = ScaleFit(
                name=f"{MIDI_NOTE_NAMES[root]}_{mode_name}",
                root_pitch_class=root,
                scale_pitch_classes=scale_pitch_classes,
                confidence=confidence,
            )
            candidates.append((distance_score, -included_count, -tonic_present, scale_fit))

    if not candidates:
        raise AudioProcessingError("Unable to fit scale")

    return min(candidates, key=lambda candidate: candidate[:3])[3]


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

    return [_adjust_note_to_scale(note, scale_fit.scale_pitch_classes) for note in notes]


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
    if not notes:
        raise AudioProcessingError("No notes available for quantization")

    quantized_notes: list[Note] = []
    previous_end_time = 0.0

    for note in notes:
        _validate_note_timing(note)
        start_time = _nearest_grid_time(note.startTime)
        duration = max(_nearest_grid_time(note.duration), QUANTIZE_GRID_SECONDS)

        if start_time < previous_end_time:
            start_time = previous_end_time

        quantized_notes.append(
            Note(
                pitch=note.pitch,
                midi_note=note.midi_note,
                startTime=float(start_time),
                duration=float(duration),
                velocity=note.velocity,
            )
        )
        previous_end_time = start_time + duration

    if not quantized_notes:
        raise AudioProcessingError("Unable to quantize notes")

    return quantized_notes


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

    total_duration = max(note.startTime + note.duration for note in notes)
    window_count = max(1, int(np.ceil(total_duration / CHORD_WINDOW_SECONDS)))
    chords: list[Chord] = []

    for window_index in range(window_count):
        start_time = window_index * CHORD_WINDOW_SECONDS
        end_time = start_time + CHORD_WINDOW_SECONDS
        window_notes = [
            note for note in notes if _note_overlaps_window(note, start_time, end_time)
        ]
        if not window_notes:
            continue

        candidate = _select_chord_candidate(window_notes, candidates, mode)
        chords.append(
            Chord(
                root=MIDI_NOTE_NAMES[candidate["root_pitch_class"]],
                type=candidate["type"],
                startTime=float(start_time),
                duration=float(CHORD_WINDOW_SECONDS),
            )
        )

    if not chords:
        raise AudioProcessingError("Unable to infer chords")

    return chords


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
            _chord_score(notes, candidate["pitch_classes"]),
            -degree_priority.get(candidate["degree"], len(tie_break_degrees)),
        ),
    )


def _chord_score(notes: list[Note], chord_pitch_classes: object) -> float:
    if not isinstance(chord_pitch_classes, frozenset):
        raise AudioProcessingError("Invalid chord candidate")

    return sum(
        note.duration
        for note in notes
        if note.midi_note % 12 in chord_pitch_classes
    )


def _note_overlaps_window(note: Note, start_time: float, end_time: float) -> bool:
    note_start = note.startTime
    note_end = note.startTime + note.duration
    return note_start < end_time and note_end > start_time


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

    _append_note_events(melody_track, melody_notes, MIDI_MELODY_CHANNEL)
    _append_chord_events(chord_track, chords)

    try:
        midi_file.save(midi_path)
    except Exception as exc:
        raise AudioProcessingError(f"Unable to write MIDI file: {midi_path}") from exc

    return midi_path


def _append_note_events(
    track: mido.MidiTrack,
    notes: list[Note],
    channel: int,
) -> None:
    current_tick = 0
    for note in notes:
        _validate_note_timing(note)
        start_tick = _seconds_to_ticks(note.startTime)
        duration_ticks = max(1, _seconds_to_ticks(note.duration))
        delta_start = max(0, start_tick - current_tick)

        track.append(
            mido.Message(
                "note_on",
                note=note.midi_note,
                velocity=note.velocity,
                channel=channel,
                time=delta_start,
            )
        )
        track.append(
            mido.Message(
                "note_off",
                note=note.midi_note,
                velocity=0,
                channel=channel,
                time=duration_ticks,
            )
        )
        current_tick = start_tick + duration_ticks


def _append_chord_events(track: mido.MidiTrack, chords: list[Chord]) -> None:
    current_tick = 0
    for chord in chords:
        start_tick = _seconds_to_ticks(chord.startTime)
        duration_ticks = max(1, _seconds_to_ticks(chord.duration))
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
