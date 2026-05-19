from pydantic import BaseModel, Field


class AudioAnalyzeRequest(BaseModel):
    audioId: str
    rawAudioPath: str
    outputDirectory: str


class AudioAnalyzeResponse(BaseModel):
    status: str
    detectedScale: str | None = Field(
        default=None,
        description="Final fitted scale name used for scale adjustment, quantization, and chord inference.",
    )
    keyConfidence: float | None = Field(
        default=None,
        description="Deterministic scale-fit confidence score for the selected detectedScale.",
    )
    originalNotes: list[str] | None = Field(
        default=None,
        description="Raw Basic Pitch note-name sequence before deterministic cleanup.",
    )
    adjustedNotes: list[str] | None = Field(
        default=None,
        description="Final API melody note-name sequence after cleanup, scale adjustment, and quantization.",
    )
    chords: list[str] | None = Field(
        default=None,
        description="Final inferred chord labels used by the product MIDI accompaniment.",
    )
    midiPath: str | None = Field(
        default=None,
        description="Path to the main product MIDI containing final melody plus inferred chords.",
    )
    previewAudioPath: str | None = Field(
        default=None,
        description="Optional WAV preview rendered from midiPath when preview generation succeeds.",
    )
    processingTimeMs: int | None = None
    errorMessage: str | None = None
