from pydantic import BaseModel


class AudioAnalyzeRequest(BaseModel):
    audioId: str
    rawAudioPath: str
    outputDirectory: str


class AudioAnalyzeResponse(BaseModel):
    status: str
    detectedScale: str | None = None
    keyConfidence: float | None = None
    originalNotes: list[str] | None = None
    adjustedNotes: list[str] | None = None
    chords: list[str] | None = None
    midiPath: str | None = None
    processingTimeMs: int | None = None
    errorMessage: str | None = None
