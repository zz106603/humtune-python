from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request, Response

from app.audio_processing import AudioProcessingError
from app.audio_processing import analyze_audio as process_audio
from app.audio_processing import Chord
from app.schemas import AudioAnalyzeRequest, AudioAnalyzeResponse

app = FastAPI()

  
@app.on_event("startup")
def log_loaded_app_path() -> None:
    print(f"python_audio_service_loaded main_path={Path(__file__).resolve()}")


@app.middleware("http")
async def log_audio_analyze_request(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/internal/audio/analyze":
        body = await request.body()
        print("RAW_HEADERS", dict(request.headers), flush=True)
        print("RAW_BODY_BYTES", len(body), flush=True)
        print("RAW_BODY_TEXT", body.decode("utf-8", errors="replace"), flush=True)

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)

    return await call_next(request)


@app.get("/health")
def health() -> Response:
    return Response(content="UP", media_type="text/plain")


@app.post(
    "/internal/audio/analyze",
    response_model=AudioAnalyzeResponse,
    response_model_exclude_none=True,
)
def analyze_audio(request: AudioAnalyzeRequest) -> AudioAnalyzeResponse:
    start_time = perf_counter()
    try:
        result = process_audio(
            audio_id=request.audioId,
            raw_audio_path=request.rawAudioPath,
            output_directory=request.outputDirectory,
        )
    except AudioProcessingError as exc:
        return AudioAnalyzeResponse(
            status="FAILED",
            errorMessage=str(exc),
        )
    except Exception:
        return AudioAnalyzeResponse(
            status="FAILED",
            errorMessage="Audio analysis failed",
        )

    processing_time_ms = int((perf_counter() - start_time) * 1000)
    return AudioAnalyzeResponse(
        status="COMPLETED",
        detectedScale=result.detectedScale,
        keyConfidence=result.keyConfidence,
        originalNotes=[note.pitch for note in result.original_notes],
        adjustedNotes=[note.pitch for note in result.adjusted_notes],
        chords=[_format_chord(chord) for chord in result.chords],
        midiPath=result.midiPath,
        previewAudioPath=result.previewAudioPath,
        processingTimeMs=processing_time_ms,
    )


def _format_chord(chord: Chord) -> str:
    if chord.type == "MAJOR":
        return chord.root
    if chord.type == "MINOR":
        return f"{chord.root}m"
    if chord.type == "DIMINISHED":
        return f"{chord.root}dim"
    return chord.root
