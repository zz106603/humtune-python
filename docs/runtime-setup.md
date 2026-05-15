# Runtime Setup

## Supported Python

- Python `3.11`

Python `3.13` is not supported for the main audio service because Basic Pitch depends on
TensorFlow versions that do not provide compatible Python `3.13` wheels.

The supported local runtime is Linux/WSL with Python `3.11`. Native Windows Python is not the
target runtime for Basic Pitch because TensorFlow wheel availability is inconsistent there.

## WSL / Linux Setup

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the local sample through the main Basic Pitch pipeline:

```bash
python scripts/manual_test.py test_data/나비야.m4a
```

Expected output includes:

- `detectedScale=...`
- `originalNoteCount=...`
- `cleanedNoteCount=...`
- `adjustedNoteCount=...`
- `quantizedNoteCount=...`
- `chordCount=...`
- `cleanedNotes=...`
- `adjustedNotes=...`
- `finalMelodyNotes=...`
- `chords=...`
- `rawMidiPath=manual-output/나비야-raw.mid`
- `cleanedMidiPath=manual-output/나비야-cleaned.mid`
- `finalMelodyMidiPath=manual-output/나비야-final-melody.mid`
- `chordMidiPath=manual-output/나비야-chords.mid`
- `combinedMidiPath=manual-output/나비야-combined.mid`
- `midiPath=manual-output/나비야.mid`

Manual MIDI files:

- `manual-output/나비야-raw.mid`: raw Basic Pitch melody events only
- `manual-output/나비야-cleaned.mid`: melody after deterministic cleanup only
- `manual-output/나비야-final-melody.mid`: final quantized melody only
- `manual-output/나비야-chords.mid`: inferred sustained chords only
- `manual-output/나비야-combined.mid`: final quantized melody plus inferred chords
- `manual-output/나비야.mid`: default service-style MIDI output

## Windows Setup

Use WSL and run the Linux setup above inside the WSL shell.

```powershell
wsl
```

Then from the WSL shell:

```bash
cd /mnt/c/Users/629jy/Documents/humtune-audio-service
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/manual_test.py test_data/나비야.m4a
```

## Runtime Notes

- Basic Pitch is now a main dependency, not an optional POC dependency.
- The service keeps the existing Spring/FastAPI request and response contract.
- The audio flow is Basic Pitch note events → deterministic cleanup → scale fitting →
  quantization → chord inference → MIDI.
- No frontend or Spring Boot process is required for local Python verification.
