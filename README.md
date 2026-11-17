# transcribe-music

A fast, local-first Python CLI, SDK, and **Marimo** workbench that converts **digital symbolic music scores** (primary source of truth) — or **sheet music PDFs** as an OMR fallback — into validated **MusicXML** enriched with **chord symbols (`<harmony>`), guitar TAB (`<string>`, `<fret>`), and ergonomic fingerings (`<fingering>`)** for:

- **Piano** (Grand Staff RH/LH splitting or native 2-staff preservation + O(N) Viterbi fingering solver)
- **Guitar** (Treble `8vb` written-pitch normalization + 6-string Viterbi TAB & chord-grip solver)
- **Violin** (4-string Viterbi position solver)
- **Cello** (Preserves Bass/Tenor/Treble clefs + 4-string closed/extended/thumb `T` solver)

Powered by **`uv`**, **`music21`**, **`converter21`**, **`pyguitarpro`**, **`duckdb`**, and optional **`marimo` + `verovio`**.

---

## Supported Input Formats

### 1. Digital Symbolic Files (Primary — 99–100% Note Fidelity, < 0.5s)
- **MusicXML**: `.musicxml`, `.xml`, `.mxl`
- **Humdrum / MEI / ABC** (via `converter21`): `.krn`, `.mei`, `.abc`
- **MIDI**: `.mid`, `.midi` (quantized cleanly + auto-split Piano Grand Staff)
- **Guitar Pro** (via `pyguitarpro`): `.gp3`, `.gp4`, `.gp5`
- **MuseScore** (when MuseScore Studio CLI is installed locally): `.mscz`, `.mscx`

**Where to get free digital scores:**
- **Classical / Public Domain**: **PDMX** (250k+ MusicXML scores), **OpenScore** (curated chamber/lieder/orchestral `.mxl`), **MuseScore.com**, **KernScores** (`.krn`), and **IMSLP** (`.mxl`/`.mid` attachments or Solo Part PDFs).
- **Modern / Rock / Pop / Anime**: **MuseScore.com** (`.mxl`/`.mscz`) or **Guitar Pro tabs** (`.gp5` via Ultimate Guitar / Songsterr).

### 2. Sheet Music PDFs (Fallback OMR)
- **`.pdf`**: Rasterized with `pymupdf` and processed in a single batched **`homr`** Vision Transformer pass via `uvx` Python 3.14. Best accuracy on clean solo instrument parts rather than full orchestral conductor scores.

---

## Quickstart

Requires **uv** (Python `3.14` managed automatically via `.python-version`).

```bash
# Core CLI + SDK
uv sync

# Optional interactive Marimo + Verovio workbench UI
uv sync --extra ui

# Development tools (pytest, ruff)
uv sync --all-extras
```

### CLI Usage

```bash
# 0. Search public digital score catalogs (MusicXML preferred), show top 5, and download your pick
uv run transcribe-music search "Moonlight Sonata"
# Or download selection #1 and enrich immediately for piano:
uv run transcribe-music search "Moonlight Sonata" --select 1 --instrument piano

# 1. Inspect parts/tracks inside a multi-part digital score
uv run transcribe-music inspect haydn_cello_concerto.mxl

# 2. Convert any digital score (or fallback PDF) with automatic part selection
uv run transcribe-music convert haydn_cello_concerto.mxl \
  --instrument cello -o haydn_cello_fingered.musicxml

# 3. Override part selection explicitly by part number or name
uv run transcribe-music score suite_or_sonata.mid \
  --instrument cello --part "Solo Cello" -o solo_cello.musicxml

# 4. Guitar Pro tab -> MusicXML with chords & ergonomic 6-string TAB/fingerings
uv run transcribe-music convert song.gp5 \
  --instrument guitar -o song_guitar.musicxml

# 5. Sheet music PDF fallback (batched homr OMR via uvx Python 3.14)
uv run transcribe-music pdf solo_cello_part.pdf \
  --instrument cello -o cello_from_pdf.musicxml

# 6. Launch the Marimo Interactive Workbench + Verovio SVG viewer + DuckDB explorer
uv run transcribe-music ui

# 7. Inspect or clear the DuckDB cache
uv run transcribe-music cache list
uv run transcribe-music cache clear
```

### Python SDK Usage

```python
from transcribe_music import Instrument, convert_music, inspect_score

inspection = inspect_score("haydn_cello_concerto.mxl")
print([(p.index, p.name, p.measure_count) for p in inspection.parts])

res = convert_music(
    input_path="haydn_cello_concerto.mxl",
    instrument=Instrument.CELLO,
    output_path="out/haydn_cello.musicxml",
)
print(res.musicxml_path, res.selected_part, res.note_count, res.chord_count, res.fingering_count)
```

---

## Architecture Highlights

1. **Digital-First Multi-Part Selection**: Isolates the target soloist part from full ensemble scores (or splits single-track Piano MIDI into Grand Staff `RH` + `LH`) while computing harmonic chroma from the **full ensemble** so solo parts inherit accurate accompaniment `<harmony>` chords.
2. **Non-Destructive MusicXML Normalization & Strict Validation**: Preserves authored measures, key/time signatures, and clef transitions (e.g., Cello Bass/Tenor/Treble) while enforcing strict bar-duration integrity (zero duplicate whole-bar rests or overfull measures).
3. **Pure-Symbolic Enrichment**:
   - **Chords (`<harmony>`)**: Symbolic pitch-class Viterbi HMM.
   - **Fingerings (`<technical><fingering>`)**: Unified O(N) Viterbi solvers for Piano, Guitar (`<string>` + `<fret>`), Violin, and Cello (including thumb position `T`).
4. **DuckDB Catalog (`catalog.duckdb`)**: Content-addressed job caching queryable via `transcribe-music cache list` or the Marimo UI (`--no-cache` bypasses the database).

## Development & Testing

```bash
uv run pytest
uv run ruff check src tests notebooks
```
