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
- **Guitar Pro**: `.gp3`, `.gp4`, `.gp5` (via `pyguitarpro`); `.gp`, `.gpx` (via MuseScore Studio CLI)
- **MuseScore & Capella**: `.mscz`, `.mscx`, `.cap` (via MuseScore Studio CLI); `.capx` (native `music21` XML parser with MuseScore fallback)
- **LilyPond**: `.ly` (via `python-ly` or `lilypond` CLI when installed)

**Built-in Score Search Catalogs (`transcribe-music search`):**
- **MusicXML Corpora**: **OpenScore Orchestra** (`MarkGotham/Hauptstimme`), **OpenScore Lieder**, **OpenScore StringQuartets**, **When-in-Rome** (`.mxl`), and optional local **PDMX** (254k+ public-domain MusicXML scores via DuckDB).
- **Symbolic & MIDI Repositories**: Expanded **Humdrum / MuseData** classical repos (`.krn`: Vivaldi concertos, Haydn symphonies/quartets/sonatas, Bach Brandenburg/WTC, Corelli, Chopin, Scarlatti), **Mutopia Project** (`.mid`/`.ly`/`.pdf`), **`music21` Corpus**, and **Archive.org**.
- **Classical Sheet Music Fallback**: **IMSLP MediaWiki API** (prioritizing Solo Part + Typeset PDFs, directing to browser verification when gated) & **Archive.org `collection:imslp`**.

### 2. Sheet Music PDFs (Fallback OMR)
- **`.pdf`**: Auto-selects **Audiveris CLI** (preferred for classical solo parts with Bass/Tenor/Treble C-clef shifts) when installed, or falls back to rasterizing with `pymupdf` and running a single batched **`homr`** Vision Transformer pass via `uvx` Python 3.14 (`--omr-backend auto|audiveris|homr`). Best accuracy on clean solo instrument parts rather than full orchestral conductor scores.

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
# 0. Search public catalogs (MusicXML > digital > solo PDFs), show top 5, and download your pick
uv run transcribe-music search "Moonlight Sonata"
# Or search classical works including IMSLP solo parts / Mutopia MIDIs:
uv run transcribe-music search "Haydn Cello Concerto"
# Or download selection #1 and enrich immediately for piano:
uv run transcribe-music search "Moonlight Sonata" --select 1 --instrument piano

# Optional: sync the 254k-score PDMX DuckDB catalog (Zenodo 15571083)
uv run transcribe-music pdmx sync --with-mid   # or --with-mxl for full .mxl archive
uv run transcribe-music pdmx status

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

# 5. Sheet music PDF fallback (Audiveris CLI when installed, else batched homr OMR)
uv run transcribe-music pdf solo_cello_part.pdf \
  --instrument cello --omr-backend auto -o cello_from_pdf.musicxml

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
