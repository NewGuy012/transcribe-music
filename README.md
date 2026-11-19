# transcribe-music

A fast, lean, local-first Python CLI and SDK that converts **digital symbolic music scores** (primary source of truth) — or **sheet music PDFs** as an OMR fallback — into validated **MusicXML** enriched with **chord symbols (`<harmony>`), guitar TAB (`<string>`, `<fret>`), and ergonomic fingerings (`<fingering>`)** for:

- **Piano** (Grand Staff RH/LH splitting or native 2-staff preservation + O(N) Viterbi fingering solver)
- **Guitar** (Treble `8vb` written-pitch normalization, custom tunings/capos, locked-TAB or 24-fret Viterbi solver, power/sus chords, and `notation|tab|both` MusicXML export)
- **Violin** (4-string Viterbi position solver)
- **Cello** (Preserves Bass/Tenor/Treble clefs + 4-string closed/extended/thumb `T` solver)

Powered by **`uv`**, **`music21`**, and the Python standard library (**`sqlite3`**, **`dataclasses`**, **`argparse`**, **`urllib`**), with optional extras for **`converter21`**, **`pymupdf`**, and **`pyguitarpro`**.

---

## Supported Input Formats

### 1. Digital Symbolic Files (Primary — 99–100% Note Fidelity, < 0.5s)
- **MusicXML**: `.musicxml`, `.xml`, `.mxl`
- **MIDI**: `.mid`, `.midi` (quantized cleanly + auto-split Piano Grand Staff)
- **Humdrum / MEI / ABC** (via optional `formats` extra: `converter21`): `.krn`, `.mei`, `.abc`
- **Guitar Pro**: `.gp3`, `.gp4`, `.gp5` (via optional `gp` extra: `pyguitarpro`); `.gp` GP7/8 (native stdlib GPIF XML parser with MuseScore fallback); `.gpx` (via MuseScore Studio CLI)
- **ASCII Guitar Tabs**: `.tab` (and `.txt` local conversion)
- **MuseScore & Capella**: `.mscz`, `.mscx`, `.cap` (via MuseScore Studio CLI); `.capx` (native `music21` XML parser with MuseScore fallback)
- **LilyPond**: `.ly` (via `python-ly` pure-parser `ly` CLI when installed)

**Built-in Fast Score Catalogs (`transcribe-music search`):**
- **MusicXML & Guitar Corpora**: **OpenScore Orchestra** (`MarkGotham/Hauptstimme`), **OpenScore Lieder**, **OpenScore StringQuartets**, **When-in-Rome** (`.mxl`), **GP Classical Guitar** (`.gp5`/`.gpx`/`.mid`), optional local **PDMX** (254k+ public-domain MusicXML/MIDI scores via SQLite), and optional local/Zenodo **DadaGP** Guitar Pro SQLite catalog.
- **Local Symbolic Corpus**: **`music21` Corpus** (`.mxl`, `.xml`, `.krn`).

### 2. Sheet Music PDFs (Fallback OMR)
- **`.pdf`**: Auto-selects **Audiveris CLI** (preferred for classical solo parts with Bass/Tenor/Treble C-clef shifts) when installed, or falls back to rasterizing with `pymupdf` (optional `pdf` extra) and running a single batched **`homr`** Vision Transformer pass via `uvx` Python 3.14 (`--omr-backend auto|audiveris|homr`). Best accuracy on clean solo instrument parts rather than full orchestral conductor scores.

---

## Quickstart

Requires **uv** (Python `3.14` managed automatically via `.python-version`).

```bash
# Core CLI + SDK (MusicXML, MIDI, native GP7/8 .gp, ASCII .tab, Capella .capx)
uv sync

# Optional extras: Humdrum/MEI/ABC (converter21), PDF OMR (pymupdf), GP3-5 (pyguitarpro)
uv sync --extra formats
uv sync --extra pdf
uv sync --extra gp
uv sync --all-extras
```

### CLI Usage

```bash
# 1. Search fast symbolic catalogs (MusicXML or GP TAB with --instrument guitar), show top 5, and download
uv run transcribe-music search "Moonlight Sonata"
uv run transcribe-music search "Asturias" --instrument guitar
# Or download selection #1 and enrich immediately for piano:
uv run transcribe-music search "Moonlight Sonata" --select 1 --instrument piano

# Optional: sync the 254k-score PDMX SQLite catalog (Zenodo 15571083)
uv run transcribe-music pdmx sync --with-mid   # or --with-mxl for full .mxl archive
uv run transcribe-music pdmx status

# Optional: index a local DadaGP / Guitar Pro folder or Zenodo record (using ZENODO_TOKEN) into SQLite
uv run transcribe-music dadagp sync --gp-dir /path/to/guitar_tabs
uv run transcribe-music dadagp status

# 2. Inspect parts/tracks inside a multi-part digital score (including guitar tuning/capo)
uv run transcribe-music inspect haydn_cello_concerto.mxl

# 3. Convert any digital score (or fallback PDF) with automatic part selection
uv run transcribe-music convert haydn_cello_concerto.mxl \
  --instrument cello -o haydn_cello_fingered.musicxml

# 4. Override part selection explicitly by part number or name
uv run transcribe-music convert suite_or_sonata.mid \
  --instrument cello --part "Solo Cello" -o solo_cello.musicxml

# 5. Guitar Pro (.gp/.gp5) or ASCII (.tab) -> linked Notation + 6-line TAB MusicXML
uv run transcribe-music convert song.gp5 \
  --instrument guitar --guitar-staff both -o song_guitar.musicxml

# 6. Sheet music PDF fallback (Audiveris CLI when installed, else batched homr OMR via --extra pdf)
uv run transcribe-music convert solo_cello_part.pdf \
  --instrument cello --omr-backend auto -o cello_from_pdf.musicxml

# 7. Inspect or clear the SQLite cache (and cached artifacts)
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
2. **Atomic MusicXML Normalization & Strict Validation**: Preserves authored measures, key/time signatures, and clef transitions (e.g., Cello Bass/Tenor/Treble) while enforcing strict bar-duration integrity (zero duplicate whole-bar rests or overfull measures) before atomically replacing the target file.
3. **Pure-Symbolic Enrichment**:
   - **Chords (`<harmony>`)**: Pure-Python symbolic pitch-class Viterbi HMM (zero `numpy` dependency).
   - **Fingerings (`<technical><fingering>`)**: Unified O(N) Viterbi solvers for Piano, Guitar (`<string>` + `<fret>`), Violin, and Cello (including thumb position `T`).
4. **SQLite Catalog (`catalog.sqlite`)**: Content-addressed canonical job caching queryable via `transcribe-music cache list` (`--no-cache` bypasses the database).

## Development & Testing

```bash
uv run pytest
uv run ruff check src tests
```
