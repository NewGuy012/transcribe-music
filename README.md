# transcribe-music

A fast, local-first Python CLI, SDK, and **Marimo** workbench that converts **YouTube videos**, **audio files**, or **sheet music PDFs** into validated **MusicXML** containing **notes, durations, measures, chord symbols (`<harmony>`), guitar TAB (`<string>`, `<fret>`), and ergonomic fingerings (`<fingering>`)** for:

- **Piano** (Grand Staff split, bundled `Transkun` v2 + `pianoplayer` in-memory optimization)
- **Guitar** (Treble `8vb` transposition, fast `MR-MT3` + 6-string Viterbi TAB & chord-grip solver)
- **Violin** (Solo/concerto salience pruning + 4-string Viterbi position solver)
- **Cello** (Solo/concerto salience pruning + `Tutti` rest cues + 4-string closed/extended/thumb solver)

Powered by modern Python tooling: **`uv`**, **`duckdb`** (content-addressed stage caching + queryable score tables), and **`marimo` + `verovio`** (reactive UI with inline SVG engraving).

---

## Quickstart

Requires **uv** (Python `3.15` managed automatically via `.python-version`). Zero system installs required (`pymupdf` handles PDFs, `static-ffmpeg` auto-fetches FFmpeg if absent, `verovio` renders scores).

```bash
uv sync --all-extras
```

### CLI Usage

```bash
# 1. YouTube video (defaults to the whole video; pass optional --start and/or --end to slice)
uv run transcribe-music youtube "https://www.youtube.com/watch?v=..." \
  --instrument cello -o suite_prelude.musicxml
uv run transcribe-music youtube "https://www.youtube.com/watch?v=..." \
  --start 0:42 --end 2:15 --instrument cello -o suite_prelude_slice.musicxml

# 2. Full local audio file (WAV / MP3 / FLAC / M4A)
uv run transcribe-music audio guitar_solo.mp3 \
  --instrument guitar -o guitar_solo.musicxml

# 3. Piano or cello concerto excerpt (isolates soloist track & marks Tutti rests)
uv run transcribe-music audio concerto.mp3 \
  --instrument piano --mode concerto -o concerto_piano.musicxml

# 4. Sheet music PDF (.pdf only; runs homr Vision Transformer OMR via uvx Python 3.14)
uv run transcribe-music pdf score.pdf \
  --instrument piano -o score_fingered.musicxml

# 5. Launch the Marimo Interactive Workbench + Verovio SVG viewer + DuckDB explorer
uv run transcribe-music ui

# 6. Inspect or clear the DuckDB cache
uv run transcribe-music cache list
uv run transcribe-music cache clear
```

### Python SDK Usage

```python
from transcribe_music import Instrument, transcribe_audio, transcribe_pdf, transcribe_youtube

res = transcribe_youtube(
    url="https://www.youtube.com/watch?v=...",
    instrument=Instrument.GUITAR,
    start="1:05",  # optional; omit start & end for the whole video
    end="2:30",
    output_path="out/guitar.musicxml",
)
print(res.musicxml_path, res.note_count, res.chord_count, res.fingering_count)
```

---

## Architecture Highlights

1. **Guaranteed Baseline MusicXML Contract**: Every score passes through `music21` structural normalization (`makeMeasures`, `makeRests`, clefs, transpositions, time/key signatures) and XML validation so downstream apps always receive valid notes, durations, and complete measures.
2. **Progressive Non-Destructive Enrichment**:
   - **Chords (`<harmony>`)**: Hybrid symbolic + acoustic CQT chroma Viterbi HMM.
   - **Fingerings (`<technical><fingering>`)**: In-memory `pianoplayer` (`Piano`), 6-string fretboard + TAB (`Guitar`), and 4-string biomechanical Viterbi solvers (`Violin`, `Cello` including thumb position `T`).
3. **DuckDB Catalog (`catalog.duckdb`)**: Content-addressed job caching plus relational `jobs`, `measures`, `notes`, and `chords` tables queryable via SQL or the Marimo UI.

## Development & Testing

```bash
uv run pytest
uv run ruff check src tests notebooks
```
