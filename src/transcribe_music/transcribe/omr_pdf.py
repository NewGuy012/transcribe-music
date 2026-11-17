"""Optical Music Recognition (OMR) fallback for multi-page sheet music PDFs via batched homr."""

from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import music21

from transcribe_music.ingest.pdf import render_pdf_pages
from transcribe_music.models import Instrument


def _build_homr_command(image_paths: list[Path]) -> list[str]:
    """Build a single batched CLI invocation for homr across one or more rendered pages."""
    resolved = [str(Path(p).resolve()) for p in image_paths]
    homr_bin = shutil.which("homr")
    if homr_bin:
        return [homr_bin, *resolved]

    uvx_bin = shutil.which("uvx") or "/opt/homebrew/bin/uvx"
    if not Path(uvx_bin).exists() and shutil.which("uvx") is None:
        raise RuntimeError(
            "PDF OMR fallback requires 'homr' or 'uvx' to run homr[cpu] in an isolated environment."
        )
    return [uvx_bin, "--python", "3.14", "--from", "homr[cpu]", "homr", *resolved]


def _run_homr_batch(image_paths: list[Path], work_dir: Path) -> list[Path]:
    """Run homr OMR once across all PDF pages and return generated MusicXML path(s)."""
    if not image_paths:
        raise ValueError("No page images supplied to homr OMR.")

    work_dir = Path(work_dir).resolve()
    for stale in list(work_dir.glob("*.musicxml")) + list(work_dir.glob("*.xml")):
        stale.unlink(missing_ok=True)

    cmd = _build_homr_command(image_paths)
    timeout_sec = max(300, 120 * len(image_paths))
    proc = subprocess.run(
        cmd, check=False, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout_sec
    )
    if proc.returncode != 0:
        raise RuntimeError(f"homr OMR failed: {proc.stderr.strip() or proc.stdout.strip()}")

    first_stem = image_paths[0].stem
    merged_candidates = [
        work_dir / f"merged_{first_stem}.musicxml",
        work_dir / f"merged_{first_stem}.xml",
    ]
    for merged in merged_candidates:
        if merged.exists():
            return [merged]

    per_page: list[Path] = []
    for img in image_paths:
        direct = img.with_suffix(".musicxml")
        if direct.exists():
            per_page.append(direct)
            continue
        matches = sorted(work_dir.glob(f"{img.stem}*.musicxml")) + sorted(
            work_dir.glob(f"{img.stem}*.xml")
        )
        if matches:
            per_page.append(matches[0])

    if per_page:
        return per_page

    any_xml = sorted(work_dir.glob("*.musicxml")) + sorted(work_dir.glob("*.xml"))
    if any_xml:
        return any_xml

    raise RuntimeError(f"homr produced no MusicXML output in {work_dir}")


def stitch_musicxml_pages(
    page_xml_paths: list[Path],
    instrument: Instrument,
    title: str | None = None,
) -> music21.stream.Score:
    """Merge multi-page MusicXML files into one continuous Score with sequential measures."""
    if not page_xml_paths:
        raise ValueError("No page MusicXML files to stitch")

    if len(page_xml_paths) == 1:
        parsed = music21.converter.parse(str(page_xml_paths[0]))
        if isinstance(parsed, music21.stream.Part):
            wrapper = music21.stream.Score()
            wrapper.insert(0, parsed)
            parsed = wrapper
        if parsed.metadata is None:
            parsed.metadata = music21.metadata.Metadata()
        if title and not parsed.metadata.title:
            parsed.metadata.title = title
        return parsed

    merged_score = music21.stream.Score()
    merged_score.metadata = music21.metadata.Metadata(title=title or f"{instrument.value.title()} Score")
    merged_parts: list[music21.stream.Part] = []

    next_measure_num = 1
    for page_idx, xml_path in enumerate(page_xml_paths):
        parsed = music21.converter.parse(str(xml_path))
        page_parts = list(parsed.parts) if isinstance(parsed, music21.stream.Score) else [parsed]

        if not merged_parts:
            for p_idx in range(len(page_parts)):
                merged_parts.append(music21.stream.Part(id=f"P{p_idx + 1}"))

        page_measure_count = 0
        for p_idx, src_part in enumerate(page_parts):
            if p_idx >= len(merged_parts):
                merged_parts.append(music21.stream.Part(id=f"P{p_idx + 1}"))
            dst_part = merged_parts[p_idx]

            measures = list(src_part.getElementsByClass(music21.stream.Measure))
            page_measure_count = max(page_measure_count, len(measures))
            for m_offset, m in enumerate(measures):
                m_copy = copy.deepcopy(m)
                m_copy.number = next_measure_num + m_offset
                # Strip redundant clefs on subsequent page boundaries
                if page_idx > 0 and m_offset == 0:
                    for redundant in list(m_copy.getElementsByClass((music21.clef.Clef,))):
                        m_copy.remove(redundant)
                dst_part.append(m_copy)

        next_measure_num += max(1, page_measure_count)

    for p in merged_parts:
        merged_score.insert(0, p)
    return merged_score


def transcribe_pdf_omr(
    pdf_path: Path,
    work_dir: Path,
    instrument: Instrument,
    dpi: int = 300,
) -> music21.stream.Score:
    """Rasterize PDF pages and run a single batched homr OMR pass into a music21 Score."""
    pages_dir = Path(work_dir) / "pdf_pages"
    image_paths = render_pdf_pages(pdf_path, pages_dir, dpi=dpi)
    xml_paths = _run_homr_batch(image_paths, pages_dir)
    return stitch_musicxml_pages(xml_paths, instrument=instrument, title=pdf_path.stem)
