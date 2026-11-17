"""Optical Music Recognition (OMR) for multi-page sheet music PDFs using homr."""

from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import music21

from transcribe_music.ingest.pdf import render_pdf_pages
from transcribe_music.models import Instrument


def _run_homr_on_image(image_path: Path) -> Path:
    """Run homr OMR on one page PNG via native CLI or isolated uvx Python 3.14."""
    expected_xml = image_path.with_suffix(".musicxml")
    if expected_xml.exists():
        expected_xml.unlink()

    homr_bin = shutil.which("homr")
    if homr_bin:
        cmd = [homr_bin, str(image_path)]
    else:
        uvx_bin = shutil.which("uvx") or "/opt/homebrew/bin/uvx"
        if not Path(uvx_bin).exists() and shutil.which("uvx") is None:
            raise RuntimeError(
                "PDF OMR requires 'homr' or 'uvx' to run homr[cpu] in an isolated environment."
            )
        cmd = [
            uvx_bin,
            "--python",
            "3.14",
            "--from",
            "homr[cpu]",
            "homr",
            str(image_path),
        ]

    proc = subprocess.run(cmd, check=False, cwd=str(image_path.parent), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"homr OMR failed on {image_path.name}: {proc.stderr.strip()}")

    if not expected_xml.exists():
        candidates = sorted(image_path.parent.glob(f"{image_path.stem}*.musicxml")) + sorted(
            image_path.parent.glob(f"{image_path.stem}*.xml")
        )
        if not candidates:
            raise RuntimeError(f"homr produced no MusicXML output for {image_path.name}")
        expected_xml = candidates[0]

    return expected_xml


def stitch_musicxml_pages(
    page_xml_paths: list[Path],
    instrument: Instrument,
    title: str | None = None,
) -> music21.stream.Score:
    """Merge multi-page MusicXML files into one continuous Score with sequential measures."""
    if not page_xml_paths:
        raise ValueError("No page MusicXML files to stitch")

    if len(page_xml_paths) == 1:
        score = music21.converter.parse(str(page_xml_paths[0]))
        if isinstance(score, music21.stream.Part):
            wrapper = music21.stream.Score()
            wrapper.insert(0, score)
            score = wrapper
        return score

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
                # Strip redundant clefs/key/time signatures on subsequent pages unless changed
                if page_idx > 0 and m_offset == 0:
                    for redundant in list(
                        m_copy.getElementsByClass((music21.clef.Clef,))
                    ):
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
    """Rasterize PDF pages and run homr OMR to produce a unified music21 Score."""
    pages_dir = Path(work_dir) / "pdf_pages"
    image_paths = render_pdf_pages(pdf_path, pages_dir, dpi=dpi)
    xml_paths = [_run_homr_on_image(img) for img in image_paths]
    return stitch_musicxml_pages(xml_paths, instrument=instrument, title=pdf_path.stem)
