"""Optical Music Recognition (OMR) fallback for PDFs via Audiveris CLI or batched homr."""

from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import music21

from transcribe_music.models import Instrument, OmrBackend
from transcribe_music.score_utils import coerce_to_score, find_executable, iter_playable_notes


def render_pdf_pages(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Render every page of a sheet music PDF into opaque RGB PNGs for OMR."""
    if not (150 <= dpi <= 600):
        raise ValueError(f"dpi must be between 150 and 600, got {dpi}")
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Sheet music input must be a .pdf file, got: {pdf_path.suffix}")

    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF OMR fallback requires the optional 'pdf' extra (`pymupdf`). "
            "Install with: uv sync --extra pdf"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    image_paths: list[Path] = []

    with pymupdf.open(str(pdf_path)) as doc:
        if len(doc) == 0:
            raise ValueError(f"PDF has zero pages: {pdf_path}")
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
            page_path = out_dir / f"page_{idx:03d}.png"
            pix.save(str(page_path))
            image_paths.append(page_path)

    return image_paths


def _find_audiveris_executable() -> str | None:
    return find_executable(
        ("audiveris", "Audiveris"),
        (
            Path("/Applications/Audiveris.app/Contents/MacOS/Audiveris"),
            Path.home() / "Applications" / "Audiveris.app" / "Contents" / "MacOS" / "Audiveris",
            Path("/opt/homebrew/bin/audiveris"),
            Path("/usr/local/bin/audiveris"),
        ),
    )


def resolve_omr_backend(preferred: OmrBackend | str = OmrBackend.AUTO) -> Literal["audiveris", "homr"]:
    """Resolve the active PDF OMR backend from user preference and local tooling."""
    backend = OmrBackend(preferred)
    if backend == OmrBackend.AUDIVERIS:
        if _find_audiveris_executable() is None:
            raise RuntimeError(
                "Audiveris OMR was requested, but 'audiveris' CLI was not found. "
                "Install Audiveris or run with --omr-backend homr."
            )
        return "audiveris"
    if backend == OmrBackend.HOMR:
        return "homr"
    return "audiveris" if _find_audiveris_executable() is not None else "homr"


def _run_audiveris_batch(pdf_path: Path, work_dir: Path) -> list[Path]:
    audiveris_bin = _find_audiveris_executable()
    if audiveris_bin is None:
        raise RuntimeError(
            "Audiveris CLI executable not found. Install Audiveris or use --omr-backend homr."
        )

    out_dir = Path(work_dir).resolve() / "audiveris_out"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        audiveris_bin,
        "-batch",
        "-transcribe",
        "-export",
        "-output",
        str(out_dir),
        "--",
        str(Path(pdf_path).resolve()),
    ]
    proc = subprocess.run(cmd, check=False, cwd=str(out_dir), capture_output=True, text=True, timeout=900)

    for ext in ("*.mxl", "*.musicxml"):
        if files := sorted(out_dir.rglob(ext)):
            return files
    if xml_files := sorted(
        p
        for p in out_dir.rglob("*.xml")
        if p.name.lower() != "book.xml" and not p.name.lower().startswith("sheet")
    ):
        return xml_files

    detail = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
    raise RuntimeError(f"Audiveris OMR produced no MusicXML output in {out_dir}: {detail}")


def _build_homr_command(image_paths: list[Path]) -> list[str]:
    resolved = [str(Path(p).resolve()) for p in image_paths]
    if homr_bin := shutil.which("homr"):
        return [homr_bin, *resolved]

    uvx_bin = find_executable(
        ("uvx",),
        (
            Path.home() / ".local" / "bin" / "uvx",
            Path("/opt/homebrew/bin/uvx"),
            Path("/usr/local/bin/uvx"),
        ),
    )
    if uvx_bin is None:
        raise RuntimeError(
            "PDF OMR fallback requires 'homr' or 'uvx' to run homr[cpu] in an isolated environment."
        )
    return [uvx_bin, "--python", "3.14", "--from", "homr[cpu]==0.7.1", "homr", *resolved]


def _run_homr_batch(image_paths: list[Path], work_dir: Path) -> list[Path]:
    if not image_paths:
        raise ValueError("No page images supplied to homr OMR.")

    work_dir = Path(work_dir).resolve()
    for stale in list(work_dir.glob("*.musicxml")) + list(work_dir.glob("*.xml")):
        stale.unlink(missing_ok=True)

    cmd = _build_homr_command(image_paths)
    proc = subprocess.run(
        cmd,
        check=False,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=max(300, 120 * len(image_paths)),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"homr OMR failed: {proc.stderr.strip() or proc.stdout.strip()}")

    first_stem = image_paths[0].stem
    for suffix in (".musicxml", ".xml"):
        if (merged := work_dir / f"merged_{first_stem}{suffix}").exists():
            return [merged]

    per_page: list[Path] = []
    for img in image_paths:
        if (direct := img.with_suffix(".musicxml")).exists():
            per_page.append(direct)
        elif matches := sorted(work_dir.glob(f"{img.stem}*.musicxml")) + sorted(
            work_dir.glob(f"{img.stem}*.xml")
        ):
            per_page.append(matches[0])
    if per_page:
        return per_page

    if any_xml := sorted(work_dir.glob("*.musicxml")) + sorted(work_dir.glob("*.xml")):
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
        score, _ = coerce_to_score(music21.converter.parse(str(page_xml_paths[0])), title)
        return score

    merged_score = music21.stream.Score()
    merged_score.metadata = music21.metadata.Metadata(title=title or f"{instrument.value.title()} Score")
    merged_parts: list[music21.stream.Part] = []
    next_measure_num = 1

    for page_idx, xml_path in enumerate(page_xml_paths):
        parsed, _ = coerce_to_score(music21.converter.parse(str(xml_path)))
        page_parts = list(parsed.parts)

        while len(merged_parts) < len(page_parts):
            merged_parts.append(music21.stream.Part(id=f"P{len(merged_parts) + 1}"))

        page_measure_count = 0
        for p_idx, src_part in enumerate(page_parts):
            dst_part = merged_parts[p_idx]
            measures = list(src_part.getElementsByClass(music21.stream.Measure))
            page_measure_count = max(page_measure_count, len(measures))
            for m_offset, m in enumerate(measures):
                m_copy = copy.deepcopy(m)
                m_copy.number = next_measure_num + m_offset
                if page_idx > 0 and m_offset == 0:
                    for redundant in list(m_copy.getElementsByClass(music21.clef.Clef)):
                        m_copy.remove(redundant)
                dst_part.append(m_copy)

        next_measure_num += max(1, page_measure_count)

    for p in merged_parts:
        merged_score.insert(0, p)
    return merged_score


def _is_omr_tab_part(part: music21.stream.Part) -> bool:
    if any(isinstance(c, music21.clef.TabClef) for c in part.recurse().getElementsByClass(music21.clef.Clef)):
        return True
    return any(
        getattr(sl, "staffLines", None) == 6
        for sl in part.recurse().getElementsByClass(music21.layout.StaffLayout)
    )


def _filter_guitar_omr_tab_staves(score: music21.stream.Score, warnings: list[str]) -> None:
    parts = list(score.parts)
    if not parts:
        return
    notation_parts = [p for p in parts if not _is_omr_tab_part(p) and any(iter_playable_notes(p))]
    tab_parts = [p for p in parts if _is_omr_tab_part(p)]
    if notation_parts and tab_parts:
        for tp in tab_parts:
            score.remove(tp)
        warnings.append(
            "Filtered out OMR-recognized TAB staff in favor of paired standard notation staff."
        )
    elif not notation_parts:
        warnings.append(
            "PDF appears to contain guitar TAB or unrecognized staves; prefer digital .gp/.gp5/.tab/.mxl input for exact TAB accuracy."
        )


def _finalize_omr_score(
    xml_paths: list[Path],
    instrument: Instrument,
    title: str,
    warnings: list[str],
) -> music21.stream.Score:
    score = stitch_musicxml_pages(xml_paths, instrument=instrument, title=title)
    if instrument == Instrument.GUITAR:
        _filter_guitar_omr_tab_staves(score, warnings)
    return score


def run_pdf_omr(
    pdf_path: Path,
    work_dir: Path,
    instrument: Instrument,
    dpi: int = 300,
    omr_backend: OmrBackend | str = OmrBackend.AUTO,
) -> tuple[music21.stream.Score, str, list[str]]:
    """Run PDF OMR (Audiveris CLI preferred when available, else batched homr) into a music21 Score."""
    pref = OmrBackend(omr_backend)
    selected = resolve_omr_backend(pref)
    warnings: list[str] = []

    if selected == "audiveris":
        try:
            xml_paths = _run_audiveris_batch(pdf_path, work_dir)
            return _finalize_omr_score(xml_paths, instrument, pdf_path.stem, warnings), "audiveris", warnings
        except Exception as exc:
            if pref == OmrBackend.AUDIVERIS:
                raise
            warnings.append(f"Audiveris OMR failed ({exc}); fell back to homr.")

    pages_dir = Path(work_dir) / "pdf_pages"
    image_paths = render_pdf_pages(pdf_path, pages_dir, dpi=dpi)
    xml_paths = _run_homr_batch(image_paths, pages_dir)
    return _finalize_omr_score(xml_paths, instrument, pdf_path.stem, warnings), "homr", warnings
