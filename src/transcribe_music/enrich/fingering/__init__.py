"""Instrument-specific fingering solvers."""

from __future__ import annotations

import music21

from transcribe_music.enrich.fingering.guitar import annotate_guitar_fingering
from transcribe_music.enrich.fingering.piano import annotate_piano_fingering
from transcribe_music.enrich.fingering.strings import annotate_bowed_string_fingering
from transcribe_music.models import Instrument
from transcribe_music.score_utils import iter_playable_notes


def _has_existing_fingering(score: music21.stream.Score, instrument: Instrument) -> bool:
    target_types: tuple[type[music21.articulations.Articulation], ...]
    if instrument == Instrument.GUITAR:
        target_types = (
            music21.articulations.Fingering,
            music21.articulations.StringIndication,
            music21.articulations.FretIndication,
        )
    else:
        target_types = (music21.articulations.Fingering,)

    for part in score.parts:
        for n in iter_playable_notes(part):
            if any(isinstance(a, target_types) for a in n.articulations):
                return True
    return False


def apply_fingering(
    score: music21.stream.Score,
    instrument: Instrument,
    *,
    preserve_existing: bool = False,
) -> music21.stream.Score:
    """Dispatch deterministic ergonomic fingering enrichment by target instrument."""
    if preserve_existing and _has_existing_fingering(score, instrument):
        return score
    if instrument == Instrument.PIANO:
        return annotate_piano_fingering(score)
    if instrument == Instrument.GUITAR:
        return annotate_guitar_fingering(score)
    if instrument in (Instrument.VIOLIN, Instrument.CELLO):
        return annotate_bowed_string_fingering(score, instrument)
    return score
