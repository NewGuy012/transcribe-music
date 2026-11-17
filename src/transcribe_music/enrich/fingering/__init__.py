"""Instrument-specific fingering solvers."""

from __future__ import annotations

import music21

from transcribe_music.enrich.fingering.guitar import annotate_guitar_fingering
from transcribe_music.enrich.fingering.piano import annotate_piano_fingering
from transcribe_music.enrich.fingering.strings import annotate_bowed_string_fingering
from transcribe_music.models import Instrument


def apply_fingering(score: music21.stream.Score, instrument: Instrument) -> music21.stream.Score:
    """Dispatch deterministic ergonomic fingering enrichment by target instrument."""
    if instrument == Instrument.PIANO:
        return annotate_piano_fingering(score)
    if instrument == Instrument.GUITAR:
        return annotate_guitar_fingering(score)
    if instrument in (Instrument.VIOLIN, Instrument.CELLO):
        return annotate_bowed_string_fingering(score, instrument)
    return score
