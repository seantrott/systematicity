"""Create dataset."""

import re
import pandas as pd

ROOT_PATH = 'data/raw/roots_celex_monosyllabic.txt'

def construct_syllable_structure(syllable, nuclei='[5@694{8312i7u$#eqFEIQVU$]'):
    """Return dict of possible onset, nucleus, and coda, using phonetic transcription."""
    nucleus = re.findall(nuclei, syllable)
    if len(nucleus) < 1:
        return None
    onset, coda = syllable.split(nucleus[0])
    return {'nucleus': nucleus[0],
            'onset': onset,
            'coda': coda}


def build_dataset(path=ROOT_PATH, min_length=3):
    """Return words, along with syllable structure."""
    entries = open(path, "r").read().split("\n")
    words = [(entry.split("\\")[0], entry.split("\\")[-1]) for entry in entries if entry != ""]
    words = [(w[0], construct_syllable_structure(w[1])) for w in words if len(w[0]) >= min_length]
    return dict(words)


data = build_dataset()