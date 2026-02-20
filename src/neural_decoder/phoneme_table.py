"""
Shared phoneme definitions used across the decoding pipeline.

Phoneme IDs in the CTC model:
  0 = CTC blank
  1..39 = the 39 ARPABET phonemes (AA, AE, ... ZH)
  40 = SIL (silence / word boundary)

This module provides canonical mappings so every other module can
import them instead of hard-coding.
"""

from typing import Dict, List

# 39 ARPABET phonemes (order must match formatCompetitionData.ipynb)
PHONE_DEF: List[str] = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

# With silence token appended
PHONE_DEF_SIL: List[str] = PHONE_DEF + ["SIL"]

# CTC blank index
BLANK_IDX: int = 0

# Number of actual phoneme classes (including SIL, excluding blank)
N_PHONEMES: int = len(PHONE_DEF_SIL)  # 40

# Total vocabulary size used by the CTC model (blank + phonemes)
N_CLASSES_CTC: int = N_PHONEMES + 1  # 41

# SIL index in the CTC model (1-indexed since 0 is blank)
SIL_IDX: int = N_PHONEMES  # 40

# Mapping from phoneme string to CTC model index (1-based for phonemes)
PHONE_TO_ID: Dict[str, int] = {p: i + 1 for i, p in enumerate(PHONE_DEF_SIL)}

# Mapping from CTC model index back to phoneme string
ID_TO_PHONE: Dict[int, str] = {v: k for k, v in PHONE_TO_ID.items()}
ID_TO_PHONE[BLANK_IDX] = "<blank>"


def phone_ids_to_str(
    ids: List[int], skip_blank: bool = True, skip_sil: bool = False
) -> str:
    """Convert a list of phoneme IDs to a human-readable string."""
    parts = []
    for i in ids:
        if skip_blank and i == BLANK_IDX:
            continue
        if skip_sil and i == SIL_IDX:
            continue
        parts.append(ID_TO_PHONE.get(i, f"<unk:{i}>"))
    return " ".join(parts)
