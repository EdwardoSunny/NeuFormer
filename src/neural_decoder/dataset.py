import re

import numpy as np
import torch
from torch.utils.data import Dataset


# Character vocabulary for dual-task CTC
# 0 = CTC blank, 1-26 = a-z, 27 = apostrophe, 28 = space (word boundary)
CHAR_VOCAB = ["-"] + list("abcdefghijklmnopqrstuvwxyz") + ["'", "|"]
CHAR_BLANK = 0
CHAR_SPACE = len(CHAR_VOCAB) - 1  # 28 = "|"
N_CHARS = len(CHAR_VOCAB)  # 29

CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}


def text_to_char_ids(text: str) -> np.ndarray:
    """Convert a transcription string to character IDs for CTC.

    Words are separated by the space token ``|``.
    Characters not in the vocabulary are skipped.
    """
    text = re.sub(r"[^a-zA-Z' ]", "", text).lower().strip()
    ids = []
    for ch in text:
        if ch == " ":
            ids.append(CHAR_SPACE)
        elif ch in CHAR_TO_ID:
            ids.append(CHAR_TO_ID[ch])
    return np.array(ids, dtype=np.int32)


class SpeechDataset(Dataset):
    def __init__(self, data, transform=None, load_chars=False):
        self.data = data
        self.transform = transform
        self.load_chars = load_chars
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        self.char_seqs = []
        self.char_seq_lens = []

        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

                if load_chars and "transcriptions" in data[day]:
                    tx = str(data[day]["transcriptions"][trial]).strip()
                    char_ids = text_to_char_ids(tx)
                    self.char_seqs.append(char_ids)
                    self.char_seq_lens.append(len(char_ids))
                elif load_chars:
                    # No transcription available — empty sequence
                    self.char_seqs.append(np.array([], dtype=np.int32))
                    self.char_seq_lens.append(0)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        items = (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )

        if self.load_chars:
            items = items + (
                torch.tensor(self.char_seqs[idx], dtype=torch.int32),
                torch.tensor(self.char_seq_lens[idx], dtype=torch.int32),
            )

        return items
