"""
Diphone vocabulary and label generation for context-aware phoneme decoding.

Based on the DCoND paper approach: using diphones as auxiliary supervision
to encourage context-aware representations.
"""

import pickle
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np


class DiphoneVocabulary:
    """
    Builds and manages a diphone vocabulary from phoneme sequences.

    A diphone is a pair (prev_phone, curr_phone) representing phonetic context.
    """

    def __init__(self, boundary_id: int = 0):
        """
        Args:
            boundary_id: Phoneme ID to use for sequence boundaries (e.g., SIL or <BND>)
        """
        self.boundary_id = boundary_id
        self.diphone_to_id: Dict[Tuple[int, int], int] = {}
        self.id_to_diphone: Dict[int, Tuple[int, int]] = {}
        self.diphone_counts: Counter = Counter()
        self.num_diphones = 0
        self.blank_id = None  # Will be set after building vocab

    def build_from_data(self, data: List[Dict], min_count: int = 1):
        """
        Build diphone vocabulary from training data.

        Args:
            data: List of day dictionaries with 'phonemes' key
            min_count: Minimum count to include a diphone (set to 1 to keep all)
        """
        print("Building diphone vocabulary...")

        # Collect all diphones from training data
        all_diphones = []
        for day in data:
            for phone_seq in day["phonemes"]:
                diphones = self._sequence_to_diphones(phone_seq)
                all_diphones.extend(diphones)

        # Count diphone frequencies
        self.diphone_counts = Counter(all_diphones)
        print(f"Found {len(self.diphone_counts)} unique diphones")

        # Filter by minimum count
        filtered_diphones = [
            diphone for diphone, count in self.diphone_counts.items()
            if count >= min_count
        ]

        # Sort for consistency
        sorted_diphones = sorted(filtered_diphones)

        # Build vocabulary mappings
        self.diphone_to_id = {
            diphone: idx for idx, diphone in enumerate(sorted_diphones)
        }
        self.id_to_diphone = {
            idx: diphone for diphone, idx in self.diphone_to_id.items()
        }

        self.num_diphones = len(self.diphone_to_id)

        # CTC blank token is the last ID
        self.blank_id = self.num_diphones

        print(f"Diphone vocabulary size: {self.num_diphones} (+ 1 blank = {self.num_diphones + 1})")

        return self

    def _sequence_to_diphones(self, phone_seq: np.ndarray) -> List[Tuple[int, int]]:
        """
        Convert a phoneme sequence to diphone pairs.

        Args:
            phone_seq: Array of phoneme IDs [p1, p2, ..., pL]

        Returns:
            List of diphone tuples [(BND, p1), (p1, p2), ..., (pL, BND)]
        """
        # Extend with boundary phonemes
        extended = [self.boundary_id] + list(phone_seq) + [self.boundary_id]

        # Create diphone pairs
        diphones = []
        for i in range(len(extended) - 1):
            prev_id = int(extended[i])
            curr_id = int(extended[i + 1])
            diphones.append((prev_id, curr_id))

        return diphones

    def phoneme_to_diphone_labels(self, phone_seq: np.ndarray) -> np.ndarray:
        """
        Convert a phoneme sequence to diphone label sequence for CTC.

        Args:
            phone_seq: Array of phoneme IDs [p1, p2, ..., pL]

        Returns:
            Array of diphone IDs for CTC training
        """
        diphones = self._sequence_to_diphones(phone_seq)

        # Convert to IDs (handle out-of-vocabulary by skipping)
        diphone_ids = []
        for diphone in diphones:
            if diphone in self.diphone_to_id:
                diphone_ids.append(self.diphone_to_id[diphone])
            # Skip OOV diphones (rare pairs not in training vocab)

        return np.array(diphone_ids, dtype=np.int32)

    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'boundary_id': self.boundary_id,
                'diphone_to_id': self.diphone_to_id,
                'id_to_diphone': self.id_to_diphone,
                'diphone_counts': self.diphone_counts,
                'num_diphones': self.num_diphones,
                'blank_id': self.blank_id,
            }, f)
        print(f"Saved diphone vocabulary to {path}")

    @classmethod
    def load(cls, path: str):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        vocab = cls(boundary_id=data['boundary_id'])
        vocab.diphone_to_id = data['diphone_to_id']
        vocab.id_to_diphone = data['id_to_diphone']
        vocab.diphone_counts = data['diphone_counts']
        vocab.num_diphones = data['num_diphones']
        vocab.blank_id = data['blank_id']

        print(f"Loaded diphone vocabulary with {vocab.num_diphones} diphones")
        return vocab

    def get_vocab_size(self) -> int:
        """Get total vocabulary size including blank token."""
        return self.num_diphones + 1  # +1 for blank

    def get_num_phonemes(self) -> int:
        """Get the number of unique phonemes (including boundary/blank)."""
        # Find all unique phoneme IDs from diphone pairs
        phonemes = set()
        for (prev, curr) in self.diphone_to_id.keys():
            phonemes.add(prev)
            phonemes.add(curr)
        return len(phonemes)

    def create_marginalization_matrix(self, num_phonemes: int, phoneme_blank_id: int) -> np.ndarray:
        """
        Create a marginalization matrix to convert diphone probabilities to phoneme probabilities.

        For each diphone (i, j), we want P(phoneme_j) to include this diphone's probability.
        So the matrix has a 1.0 at position [diphone_id, j].

        Args:
            num_phonemes: Total number of phoneme classes (including blank)
            phoneme_blank_id: The blank token ID for phonemes (usually the last index)

        Returns:
            Matrix of shape [num_diphones+1, num_phonemes] where entry [d, p] = 1.0
            if diphone d has current phoneme p, else 0.0
        """
        # Initialize marginalization matrix
        # Shape: [num_diphones + 1 (for blank), num_phonemes]
        marg_matrix = np.zeros((self.num_diphones + 1, num_phonemes), dtype=np.float32)

        # For each diphone (prev, curr) at index d, set matrix[d, curr] = 1.0
        for (prev, curr), diphone_id in self.diphone_to_id.items():
            # The diphone (prev, curr) contributes to phoneme 'curr'
            marg_matrix[diphone_id, curr] = 1.0

        # The diphone blank token maps to the phoneme blank token
        marg_matrix[self.blank_id, phoneme_blank_id] = 1.0

        return marg_matrix


def build_and_save_diphone_vocab(
    dataset_path: str,
    output_path: str,
    boundary_id: int = 0,
    min_count: int = 1,
):
    """
    Convenience function to build and save diphone vocabulary.

    Args:
        dataset_path: Path to pickled dataset
        output_path: Where to save the vocabulary
        boundary_id: ID of boundary phoneme (0 for blank/SIL)
        min_count: Minimum diphone frequency to include
    """
    import pickle

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    vocab = DiphoneVocabulary(boundary_id=boundary_id)
    vocab.build_from_data(data['train'], min_count=min_count)
    vocab.save(output_path)

    # Print some statistics
    print(f"\nTop 20 most common diphones:")
    for (prev, curr), count in vocab.diphone_counts.most_common(20):
        prev_name = f"P{prev}" if prev != boundary_id else "BND"
        curr_name = f"P{curr}" if curr != boundary_id else "BND"
        print(f"  ({prev_name}, {curr_name}): {count}")

    return vocab


if __name__ == "__main__":
    # Build diphone vocabulary from training data
    dataset_path = "/home/edward/neural_seq_decoder/ptDecoder_ctc"
    output_path = "/home/edward/neural_seq_decoder/diphone_vocab.pkl"

    vocab = build_and_save_diphone_vocab(
        dataset_path=dataset_path,
        output_path=output_path,
        boundary_id=0,  # Use phoneme ID 0 as boundary
        min_count=1,     # Keep all diphones
    )
