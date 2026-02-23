"""
Set up the lm_dir/ for CTC beam search decoding with KenLM.

Downloads the LibriSpeech 4-gram KenLM model via torchaudio and ensures
lexicon.txt and tokens.txt are in place.

Usage
-----
    uv run python scripts/setup_lm_dir.py

After running, lm_dir/ will contain:
    lexicon.txt   125k words → ARPABET phoneme sequences
    tokens.txt    41 CTC tokens (blank + 39 phones + SIL)
    lm.bin        LibriSpeech 4-gram KenLM binary (~3GB)

Then run eval with:
    uv run python scripts/eval_posterior_constrained.py \\
        --model_path logs/speech_logs/conformer_v2 \\
        --dataset_path ptDecoder_ctc \\
        --partition test \\
        --lm_dir lm_dir \\
        --lm_weight 3.23 \\
        --beam_size 1500
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

LM_DIR = os.path.join(os.path.dirname(__file__), "..", "lm_dir")


def main():
    os.makedirs(LM_DIR, exist_ok=True)

    # ---- 1. Download LibriSpeech 4-gram KenLM ----
    lm_dst = os.path.join(LM_DIR, "lm.bin")
    if os.path.exists(lm_dst) and os.path.getsize(lm_dst) > 1_000_000:
        print(
            f"lm.bin already exists ({os.path.getsize(lm_dst) / 1e9:.1f} GB), skipping download."
        )
    else:
        print("Downloading LibriSpeech 4-gram KenLM (~3 GB) ...")
        from torchaudio.models.decoder import download_pretrained_files

        files = download_pretrained_files("librispeech-4-gram")

        # Remove any stale symlink
        if os.path.islink(lm_dst):
            os.remove(lm_dst)

        shutil.copy(files.lm, lm_dst)
        print(f"Copied lm.bin to {lm_dst} ({os.path.getsize(lm_dst) / 1e9:.1f} GB)")

    # ---- 2. Generate tokens.txt ----
    tok_dst = os.path.join(LM_DIR, "tokens.txt")
    if os.path.exists(tok_dst):
        print(f"tokens.txt already exists, skipping.")
    else:
        from neural_decoder.ngram_decoder import _write_tokens_file

        _write_tokens_file(tok_dst)
        print(f"Wrote {tok_dst}")

    # ---- 3. Generate lexicon.txt from lexicon_numbers.txt ----
    lex_dst = os.path.join(LM_DIR, "lexicon.txt")
    if os.path.exists(lex_dst):
        print(f"lexicon.txt already exists, skipping.")
    else:
        # Look for lexicon_numbers.txt in common locations
        candidates = [
            os.path.join(LM_DIR, "lexicon_numbers.txt"),
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "lm_extracted",
                "languageModel",
                "lexicon_numbers.txt",
            ),
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "language_model",
                "pretrained_language_models",
                "languageModel",
                "lexicon_numbers.txt",
            ),
        ]
        num_lex = None
        for c in candidates:
            if os.path.exists(c):
                num_lex = c
                break

        if num_lex:
            from neural_decoder.ngram_decoder import _convert_numeric_lexicon

            _convert_numeric_lexicon(num_lex, lex_dst)
            print(f"Converted {num_lex} -> {lex_dst}")
        else:
            print(
                "WARNING: No lexicon_numbers.txt found. "
                "Place it in lm_dir/ and re-run, or provide lexicon.txt manually."
            )
            print("Searched:", candidates)

    # ---- Done ----
    print()
    print("lm_dir contents:")
    for f in sorted(os.listdir(LM_DIR)):
        path = os.path.join(LM_DIR, f)
        size = (
            os.path.getsize(path) if not os.path.islink(path) else os.path.getsize(path)
        )
        if size > 1e9:
            print(f"  {f:30s} {size / 1e9:.1f} GB")
        elif size > 1e6:
            print(f"  {f:30s} {size / 1e6:.1f} MB")
        else:
            print(f"  {f:30s} {size / 1e3:.1f} KB")

    print()
    print("Done! Now run:")
    print()
    print("  uv run python scripts/eval_posterior_constrained.py \\")
    print("      --model_path logs/speech_logs/conformer_v2 \\")
    print("      --dataset_path ptDecoder_ctc \\")
    print("      --partition test \\")
    print("      --lm_dir lm_dir \\")
    print("      --lm_weight 3.23 \\")
    print("      --beam_size 1500")


if __name__ == "__main__":
    main()
