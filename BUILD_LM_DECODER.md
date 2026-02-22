# Building the `lm_decoder` C++ Module

The WFST n-gram decoder (`lm_decoder`) is a pybind11 C++ extension from the NEJM
brain-to-text repo. It provides Kaldi-based CTC-WFST beam search with n-gram LM
integration. This document describes how to build and install it.

## Prerequisites

| Requirement   | Version           | Notes                              |
|---------------|-------------------|------------------------------------|
| Python        | 3.9               | Must match exactly (pybind11 ABI)  |
| PyTorch       | 1.13.1 (CPU)      | libtorch is downloaded by CMake    |
| CMake         | >= 3.14           | `sudo apt install cmake`           |
| GCC           | >= 10.1           | `sudo apt install build-essential` |
| zlib          | any               | `sudo apt install zlib1g-dev`      |

## Quick Setup (conda)

This creates a dedicated conda environment and builds the module:

```bash
# 1. Create and activate a Python 3.9 environment
conda create -n lm_decoder python=3.9 -y
conda activate lm_decoder

# 2. Install Python dependencies
pip install --upgrade pip
pip install torch==1.13.1 numpy==1.24.4

# 3. Navigate to the NEJM repo's build directory
cd /path/to/nejm-brain-to-text/language_model/runtime/server/x86

# 4. Clean any previous builds
rm -rf build/ fc_base/

# 5. Build and install
python setup.py install

# 6. Verify
python -c "import lm_decoder; print('lm_decoder imported successfully')"
```

## What the Build Does

The `setup.py` invokes CMake which:

1. Downloads and builds these dependencies via `FetchContent`:
   - **gflags** 2.2.1
   - **glog** 0.4.0
   - **googletest** 1.10.0
   - **boost** 1.75.0 (headers only)
   - **cnpy** (NumPy file I/O)
   - **libtorch** 1.13.1 CPU
   - **OpenFST** 1.6.5 (built from source with custom patches)
   - **hiredis** + **redis-plus-plus** (Redis client, linked but not used in our pipeline)

2. Compiles the Kaldi lattice decoder sources (under `kaldi/`)

3. Compiles the decoder library (`brain_speech_decoder.cc`, `ctc_wfst_beam_search.cc`, etc.)

4. Builds the `lm_decoder` pybind11 module (`python/lm_decoder.cc`)

The first build takes ~10-15 minutes due to downloading and compiling dependencies.
Subsequent builds are fast if the `fc_base/` directory is preserved.

## Important Notes

### ABI Compatibility

The `CMakeLists.txt` sets `-D_GLIBCXX_USE_CXX11_ABI=0` to match PyTorch's ABI.
If you get symbol errors at import time, ensure your PyTorch was built with the
same ABI setting:

```python
import torch
print(torch._C._GLIBCXX_USE_CXX11_ABI)  # Should print False
```

### Using with a Different Python Environment

The `lm_decoder.*.so` shared library is specific to the Python version it was
built against. If you need to use it in a different environment (e.g., Python 3.12),
you can copy the `.so` file to your `sys.path`, but it will only work if the
Python minor version matches (3.9).

For a different Python version, you would need to rebuild with that Python.

### Required FST Files

After building, you need pre-compiled FST files to actually run the decoder:

- `TLG.fst` — Token + Lexicon + Grammar (required)
- `words.txt` — Word symbol table (required)
- `G.fst` — Grammar FST for lattice rescoring (optional)
- `G_no_prune.fst` — Unpruned grammar for better rescoring (optional)

The NEJM repo includes a 1-gram model at:
```
language_model/pretrained_language_models/openwebtext_1gram_lm_sil/
```

Higher-order n-gram models (3-gram, 5-gram) produce better results but require
more memory and are not included in the repo.

## API Reference

Once installed, the module provides:

```python
import lm_decoder

# Configuration
opts = lm_decoder.DecodeOptions(
    max_active=7000,
    min_active=200,
    beam=17.0,
    lattice_beam=8.0,
    acoustic_scale=0.3,
    ctc_blank_skip_threshold=1.0,
    length_penalty=0.0,
    nbest=100,
)

resource = lm_decoder.DecodeResource(
    TLG_path,           # Path to TLG.fst
    G_path,             # Path to G.fst (or "" to skip)
    rescore_G_path,     # Path to G_no_prune.fst (or "" to skip)
    words_path,         # Path to words.txt
    unit_path,          # Path to units.txt (or "" for speech)
)

# Create decoder
decoder = lm_decoder.BrainSpeechDecoder(resource, opts)

# Decode
#   logits: [T, 41] float32 — raw logits (log_softmax applied internally)
#   log_priors: [T, 41] float32 — subtracted from log-probs (zeros = no prior)
#   blank_penalty: float — log(penalty) subtracted from blank column
lm_decoder.DecodeNumpy(decoder, logits, log_priors, blank_penalty)

decoder.FinishDecoding()
decoder.Rescore()  # optional lattice rescoring

# Get results
for r in decoder.result():
    print(r.sentence, r.ac_score, r.lm_score)

decoder.Reset()  # reset for next utterance
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'lm_decoder'` | Build not installed. Run `python setup.py install` |
| CMake can't find Python | Set `-DPYTHON_EXECUTABLE=$(which python)` |
| OpenFST configure fails | Ensure GCC >= 10.1 is on PATH. Try `ml gcc/10.1.0` on SLURM |
| Symbol errors on import | ABI mismatch. Rebuild with matching PyTorch version |
| `GLIBCXX_3.4.29 not found` | GCC too old. Need GCC >= 10.1 |
| Build fails downloading libtorch | Network issue. Download manually and set `FETCHCONTENT_SOURCE_DIR_LIBTORCH` |
