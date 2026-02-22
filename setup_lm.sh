#!/bin/bash
set -e

# ============================================================
# Build the lm_decoder C++ pybind11 module
# ============================================================
# Creates a conda env (b2txt25_lm) with Python 3.9 + PyTorch 1.13.1
# and builds the Kaldi-based WFST CTC decoder.
#
# Run from the repo root:
#   bash setup_lm.sh
# ============================================================

# Ensure that the script is run from the root directory of the project
if [ ! -f "setup_lm.sh" ]; then
    echo "ERROR: This script must be run from the root directory of the project."
    exit 1
fi

# Check that the C++ source tree exists
if [ ! -f "language_model/runtime/server/x86/CMakeLists.txt" ]; then
    echo "ERROR: language_model/runtime/server/x86/CMakeLists.txt not found."
    echo "       The C++ source tree is missing."
    exit 1
fi

# Ensure that build and fc_base directories don't exist (clean build)
if [ -d "language_model/runtime/server/x86/build" ]; then
    echo "ERROR: language_model/runtime/server/x86/build directory already exists."
    echo "       Remove it first: rm -rf language_model/runtime/server/x86/build"
    exit 1
fi

if [ -d "language_model/runtime/server/x86/fc_base" ]; then
    echo "ERROR: language_model/runtime/server/x86/fc_base directory already exists."
    echo "       Remove it first: rm -rf language_model/runtime/server/x86/fc_base"
    exit 1
fi

# Check prerequisites
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake is not installed. Install with: sudo apt-get install cmake"
    exit 1
fi

if ! command -v gcc &> /dev/null; then
    echo "ERROR: GCC is not installed. Install with: sudo apt-get install build-essential"
    exit 1
fi

# Check GCC version (need >= 10.1)
GCC_VERSION=$(gcc -dumpversion)
GCC_MAJOR=$(echo "$GCC_VERSION" | cut -d. -f1)
if [ "$GCC_MAJOR" -lt 10 ]; then
    echo "ERROR: GCC >= 10.1 is required. Found GCC $GCC_VERSION."
    echo "       On SLURM: ml gcc/10.1.0"
    echo "       On Ubuntu: sudo apt install gcc-10 g++-10"
    exit 1
fi

echo "Prerequisites OK (CMake, GCC $GCC_VERSION)"

# Prefer mamba if available, fall back to conda
if command -v mamba &> /dev/null; then
    CONDA_CMD=mamba
elif command -v conda &> /dev/null; then
    CONDA_CMD=conda
else
    echo "ERROR: Neither conda nor mamba found. Install conda/miniforge first."
    exit 1
fi

echo "Using $CONDA_CMD"

# Ensure conda shell functions are available
if [ -n "$CONDA_EXE" ]; then
    CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
else
    CONDA_BASE=$($CONDA_CMD info --base 2>/dev/null)
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/mamba.sh"
fi

ENV_NAME="b2txt25_lm"

# Create conda environment with Python 3.9
echo ""
echo "Creating conda environment '$ENV_NAME' with Python 3.9..."
$CONDA_CMD create -n "$ENV_NAME" python=3.9 -y

# Activate the environment
conda activate "$ENV_NAME"
echo "Activated environment: $(python --version) at $(which python)"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install \
    torch==1.13.1 \
    numpy==1.24.4 \
    redis==5.0.6 \
    tqdm==4.66.4 \
    g2p_en==2.1.0 \
    omegaconf==2.3.0 \
    huggingface-hub==0.23.4 \
    transformers==4.40.0 \
    tokenizers==0.19.1 \
    accelerate==0.33.0 \
    editdistance==0.8.1 \
    scipy==1.11.1 \
    scikit-learn==1.6.1

# Build and install lm_decoder
echo ""
echo "Building lm_decoder C++ module (this takes ~10-15 minutes on first build)..."
cd language_model/runtime/server/x86
python setup.py install
cd ../../../..

# Verify
echo ""
echo "Verifying installation..."
python -c "import lm_decoder; print('lm_decoder imported successfully!')"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "To use:"
echo "  conda activate $ENV_NAME"
echo "  python scripts/eval_posterior_constrained.py \\"
echo "      --model_path /path/to/checkpoint \\"
echo "      --dataset_path /path/to/ptDecoder_ctc \\"
echo "      --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil"
echo ""
echo "For LLM rescoring, add:  --do_llm --llm_model facebook/opt-6.7b"
echo "============================================================"
