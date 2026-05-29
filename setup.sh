#!/bin/bash
# =============================================================================
# AbDistill Setup Script
# =============================================================================

set -e
WORKDIR=$(pwd)
WEIGHTS_DIR="${WORKDIR}/boltz_weights"

echo "============================================"
echo " AbDistill Setup "
echo "============================================"

# 1. System packages & Smina Binary Download
apt-get update -qq || true
apt-get install -y -qq git curl wget rsync build-essential gfortran || true

echo "[1/4] Installing Smina..."
if ! command -v smina &>/dev/null; then
    wget -q https://sourceforge.net/projects/smina/files/smina.static/download -O /usr/local/bin/smina
    chmod +x /usr/local/bin/smina
fi

# 2. Python environment — forced isolated conda environment
echo "[2/4] Creating conda environment (abdistill, Python 3.11)..."
if [ ! -f "/opt/miniconda/bin/conda" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda
fi
export PATH="/opt/miniconda/bin:$PATH"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true

ENV_DIR="/opt/miniconda/envs/abdistill"

if [ -d "$ENV_DIR" ]; then
    echo "  abdistill env already exists, skipping creation"
else
    # CRITICAL FIX: Use -p to force the exact directory and ignore corrupt .condarc
    # Added bioconda for anarci (IMGT numbering) and standard data science tools
    conda create -y -p "$ENV_DIR" -c conda-forge -c bioconda python=3.11 pip numpy scipy hmmer anarci biopython scikit-learn matplotlib seaborn
fi

# 3. Install Python Packages
echo "[3/4] Installing packages into abdistill (conda, Python 3.11)..."

ENV_PIP="${ENV_DIR}/bin/pip"
ENV_PYTHON="${ENV_DIR}/bin/python"

$ENV_PIP install --upgrade pip -q

# Install Boltz (This handles Torch automatically)
$ENV_PIP install boltz -q

# Install AntiFold strictly from GitHub
$ENV_PIP install git+https://github.com/oxpig/AntiFold.git -q

# Install our custom stack (AbLang2, UniMol, ESM, PyG, Data dependencies)
$ENV_PIP install pandas tqdm rdkit ablang2 unimol_tools fair-esm torch_geometric -q

# Install torch_scatter safely via pre-compiled PyG wheels
echo "[3.5/4] Installing torch_scatter (PyG Wheels)..."
$ENV_PYTHON -c "
import torch
import subprocess
import sys

# Get PyTorch version (e.g. 2.5.1)
pt_version = torch.__version__.split('+')[0]
cu_version = 'cu121'

# Build the exact PyG URL for your environment
url = f'https://data.pyg.org/whl/torch-{pt_version}+{cu_version}.html'
print(f'Downloading pre-compiled wheel from: {url}')

# Install using pip and the specific URL
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch_scatter', '-f', url, '-q'])
"

# 4. Boltz-2 Weights Cache Fix
echo "[4/4] Setting up Boltz-2 weights cache..."
mkdir -p "${WEIGHTS_DIR}"

if [ -f "${WEIGHTS_DIR}/boltz2_conf.ckpt" ] && [ -f "${WEIGHTS_DIR}/boltz2_aff.ckpt" ]; then
    echo "  Found pre-existing weights — skipping download."
else
    echo "  Downloading Boltz-2 weights to prevent runtime redownload..."
    $ENV_PYTHON -c "
from boltz.main import download_boltz2
import pathlib
try:
    download_boltz2(cache=pathlib.Path('${WEIGHTS_DIR}'))
    print('  Weights downloaded to ${WEIGHTS_DIR}')
except Exception as e:
    print(f'  Will download on first run: {e}')
" || true
fi

echo "============================================"
echo " Setup complete!"
echo " Activate env with: conda activate abdistill"
echo "============================================"
