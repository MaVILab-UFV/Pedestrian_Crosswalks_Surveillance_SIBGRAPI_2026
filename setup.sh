#!/usr/bin/env bash
# =============================================================================
# setup.sh — prepara o ambiente para rodar os pipelines de infração.
#
#   1) clona Depth-Anything-3 e sam-3d-objects (repositórios ORIGINAIS) em
#      third_party/, fixados nos commits usados no artigo;
#   2) aplica no SAM3D os ajustes de versão para CUDA 12.8 (patches/);
#   3) cria o ambiente conda (environment.yml) e instala torch, SAM3D, DA3 e
#      as dependências deste repositório (requirements.txt);
#   4) baixa os checkpoints do SAM3D (exige acesso aprovado no Hugging Face).
#
# Requisitos: Linux x86-64, GPU NVIDIA (o SAM3D pede >= 32 GB de VRAM),
# git e conda/mamba.
#
# Uso:
#   ./setup.sh                    # tudo
#   PULAR_ENV=1 ./setup.sh        # só clona/aplica patch (ambiente já existe)
#   PULAR_CHECKPOINTS=1 ./setup.sh
#   ENV_NAME=outro_nome ./setup.sh
# =============================================================================
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$RAIZ/third_party"
ENV_NAME="${ENV_NAME:-infracao-faixa}"

DA3_URL="https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_COMMIT="41736238f5bced4debf3f2a12375d2466874866d"
DA3_DIR="$TP/Depth-Anything-3"

SAM3D_URL="https://github.com/facebookresearch/sam-3d-objects.git"
SAM3D_COMMIT="81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd"
SAM3D_DIR="$TP/sam-3d-objects"
SAM3D_TAG="hf"   # checkpoints/<TAG>/pipeline.yaml (igual a CFG.SAM3D_CONFIG_TAG)

TORCH_INDEX="https://download.pytorch.org/whl/cu128"
KAOLIN_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html"

log() { echo -e "\n==> $*"; }

clonar() {  # clonar <url> <pasta> <commit>
    local url="$1" dir="$2" commit="$3"
    if [ ! -d "$dir/.git" ]; then
        git clone "$url" "$dir"
    fi
    git -C "$dir" fetch --quiet origin
    git -C "$dir" checkout --quiet "$commit"
    echo "    $(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}

# -----------------------------------------------------------------------------
log "1/4 Clonando repositórios de terceiros em third_party/"
mkdir -p "$TP"
clonar "$DA3_URL" "$DA3_DIR" "$DA3_COMMIT"
clonar "$SAM3D_URL" "$SAM3D_DIR" "$SAM3D_COMMIT"

# -----------------------------------------------------------------------------
log "2/4 Aplicando ajustes de versão do SAM3D (CUDA 12.8)"
PATCH="$RAIZ/patches/sam3d-cu128.patch"
if git -C "$SAM3D_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "    patch já aplicado"
else
    git -C "$SAM3D_DIR" apply "$PATCH"
    echo "    patch aplicado"
fi

# -----------------------------------------------------------------------------
if [ "${PULAR_ENV:-0}" != "1" ]; then
    log "3/4 Criando ambiente conda '$ENV_NAME' e instalando dependências"
    CONDA="$(command -v mamba || command -v conda)"
    eval "$(conda shell.bash hook)"
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        "$CONDA" env create -f "$RAIZ/environment.yml" -n "$ENV_NAME"
    fi
    conda activate "$ENV_NAME"

    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url "$TORCH_INDEX"

    # SAM3D, seguindo doc/setup.md do repositório original (com índices cu128)
    export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com $TORCH_INDEX"
    pip install -e "$SAM3D_DIR[dev]"
    pip install -e "$SAM3D_DIR[p3d]"
    PIP_FIND_LINKS="$KAOLIN_LINKS" pip install -e "$SAM3D_DIR[inference]"
    python "$SAM3D_DIR/patching/hydra"

    # Depth-Anything-3
    pip install -e "$DA3_DIR"

    # dependências próprias dos pipelines
    pip install -r "$RAIZ/requirements.txt"
else
    log "3/4 PULAR_ENV=1: ambiente não foi criado/alterado"
fi

# -----------------------------------------------------------------------------
if [ "${PULAR_CHECKPOINTS:-0}" != "1" ]; then
    log "4/4 Checkpoints do SAM3D"
    if [ -f "$SAM3D_DIR/checkpoints/$SAM3D_TAG/pipeline.yaml" ]; then
        echo "    já existem em checkpoints/$SAM3D_TAG"
    else
        echo "    Requer acesso aprovado em https://huggingface.co/facebook/sam-3d-objects"
        echo "    e login (hf auth login)."
        hf download --repo-type model --local-dir "$SAM3D_DIR/checkpoints/$SAM3D_TAG-download" \
            --max-workers 1 facebook/sam-3d-objects
        mv "$SAM3D_DIR/checkpoints/$SAM3D_TAG-download/checkpoints" "$SAM3D_DIR/checkpoints/$SAM3D_TAG"
        rm -rf "$SAM3D_DIR/checkpoints/$SAM3D_TAG-download"
    fi
else
    log "4/4 PULAR_CHECKPOINTS=1: checkpoints do SAM3D não foram baixados"
fi

log "Pronto. Os pesos do DA3 e do YOLO são baixados automaticamente na 1ª execução."
echo "    conda activate $ENV_NAME && python src/pipeline_infracao_vtempo2.py"
