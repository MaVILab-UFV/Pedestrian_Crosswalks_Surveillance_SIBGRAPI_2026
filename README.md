# Detecção de infração na faixa de pedestres

Pipelines que detectam quando um **automóvel e uma pessoa ocupam a faixa de
pedestres ao mesmo tempo**, combinando detecção 2D (YOLO), profundidade
monocular ([Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3))
e reconstrução 3D do veículo ([SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)).
A faixa vira uma caixa orientada (OBB) em 3D, e a infração é testada pela
interseção da mesh do veículo com essa caixa.

## Estrutura

```
src/
  geometria.py                   # funções 3D (OBB da faixa, interseção mesh x OBB, ...)
  pipeline_infracao.py           # versão original: 3D sobre JSONs de detecção já existentes
  pipeline_infracao_v2.py        # YOLO próprio + fluxos 2D/3D + cache em disco
  pipeline_infracao_vtempo2.py   # v2 + medição de tempo por etapa + fluxos extras
data/
  video/          # vídeos de entrada (MVI_xxxx.mov, .mp4, ...)
  calib/          # pontos da faixa por vídeo (MVI_xxxx_pontos_faixa.npy)
  output_naive/   # JSONs de detecção 2D (só para pipeline_infracao.py)
patches/
  sam3d-cu128.patch   # ajustes de versão do SAM3D para CUDA 12.8
setup.sh, environment.yml, requirements.txt
third_party/      # criada pelo setup.sh (DA3 e SAM3D; fora do git)
dados/            # criada na execução (cache e saídas; fora do git)
```

## Requisitos

- Linux x86-64 com GPU NVIDIA. O SAM3D pede pelo menos 32 GB de VRAM.
- `git` e `conda` (ou `mamba`).
- Acesso aprovado aos checkpoints do SAM3D em
  <https://huggingface.co/facebook/sam-3d-objects>, e login com `hf auth login`.

## Instalação

### Automática

```bash
git clone <url-deste-repositorio>
cd <pasta-do-repositorio>
./setup.sh
```

O `setup.sh`:

1. clona o Depth-Anything-3 e o sam-3d-objects dos repositórios originais em
   `third_party/`, fixados nos commits usados no trabalho;
2. aplica `patches/sam3d-cu128.patch` no SAM3D;
3. cria o ambiente conda `infracao-faixa` e instala torch 2.7.0 (cu128), SAM3D,
   DA3 e `requirements.txt`;
4. baixa os checkpoints do SAM3D em `third_party/sam-3d-objects/checkpoints/hf/`.

Variáveis opcionais: `ENV_NAME=<nome>` muda o nome do ambiente,
`PULAR_ENV=1` só clona e aplica o patch, e `PULAR_CHECKPOINTS=1` não baixa os
checkpoints.

### Manual

Se preferir instalar por conta própria, os pipelines só esperam encontrar os
dois repositórios nestes caminhos:

```bash
mkdir -p third_party && cd third_party

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
git -C Depth-Anything-3 checkout 41736238f5bced4debf3f2a12375d2466874866d

git clone https://github.com/facebookresearch/sam-3d-objects.git
git -C sam-3d-objects checkout 81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd
git -C sam-3d-objects apply ../../patches/sam3d-cu128.patch
```

Depois, instale o SAM3D seguindo o
[doc/setup.md](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
dele (incluindo os checkpoints em `checkpoints/hf/`), o DA3 com
`pip install -e third_party/Depth-Anything-3` e, por fim,
`pip install -r requirements.txt`.

### Pesos baixados automaticamente

- **DA3** (`depth-anything/DA3-LARGE-1.1`): baixado do Hugging Face na primeira
  execução, para `~/.cache/huggingface/hub/` (ou `$HF_HOME`).
- **YOLO** (`yolo11l.pt`, `yolo11x-seg.pt`): baixados pelo Ultralytics na
  primeira execução, na pasta de onde o script for chamado.

## Dados de entrada

| Pasta | Conteúdo | Usado por |
|---|---|---|
| `data/video/` | vídeos (`.mov`, `.mp4`, `.avi`, `.mkv`, `.m4v`) | todos |
| `data/calib/` | `<video>_pontos_faixa.npy`: pontos (u, v) em pixels que delimitam a faixa, em ordem ao redor do contorno | todos |
| `data/output_naive/` | JSONs de detecção 2D com prefixo `<video>_` (subpastas por modelo) | só `pipeline_infracao.py` |

Os arquivos são associados pelo nome-base do vídeo: `MVI_7693.mov` usa
`MVI_7693_pontos_faixa.npy` e `MVI_7693_*.json`.

## Execução

Com o ambiente ativo (`conda activate infracao-faixa`), a partir de qualquer pasta:

```bash
python src/pipeline_infracao_vtempo2.py
```

Os parâmetros ficam na classe `CFG`, no topo de cada script. O principal é
`CFG.FLUXO`, nos pipelines `v2` e `vtempo2`:

| Fluxo | O que faz | `v2` | `vtempo2` |
|---|---|:-:|:-:|
| `2d` | YOLO bbox + interseção 2D com a faixa | ✓ | ✓ |
| `2d_coocorrencia` | igual ao `2d`, só nos frames com automóvel e pessoa | ✓ | ✓ |
| `completo` | pipeline 3D: YOLO, DA3, OBB (PCA), YOLO-seg, SAM3D e interseção 3D | ✓ | ✓ |
| `nuvem_mascara` | como o `completo`, mas sem SAM3D: usa a nuvem de pontos sob a máscara do veículo | | ✓ |
| `pontos` | como o `completo`, com a OBB montada direto dos pontos da faixa | | ⚠ |

⚠ O fluxo `pontos` chama `geometria.construir_obb_faixa_pontos`, que ainda não
está em `src/geometria.py`.

## Saídas

- **`pipeline_infracao.py`**: um `<json>_infracoes_3d.json` ao lado de cada JSON
  de entrada em `data/output_naive/`.
- **`v2` e `vtempo2`**: tudo em `dados/<video>/`. O cache de cada etapa fica em
  `yolo_bb/`, `co_ocorrencia/`, `da3/`, `obb_faixa/`, `yolo_seg/` e `sam3d/`, e o
  resultado final em `infracoes/` ou `interseccao_2d/`.
- **`vtempo2`**, além disso: `tempos_execucao.csv` (tempo de cada etapa por
  frame) e `tempos_execucao_resumo.csv` (tabela frame × etapa com o total).
