# Pedestrian Crosswalks Surveillance

## Installation & Setup

The pipeline depends on two external projects, [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) and [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3). Install both first, following their official guides, inside the `third_party/` folder of this repository. Then install the requirements of this pipeline in the same environment.

1. **Prerequisites**
   * Linux x86-64 with an NVIDIA GPU. SAM 3D Objects requires at least 32 GB of VRAM.
   * `git` and `conda` (or `mamba`).

2. **Clone the Repository**
   ```bash
   git clone https://github.com/MaVILab-UFV/Pedestrian_Crosswalks_Surveillance_SIBGRAPI_2026.git
   cd Pedestrian_Crosswalks_Surveillance_SIBGRAPI_2026
   ```

3. **Install SAM 3D Objects**
   Clone it into `third_party/`:
   ```bash
   cd third_party
   git clone https://github.com/facebookresearch/sam-3d-objects.git
   cd sam-3d-objects
   ```
   Then follow its official [setup guide](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md), from inside `third_party/sam-3d-objects/`. The guide creates the `sam3d-objects` conda environment and downloads the checkpoints, which require approved access on [Hugging Face](https://huggingface.co/facebook/sam-3d-objects). Keep the default checkpoint tag (`hf`): the pipeline expects the checkpoints at `third_party/sam-3d-objects/checkpoints/hf/`.

4. **Install Depth-Anything-3**
   With the `sam3d-objects` environment active, clone it into `third_party/`:
   ```bash
   conda activate sam3d-objects
   cd third_party
   git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
   cd Depth-Anything-3
   ```
   Then follow the installation steps of its official [README](https://github.com/ByteDance-Seed/Depth-Anything-3#-installation). Install it in the same `sam3d-objects` environment, which already provides PyTorch, so the basic install (`pip install -e .`) is enough.

5. **Install the Pipeline Requirements**
   Back at the repository root, with the `sam3d-objects` environment active:
   ```bash
   pip install -r requirements.txt
   ```

   At the end, the repository should look like this:
   ```
   Pedestrian_Crosswalks_Surveillance_SIBGRAPI_2026/
   ├── data/
   ├── src/
   ├── third_party/
   │   ├── Depth-Anything-3/
   │   └── sam-3d-objects/
   │       └── checkpoints/hf/
   └── requirements.txt
   ```

   The paper used SAM 3D Objects at commit `81a82373` and Depth-Anything-3 at commit `41736238`. To reproduce it exactly, run `git checkout <commit>` inside each folder right after cloning.

6. **Model Weights**
   No manual download is needed for these:
   * **Depth-Anything-3** (`depth-anything/DA3-LARGE-1.1`) is downloaded from Hugging Face on the first run, into `~/.cache/huggingface/hub/` (or `$HF_HOME`).
   * **YOLO** (`yolo11l.pt`, `yolo11x-seg.pt`) is downloaded by Ultralytics on the first run, into the directory the script is called from.

---

## Dataset

The dataset used in the paper is available on Zenodo: **[TODO: Zenodo link]**.

Download it and extract its contents into the `data/` folder of this repository, so that it looks like this:

```
data/
├── video/           # videos (.mov, .mp4, .avi, .mkv, .m4v)
└── calib/           # <video>_pontos_faixa.npy, one per video
```

| Folder | Content |
|---|---|
| `data/video/` | the input videos |
| `data/calib/` | `<video>_pontos_faixa.npy`: (u, v) pixel points outlining the crosswalk, ordered around its contour. The camera is fixed, so one file covers the whole video. |

Files are matched by the video base name: `MVI_7693.mov` uses `MVI_7693_pontos_faixa.npy`.

---

## Basic Usage

The main pipeline of the paper is `src/pipeline_infracao.py`. It runs every stage itself: YOLO detection (`yolo11l.pt`), co-occurrence filtering, Depth-Anything-3, the crosswalk OBB, YOLO segmentation (`yolo11x-seg.pt`), SAM 3D Objects and the 3D violation test.

1. **Configure Parameters**
   Open `src/pipeline_infracao.py` and adjust the `CFG` class at the top of the file. The default values run the full method of the paper (`FLUXO = "completo"`). The most relevant parameters are:
   * `FLUXO`: pipeline mode (see the table below).
   * `SUFIXO_SAIDA`: suffix of the output file (`"_completo"` by default).
   * `RECOMPUTAR_*`: forces a stage to run again even if its result is already cached.

   | `FLUXO` | Description |
   |---|---|
   | `completo` (default) | full method: YOLO, co-occurrence filter, Depth-Anything-3, crosswalk OBB, YOLO-seg, SAM 3D Objects and 3D intersection |
   | `2d` | 2D baseline: YOLO bounding boxes + 2D intersection with the crosswalk, on every frame |
   | `2d_coocorrencia` | same as `2d`, only on the visually ambiguous frames selected by the co-occurrence filter |

2. **Run the Pipeline**
   With the environment active, from any directory:
   ```bash
   conda activate sam3d-objects
   python src/pipeline_infracao.py
   ```
   The pipeline processes every video in `data/video/` that has a matching file in `data/calib/`.

3. **Outputs**
   Everything goes under `dados/<video>/`:
   * `infracoes/<video><SUFIXO_SAIDA>.json`: final result of the `completo` mode. For each processed frame, it reports whether the vehicle mesh intersects the crosswalk, whether a pedestrian is present on it, and whether this configures a violation.
   * `interseccao_2d/`: results of the `2d` and `2d_coocorrencia` modes.
   * `yolo_bb/`, `co_ocorrencia/`, `da3/`, `obb_faixa/`, `yolo_seg/` and `sam3d/`: cache of each stage. If the pipeline is run again, cached stages are not recomputed.

---