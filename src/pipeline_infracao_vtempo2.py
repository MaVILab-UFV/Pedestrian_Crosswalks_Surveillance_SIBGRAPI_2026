from __future__ import annotations
import os
import sys
import json
import glob
import time
import tempfile
import traceback
import numpy as np
import pandas as pd
import cv2
import open3d as o3d

import geometria as G

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CFG:
    VIDEOS_DIR = os.path.join(RAIZ, "data", "video")
    VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv", ".m4v")

    PONTOS_FAIXA_DIR = os.path.join(RAIZ, "data", "calib")

    FLUXO = "completo"

    DADOS_DIR = os.path.join(RAIZ, "dados")
    SUFIXO_SAIDA = "_completo"
    SUFIXO_SAIDA_PONTOS = "_pontos"
    SUFIXO_SAIDA_NUVEM_MASCARA = "_nuvem_mascara"
    SOBRESCREVER = False

    RECOMPUTAR_YOLO_BB = False
    RECOMPUTAR_COOCORRENCIA = False
    RECOMPUTAR_DA3 = False
    RECOMPUTAR_OBB = False
    RECOMPUTAR_OBB_PONTOS = False
    RECOMPUTAR_YOLO_SEG = False
    RECOMPUTAR_SAM3D = False

    YOLO_BB_MODEL = "yolo11l.pt"
    YOLO_BB_CONF = 0.25
    YOLO_BB_DEVICE = "cuda:0"
    YOLO_BB_BATCH = 32
    YOLO_BB_CLASSES = [0, 1, 2, 3, 5, 7]
    FAIXA_2D_MARGEM_PX = 10

    YOLO_SEG_MODEL = "yolo11x-seg.pt"
    YOLO_SEG_CONF = 0.25
    YOLO_SEG_DEVICE = "cuda:0"

    DA3_REPO_SRC = os.path.join(RAIZ, "third_party", "Depth-Anything-3", "src")
    DA3_MODEL = "depth-anything/DA3-LARGE-1.1"
    DA3_PROCESS_RES = 1274

    SAM3D_REPO_DIR = os.path.join(RAIZ, "third_party", "sam-3d-objects")
    SAM3D_NOTEBOOK_DIR = os.path.join(RAIZ, "third_party", "sam-3d-objects", "notebook")
    SAM3D_CONFIG_TAG = "hf"
    SAM3D_SEED = 42
    SAM3D_MAX_LADO = None

    BOX_HEIGHT = 2.5
    OBB_MARGIN = 0.05
    OBB_TOL_SOLO = 0.3
    OBB_MODO = "acima_do_solo"

    SELECAO_FRAMES = "auto_e_pessoa"
    MAX_FRAMES = None

    MIN_PONTOS_MESH = 50
    MESH_N_AMOSTRAS = 20000
    MIN_PONTOS_PESSOA = 30
    Z_MIN, Z_MAX = 0.1, 80.0

    MIN_PONTOS_VEICULO_PC = 50
    EROSAO_MASCARA_PX = 3
    SOR_NB_VIZINHOS = 20
    SOR_STD_RATIO = 2.0

    VERBOSE = True


def log(*a):
    if CFG.VERBOSE:
        print(*a, flush=True)


def salvar_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def carregar_json(path):
    with open(path) as f:
        return json.load(f)


def salvar_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


COLUNAS_TEMPOS = ["video", "frame", "etapa", "escopo", "obj_idx", "tempo_s", "origem", "timestamp"]

_LINHAS_TEMPOS: list[dict] = []


def dataframe_tempos() -> pd.DataFrame:
    if not _LINHAS_TEMPOS:
        return pd.DataFrame(columns=COLUNAS_TEMPOS)
    return pd.DataFrame(_LINHAS_TEMPOS, columns=COLUNAS_TEMPOS)


def registrar_tempo(video, frame, etapa, tempo_s, origem, escopo="frame", obj_idx=None):
    _LINHAS_TEMPOS.append({
        "video": video,
        "frame": frame,
        "etapa": etapa,
        "escopo": escopo,
        "obj_idx": obj_idx,
        "tempo_s": float(tempo_s) if tempo_s is not None else float("nan"),
        "origem": origem,
        "timestamp": pd.Timestamp.now().isoformat(),
    })


def caminho_tempos_csv(dados_dir):
    return os.path.join(dados_dir, "tempos_execucao.csv")


def carregar_historico_tempos(dados_dir):
    path = caminho_tempos_csv(dados_dir)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            log(f"  não consegui ler histórico de tempos em {path}; iniciando vazio")
    return pd.DataFrame(columns=COLUNAS_TEMPOS)


def resgatar_tempo_original(historico, video, etapa, frame=None, escopo="frame", obj_idx=None):
    if historico is None or historico.empty:
        log(f"    sem histórico de tempos para resgatar etapa='{etapa}' "
            f"frame={frame} (video={video}); tempo será NaN")
        return float("nan")

    filtro = (
        (historico["video"] == video)
        & (historico["etapa"] == etapa)
        & (historico["escopo"] == escopo)
    )
    if escopo == "frame" and frame is not None:
        frames_hist = pd.to_numeric(historico["frame"], errors="coerce")
        filtro &= (frames_hist == float(frame))
    if obj_idx is not None and "obj_idx" in historico.columns:
        obj_hist = pd.to_numeric(historico["obj_idx"], errors="coerce")
        filtro &= (obj_hist == float(obj_idx))

    achados = historico[filtro]
    if achados.empty:
        log(f"    sem histórico de tempos para resgatar etapa='{etapa}' "
            f"frame={frame} (video={video}); tempo será NaN")
        return float("nan")

    calculados = achados[achados["origem"] != "cache"]
    alvo = calculados if not calculados.empty else achados
    return float(alvo.iloc[-1]["tempo_s"])


def salvar_tempos(dados_dir, base):
    df = dataframe_tempos()
    novas = df[df["video"] == base]
    if novas.empty:
        return
    path = caminho_tempos_csv(dados_dir)
    historico = carregar_historico_tempos(dados_dir)
    combinado = pd.concat([historico, novas], ignore_index=True) if not historico.empty else novas
    combinado = combinado.drop_duplicates(
        subset=["video", "frame", "etapa", "escopo", "obj_idx", "tempo_s", "origem"]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    combinado.to_csv(path, index=False)
    _LINHAS_TEMPOS[:] = [l for l in _LINHAS_TEMPOS if l["video"] != base]


def resumo_tempos_por_video(base):
    df = dataframe_tempos()
    df_video = df[(df["video"] == base) & (df["escopo"] == "frame")]
    if df_video.empty:
        return pd.DataFrame()
    pivot = df_video.pivot_table(index="frame", columns="etapa", values="tempo_s",
                                 aggfunc="sum", fill_value=0.0)
    pivot["tempo_total_s"] = pivot.sum(axis=1)
    return pivot.reset_index().sort_values("frame")


class LeitorVideo:
    def __init__(self, path):
        self.path = path
        self.backend = None
        try:
            import decord
            self.vr = decord.VideoReader(path)
            self.backend = "decord"
            self.n = len(self.vr)
            try:
                self.fps = float(self.vr.get_avg_fps())
            except Exception:
                self.fps = 0.0
        except Exception:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Não consegui abrir o vídeo: {path}")
            self.backend = "opencv"
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        log(f"Vídeo aberto via {self.backend} | {self.n} frames | {self.fps:.2f} fps")

    def ler_bgr(self, idx):
        if idx < 0 or idx >= self.n:
            return None
        if self.backend == "decord":
            frame_rgb = self.vr[idx].asnumpy()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = self.cap.read()
            return frame_bgr if ok else None


def ordenar_pontos_poligono(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    centro = pts.mean(axis=0)
    angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    return pts[np.argsort(angulos)]


def expandir_poligono(pts, margem=10):
    centro = pts.mean(axis=0)
    direcoes = pts - centro
    normas = np.linalg.norm(direcoes, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    direcoes_norm = direcoes / normas
    return (pts + direcoes_norm * margem).astype(np.float32)


def construir_poligono_faixa_2d(pontos_faixa_px, margem=10):
    ordenados = ordenar_pontos_poligono(pontos_faixa_px)
    return expandir_poligono(ordenados, margem=margem)


def check_bbox_intersect_polygon(bbox_xyxy, polygon_pts):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    poly = polygon_pts.reshape((-1, 1, 2)).astype(np.float32)

    candidatos_bbox = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2),
        ((x1 + x2) / 2, (y1 + y2) / 2),
    ]
    for pt in candidatos_bbox:
        if cv2.pointPolygonTest(poly, pt, False) >= 0:
            return True

    for pt in polygon_pts:
        px, py = float(pt[0]), float(pt[1])
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True

    return False


def bboxes_intersectam(bbox_a, bbox_b):
    x1a, y1a, x2a, y2a = bbox_a
    x1b, y1b, x2b, y2b = bbox_b
    return max(x1a, x1b) < min(x2a, x2b) and max(y1a, y1b) < min(y2a, y2b)


def is_motociclista(pessoa_bbox, lista_motos):
    return any(bboxes_intersectam(pessoa_bbox, moto["bbox"]) for moto in lista_motos)


def mascara_de_bbox_retangular(bbox, w, h):
    x1, y1, x2, y2 = [int(round(c)) for c in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def interseccao_2d(vulneraveis, automoveis, poligono_faixa):
    pares = []
    algum_risco = False
    for v in vulneraveis:
        v_bbox = v["bbox"]
        v_na_faixa = check_bbox_intersect_polygon(v_bbox, poligono_faixa)
        for a in automoveis:
            a_bbox = a["bbox"]
            a_na_faixa = check_bbox_intersect_polygon(a_bbox, poligono_faixa)
            v_a = bboxes_intersectam(v_bbox, a_bbox)
            risco = bool(v_na_faixa and a_na_faixa and v_a)
            algum_risco = algum_risco or risco
            pares.append({
                "vulneravel_cls": v.get("cls"), "veiculo_cls": a.get("cls"),
                "vulneravel_na_faixa": bool(v_na_faixa),
                "veiculo_na_faixa": bool(a_na_faixa),
                "vulneravel_veiculo": bool(v_a),
                "risco": risco,
            })
    return {"algum_risco": bool(algum_risco), "pares": pares}


class YoloBBox:
    def __init__(self):
        from ultralytics import YOLO
        log(f"Carregando YOLO bbox ({CFG.YOLO_BB_MODEL}) ...")
        self.model = YOLO(CFG.YOLO_BB_MODEL)
        log("YOLO bbox carregado.")

    def detectar_video(self, leitor, poligono_faixa_px):
        eventos = []
        n = leitor.n
        for ini in range(0, n, CFG.YOLO_BB_BATCH):
            fim = min(ini + CFG.YOLO_BB_BATCH, n)
            pares = [(i, leitor.ler_bgr(i)) for i in range(ini, fim)]
            pares = [(i, f) for i, f in pares if f is not None]
            if not pares:
                continue
            idxs, frames = zip(*pares)
            results = self.model.predict(list(frames), conf=CFG.YOLO_BB_CONF,
                                         device=CFG.YOLO_BB_DEVICE, verbose=False)
            for fidx, res in zip(idxs, results):
                ev = self._evento_do_frame(fidx, res, poligono_faixa_px, leitor.fps)
                if ev is not None:
                    eventos.append(ev)
            if CFG.VERBOSE:
                log(f"    ... YOLO bbox {fim}/{n} frames")
        return eventos

    def _evento_do_frame(self, fidx, res, poligono, fps):
        if res.boxes is None or len(res.boxes) == 0:
            return None

        pessoas, motos, outros_vuln, automoveis = [], [], [], []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in CFG.YOLO_BB_CLASSES:
                continue
            coords = box.xyxy[0].cpu().numpy().tolist()
            if not check_bbox_intersect_polygon(coords, poligono):
                continue

            entry = {"cls": self.model.names[cls_id].upper(), "bbox": coords,
                     "conf": round(float(box.conf[0]), 4)}
            if cls_id == 0:
                pessoas.append(entry)
            elif cls_id == 1:
                outros_vuln.append(entry)
            elif cls_id == 3:
                motos.append(entry)
            elif cls_id in (2, 5, 7):
                automoveis.append(entry)

        pedestres_reais = [p for p in pessoas if not is_motociclista(p["bbox"], motos)]
        vulneraveis = pedestres_reais + outros_vuln
        automoveis = automoveis + motos

        if not (vulneraveis or automoveis):
            return None

        return {
            "frame": fidx,
            "timestamp_ms": round((fidx / fps) * 1000, 2) if fps else None,
            "alerta_risco_2d": bool(vulneraveis and automoveis),
            "deteccoes": {"vulneraveis": vulneraveis, "automoveis": automoveis},
        }


class YoloSegMascara:
    def __init__(self):
        from ultralytics import YOLO
        log(f"Carregando YOLO-seg ({CFG.YOLO_SEG_MODEL}) p/ máscaras do SAM3D ...")
        self.model = YOLO(CFG.YOLO_SEG_MODEL)
        log("YOLO-seg carregado.")

    def segmentar_frame(self, frame_bgr):
        results = self.model.predict(frame_bgr, conf=CFG.YOLO_SEG_CONF,
                                     device=CFG.YOLO_SEG_DEVICE,
                                     classes=CFG.YOLO_BB_CLASSES,
                                     retina_masks=True, verbose=False)[0]
        polys = []
        if results.masks is not None:
            for i, poly in enumerate(results.masks.xy):
                polys.append({"cls_id": int(results.boxes.cls[i]),
                              "polygon": np.asarray(poly, dtype=np.float64).tolist()})
        return polys

    @staticmethod
    def mascara_de_bbox(polys_frame, bbox, w_img, h_img):
        x1, y1, x2, y2 = [int(round(c)) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        if x2 <= x1 or y2 <= y1 or not polys_frame:
            return mask_full

        box_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 1

        best_idx, best_overlap = -1, 0
        for i, item in enumerate(polys_frame):
            poligono = np.asarray(item["polygon"], dtype=np.float64).round().astype(np.int32)
            if len(poligono) < 3:
                continue
            temp_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [poligono], 1)
            overlap = int(np.logical_and(temp_mask, box_mask).sum())
            if overlap > best_overlap:
                best_overlap, best_idx = overlap, i

        if best_idx >= 0:
            poligono_final = np.asarray(polys_frame[best_idx]["polygon"],
                                        dtype=np.float64).round().astype(np.int32)
            cv2.fillPoly(mask_full, [poligono_final], 255)
        return mask_full


class DA3:
    def __init__(self):
        if CFG.DA3_REPO_SRC not in sys.path:
            sys.path.insert(0, CFG.DA3_REPO_SRC)
        import torch
        from depth_anything_3.api import DepthAnything3
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Carregando DA3 ({CFG.DA3_MODEL}) em {self.device} ...")
        self.model = DepthAnything3.from_pretrained(CFG.DA3_MODEL).to(self.device)
        self.model.eval()
        log("DA3 carregado.")

    def inferir(self, frame_bgr):
        from PIL import Image
        h_img, w_img = frame_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        pred = self.model.inference(
            image=[pil],
            process_res=CFG.DA3_PROCESS_RES,
            process_res_method="upper_bound_resize",
            export_dir=None,
            export_format="glb",
        )
        depth = np.asarray(pred.depth[0], dtype=np.float64)
        K = np.asarray(pred.intrinsics[0], dtype=np.float64)
        hp, wp = depth.shape

        if (wp, hp) != (w_img, h_img):
            depth = cv2.resize(depth.astype(np.float32), (w_img, h_img),
                               interpolation=cv2.INTER_NEAREST).astype(np.float64)
            K = G.escalar_intrinseca(K, (wp, hp), (w_img, h_img))
        return depth, K


class SAM3D:
    def __init__(self):
        if CFG.SAM3D_NOTEBOOK_DIR not in sys.path:
            sys.path.insert(0, CFG.SAM3D_NOTEBOOK_DIR)
        import torch
        from inference import Inference, load_image, load_masks
        from pytorch3d.transforms import quaternion_to_matrix, Transform3d
        self.torch = torch
        self.load_image = load_image
        self.load_masks = load_masks
        self._quat_to_mat = quaternion_to_matrix
        self._Transform3d = Transform3d

        config_path = os.path.join(CFG.SAM3D_REPO_DIR, "checkpoints",
                                   CFG.SAM3D_CONFIG_TAG, "pipeline.yaml")
        log(f"Carregando SAM3D ({config_path}) ...")
        self.inference = Inference(config_path, compile=False)

        self.R_yup_to_zup = torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32)
        self.R_flip_z = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32)
        self.R_pt3d_to_cam = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)
        log("SAM3D carregado.")

    def _pointmap(self, depth, K):
        torch = self.torch
        d = depth.copy().astype(np.float32)
        d[d <= 0] = np.nan
        H, W = d.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        Z = d
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy
        pointmap = np.stack([-X, -Y, Z], axis=-1)
        return torch.tensor(pointmap, dtype=torch.float32)

    def _transform_vertices(self, vertices, rotation, translation, scale):
        torch = self.torch
        if isinstance(vertices, np.ndarray):
            vertices = torch.tensor(vertices, dtype=torch.float32)
        vertices = vertices.unsqueeze(0)
        vertices = vertices @ self.R_flip_z.to(vertices.device)
        vertices = vertices @ self.R_yup_to_zup.to(vertices.device)
        R_mat = self._quat_to_mat(rotation.to(vertices.device))
        tfm = self._Transform3d(dtype=vertices.dtype, device=vertices.device)
        tfm = (tfm.scale(scale)
                  .rotate(R_mat)
                  .translate(translation[0], translation[1], translation[2]))
        vertices_world = tfm.transform_points(vertices)
        vertices_world = vertices_world @ self.R_pt3d_to_cam.to(vertices_world.device)
        return vertices_world[0]

    def levantar_mesh(self, frame_bgr, mask_bin, depth, K):
        img = frame_bgr
        m = mask_bin
        d = depth
        K_use = K

        if CFG.SAM3D_MAX_LADO:
            h, w = img.shape[:2]
            lado = max(h, w)
            if lado > CFG.SAM3D_MAX_LADO:
                s = CFG.SAM3D_MAX_LADO / lado
                nw, nh = int(round(w * s)), int(round(h * s))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
                d = cv2.resize(d.astype(np.float32), (nw, nh),
                               interpolation=cv2.INTER_NEAREST).astype(np.float64)
                K_use = G.escalar_intrinseca(K, (w, h), (nw, nh))

        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "frame.png")
            masks_dir = os.path.join(tmp, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            cv2.imwrite(img_path, img)
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img_rgba[:, :, 3] = m
            cv2.imwrite(os.path.join(masks_dir, "1.png"), img_rgba)

            image = self.load_image(img_path)
            masks = self.load_masks(masks_dir, indices_list=[1], extension=".png")
            if not masks:
                return None, None

            pointmap = self._pointmap(d, K_use)
            out = self.inference(image, masks[0], seed=CFG.SAM3D_SEED, pointmap=pointmap)

        mesh = out["glb"]
        S = out["scale"][0].cpu().float()
        T = out["translation"][0].cpu().float()
        R = out["rotation"].squeeze().cpu().float()
        verts = self._transform_vertices(mesh.vertices, R, T, S)
        verts = verts.cpu().numpy().astype(np.float64)
        faces = np.asarray(mesh.faces) if hasattr(mesh, "faces") else None
        return verts, faces


def _casa_prefixo(nome_arquivo, base):
    if not nome_arquivo.startswith(base):
        return False
    resto = nome_arquivo[len(base):]
    return resto == "" or resto[0] in ("_", ".", "-")


def encontrar_videos(videos_dir):
    achados = []
    for nome in sorted(os.listdir(videos_dir)):
        ext = os.path.splitext(nome)[1].lower()
        if ext in CFG.VIDEO_EXTS:
            base = os.path.splitext(nome)[0]
            achados.append((base, os.path.join(videos_dir, nome)))
    return achados


def carregar_pontos_faixa(base):
    candidatos = [p for p in glob.glob(os.path.join(CFG.PONTOS_FAIXA_DIR, "*.npy"))
                  if _casa_prefixo(os.path.basename(p), base)]
    if not candidatos:
        raise FileNotFoundError(
            f"Pontos de faixa não encontrados para '{base}' em {CFG.PONTOS_FAIXA_DIR} "
            f"(esperado algo como {base}_pontos_faixa.npy).")
    if len(candidatos) > 1:
        log(f"  múltiplos arquivos de faixa para {base}; usando {os.path.basename(candidatos[0])}")
    pts = np.load(sorted(candidatos)[0])
    return pts.astype(np.float64).reshape(-1, 2)


def pasta_dados_video(base):
    d = os.path.join(CFG.DADOS_DIR, base)
    for sub in ("yolo_bb", "interseccao_2d", "co_ocorrencia", "da3", "obb_faixa",
                "obb_faixa_pontos", "yolo_seg", "sam3d", "infracoes"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


def obter_deteccoes_yolo_bb(dados_dir, base, leitor, yolo_bb, poligono_faixa_px, historico):
    cache_path = os.path.join(dados_dir, "yolo_bb", "deteccoes.json")
    if os.path.exists(cache_path) and not CFG.RECOMPUTAR_YOLO_BB:
        log("  YOLO bbox: usando cache")
        eventos = carregar_json(cache_path)["eventos"]
        tempo_total = resgatar_tempo_original(historico, base, "yolo_bbox", escopo="lote")
        origem = "cache"
    else:
        t0 = time.perf_counter()
        eventos = yolo_bb.detectar_video(leitor, poligono_faixa_px)
        tempo_total = time.perf_counter() - t0
        origem = "calculado"
        salvar_json(cache_path, {
            "metadados": {"video": base, "modelo": CFG.YOLO_BB_MODEL, "conf": CFG.YOLO_BB_CONF,
                          "batch_size": CFG.YOLO_BB_BATCH,
                          "margem_faixa_px": CFG.FAIXA_2D_MARGEM_PX,
                          "n_frames_video": leitor.n, "n_frames_com_deteccao": len(eventos)},
            "eventos": eventos,
        })

    registrar_tempo(base, None, "yolo_bbox", tempo_total, origem, escopo="lote")
    if eventos and not pd.isna(tempo_total):
        tempo_rateado = tempo_total / len(eventos)
        for e in eventos:
            registrar_tempo(base, e["frame"], "yolo_bbox", tempo_rateado, origem, escopo="frame")
    return eventos


def selecionar_eventos(eventos):
    modo = CFG.SELECAO_FRAMES
    sel = []
    for e in eventos:
        det = e["deteccoes"]
        tem_auto = len(det["automoveis"]) > 0
        tem_vuln = len(det["vulneraveis"]) > 0
        if modo == "auto_e_pessoa" and not (tem_auto and tem_vuln):
            continue
        if modo == "qualquer_auto" and not tem_auto:
            continue
        sel.append(e)
    if CFG.MAX_FRAMES:
        sel = sel[:CFG.MAX_FRAMES]
    return sel


def obter_eventos_selecionados(dados_dir, base, eventos_brutos, historico):
    cache_path = os.path.join(dados_dir, "co_ocorrencia", "eventos_selecionados.json")
    if os.path.exists(cache_path) and not CFG.RECOMPUTAR_COOCORRENCIA:
        log("  Co-ocorrência: usando cache")
        sel = carregar_json(cache_path)["eventos"]
        tempo_total = resgatar_tempo_original(historico, base, "co_ocorrencia", escopo="lote")
        origem = "cache"
    else:
        t0 = time.perf_counter()
        sel = selecionar_eventos(eventos_brutos)
        tempo_total = time.perf_counter() - t0
        origem = "calculado"
        salvar_json(cache_path, {
            "metadados": {"video": base, "criterio": CFG.SELECAO_FRAMES,
                          "max_frames": CFG.MAX_FRAMES,
                          "n_eventos_brutos": len(eventos_brutos),
                          "n_eventos_selecionados": len(sel)},
            "eventos": sel,
        })

    registrar_tempo(base, None, "co_ocorrencia", tempo_total, origem, escopo="lote")
    if sel and not pd.isna(tempo_total):
        tempo_rateado = tempo_total / len(sel)
        for e in sel:
            registrar_tempo(base, e["frame"], "co_ocorrencia", tempo_rateado, origem, escopo="frame")
    return sel


def obter_da3(dados_dir, base, fidx, frame_bgr, da3, historico):
    cache_path = os.path.join(dados_dir, "da3", f"frame_{fidx:06d}.npz")
    if os.path.exists(cache_path) and not CFG.RECOMPUTAR_DA3:
        d = np.load(cache_path)
        depth, K = d["depth"].astype(np.float64), d["K"].astype(np.float64)
        tempo = resgatar_tempo_original(historico, base, "da3", frame=fidx)
        origem = "cache"
    else:
        t0 = time.perf_counter()
        depth, K = da3.inferir(frame_bgr)
        tempo = time.perf_counter() - t0
        origem = "calculado"
        salvar_npz(cache_path, depth=depth.astype(np.float32), K=K.astype(np.float64))

    registrar_tempo(base, fidx, "da3", tempo, origem)
    return depth, K


def obter_obb_faixa(dados_dir, base, fidx, pontos_faixa_px, depth, K, historico):
    cache_path = os.path.join(dados_dir, "obb_faixa", f"frame_{fidx:06d}.npz")
    if os.path.exists(cache_path) and not CFG.RECOMPUTAR_OBB:
        d = np.load(cache_path)
        obb = o3d.geometry.OrientedBoundingBox(center=d["center"], R=d["R"], extent=d["extent"])
        obb.color = (0, 0, 1)
        tempo = resgatar_tempo_original(historico, base, "obb_faixa", frame=fidx)
        registrar_tempo(base, fidx, "obb_faixa", tempo, "cache")
        return obb

    t0 = time.perf_counter()
    faixa3d = G.projetar_pixels_3d(pontos_faixa_px, depth, K)
    if len(faixa3d) < 3:
        raise RuntimeError("faixa sem profundidade válida")
    obb = G.construir_obb_faixa(faixa3d, box_height=CFG.BOX_HEIGHT,
                                margin=CFG.OBB_MARGIN, tol_solo=CFG.OBB_TOL_SOLO,
                                modo=CFG.OBB_MODO)
    tempo = time.perf_counter() - t0
    salvar_npz(cache_path, center=np.asarray(obb.center), R=np.asarray(obb.R),
              extent=np.asarray(obb.extent), faixa3d=faixa3d)
    registrar_tempo(base, fidx, "obb_faixa", tempo, "calculado")
    return obb


def obter_obb_faixa_pontos(dados_dir, base, fidx, pontos_faixa_px, depth, K, historico):
    cache_path = os.path.join(dados_dir, "obb_faixa_pontos", f"frame_{fidx:06d}.npz")
    if os.path.exists(cache_path) and not CFG.RECOMPUTAR_OBB_PONTOS:
        d = np.load(cache_path)
        obb = o3d.geometry.OrientedBoundingBox(center=d["center"], R=d["R"], extent=d["extent"])
        obb.color = (0, 1, 0)
        tempo = resgatar_tempo_original(historico, base, "obb_faixa_pontos", frame=fidx)
        registrar_tempo(base, fidx, "obb_faixa_pontos", tempo, "cache")
        return obb

    t0 = time.perf_counter()
    faixa3d = G.projetar_pixels_3d(pontos_faixa_px, depth, K)
    if len(faixa3d) < 3:
        raise RuntimeError("faixa sem profundidade válida")
    obb = G.construir_obb_faixa_pontos(faixa3d, box_height=CFG.BOX_HEIGHT,
                                       margin=CFG.OBB_MARGIN, tol_solo=CFG.OBB_TOL_SOLO,
                                       modo=CFG.OBB_MODO)
    tempo = time.perf_counter() - t0
    salvar_npz(cache_path, center=np.asarray(obb.center), R=np.asarray(obb.R),
              extent=np.asarray(obb.extent), faixa3d=faixa3d)
    registrar_tempo(base, fidx, "obb_faixa_pontos", tempo, "calculado")
    return obb


def obter_poligonos_yolo_seg(dados_dir, base, fidx, frame_bgr, yolo_seg, historico):
    cache_path = os.path.join(dados_dir, "yolo_seg", f"frame_{fidx:06d}.json")
    if os.path.exists(cache_path) and not CFG.RECOMPUTAR_YOLO_SEG:
        polys = carregar_json(cache_path)["poligonos"]
        tempo = resgatar_tempo_original(historico, base, "yolo_seg", frame=fidx)
        registrar_tempo(base, fidx, "yolo_seg", tempo, "cache")
        return polys

    t0 = time.perf_counter()
    polys = yolo_seg.segmentar_frame(frame_bgr)
    tempo = time.perf_counter() - t0
    salvar_json(cache_path, {"frame": fidx, "poligonos": polys})
    registrar_tempo(base, fidx, "yolo_seg", tempo, "calculado")
    return polys


def obter_mesh_sam3d(dados_dir, base, fidx, obj_idx, frame_bgr, mask, depth, K, sam3d, historico):
    cache_path = os.path.join(dados_dir, "sam3d", f"frame_{fidx:06d}_obj{obj_idx:02d}.npz")
    if os.path.exists(cache_path) and not CFG.RECOMPUTAR_SAM3D:
        d = np.load(cache_path)
        verts = d["verts"]
        faces = d["faces"] if d["faces"].size else None
        tempo = resgatar_tempo_original(historico, base, "sam3d", frame=fidx, obj_idx=obj_idx)
        registrar_tempo(base, fidx, "sam3d", tempo, "cache", obj_idx=obj_idx)
        if verts.size == 0:
            return None, None
        return verts, faces

    t0 = time.perf_counter()
    verts, faces = sam3d.levantar_mesh(frame_bgr, mask, depth, K)
    tempo = time.perf_counter() - t0
    salvar_npz(cache_path,
              verts=verts if verts is not None else np.empty((0, 3)),
              faces=faces if faces is not None else np.empty((0, 3), dtype=np.int64))
    registrar_tempo(base, fidx, "sam3d", tempo, "calculado", obj_idx=obj_idx)
    return verts, faces


def _veiculo_pontos_limpos_na_obb(depth, K, mask, obb, z_min, z_max, min_pontos,
                                   erosao_px=3, sor_nb_vizinhos=20, sor_std_ratio=2.0,
                                   retornar_pontos=False):
    n_pixels_mascara_bruta = int(np.count_nonzero(mask))

    m = mask
    if erosao_px and erosao_px > 0 and np.any(m):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * erosao_px + 1, 2 * erosao_px + 1))
        m_erodida = cv2.erode(m, kernel)
        if np.any(m_erodida):
            m = m_erodida
    n_pixels_mascara_erodida = int(np.count_nonzero(m))

    def _saida(presente, n_dentro, pts_antes=None, pts_depois=None,
                pts_removidos=None, pts_dentro=None):
        r = {
            "presente": presente,
            "n_dentro": n_dentro,
            "n_pixels_mascara_bruta": n_pixels_mascara_bruta,
            "n_pixels_mascara_erodida": n_pixels_mascara_erodida,
            "n_pontos_antes_sor": 0 if pts_antes is None else int(len(pts_antes)),
            "n_pontos_depois_sor": 0 if pts_depois is None else int(len(pts_depois)),
        }
        if retornar_pontos:
            vazio = np.empty((0, 3), dtype=np.float64)
            r["pts_antes_sor"] = vazio if pts_antes is None else pts_antes
            r["pts_depois_sor"] = vazio if pts_depois is None else pts_depois
            r["pts_removidos_sor"] = vazio if pts_removidos is None else pts_removidos
            r["pts_dentro_obb"] = vazio if pts_dentro is None else pts_dentro
        return r

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return _saida(False, 0)

    uv = np.stack([xs, ys], axis=1)
    pts = G.projetar_pixels_3d(uv, depth, K)
    if len(pts) == 0:
        return _saida(False, 0)

    z = pts[:, 2]
    pts_antes_sor = pts[(z > z_min) & (z < z_max)]
    if len(pts_antes_sor) == 0:
        return _saida(False, 0)

    pts_depois_sor = pts_antes_sor
    pts_removidos_sor = np.empty((0, 3), dtype=np.float64)
    if sor_nb_vizinhos and sor_nb_vizinhos > 0 and len(pts_antes_sor) > sor_nb_vizinhos:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_antes_sor)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=sor_nb_vizinhos,
                                                std_ratio=sor_std_ratio)
        pts_depois_sor = pts_antes_sor[ind]
        if retornar_pontos:
            pts_removidos_sor = np.delete(pts_antes_sor, ind, axis=0)

    idx = G.pontos_dentro_obb(obb, pts_depois_sor)
    n = int(len(idx))
    return _saida(bool(n >= min_pontos), n, pts_antes_sor, pts_depois_sor,
                 pts_removidos_sor, pts_depois_sor[idx])


def processar_deteccoes(dados_dir, base, fidx, evento, frame_bgr, depth, K, obb, sam3d, yolo_seg, historico):
    det = evento["deteccoes"]
    h_img, w_img = frame_bgr.shape[:2]

    res_autos, mesh_intersecta = [], False
    res_vuln, pessoa_presente = [], False
    tempo_interseccao_3d = 0.0

    polys_frame = obter_poligonos_yolo_seg(dados_dir, base, fidx, frame_bgr, yolo_seg, historico) \
        if det["automoveis"] else []

    for obj_idx, obj in enumerate(det["automoveis"]):
        info = {"cls": obj["cls"], "conf": round(float(obj.get("conf", 0)), 4),
                "intersecta": False, "n_dentro": 0, "fracao": 0.0}
        try:
            mask = YoloSegMascara.mascara_de_bbox(polys_frame, obj["bbox"], w_img, h_img)
            if mask.sum() == 0:
                info["erro"] = "mascara_vazia_yolo_seg"
                res_autos.append(info)
                continue
            verts, faces = obter_mesh_sam3d(dados_dir, base, fidx, obj_idx, frame_bgr, mask,
                                           depth, K, sam3d, historico)
            if verts is not None and len(verts):
                t0 = time.perf_counter()
                r = G.mesh_intersecta_obb(verts, faces, obb,
                                          min_pontos=CFG.MIN_PONTOS_MESH,
                                          n_amostras=CFG.MESH_N_AMOSTRAS)
                tempo_interseccao_3d += time.perf_counter() - t0
                info.update(intersecta=r["intersecta"], n_dentro=r["n_dentro"],
                            fracao=r["fracao"])
                mesh_intersecta = mesh_intersecta or r["intersecta"]
            else:
                info["erro"] = "mesh_vazia"
        except Exception as ex:
            info["erro"] = f"sam3d: {ex}"
        res_autos.append(info)

    for obj in det["vulneraveis"]:
        mask = mascara_de_bbox_retangular(obj["bbox"], w_img, h_img)
        t0 = time.perf_counter()
        chk = G.pontos_cena_sob_mascara_na_obb(depth, K, mask, obb,
                                               z_min=CFG.Z_MIN, z_max=CFG.Z_MAX,
                                               min_pontos=CFG.MIN_PONTOS_PESSOA)
        tempo_interseccao_3d += time.perf_counter() - t0
        res_vuln.append({"cls": obj["cls"], "conf": round(float(obj.get("conf", 0)), 4),
                         "presente": chk["presente"], "n_dentro": chk["n_dentro"]})
        pessoa_presente = pessoa_presente or chk["presente"]

    registrar_tempo(base, fidx, "interseccao_3d", tempo_interseccao_3d, "calculado")

    return {
        "frame": fidx,
        "timestamp_ms": evento.get("timestamp_ms"),
        "alerta_risco_2d": evento.get("alerta_risco_2d"),
        "mesh_intersecta_faixa": bool(mesh_intersecta),
        "pessoa_presente_faixa": bool(pessoa_presente),
        "infracao": bool(mesh_intersecta and pessoa_presente),
        "automoveis": res_autos,
        "vulneraveis": res_vuln,
    }


def processar_deteccoes_nuvem_mascara(dados_dir, base, fidx, evento, frame_bgr, depth, K,
                                       obb, yolo_seg, historico):
    det = evento["deteccoes"]
    h_img, w_img = frame_bgr.shape[:2]

    res_autos, mesh_intersecta = [], False
    res_vuln, pessoa_presente = [], False
    tempo_interseccao_3d = 0.0

    polys_frame = obter_poligonos_yolo_seg(dados_dir, base, fidx, frame_bgr, yolo_seg, historico) \
        if det["automoveis"] else []

    for obj in det["automoveis"]:
        info = {"cls": obj["cls"], "conf": round(float(obj.get("conf", 0)), 4),
                "modo": "nuvem_mascara",
                "intersecta": False, "n_dentro": 0, "fracao": 0.0}
        try:
            mask = YoloSegMascara.mascara_de_bbox(polys_frame, obj["bbox"], w_img, h_img)
            if mask.sum() == 0:
                info["erro"] = "mascara_vazia_yolo_seg"
                res_autos.append(info)
                continue
            t0 = time.perf_counter()
            chk = _veiculo_pontos_limpos_na_obb(
                depth, K, mask, obb, z_min=CFG.Z_MIN, z_max=CFG.Z_MAX,
                min_pontos=CFG.MIN_PONTOS_VEICULO_PC, erosao_px=CFG.EROSAO_MASCARA_PX,
                sor_nb_vizinhos=CFG.SOR_NB_VIZINHOS, sor_std_ratio=CFG.SOR_STD_RATIO)
            tempo_interseccao_3d += time.perf_counter() - t0
            info.update(intersecta=bool(chk["presente"]), n_dentro=int(chk["n_dentro"]),
                        n_pixels_mascara_bruta=chk["n_pixels_mascara_bruta"],
                        n_pixels_mascara_erodida=chk["n_pixels_mascara_erodida"],
                        n_pontos_antes_sor=chk["n_pontos_antes_sor"],
                        n_pontos_depois_sor=chk["n_pontos_depois_sor"])
            mesh_intersecta = mesh_intersecta or chk["presente"]
        except Exception as ex:
            info["erro"] = f"nuvem_mascara: {ex}"
        res_autos.append(info)

    for obj in det["vulneraveis"]:
        mask = mascara_de_bbox_retangular(obj["bbox"], w_img, h_img)
        t0 = time.perf_counter()
        chk = G.pontos_cena_sob_mascara_na_obb(depth, K, mask, obb,
                                               z_min=CFG.Z_MIN, z_max=CFG.Z_MAX,
                                               min_pontos=CFG.MIN_PONTOS_PESSOA)
        tempo_interseccao_3d += time.perf_counter() - t0
        res_vuln.append({"cls": obj["cls"], "conf": round(float(obj.get("conf", 0)), 4),
                         "presente": chk["presente"], "n_dentro": chk["n_dentro"]})
        pessoa_presente = pessoa_presente or chk["presente"]

    registrar_tempo(base, fidx, "interseccao_3d", tempo_interseccao_3d, "calculado")

    return {
        "frame": fidx,
        "timestamp_ms": evento.get("timestamp_ms"),
        "alerta_risco_2d": evento.get("alerta_risco_2d"),
        "mesh_intersecta_faixa": bool(mesh_intersecta),
        "pessoa_presente_faixa": bool(pessoa_presente),
        "infracao": bool(mesh_intersecta and pessoa_presente),
        "automoveis": res_autos,
        "vulneraveis": res_vuln,
    }


def montar_saida(base, video_path, frames_out, obb_metodo="pca"):
    resumo = {
        "frames_processados": len(frames_out),
        "frames_com_intersecao_mesh": [r["frame"] for r in frames_out if r.get("mesh_intersecta_faixa")],
        "frames_com_pessoa": [r["frame"] for r in frames_out if r.get("pessoa_presente_faixa")],
        "frames_infracao": [r["frame"] for r in frames_out if r.get("infracao")],
    }
    return {
        "metadados": {
            "video": base,
            "video_path": video_path,
            "modelo_yolo_bb": CFG.YOLO_BB_MODEL,
            "modelo_yolo_seg": CFG.YOLO_SEG_MODEL,
            "modelo_depth": CFG.DA3_MODEL,
            "selecao_frames": CFG.SELECAO_FRAMES,
            "box_height_m": CFG.BOX_HEIGHT,
            "obb_modo": CFG.OBB_MODO,
            "obb_metodo": obb_metodo,
            "min_pontos_mesh": CFG.MIN_PONTOS_MESH,
            "referencial": "camera (extrinseca NAO aplicada)",
            "definicao_infracao": "automovel(mesh) intersecta a faixa E pessoa presente na faixa",
        },
        "resumo": resumo,
        "frames": frames_out,
    }


def caminho_saida_video(dados_dir, base):
    return os.path.join(dados_dir, "infracoes", f"{base}{CFG.SUFIXO_SAIDA}.json")


def caminho_saida_2d(dados_dir, base):
    return os.path.join(dados_dir, "interseccao_2d", f"{base}_interseccao_2d.json")


def caminho_saida_2d_coocorrencia(dados_dir, base):
    return os.path.join(dados_dir, "interseccao_2d", f"{base}_interseccao_2d_coocorrencia.json")


def caminho_saida_pontos(dados_dir, base):
    return os.path.join(dados_dir, "infracoes", f"{base}{CFG.SUFIXO_SAIDA_PONTOS}.json")


def caminho_saida_nuvem_mascara(dados_dir, base):
    return os.path.join(dados_dir, "infracoes", f"{base}{CFG.SUFIXO_SAIDA_NUVEM_MASCARA}.json")


def _frames_com_interseccao_2d(base, eventos, poligono_2d):
    frames_out = []
    for evento in eventos:
        det = evento["deteccoes"]
        t0 = time.perf_counter()
        inter = interseccao_2d(det["vulneraveis"], det["automoveis"], poligono_2d)
        tempo = time.perf_counter() - t0
        registrar_tempo(base, evento["frame"], "interseccao_2d", tempo, "calculado")
        frames_out.append({
            "frame": evento["frame"],
            "timestamp_ms": evento.get("timestamp_ms"),
            "alerta_risco_2d": evento.get("alerta_risco_2d"),
            "interseccao_2d": inter,
        })
    return frames_out


def _anexar_tempos_e_salvar(dados_dir, base, frames_out):
    resumo = resumo_tempos_por_video(base)
    tempo_total_video = 0.0
    if not resumo.empty:
        mapa = dict(zip(resumo["frame"], resumo["tempo_total_s"]))
        for item in frames_out:
            item["tempo_total_execucao_s"] = round(mapa.get(item["frame"], float("nan")), 4)
        resumo_path = os.path.join(dados_dir, "tempos_execucao_resumo.csv")
        resumo.to_csv(resumo_path, index=False)
        tempo_total_video = float(resumo["tempo_total_s"].sum())
        log(f"  tempo total de execução (soma por frame): {tempo_total_video:.2f}s "
            f"-> {resumo_path}")
    salvar_tempos(dados_dir, base)
    return tempo_total_video


def processar_video_2d(base, video_path, pontos_faixa_px, yolo_bb):
    dados_dir = pasta_dados_video(base)
    historico = carregar_historico_tempos(dados_dir)
    leitor = LeitorVideo(video_path)
    poligono_2d = construir_poligono_faixa_2d(pontos_faixa_px, margem=CFG.FAIXA_2D_MARGEM_PX)

    eventos = obter_deteccoes_yolo_bb(dados_dir, base, leitor, yolo_bb, poligono_2d, historico)
    log(f"  YOLO bbox: {len(eventos)} frames com detecção na faixa")

    frames_out = _frames_com_interseccao_2d(base, eventos, poligono_2d)

    n_risco = sum(1 for f in frames_out if f["interseccao_2d"]["algum_risco"])
    tempo_total_video = _anexar_tempos_e_salvar(dados_dir, base, frames_out)
    out = {
        "metadados": {
            "video": base, "video_path": video_path,
            "modelo_yolo_bb": CFG.YOLO_BB_MODEL,
            "margem_faixa_px": CFG.FAIXA_2D_MARGEM_PX,
            "n_frames_processados": len(frames_out),
            "n_frames_com_risco_2d": n_risco,
            "tempo_total_execucao_video_s": round(tempo_total_video, 2),
        },
        "frames": frames_out,
    }
    out_path = caminho_saida_2d(dados_dir, base)
    salvar_json(out_path, out)
    log(f"  salvo: {out_path}  (frames com risco 2D: {n_risco})")


def processar_video_2d_coocorrencia(base, video_path, pontos_faixa_px, yolo_bb):
    dados_dir = pasta_dados_video(base)
    historico = carregar_historico_tempos(dados_dir)
    leitor = LeitorVideo(video_path)
    poligono_2d = construir_poligono_faixa_2d(pontos_faixa_px, margem=CFG.FAIXA_2D_MARGEM_PX)

    eventos_brutos = obter_deteccoes_yolo_bb(dados_dir, base, leitor, yolo_bb, poligono_2d, historico)
    log(f"  YOLO bbox: {len(eventos_brutos)} frames com detecção na faixa")

    eventos_sel = obter_eventos_selecionados(dados_dir, base, eventos_brutos, historico)
    log(f"  Co-ocorrência ({CFG.SELECAO_FRAMES}): {len(eventos_sel)} frames selecionados")

    frames_out = _frames_com_interseccao_2d(base, eventos_sel, poligono_2d)

    n_risco = sum(1 for f in frames_out if f["interseccao_2d"]["algum_risco"])
    tempo_total_video = _anexar_tempos_e_salvar(dados_dir, base, frames_out)
    out = {
        "metadados": {
            "video": base, "video_path": video_path,
            "modelo_yolo_bb": CFG.YOLO_BB_MODEL,
            "margem_faixa_px": CFG.FAIXA_2D_MARGEM_PX,
            "criterio_coocorrencia": CFG.SELECAO_FRAMES,
            "n_frames_brutos": len(eventos_brutos),
            "n_frames_processados": len(frames_out),
            "n_frames_com_risco_2d": n_risco,
            "tempo_total_execucao_video_s": round(tempo_total_video, 2),
        },
        "frames": frames_out,
    }
    out_path = caminho_saida_2d_coocorrencia(dados_dir, base)
    salvar_json(out_path, out)
    log(f"  salvo: {out_path}  (frames com risco 2D: {n_risco})")


def processar_video_pontos(base, video_path, pontos_faixa_px, yolo_bb, yolo_seg, da3, sam3d):
    dados_dir = pasta_dados_video(base)
    historico = carregar_historico_tempos(dados_dir)
    leitor = LeitorVideo(video_path)
    poligono_2d = construir_poligono_faixa_2d(pontos_faixa_px, margem=CFG.FAIXA_2D_MARGEM_PX)

    eventos_brutos = obter_deteccoes_yolo_bb(dados_dir, base, leitor, yolo_bb, poligono_2d, historico)
    log(f"  YOLO bbox: {len(eventos_brutos)} frames com detecção na faixa")

    eventos_sel = obter_eventos_selecionados(dados_dir, base, eventos_brutos, historico)
    log(f"  Co-ocorrência ({CFG.SELECAO_FRAMES}): {len(eventos_sel)} frames selecionados")

    if not eventos_sel:
        log("  Nenhum frame após o filtro de co-ocorrência; nada a reconstruir em 3D.")
        _anexar_tempos_e_salvar(dados_dir, base, [])
        salvar_json(caminho_saida_pontos(dados_dir, base),
                   montar_saida(base, video_path, [], obb_metodo="pontos"))
        return

    resultados = {}
    for n, evento in enumerate(eventos_sel, 1):
        fidx = evento["frame"]
        frame_bgr = leitor.ler_bgr(fidx)
        if frame_bgr is None:
            resultados[fidx] = {"frame": fidx, "erro": "frame_nao_lido"}
            continue

        try:
            depth, K = obter_da3(dados_dir, base, fidx, frame_bgr, da3, historico)
            obb = obter_obb_faixa_pontos(dados_dir, base, fidx, pontos_faixa_px, depth, K, historico)
        except Exception as ex:
            resultados[fidx] = {"frame": fidx, "erro": f"da3/obb_pontos: {ex}"}
            continue

        try:
            resultados[fidx] = processar_deteccoes(dados_dir, base, fidx, evento, frame_bgr,
                                                    depth, K, obb, sam3d, yolo_seg, historico)
        except Exception:
            resultados[fidx] = {"frame": fidx, "erro": traceback.format_exc(limit=2)}

        if CFG.VERBOSE and (n % 50 == 0 or n == len(eventos_sel)):
            log(f"    ... {n}/{len(eventos_sel)} frames")

    frames_out = [resultados[f] for f in sorted(resultados)]
    tempo_total_video = _anexar_tempos_e_salvar(dados_dir, base, frames_out)
    out = montar_saida(base, video_path, frames_out, obb_metodo="pontos")
    out["metadados"]["tempo_total_execucao_video_s"] = round(tempo_total_video, 2)
    out_path = caminho_saida_pontos(dados_dir, base)
    salvar_json(out_path, out)
    n_inf = len(out["resumo"]["frames_infracao"])
    log(f"  salvo: {out_path}  (infrações via pontos: {n_inf})")


def processar_video_nuvem_mascara(base, video_path, pontos_faixa_px, yolo_bb, yolo_seg, da3):
    dados_dir = pasta_dados_video(base)
    historico = carregar_historico_tempos(dados_dir)
    leitor = LeitorVideo(video_path)
    poligono_2d = construir_poligono_faixa_2d(pontos_faixa_px, margem=CFG.FAIXA_2D_MARGEM_PX)

    eventos_brutos = obter_deteccoes_yolo_bb(dados_dir, base, leitor, yolo_bb, poligono_2d, historico)
    log(f"  YOLO bbox: {len(eventos_brutos)} frames com detecção na faixa")

    eventos_sel = obter_eventos_selecionados(dados_dir, base, eventos_brutos, historico)
    log(f"  Co-ocorrência ({CFG.SELECAO_FRAMES}): {len(eventos_sel)} frames selecionados")

    if not eventos_sel:
        log("  Nenhum frame após o filtro de co-ocorrência; nada a reconstruir em 3D.")
        _anexar_tempos_e_salvar(dados_dir, base, [])
        out = montar_saida(base, video_path, [], obb_metodo="pca")
        out["metadados"]["modo_veiculo"] = "nuvem_mascara"
        out["metadados"]["min_pontos_veiculo_pc"] = CFG.MIN_PONTOS_VEICULO_PC
        out["metadados"]["erosao_mascara_px"] = CFG.EROSAO_MASCARA_PX
        out["metadados"]["sor_nb_vizinhos"] = CFG.SOR_NB_VIZINHOS
        out["metadados"]["sor_std_ratio"] = CFG.SOR_STD_RATIO
        out["metadados"]["min_pontos_mesh"] = None
        salvar_json(caminho_saida_nuvem_mascara(dados_dir, base), out)
        return

    resultados = {}
    for n, evento in enumerate(eventos_sel, 1):
        fidx = evento["frame"]
        frame_bgr = leitor.ler_bgr(fidx)
        if frame_bgr is None:
            resultados[fidx] = {"frame": fidx, "erro": "frame_nao_lido"}
            continue

        try:
            depth, K = obter_da3(dados_dir, base, fidx, frame_bgr, da3, historico)
            obb = obter_obb_faixa(dados_dir, base, fidx, pontos_faixa_px, depth, K, historico)
        except Exception as ex:
            resultados[fidx] = {"frame": fidx, "erro": f"da3/obb: {ex}"}
            continue

        try:
            resultados[fidx] = processar_deteccoes_nuvem_mascara(
                dados_dir, base, fidx, evento, frame_bgr, depth, K, obb, yolo_seg, historico)
        except Exception:
            resultados[fidx] = {"frame": fidx, "erro": traceback.format_exc(limit=2)}

        if CFG.VERBOSE and (n % 50 == 0 or n == len(eventos_sel)):
            log(f"    ... {n}/{len(eventos_sel)} frames")

    frames_out = [resultados[f] for f in sorted(resultados)]
    tempo_total_video = _anexar_tempos_e_salvar(dados_dir, base, frames_out)
    out = montar_saida(base, video_path, frames_out, obb_metodo="pca")
    out["metadados"]["tempo_total_execucao_video_s"] = round(tempo_total_video, 2)
    out["metadados"]["modo_veiculo"] = "nuvem_mascara"
    out["metadados"]["min_pontos_veiculo_pc"] = CFG.MIN_PONTOS_VEICULO_PC
    out["metadados"]["erosao_mascara_px"] = CFG.EROSAO_MASCARA_PX
    out["metadados"]["sor_nb_vizinhos"] = CFG.SOR_NB_VIZINHOS
    out["metadados"]["sor_std_ratio"] = CFG.SOR_STD_RATIO
    out["metadados"]["min_pontos_mesh"] = None
    out_path = caminho_saida_nuvem_mascara(dados_dir, base)
    salvar_json(out_path, out)
    n_inf = len(out["resumo"]["frames_infracao"])
    log(f"  salvo: {out_path}  (infrações via nuvem_mascara: {n_inf})")


def processar_video(base, video_path, pontos_faixa_px, yolo_bb, yolo_seg, da3, sam3d):
    dados_dir = pasta_dados_video(base)
    historico = carregar_historico_tempos(dados_dir)
    leitor = LeitorVideo(video_path)
    poligono_2d = construir_poligono_faixa_2d(pontos_faixa_px, margem=CFG.FAIXA_2D_MARGEM_PX)

    eventos_brutos = obter_deteccoes_yolo_bb(dados_dir, base, leitor, yolo_bb, poligono_2d, historico)
    log(f"  YOLO bbox: {len(eventos_brutos)} frames com detecção na faixa")

    eventos_sel = obter_eventos_selecionados(dados_dir, base, eventos_brutos, historico)
    log(f"  Co-ocorrência ({CFG.SELECAO_FRAMES}): {len(eventos_sel)} frames selecionados")

    if not eventos_sel:
        log("  Nenhum frame após o filtro de co-ocorrência; nada a reconstruir em 3D.")
        _anexar_tempos_e_salvar(dados_dir, base, [])
        salvar_json(caminho_saida_video(dados_dir, base), montar_saida(base, video_path, []))
        return

    resultados = {}
    for n, evento in enumerate(eventos_sel, 1):
        fidx = evento["frame"]
        frame_bgr = leitor.ler_bgr(fidx)
        if frame_bgr is None:
            resultados[fidx] = {"frame": fidx, "erro": "frame_nao_lido"}
            continue

        try:
            depth, K = obter_da3(dados_dir, base, fidx, frame_bgr, da3, historico)
            obb = obter_obb_faixa(dados_dir, base, fidx, pontos_faixa_px, depth, K, historico)
        except Exception as ex:
            resultados[fidx] = {"frame": fidx, "erro": f"da3/obb: {ex}"}
            continue

        try:
            resultados[fidx] = processar_deteccoes(dados_dir, base, fidx, evento, frame_bgr,
                                                   depth, K, obb, sam3d, yolo_seg, historico)
        except Exception:
            resultados[fidx] = {"frame": fidx, "erro": traceback.format_exc(limit=2)}

        if CFG.VERBOSE and (n % 50 == 0 or n == len(eventos_sel)):
            log(f"    ... {n}/{len(eventos_sel)} frames")

    frames_out = [resultados[f] for f in sorted(resultados)]
    tempo_total_video = _anexar_tempos_e_salvar(dados_dir, base, frames_out)
    out = montar_saida(base, video_path, frames_out)
    out["metadados"]["tempo_total_execucao_video_s"] = round(tempo_total_video, 2)
    out_path = caminho_saida_video(dados_dir, base)
    salvar_json(out_path, out)
    n_inf = len(out["resumo"]["frames_infracao"])
    log(f"  salvo: {out_path}  (infrações: {n_inf})")


def _main_fluxo_2d(videos):
    log("== Fluxo 2D em LOTE (YOLO bbox + interseccao_2d) ==")
    yolo_bb = YoloBBox()

    for base, video_path in videos:
        log(f"\n================ VÍDEO: {base} ================")
        try:
            pontos = carregar_pontos_faixa(base)
        except FileNotFoundError as ex:
            log(f"{ex}\n  (pulando vídeo)")
            continue

        out_path = caminho_saida_2d(os.path.join(CFG.DADOS_DIR, base), base)
        if not CFG.SOBRESCREVER and os.path.exists(out_path):
            log(f"Já processado ({out_path}); pulando.")
            continue

        try:
            processar_video_2d(base, video_path, pontos, yolo_bb)
        except Exception:
            log("Falha no vídeo:\n" + traceback.format_exc(limit=3))

    log("\nLote concluído (fluxo 2D).")


def _main_fluxo_2d_coocorrencia(videos):
    log("== Fluxo 2D+co-ocorrência em LOTE (YOLO bbox + co-ocorrência + interseccao_2d) ==")
    yolo_bb = YoloBBox()

    for base, video_path in videos:
        log(f"\n================ VÍDEO: {base} ================")
        try:
            pontos = carregar_pontos_faixa(base)
        except FileNotFoundError as ex:
            log(f"{ex}\n  (pulando vídeo)")
            continue

        out_path = caminho_saida_2d_coocorrencia(os.path.join(CFG.DADOS_DIR, base), base)
        if not CFG.SOBRESCREVER and os.path.exists(out_path):
            log(f"Já processado ({out_path}); pulando.")
            continue

        try:
            processar_video_2d_coocorrencia(base, video_path, pontos, yolo_bb)
        except Exception:
            log("Falha no vídeo:\n" + traceback.format_exc(limit=3))

    log("\nLote concluído (fluxo 2D+co-ocorrência).")


def _main_fluxo_pontos(videos):
    log("== Fluxo PONTOS em LOTE (YOLO bbox + YOLO-seg + DA3 + SAM3D, OBB direto dos pontos, com cache) ==")
    yolo_bb = YoloBBox()
    yolo_seg = YoloSegMascara()
    da3 = DA3()
    sam3d = SAM3D()

    for base, video_path in videos:
        log(f"\n================ VÍDEO: {base} ================")
        try:
            pontos = carregar_pontos_faixa(base)
        except FileNotFoundError as ex:
            log(f"{ex}\n  (pulando vídeo)")
            continue

        out_path = caminho_saida_pontos(os.path.join(CFG.DADOS_DIR, base), base)
        if not CFG.SOBRESCREVER and os.path.exists(out_path):
            log(f"Já processado ({out_path}); pulando.")
            continue

        try:
            processar_video_pontos(base, video_path, pontos, yolo_bb, yolo_seg, da3, sam3d)
        except Exception:
            log("Falha no vídeo:\n" + traceback.format_exc(limit=3))

    log("\nLote concluído (fluxo PONTOS).")


def _main_fluxo_nuvem_mascara(videos):
    log("== Fluxo NUVEM_MASCARA em LOTE (YOLO bbox + YOLO-seg + DA3, OBB via PCA, SEM SAM3D, com cache) ==")
    yolo_bb = YoloBBox()
    yolo_seg = YoloSegMascara()
    da3 = DA3()

    for base, video_path in videos:
        log(f"\n================ VÍDEO: {base} ================")
        try:
            pontos = carregar_pontos_faixa(base)
        except FileNotFoundError as ex:
            log(f"{ex}\n  (pulando vídeo)")
            continue

        out_path = caminho_saida_nuvem_mascara(os.path.join(CFG.DADOS_DIR, base), base)
        if not CFG.SOBRESCREVER and os.path.exists(out_path):
            log(f"Já processado ({out_path}); pulando.")
            continue

        try:
            processar_video_nuvem_mascara(base, video_path, pontos, yolo_bb, yolo_seg, da3)
        except Exception:
            log("Falha no vídeo:\n" + traceback.format_exc(limit=3))

    log("\nLote concluído (fluxo NUVEM_MASCARA).")


def _main_fluxo_completo(videos):
    log("== Fluxo completo em LOTE (YOLO bbox + YOLO-seg + DA3 + SAM3D, com cache) ==")

    yolo_bb = YoloBBox()
    yolo_seg = YoloSegMascara()
    da3 = DA3()
    sam3d = SAM3D()

    for base, video_path in videos:
        log(f"\n================ VÍDEO: {base} ================")
        try:
            pontos = carregar_pontos_faixa(base)
        except FileNotFoundError as ex:
            log(f"{ex}\n  (pulando vídeo)")
            continue

        out_path = caminho_saida_video(os.path.join(CFG.DADOS_DIR, base), base)
        if not CFG.SOBRESCREVER and os.path.exists(out_path):
            log(f"Já processado ({out_path}); pulando.")
            continue

        try:
            processar_video(base, video_path, pontos, yolo_bb, yolo_seg, da3, sam3d)
        except Exception:
            log("Falha no vídeo:\n" + traceback.format_exc(limit=3))

    log("\nLote concluído.")


def main():
    videos = encontrar_videos(CFG.VIDEOS_DIR)
    if not videos:
        log(f"Nenhum vídeo encontrado em {CFG.VIDEOS_DIR}")
        return
    log(f"{len(videos)} vídeos encontrados.")

    if CFG.FLUXO == "2d":
        _main_fluxo_2d(videos)
    elif CFG.FLUXO == "2d_coocorrencia":
        _main_fluxo_2d_coocorrencia(videos)
    elif CFG.FLUXO == "pontos":
        _main_fluxo_pontos(videos)
    elif CFG.FLUXO == "nuvem_mascara":
        _main_fluxo_nuvem_mascara(videos)
    else:
        _main_fluxo_completo(videos)


if __name__ == "__main__":
    main()