#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import json
import tempfile
import traceback
import numpy as np
import cv2
import time

import geometria as G

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CFG:
    VIDEOS_DIR = os.path.join(RAIZ, "data", "video")
    VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv", ".m4v")

    RESULTS_ROOT = os.path.join(RAIZ, "data", "output_naive")

    PONTOS_FAIXA_DIR = os.path.join(RAIZ, "data", "calib")

    SUFIXO_SAIDA = "_infracoes_3d"
    SOBRESCREVER = False

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

    VERBOSE = True
    SALVAR_DEBUG_PRIMEIRO = True
    DEBUG_DIR = os.path.join(RAIZ, "dados", "debug_3d")


def log(*a):
    if CFG.VERBOSE:
        print(*a, flush=True)


class LeitorVideo:
    def __init__(self, path):
        self.path = path
        self.backend = None
        try:
            import decord
            self.vr = decord.VideoReader(path)
            self.backend = "decord"
            self.n = len(self.vr)
        except Exception:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Não consegui abrir o vídeo: {path}")
            self.backend = "opencv"
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log(f"Vídeo aberto via {self.backend} | {self.n} frames")

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


import glob


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


def encontrar_jsons_do_video(results_root, base):
    sufixo_out = CFG.SUFIXO_SAIDA + ".json"
    achados = []
    for caminho in glob.glob(os.path.join(results_root, "**", "*.json"), recursive=True):
        nome = os.path.basename(caminho)
        if nome.endswith(sufixo_out):
            continue
        if _casa_prefixo(nome, base):
            achados.append(caminho)
    return sorted(achados)


def caminho_saida(json_in):
    d = os.path.dirname(json_in)
    stem = os.path.splitext(os.path.basename(json_in))[0]
    return os.path.join(d, stem + CFG.SUFIXO_SAIDA + ".json")


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


def processar_deteccoes(evento, frame_bgr, depth, K, obb, sam3d):
    fidx = evento["frame"]
    det = evento["deteccoes"]
    h_img, w_img = frame_bgr.shape[:2]

    res_autos, mesh_intersecta = [], False
    res_vuln, pessoa_presente = [], False

    for obj in det["automoveis"]:
        mask = G.mascara_de_poligono(obj["polygon"], w_img, h_img)
        info = {"cls": obj["cls"], "conf": round(float(obj.get("conf", 0)), 4),
                "intersecta": False, "n_dentro": 0, "fracao": 0.0}
        try:
            verts, faces = sam3d.levantar_mesh(frame_bgr, mask, depth, K)
            if verts is not None and len(verts):
                r = G.mesh_intersecta_obb(verts, faces, obb,
                                          min_pontos=CFG.MIN_PONTOS_MESH,
                                          n_amostras=CFG.MESH_N_AMOSTRAS)
                info.update(intersecta=r["intersecta"], n_dentro=r["n_dentro"],
                            fracao=r["fracao"])
                mesh_intersecta = mesh_intersecta or r["intersecta"]
            else:
                info["erro"] = "mesh_vazia"
        except Exception as ex:
            info["erro"] = f"sam3d: {ex}"
        res_autos.append(info)

    for obj in det["vulneraveis"]:
        mask = G.mascara_de_poligono(obj["polygon"], w_img, h_img)
        chk = G.pontos_cena_sob_mascara_na_obb(depth, K, mask, obb,
                                               z_min=CFG.Z_MIN, z_max=CFG.Z_MAX,
                                               min_pontos=CFG.MIN_PONTOS_PESSOA)
        res_vuln.append({"cls": obj["cls"], "conf": round(float(obj.get("conf", 0)), 4),
                         "presente": chk["presente"], "n_dentro": chk["n_dentro"]})
        pessoa_presente = pessoa_presente or chk["presente"]

    return {
        "frame": fidx,
        "timestamp_ms": evento.get("timestamp_ms"),
        "alerta_risco_json": evento.get("alerta_risco"),
        "mesh_intersecta_faixa": bool(mesh_intersecta),
        "pessoa_presente_faixa": bool(pessoa_presente),
        "infracao": bool(mesh_intersecta and pessoa_presente),
        "automoveis": res_autos,
        "vulneraveis": res_vuln,
    }


def montar_saida(json_in, meta_json, frames_out):
    resumo = {
        "frames_processados": len(frames_out),
        "frames_com_intersecao_mesh": [r["frame"] for r in frames_out if r.get("mesh_intersecta_faixa")],
        "frames_com_pessoa": [r["frame"] for r in frames_out if r.get("pessoa_presente_faixa")],
        "frames_infracao": [r["frame"] for r in frames_out if r.get("infracao")],
    }
    return {
        "metadados": {
            "origem_json": json_in,
            "metadados_json_origem": meta_json,
            "modelo_depth": CFG.DA3_MODEL,
            "selecao_frames": CFG.SELECAO_FRAMES,
            "box_height_m": CFG.BOX_HEIGHT,
            "obb_modo": CFG.OBB_MODO,
            "min_pontos_mesh": CFG.MIN_PONTOS_MESH,
            "referencial": "camera (extrinseca NAO aplicada)",
            "definicao_infracao": "automovel(mesh) intersecta a faixa E pessoa presente na faixa",
        },
        "resumo": resumo,
        "frames": frames_out,
    }


def processar_video(base, video_path, jsons_in, pontos_faixa_px, da3, sam3d):
    leitor = LeitorVideo(video_path)

    tarefas, metas, resultados = {}, {}, {}
    for jp in jsons_in:
        with open(jp) as f:
            dj = json.load(f)
        sel = selecionar_eventos(dj["eventos"])
        tarefas[jp] = {e["frame"]: e for e in sel}
        metas[jp] = dj.get("metadados", {})
        resultados[jp] = {}
        log(f"  {os.path.relpath(jp, CFG.RESULTS_ROOT)}: {len(sel)} frames selecionados")

    todos_frames = sorted(set().union(*[set(t.keys()) for t in tarefas.values()])) \
        if tarefas else []
    log(f"  {len(todos_frames)} frames únicos a reconstruir (DA3 roda 1x por frame)")

    for n, fidx in enumerate(todos_frames, 1):
        frame_bgr = leitor.ler_bgr(fidx)
        if frame_bgr is None:
            for jp, fmap in tarefas.items():
                if fidx in fmap:
                    resultados[jp][fidx] = {"frame": fidx, "erro": "frame_nao_lido"}
            continue
        try:
            depth, K = da3.inferir(frame_bgr)
            faixa3d = G.projetar_pixels_3d(pontos_faixa_px, depth, K)
            if len(faixa3d) < 3:
                raise RuntimeError("faixa sem profundidade válida")
            obb = G.construir_obb_faixa(faixa3d, box_height=CFG.BOX_HEIGHT,
                                        margin=CFG.OBB_MARGIN, tol_solo=CFG.OBB_TOL_SOLO,
                                        modo=CFG.OBB_MODO)
        except Exception as ex:
            for jp, fmap in tarefas.items():
                if fidx in fmap:
                    resultados[jp][fidx] = {"frame": fidx, "erro": f"da3/obb: {ex}"}
            continue

        for jp, fmap in tarefas.items():
            ev = fmap.get(fidx)
            if ev is None:
                continue
            try:
                resultados[jp][fidx] = processar_deteccoes(ev, frame_bgr, depth, K, obb, sam3d)
            except Exception:
                resultados[jp][fidx] = {"frame": fidx, "erro": traceback.format_exc(limit=2)}

        if CFG.VERBOSE and (n % 50 == 0 or n == len(todos_frames)):
            log(f"    ... {n}/{len(todos_frames)} frames")

    for jp in jsons_in:
        frames_out = [resultados[jp][f] for f in sorted(resultados[jp])]
        out = montar_saida(jp, metas[jp], frames_out)
        out_path = caminho_saida(jp)
        with open(out_path, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        n_inf = len(out["resumo"]["frames_infracao"])
        log(f"  salvo: {os.path.relpath(out_path, CFG.RESULTS_ROOT)}  (infrações: {n_inf})")


def main():
    log("== Pipeline de infração 3D em LOTE (DA3 + SAM3D) ==")

    videos = encontrar_videos(CFG.VIDEOS_DIR)
    if not videos:
        log(f"Nenhum vídeo encontrado em {CFG.VIDEOS_DIR}")
        return
    log(f"{len(videos)} vídeos encontrados.")

    da3 = DA3()
    sam3d = SAM3D()

    for base, video_path in videos:
        log(f"\n================ VÍDEO: {base} ================")
        try:
            pontos = carregar_pontos_faixa(base)
        except FileNotFoundError as ex:
            log(f"{ex}\n  (pulando vídeo)")
            continue

        jsons = encontrar_jsons_do_video(CFG.RESULTS_ROOT, base)
        if not jsons:
            log(f"Nenhum JSON de entrada com prefixo '{base}' em {CFG.RESULTS_ROOT} (pulando)")
            continue

        if not CFG.SOBRESCREVER:
            pendentes = [jp for jp in jsons if not os.path.exists(caminho_saida(jp))]
            if len(pendentes) < len(jsons):
                log(f"Pulando {len(jsons) - len(pendentes)} JSONs já processados")
            jsons = pendentes
        if not jsons:
            log("Tudo já processado para este vídeo.")
            continue

        log(f"{len(jsons)} JSON(s) de entrada para este vídeo.")
        try:
            processar_video(base, video_path, jsons, pontos, da3, sam3d)
        except Exception:
            log("Falha no vídeo:\n" + traceback.format_exc(limit=3))

    log("\nLote concluído.")


if __name__ == "__main__":
    main()
