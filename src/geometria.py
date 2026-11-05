
# Funções geométricas dos pipelines de infração
# projeta pixels em 3D a partir do depth e dos intrínsecos
# constrói a OBB da faixa de pedestres e 
# testa se a mesh do automóvel ou a nuvem de pontos sob uma máscara invadem essa OBB.

from __future__ import annotations
import numpy as np
import cv2
import open3d as o3d


def mascara_de_poligono(polygon, w: int, h: int) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    pts = np.round(pts).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def escalar_intrinseca(K: np.ndarray, src_wh, dst_wh) -> np.ndarray:
    (sw, sh), (dw, dh) = src_wh, dst_wh
    sx, sy = dw / float(sw), dh / float(sh)
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def projetar_pixels_3d(pixels_uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    uv = np.asarray(pixels_uv, dtype=np.int64)
    u = np.clip(uv[:, 0], 0, w - 1)
    v = np.clip(uv[:, 1], 0, h - 1)
    z = depth[v, u].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    u, v, z = u[valid], v[valid], z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def nuvem_cena_camera(depth: np.ndarray, K: np.ndarray, rgb: np.ndarray | None = None,
                      z_min: float = 0.1, z_max: float = 80.0):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten().astype(np.float64)
    mask = np.isfinite(z) & (z > z_min) & (z < z_max)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uf, vf, zf = u.flatten()[mask], v.flatten()[mask], z[mask]
    x = (uf - cx) * zf / fx
    y = (vf - cy) * zf / fy
    pts = np.stack([x, y, zf], axis=1)

    cores = None
    if rgb is not None:
        cores = rgb.reshape(-1, 3)[mask]
    return pts, cores, mask


def pcd_open3d(pts: np.ndarray, cores: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    if cores is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(cores, dtype=np.float64))
    return pcd


def construir_obb_faixa(pts3d: np.ndarray, box_height: float = 2.5,
                        margin: float = 0.05, tol_solo: float = 0.3,
                        modo: str = "acima_do_solo"
                        ) -> o3d.geometry.OrientedBoundingBox:
    P = np.asarray(pts3d, dtype=np.float64)
    if P.shape[0] < 3:
        raise ValueError("São necessários >= 3 pontos 3D válidos para a faixa.")

    centro = P.mean(axis=0)
    Q = P - centro
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    n = Vt[2]

    if n[1] > 0:
        n = -n

    R = np.stack([e1, e2, n], axis=1)

    proj = Q @ R
    span = proj.max(axis=0) - proj.min(axis=0)

    if modo == "acima_do_solo":
        altura = box_height + tol_solo
        centro = centro + n * (box_height - tol_solo) / 2.0
    else:
        altura = box_height

    extent = np.array([span[0] + margin, span[1] + margin, altura], dtype=np.float64)
    obb = o3d.geometry.OrientedBoundingBox(center=centro, R=R, extent=extent)
    obb.color = (0, 0, 1)
    return obb


def pontos_dentro_obb(obb: o3d.geometry.OrientedBoundingBox, pontos: np.ndarray) -> np.ndarray:
    if len(pontos) == 0:
        return np.array([], dtype=np.int64)
    v = o3d.utility.Vector3dVector(np.asarray(pontos, dtype=np.float64))
    idx = obb.get_point_indices_within_bounding_box(v)
    return np.asarray(idx, dtype=np.int64)


def amostrar_mesh(vertices: np.ndarray, faces: np.ndarray | None,
                  n_amostras: int = 20000) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if faces is None or len(faces) == 0:
        return vertices
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
        amostras, _ = trimesh.sample.sample_surface(m, n_amostras)
        return np.vstack([vertices, np.asarray(amostras, dtype=np.float64)])
    except Exception:
        return vertices


def mesh_intersecta_obb(vertices: np.ndarray, faces: np.ndarray | None,
                        obb: o3d.geometry.OrientedBoundingBox,
                        min_pontos: int = 50, n_amostras: int = 20000):
    pts = amostrar_mesh(vertices, faces, n_amostras=n_amostras)
    idx = pontos_dentro_obb(obb, pts)
    n_dentro = int(len(idx))
    n_total = int(len(pts))
    frac = n_dentro / n_total if n_total else 0.0
    return {
        "intersecta": bool(n_dentro >= min_pontos),
        "n_dentro": n_dentro,
        "n_total": n_total,
        "fracao": round(frac, 4),
    }


def pontos_cena_sob_mascara_na_obb(depth: np.ndarray, K: np.ndarray,
                                   mask_bin: np.ndarray,
                                   obb: o3d.geometry.OrientedBoundingBox,
                                   z_min: float = 0.1, z_max: float = 80.0,
                                   min_pontos: int = 30):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return {"presente": False, "n_dentro": 0}
    uv = np.stack([xs, ys], axis=1)
    pts = projetar_pixels_3d(uv, depth, K)
    if len(pts) == 0:
        return {"presente": False, "n_dentro": 0}
    z = pts[:, 2]
    pts = pts[(z > z_min) & (z < z_max)]
    idx = pontos_dentro_obb(obb, pts)
    n = int(len(idx))
    return {"presente": bool(n >= min_pontos), "n_dentro": n}
