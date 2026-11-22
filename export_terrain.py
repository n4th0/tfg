#!/usr/bin/env python3
"""Exporta baldosas de terrain_tiles.npz (eval_suite --record) a escenas de unitree_mujoco con el Go2.

Crea dentro de la carpeta del robot (unitree_robots/go2):
    scene_<nombre>.xml                    escena: incluye go2.xml + terreno
    assets/terrain/<nombre>/hfield.bin    mapa de alturas (float32), o .png con --format png

    python3 export_terrain.py terrain_tiles.npz --list
    python3 export_terrain.py terrain_tiles.npz --type stairs_inv --row 9
    python3 export_terrain.py terrain_tiles.npz --all
"""
import argparse
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

try:
    import mujoco
except ImportError:
    mujoco = None

TYPES = ["flat", "rough", "slope_up", "slope_down", "stairs_pyr", "stairs_inv", "boxes", "obstacles", "waves"]
ORIENT = None  # 0..3: bit0 invierte filas, bit1 invierte columnas (se calibra con MuJoCo)
MASK = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # los rayos solo ven el grupo 0 (el terreno), no el robot


def rasterize_top(V, F, xs, ys):
    """Altura de la superficie más alta en cada celda de la rejilla (las caras verticales no cuentan)."""
    H = np.full((len(ys), len(xs)), -np.inf)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    for a, b, c in F:
        p0, p1, p2 = V[a], V[b], V[c]
        den = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(den) < 1e-12:
            continue
        i0 = max(int(np.ceil((min(p0[0], p1[0], p2[0]) - xs[0]) / dx - 1e-9)), 0)
        i1 = min(int(np.floor((max(p0[0], p1[0], p2[0]) - xs[0]) / dx + 1e-9)), len(xs) - 1)
        j0 = max(int(np.ceil((min(p0[1], p1[1], p2[1]) - ys[0]) / dy - 1e-9)), 0)
        j1 = min(int(np.floor((max(p0[1], p1[1], p2[1]) - ys[0]) / dy + 1e-9)), len(ys) - 1)
        if i1 < i0 or j1 < j0:
            continue
        X, Y = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1])
        l0 = ((p1[1] - p2[1]) * (X - p2[0]) + (p2[0] - p1[0]) * (Y - p2[1])) / den
        l1 = ((p2[1] - p0[1]) * (X - p2[0]) + (p0[0] - p2[0]) * (Y - p2[1])) / den
        l2 = 1.0 - l0 - l1
        ok = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        if not ok.any():
            continue
        z = l0 * p0[2] + l1 * p1[2] + l2 * p2[2]
        sub = H[j0:j1 + 1, i0:i1 + 1]
        sub[ok] = np.maximum(sub[ok], z[ok])
    return H


def fill_holes(H):
    bad = ~np.isfinite(H)
    if bad.any():
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        H = H[tuple(idx)]
    return H


def build_grid(V, F, cell, pad):
    V = np.asarray(V, np.float64).copy()
    F = np.asarray(F, np.int64)
    V[:, :2] -= (V[:, :2].min(0) + V[:, :2].max(0)) / 2  # centro de la baldosa = punto de aparición
    (xmin, ymin), (xmax, ymax) = V[:, :2].min(0), V[:, :2].max(0)
    nx, ny = int(round((xmax - xmin) / cell)) + 1, int(round((ymax - ymin) / cell)) + 1
    xs, ys = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    H = fill_holes(rasterize_top(V, F, xs, ys))
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    p = int(round(pad / cell))
    if p > 0:  # prolonga el borde para que el robot no caiga al vacío
        H = np.pad(H, p, mode="edge")
        xs, ys = xmin + dx * np.arange(-p, nx + p), ymin + dy * np.arange(-p, ny + p)
    return xs, ys, H, (xmin, xmax, ymin, ymax)


def spawn_height(xs, ys, H):
    w = max(int(0.25 / (xs[1] - xs[0])), 1)
    i, j = int(np.argmin(np.abs(xs))), int(np.argmin(np.abs(ys)))
    return float(np.median(H[j - w:j + w + 1, i - w:i + w + 1]))


def orient(N, k):
    if k & 1:
        N = N[::-1]
    if k & 2:
        N = N[:, ::-1]
    return np.ascontiguousarray(N)


def write_hf(path, N, fmt):
    if fmt == "bin":
        with open(path, "wb") as f:
            np.array(N.shape, dtype=np.int32).tofile(f)  # nrow, ncol
            N.astype(np.float32).tofile(f)
    else:
        Image.fromarray(np.clip(np.rint(N * 255), 0, 255).astype(np.uint8)).save(path)


def expected_at(xs, ys, Hs, pts):
    i = np.clip(np.rint((pts[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1)
    j = np.clip(np.rint((pts[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1)
    return Hs[j, i]


def ray_h(m, d, gid, x, y):
    dist = mujoco.mj_ray(m, d, np.array([x, y, 50.0]), np.array([0.0, 0.0, -1.0]), MASK, 1, -1, gid)
    return 50.0 - dist if dist >= 0 else np.nan


def calibrate(N, xs, ys, Hs, zmin, zr, fmt, pts):
    """Error mediano (m) de cada una de las 4 orientaciones frente a rayos reales de MuJoCo."""
    sx, sy, cx, cy = (xs[-1] - xs[0]) / 2, (ys[-1] - ys[0]) / 2, (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
    exp, errs = expected_at(xs, ys, Hs, pts), []
    with tempfile.TemporaryDirectory() as td:
        for k in range(4):
            fn = f"h{k}.{fmt}"
            write_hf(Path(td) / fn, orient(N, k), fmt)
            xml = (f'<mujoco><asset><hfield name="h" file="{fn}" size="{sx:.6f} {sy:.6f} {zr:.6f} 0.2"/></asset>'
                   f'<worldbody><geom type="hfield" hfield="h" pos="{cx:.6f} {cy:.6f} {zmin:.6f}"/></worldbody></mujoco>')
            (Path(td) / f"s{k}.xml").write_text(xml)
            m = mujoco.MjModel.from_xml_path(str(Path(td) / f"s{k}.xml"))
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            gid = np.zeros(1, np.int32)
            meas = np.array([ray_h(m, d, gid, x, y) for x, y in pts])
            errs.append(float(np.nanmedian(np.abs(meas - exp))))
    return errs


def write_scene(path, name, ref, go2_xml, sx, sy, zr, cx, cy, zpos):
    path.write_text(f"""<mujoco model="{name}">
  <include file="{go2_xml}"/>
  <statistic center="0 0 0.1" extent="0.8"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <hfield name="terrain" file="{ref}" size="{sx:.6f} {sy:.6f} {zr:.6f} 0.2"/>
    <material name="terrain_mat" rgba="0.55 0.58 0.50 1" specular="0" shininess="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="terrain_geom" type="hfield" hfield="terrain" material="terrain_mat"
          pos="{cx:.6f} {cy:.6f} {zpos:.6f}" friction="1.0 0.005 0.0001" contype="1" conaffinity="1"/>
  </worldbody>
</mujoco>
""")


def verify(scene, xs, ys, Hs, pts):
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link") >= 0, "la escena no contiene el body base_link del Go2"
    gid = np.zeros(1, np.int32)
    meas = np.array([ray_h(m, d, gid, x, y) for x, y in pts])
    err = np.abs(meas - expected_at(xs, ys, Hs, pts))
    return float(np.nanmedian(err)), float(np.nanmax(err)), ray_h(m, d, gid, 0.0, 0.0)


def export_tile(V, F, name, a, go2_dir, meshdir):
    global ORIENT
    xs, ys, H, (xmin, xmax, ymin, ymax) = build_grid(V, F, a.cell, a.pad)
    z0 = spawn_height(xs, ys, H)
    Hs = H - z0  # el suelo bajo el robot queda en z = 0
    zmin, zmax = float(Hs.min()), float(Hs.max())
    zr = max(zmax - zmin, 1e-3)
    N = (Hs - zmin) / zr
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(xmin * 0.9, xmax * 0.9, 300), rng.uniform(ymin * 0.9, ymax * 0.9, 300)]

    k = ORIENT if ORIENT is not None else 0
    if mujoco is None:
        print("  AVISO: sin el paquete 'mujoco' no se calibra la orientación ni se verifica la escena")
    elif ORIENT is None and (zmax - zmin) > 0.02:
        errs = calibrate(N, xs, ys, Hs, zmin, zr, a.format, pts)
        if max(errs) - min(errs) > 0.005:
            ORIENT = k = int(np.argmin(errs))
            print(f"  orientación calibrada con MuJoCo: opción {k} (errores medianos {[round(e, 4) for e in errs]} m)")

    tdir = go2_dir / "assets" / "terrain" / name
    tdir.mkdir(parents=True, exist_ok=True)
    hf_path = tdir / f"hfield.{a.format}"
    write_hf(hf_path, orient(N, k), a.format)
    sx, sy, cx, cy = (xs[-1] - xs[0]) / 2, (ys[-1] - ys[0]) / 2, (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
    scene = go2_dir / f"scene_{name}.xml"
    refs = [os.path.relpath(hf_path, go2_dir / meshdir)] if meshdir else []
    refs.append(os.path.relpath(hf_path, go2_dir))
    res, last = None, None
    for ref in refs:
        write_scene(scene, name, ref.replace(os.sep, "/"), "go2.xml", sx, sy, zr, cx, cy, zmin)
        if mujoco is None:
            break
        try:
            res = verify(scene, xs, ys, Hs, pts)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise SystemExit(f"MuJoCo no pudo cargar {scene}: {last}")
    print(f"[OK] {name}: relieve {zmax - zmin:.2f} m | suelo de aparición a {z0:+.3f} m en Isaac (desplazado a 0) | {scene}")
    if res:
        med, mx, h0 = res
        flag = "" if med < 0.01 else "  <-- REVISAR: el heightfield no coincide con el terreno"
        print(f"     verificación con rayos: error mediano {med * 1000:.1f} mm, máximo {mx * 1000:.0f} mm | altura en (0,0): {h0:+.3f} m{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--go2-dir", default="~/tfg/unitree_mujoco/unitree_robots/go2")
    ap.add_argument("--row", type=int, default=9, help="fila de dificultad (0 fácil ... 9 difícil)")
    ap.add_argument("--type", choices=TYPES)
    ap.add_argument("--sub", type=int, default=0, help="cuál de las columnas de ese tipo (0 o 1)")
    ap.add_argument("--col", type=int, help="columna global (alternativa a --type)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cell", type=float, default=0.025, help="tamaño de celda del heightfield (m)")
    ap.add_argument("--pad", type=float, default=3.0, help="borde prolongado alrededor de la baldosa (m)")
    ap.add_argument("--format", choices=["bin", "png"], default="bin")
    a = ap.parse_args()

    data = np.load(a.npz)
    tiles = sorted({(int(k.split("_")[1]), int(k.split("_")[2])) for k in data.files if k.startswith("v_")})
    ncols = max(c for _, c in tiles) + 1
    cpt = ncols // len(TYPES) if ncols % len(TYPES) == 0 else None
    if a.list:
        print(f"{len(tiles)} baldosas | filas 0..{max(r for r, _ in tiles)} | {ncols} columnas | {cpt} por tipo")
        print("tipos (orden):", TYPES)
        print("ejemplo: --type stairs_inv --row 9" + (" --sub 1" if cpt and cpt > 1 else ""))
        return

    go2_dir = Path(os.path.expanduser(a.go2_dir)).resolve()
    assert (go2_dir / "go2.xml").exists(), f"no existe {go2_dir}/go2.xml: indica --go2-dir"
    m_ = re.search(r'meshdir="([^"]*)"', (go2_dir / "go2.xml").read_text())
    meshdir = m_.group(1) if m_ else None

    if a.all:
        sel = tiles
    else:
        if a.col is not None:
            col = a.col
        else:
            assert a.type and cpt, "usa --type (y un .npz de 9 o 18 columnas) o --col"
            assert a.sub < cpt, f"--sub debe ser < {cpt}"
            col = TYPES.index(a.type) * cpt + a.sub
        assert (a.row, col) in tiles, f"no existe la baldosa fila {a.row}, columna {col}. Usa --list"
        sel = [(a.row, col)]

    for r, c in sel:
        name = f"{TYPES[c // cpt]}_L{r}_s{c % cpt}" if cpt else f"L{r}_C{c}"
        export_tile(data[f"v_{r}_{c}"], data[f"f_{r}_{c}"], name, a, go2_dir, meshdir)
    print("\nAhora: en simulate/config.yaml pon robot_scene: \"scene_<nombre>.xml\" y arranca el simulador.")


if __name__ == "__main__":
    main()
