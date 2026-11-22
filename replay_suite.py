import argparse
import os

import numpy as np


"""

❯ python3 replay_suite.py --go2_xml ../unitree_mujoco/unitree_robots/go2/go2.xml eval_suite/ policy_vel_9999

"""

ap = argparse.ArgumentParser()
ap.add_argument("dir", help="carpeta de resultados, p. ej. eval_vid")
ap.add_argument("policy", help="nombre de la política (subcarpeta)")
ap.add_argument("--go2_xml", required=True, help="unitree_mujoco/unitree_robots/go2/go2.xml")
ap.add_argument("--levels", type=int, nargs="+", default=[0, 4, 9], help="niveles de dificultad a grabar")
ap.add_argument("--types", nargs="+", default=None, help="solo estos tipos de terreno")
ap.add_argument("--all", action="store_true", help="los 10 niveles")
ap.add_argument("--skip", type=int, default=2, help="1 de cada N pasos (2 = 25 fps en tiempo real)")
ap.add_argument("--col", type=int, default=0, help="0 o 1: cuál de las dos columnas de cada tipo")
ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
a = ap.parse_args()

import mujoco
import pyvista as pv

d = np.load(f"{a.dir}/{a.policy}/traj.npz")
tiles = np.load(f"{a.dir}/terrain_tiles.npz")
traj, dt = d["traj"], float(d["dt"])
types, cpt, ncols = [str(t) for t in d["types"]], int(d["cpt"]), int(d["ncols"])
tile, ok, fell, surv, prm = d["tile"], d["success"], d["fell"], d["survival_s"], d["param"]
T, ROWS = traj.shape[0], int(d["row"].max()) + 1

m = mujoco.MjModel.from_xml_path(os.path.expanduser(a.go2_xml))
md = mujoco.MjData(m)
qadr = []
for n in [str(x) for x in d["joint_names"]]:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    assert j >= 0, f"la articulación {n} no existe en el XML"
    qadr.append(int(m.jnt_qposadr[j]))


def poly(V, F):
    return pv.PolyData(np.asarray(V, float), np.column_stack([np.full(len(F), 3), F]).ravel())


geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2]
robot_meshes = []
for g in geoms:
    mid = m.geom_dataid[g]
    va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
    F = m.mesh_face[fa:fa + fn]
    if F.max() >= vn:
        F = F - va
    mat = m.geom_matid[g]
    rgb = (m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g])[:3]
    robot_meshes.append((g, poly(m.mesh_vert[va:va + vn], F), rgb))

out = f"{a.dir}/{a.policy}/videos"
os.makedirs(out, exist_ok=True)


def make_clip(r, ti):
    tid = r * ncols + ti * cpt + a.col
    idx = np.nonzero(tile == tid)[0]
    maj = ok[idx].mean() >= 0.5  # robot representativo: el resultado mayoritario de su baldosa
    env = int(idx[ok[idx] == float(maj)][0])
    n_valid = T if not fell[env] else max(1, int(round(surv[env] / dt)) - 1)
    ks = list(range(0, n_valid, a.skip))
    ks += [ks[-1]] * int(1.0 / (dt * a.skip))  # un segundo con la imagen final
    res = "exito" if ok[env] else ("caida" if fell[env] else "sin avanzar")
    label = (f"{types[ti]} | nivel {r} (parametro {prm[tile == tid].mean():.2f}) | "
             f"{int(ok[idx].sum())}/{len(idx)} robots lo logran | este robot: {res}")
    pl = pv.Plotter(off_screen=True, window_size=a.size)
    pl.set_background("white")
    V, F = tiles[f"v_{r}_{ti * cpt + a.col}"], tiles[f"f_{r}_{ti * cpt + a.col}"]
    terr = poly(V, F)
    terr.point_data["z"] = V[:, 2]
    pl.add_mesh(terr, scalars="z", cmap="terrain", clim=(V[:, 2].min(), V[:, 2].max() + 1e-3), show_scalar_bar=False)
    acts = [(g, pl.add_mesh(p_, color=rgb, smooth_shading=False)) for g, p_, rgb in robot_meshes]
    path = f"{out}/{ti:02d}_{types[ti]}_c{a.col}_L{r}.mp4"
    pl.open_movie(path, framerate=int(round(1 / (dt * a.skip))))
    w, x, y, z = traj[0, env][3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    cam = np.array([-2.0 * c - 1.6 * s, -2.0 * s + 1.6 * c, 1.1])  # detrás y a la izquierda, según el rumbo inicial
    for k in ks:
        q = traj[k, env]
        md.qpos[:7] = q[:7]  # posición y cuaternión (w, x, y, z), igual convención en Isaac y MuJoCo
        for ad, v in zip(qadr, q[7:]):
            md.qpos[ad] = v
        mujoco.mj_kinematics(m, md)
        for g, act in acts:
            M = np.eye(4)
            M[:3, :3] = md.geom_xmat[g].reshape(3, 3)
            M[:3, 3] = md.geom_xpos[g]
            act.user_matrix = M
        pos = q[:3].astype(float)
        pl.camera.focal_point = pos
        pl.camera.position = pos + cam
        pl.camera.up = (0, 0, 1)
        # pl.add_text(f"{label} | t = {k * dt:4.1f} s", name="info", font_size=9, color="black")
        pl.write_frame()
    pl.close()
    print("guardado", path)

#
# levels = range(ROWS) if a.all else [r for r in a.levels if r < ROWS]
# for ti, t in enumerate(types):
#     if a.types and t not in a.types:
#         continue
#     if t == "flat":
#         for r in ([0] if t == "flat" else levels):
#             make_clip(r, ti)



levels = range(ROWS) if a.all else [r for r in a.levels if r < ROWS]

for ti, t in enumerate(types):
    if a.types and t not in a.types:
        continue

    for r in levels:
        make_clip(r, ti)
