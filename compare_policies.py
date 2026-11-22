import argparse
import itertools
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

"""
❯ python3 compare_policies.py eval_suite

"""


ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default="eval_suite")
ap.add_argument("--boot", type=int, default=5000)
a = ap.parse_args()
rng = np.random.default_rng(0)
out = os.path.join(a.dir, "analysis")
os.makedirs(out, exist_ok=True)
pols = sorted(d for d in os.listdir(a.dir) if os.path.isfile(f"{a.dir}/{d}/terrain.csv"))
rd = lambda p, f: pd.read_csv(f"{a.dir}/{p}/{f}.csv")  # noqa: E731
T, C, P = ({p: rd(p, f) for p in pols} for f in ("terrain", "commands", "push"))
types = list(dict.fromkeys(T[pols[0]].ttype))
cmds = list(dict.fromkeys(C[pols[0]].cmd))


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(x[rng.integers(0, len(x), (a.boot, len(x)))].mean(1), [2.5, 97.5])
    return x.mean(), lo, hi


def fmt(m, lo, hi, d=3):
    return "—" if np.isnan(m) else f"{m:.{d}g} [{lo:.{d}g}, {hi:.{d}g}]"


def show(df, name, idx=False):
    df.to_csv(f"{out}/{name}.csv", index=idx)
    print(f"\n== {name} ==\n{df.to_string(index=idx)}")


# unidad de análisis del terreno: la baldosa (los 20 robots de una baldosa no son independientes)
tiles = {p: T[p].groupby("tile").agg(ttype=("ttype", "first"), row=("row", "first"), success=("success", "mean")) for p in pols}

# T1: éxito por terreno (media sobre baldosas, IC95 bootstrap)
rows = []
for p in pols:
    tl = tiles[p]
    r = {"política": p, "TODOS %": fmt(*ci(tl.success * 100))}
    r.update({t: fmt(*ci(tl.success[tl.ttype == t] * 100)) for t in types})
    rows.append(r)
show(pd.DataFrame(rows), "t1_exito_terreno")


# T2: dificultad máxima superada (primer nivel con éxito < 50 % corta la cuenta)
def max_ok(df, t):
    g = df[df.ttype == t].groupby("row").agg(s=("success", "mean"), prm=("param", "mean"), rel=("relief_m", "mean"))
    ok = -1
    for r_, s in g.s.items():
        if s < 0.5:
            break
        ok = r_
    return "ninguna" if ok < 0 else f"fila {ok} (nominal {g.loc[ok, 'prm']:.3g}; relieve medido {g.loc[ok, 'rel']:.2f} m)"


show(pd.DataFrame([{"política": p, **{t: max_ok(T[p], t) for t in types if t != "flat"}} for p in pols]), "t2_dificultad_maxima")


# T3/T4: comandos (plano)
def cmd_table(col, name, scale=1.0, surv=False):
    rows = []
    for p in pols:
        r = {"política": p}
        for c in cmds:
            d = C[p][C[p].cmd == c]
            d = d[d.fell == 0] if surv else d
            r[c] = fmt(*ci(d[col] * scale))
        rows.append(r)
    show(pd.DataFrame(rows), name)


cmd_table("success", "t3_exito_comandos_pct", 100)
cmd_table("track_lin", "t4_tracking_lineal_0a1", surv=True)
cmd_table("ang_err", "t4b_error_yaw_rad_s", surv=True)

# T5: calidad de la marcha en plano a 0.6 m/s (solo robots que no caen)
Q = ["track_lin", "lin_rmse", "ang_err", "lateral_m", "cot", "tau2_per_m", "act_rate", "act_jerk", "jacc_rms", "tilt_deg",
     "w_xy_rms", "vz_rms", "duty", "step_hz", "slip_ms", "fz_peak_bw", "tau_peak", "tau_sat", "body_contact"]
rows = {}
for p in pols:
    d = C[p][(C[p].cmd == "fwd_0.6") & (C[p].fell == 0)]
    rows[p] = {q: fmt(*ci(d[q])) for q in Q}
show(pd.DataFrame(rows), "t5_calidad_plano_fwd0.6", idx=True)

# T6: calidad en terreno difícil (media sobre robots que no caen)
for q in ["body_contact", "tau_sat", "slip_ms", "tilt_max_deg"]:
    rows = []
    for p in pols:
        d = T[p][T[p].fell == 0]
        rows.append({"política": p, **{t: f"{d[d.ttype == t][q].mean():.3g}" for t in types}})
    show(pd.DataFrame(rows), f"t6_{q}_por_terreno")

# T7: empujones
rows = []
for p in pols:
    d = P[p][P[p].alive_push == 1]
    r, okmax, still = {"política": p}, 0.0, True
    for m in sorted(d.mag.unique()):
        s = d[d.mag == m]
        sv = 1 - s.fell
        r[f"sup. {m} m/s %"] = fmt(*ci(sv * 100))
        rec = s.recovery_s[s.fell == 0]
        r[f"recup. {m} mediana s (% sin recuperar)"] = (f"{rec.median():.2f}" if rec.notna().any() else "—") + f" ({rec.isna().mean() * 100:.0f} %)"
        if still and sv.mean() >= 0.9:
            okmax = m
        else:
            still = False
    r["empujón máx. con ≥90 % sup. (m/s)"] = okmax
    rows.append(r)
show(pd.DataFrame(rows), "t7_empujones")

# T8: diferencias pareadas de éxito en terreno (misma baldosa, A−B)
rows = [{"A": A, "B": B, "Δ éxito pp (A−B) [IC95]": fmt(*ci((tiles[A].success - tiles[B].success) * 100))}
        for A, B in itertools.combinations(pols, 2)]
show(pd.DataFrame(rows), "t8_diferencias_pareadas")

# Figuras
rough = [t for t in types if t != "flat"]
fig, axs = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
for ax, t in zip(axs.ravel(), rough):
    for p in pols:
        g = T[p][T[p].ttype == t].groupby("row").success.mean() * 100
        ax.plot(g.index, g.values, marker="o", ms=3, label=p)
    ax.axhline(50, ls=":", c="gray")
    ax.set_title(t)
    ax.set_xlabel("fila de dificultad")
axs[0, 0].set_ylabel("éxito (%)")
axs[0, 0].legend(fontsize=7)
fig.savefig(f"{out}/fig_exito_vs_dificultad.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(1.4 * len(cmds) + 3, 3.6))
w = 0.8 / len(pols)
for k, p in enumerate(pols):
    st = np.array([ci(C[p][(C[p].cmd == c) & (C[p].fell == 0)].track_lin) for c in cmds])
    ax.bar(np.arange(len(cmds)) + k * w, st[:, 0], w, yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], capsize=2, label=p)
ax.set_xticks(np.arange(len(cmds)) + 0.4 - w / 2)
ax.set_xticklabels(cmds, rotation=20)
ax.set_ylabel("tracking lineal (0-1)")
ax.legend(fontsize=7)
fig.savefig(f"{out}/fig_tracking_comandos.pdf", bbox_inches="tight")
print("\nguardado en", out)
