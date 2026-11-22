#!/usr/bin/env python3
"""Uso: python resumen_run.py LOG_TFEVENTS NOMBRE   (ej. m1c1)
Imprime un resumen corto (pégalo en el prompt) y guarda 3 figuras en figuras/NOMBRE_*.png"""
import sys, os, json
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log, nombre = sys.argv[1], sys.argv[2]
ea = EventAccumulator(log, size_guidance={"scalars": 0}); ea.Reload()
rows = [(t, e.step, e.value, e.wall_time) for t in ea.Tags()["scalars"] for e in ea.Scalars(t)]
df = pd.DataFrame(rows, columns=["tag", "step", "v", "wall"])
p = df[~df.tag.str.endswith("/time")].pivot_table(index="step", columns="tag", values="v", aggfunc="last")
g = lambda t: p[t] if t in p else pd.Series(float("nan"), index=p.index)
r, k = g("Train/mean_reward"), max(10, len(p) // 6)
first = lambda s, thr: (int(s[s >= thr].index.min()) if (s >= thr).any() else None)
cambio = (r.iloc[-k:].mean() - r.iloc[-2*k:-k].mean()) / abs(r.iloc[-2*k:-k].mean())
R = lambda x, d=3: None if pd.isna(x) else round(float(x), d)

res = {
 "iteraciones": len(p), "duracion_min": R((df.wall.max() - df.wall.min()) / 60, 1),
 "pasos_por_s_medio": round(float(g("Perf/total_fps").iloc[10:].mean())),
 "recompensa": {"inicial": R(r.iloc[0]), "final": R(r.iloc[-1]), f"media_ultimas_{k}_it": R(r.iloc[-k:].mean()),
                f"std_ultimas_{k}_it": R(r.iloc[-k:].std()), "iter_supera_0": first(r.rolling(9, min_periods=1).mean(), 0),
                "cambio_ultima_ventana_vs_anterior_pct": R(cambio * 100, 1)},
 "longitud_episodio": {"inicial": R(g("Train/mean_episode_length").iloc[0]), "final": R(g("Train/mean_episode_length").iloc[-1])},
 "error_vel_lineal_xy": {"inicial": R(g("Metrics/base_velocity/error_vel_xy").iloc[0]), "final": R(g("Metrics/base_velocity/error_vel_xy").iloc[-1])},
 "error_vel_angular_yaw": {"inicial": R(g("Metrics/base_velocity/error_vel_yaw").iloc[0]), "final": R(g("Metrics/base_velocity/error_vel_yaw").iloc[-1])},
 "success_rate": {"inicial": R(g("Metrics/success_rate").iloc[0]), "final": R(g("Metrics/success_rate").iloc[-1]),
                  "iter_supera_0.9": first(g("Metrics/success_rate"), 0.9)},
 "terminaciones_final": {"time_out": R(g("Episode_Termination/time_out").iloc[-1]), "contacto_base": R(g("Episode_Termination/base_contact").iloc[-1]),
                         "pico_contacto_base": R(g("Episode_Termination/base_contact").max()), "iter_pico": int(g("Episode_Termination/base_contact").idxmax())},
 "terminos_recompensa_final": {c.replace("Episode_Reward/", ""): R(p[c].iloc[-1], 4) for c in p.columns if c.startswith("Episode_Reward/")},
 "ppo": {"entropia": [R(g("Loss/entropy").iloc[0]), R(g("Loss/entropy").iloc[-1])], "std_politica": [R(g("Policy/mean_std").iloc[0]), R(g("Policy/mean_std").iloc[-1])],
         "loss_critico": [R(g("Loss/value").iloc[0], 4), R(g("Loss/value").iloc[-1], 4)]},
}
print(json.dumps(res, indent=1, ensure_ascii=False))

os.makedirs("figuras", exist_ok=True)
def fig(suffix, series, ylabel, ylim=None):
    plt.figure(figsize=(6.5, 3.4))
    for tag, lab in series: plt.plot(g(tag).index, g(tag), label=lab)
    plt.xlabel("Iteración"); plt.ylabel(ylabel); plt.grid(alpha=.3)
    if ylim: plt.ylim(*ylim)
    if len(series) > 1: plt.legend()
    plt.tight_layout(); plt.savefig(f"figuras/{nombre}_{suffix}.png", dpi=200); plt.close()
fig("recompensa", [("Train/mean_reward", "Recompensa media")], "Recompensa media")
fig("seguimiento", [("Metrics/base_velocity/error_vel_xy", "Error lineal (xy)"), ("Metrics/base_velocity/error_vel_yaw", "Error angular (yaw)")], "Error")
fig("exito", [("Metrics/success_rate", "Success rate"), ("Episode_Termination/base_contact", "Contacto de la base")], "Fracción", (0, 1.05))
print(f"\n# Figuras: figuras/{nombre}_recompensa.png, _seguimiento.png, _exito.png", file=sys.stderr)
