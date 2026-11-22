# Resultados (BORRADOR autogenerado — revisar y redactar)

## 1. Configuración del experimento

| Parámetro | Valor |
|---|---|
| Tarea | `Isaac-Velocity-Rough-Unitree-Go2-v0` |
| Entornos paralelos | 4096 |
| Iteraciones registradas | 1500 |
| Duración del entrenamiento | 29.4 min |
| Throughput medio | 84,011 pasos/s |
| Tiempo por iteración (final) | 1.14 s (recolección 1.07 s, aprendizaje 0.067 s) |

## 2. Curva de aprendizaje

| Métrica | Inicio | Final |
|---|---|---|
| Recompensa media | -0.70 | 23.37 (media últimas 250 it.: 22.83 ± 0.82) |
| Longitud de episodio | 13 | 950 |
| Error velocidad lineal xy | 0.835 | 0.227 |
| Error velocidad angular yaw | 0.791 | 0.280 |
| Success rate | — | 0.99 |
| Terminación por time-out | — | 88.2 % |
| Terminación por contacto de base | pico 66.8 % (it. 43) | 11.83 % |

Hitos: recompensa > 0 en la it. 63, > 10 en 165, > 20 en 631, > 30 en None; episodios > 900 pasos en la it. 44; success rate > 0,5 en la it. 444 y > 0,9 en la it. 667.

## 3. Términos de recompensa (valor final)

| Término | Valor |
|---|---|
| `track_lin_vel_xy_exp` | +1.1345 |
| `track_ang_vel_z_exp` | +0.5088 |
| `dof_acc_l2` | -0.1297 |
| `ang_vel_xy_l2` | -0.1156 |
| `dof_torques_l2` | -0.0992 |
| `action_rate_l2` | -0.0761 |
| `lin_vel_z_l2` | -0.0501 |
| `feet_air_time` | -0.0014 |
| `dof_pos_limits` | +0.0000 |
| `flat_orientation_l2` | +0.0000 |

## 4. Diagnóstico de PPO

| Métrica | Inicio | Final |
|---|---|---|
| Entropía | 17.04 | 7.51 |
| Std media de la política | 1.001 | 0.457 |
| Pérdida del crítico | 0.4288 | 0.0177 |
| Learning rate | 1.00e-05 | 3.84e-04 (pico 2.92e-03) |

## 5. Avisos para la redacción

- feet_air_time es negativo al final: los pasos son más cortos que el umbral de tiempo en el aire.
- Todas las métricas son de entrenamiento (con ruido de exploración). No demuestran buen comportamiento en evaluación/despliegue; hace falta play.py o una prueba sim-to-sim.
