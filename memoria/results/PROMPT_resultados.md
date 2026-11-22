# Plantilla de prompt — Apartado de Resultados (TFG)

**Cómo usarla:** ejecuta `analizar_run.py` sobre tu log, y pega en un chat nuevo este prompt
con los tres huecos rellenos (`<<...>>`). Adjunta también las figuras PNG si la IA admite imágenes.

---

Eres mi tutor técnico y me ayudas a redactar el apartado **«Resultados»** de mi TFG
(aprendizaje por refuerzo para locomoción del robot Unitree Go2 con Isaac Lab + rsl_rl/PPO).

## Entradas
**1. Comando de entrenamiento exacto**
```
<<PEGA AQUÍ EL COMANDO>>
```
**2. Resumen de métricas (`resumen.json` generado por analizar_run.py)**
```json
<<PEGA AQUÍ resumen.json>>
```
**3. Contexto que no está en el log** (rellena lo que sepas; si no lo sabes, escribe "desconocido"):
- Hiperparámetros PPO / `agent.yaml` y `env.yaml` relevantes: <<...>>
- Pesos de las recompensas: <<...>>
- Rango de comandos de velocidad (lin_vel_x, lin_vel_y, ang_vel_z): <<...>>
- Hardware (GPU) y versión de Isaac Sim / Isaac Lab: <<...>>
- Semillas utilizadas / nº de ejecuciones: <<...>>
- Evaluación posterior (play.py, sim-to-sim, robot real): <<ninguna / describir>>

## Qué quiero
Redacta en **español académico, tercera persona / impersonal**, el apartado de Resultados con esta estructura:
1. **Configuración experimental** (1 párrafo + tabla).
2. **Evolución del aprendizaje**: recompensa, longitud de episodio, fases identificables
   (usa los hitos de iteración del JSON; describe cada fase y por qué ocurre).
3. **Seguimiento de velocidad y estabilidad**: errores lineal/angular, success rate, terminaciones.
4. **Análisis de los términos de recompensa**: qué domina, qué penalizaciones pesan, qué implica.
5. **Diagnóstico del algoritmo (PPO)**: entropía, std de la política, critic loss, learning rate adaptativo.
6. **Rendimiento computacional**: throughput, tiempo por iteración.
7. **Limitaciones y amenazas a la validez**.
8. **Conclusiones del apartado** (3-5 frases).

Incluye referencias a las figuras como `Figura X` con pie de figura propuesto, usando:
`fig_recompensa_episodio.png`, `fig_seguimiento.png`, `fig_terminos_recompensa.png`, `fig_ppo.png`.

## Reglas estrictas (importantes)
- **Usa solo cifras presentes en el JSON.** No inventes valores, iteraciones ni porcentajes. Si falta un dato, escribe `[FALTA: …]`.
- **Distingue observación de interpretación.** Las explicaciones causales ("la entropía cae porque…") deben ir como interpretación razonada, no como hecho medido.
- **No afirmes convergencia** salvo que la recompensa se estabilice claramente en la parte final (mira `std_ventana_final` y si el máximo coincide con la última iteración).
- Las métricas son de **entrenamiento** (con ruido de exploración). No las presentes como rendimiento de la política desplegada salvo que haya evaluación aparte.
- Revisa los `avisos` del JSON y trátalos explícitamente en «Limitaciones».
- Una sola ejecución = sin intervalos de confianza; dilo si no hay varias semillas.
- Formato de salida: **Markdown** (o LaTeX si te lo pido), tablas con las mismas cifras del JSON, redondeadas a 2-3 decimales.
- Termina con una lista breve de **lo que debería verificar yo** antes de entregar.
