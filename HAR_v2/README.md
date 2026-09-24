# Human Activity Recognition — Pose 2D (heurística)

Detector basado en reglas geométricas sobre keypoints 2D (sin ML).

## Datos

Los `.npy` viven en `./videos/` (clases `0`…`7`, clips, `user_XXX/`).

## Ejecución rápida

```bash
cd /home/ignacio/Escritorio/Company/Tecnica_Heuristica_HAR

# Usa por defecto:
# videos/1/...002920_002950_1/user_136/poses_full.npy
python3 detect_actions.py
python3 validate_realtime.py

# Otro clip (ruta relativa a videos/)
python3 detect_actions.py 2/CLIP/user_183/poses_full.npy
```

```bash
python3 -m unittest tests.test_har_pose -v
```

El FPS se lee de `meta.json` del clip si no pasas `--fps`.
Salida en `output/<clase>/<CLIP>/`.
