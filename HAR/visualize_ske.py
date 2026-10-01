"""
Abre una ventana con el esqueleto y guarda el mismo vídeo en temp_output.
Ese archivo se sustituye en cada ejecución.

A la derecha, cada acción aparece cuando termina, con su intervalo.
La siguiente acción ocupa la fila de debajo.

Uso:
  python visualize_ske.py /ruta/user_x/poses_full.npy

Espacio: pausa. Flechas: un frame. Q: salir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_acciones.detectar import acciones_en_frame, detectar
from har_acciones.dibujar import (
    guardar_temporal,
    lienzo_esqueleto,
    recolorear_esqueleto_activo,
    reproducir,
    unir_lista,
)
from har_acciones.pose import EJEMPLO_POSES, cargar_poses, fps_y_video, suavizar

ANCHO, ALTO = 960, 720


def main() -> None:
    parser = argparse.ArgumentParser(description="Ventana con el esqueleto y la lista de acciones.")
    parser.add_argument("poses", type=Path, nargs="?", default=EJEMPLO_POSES, help="poses_full.npy")
    args = parser.parse_args()

    if not args.poses.is_file():
        raise SystemExit(f"No está el npy: {args.poses}")

    poses = suavizar(cargar_poses(args.poses), ventana=3)
    fps, _video = fps_y_video(args.poses)
    acciones = detectar(poses, fps)
    temporal = ROOT / "temp_output" / "esqueleto.mp4"
    print(f"Temporal: {temporal}")
    print("Espacio: pausa. Flechas: un frame. Q: salir.")

    def construir(indice: int):
        actuales = acciones_en_frame(acciones, indice)
        lienzo, desplazado = lienzo_esqueleto(poses[indice], ANCHO, ALTO)
        recolorear_esqueleto_activo(lienzo, desplazado, actuales)
        return unir_lista(lienzo, acciones, indice, fps)

    guardar_temporal(temporal, fps, len(poses), construir)
    reproducir("Esqueleto", len(poses), fps, construir)


if __name__ == "__main__":
    main()
