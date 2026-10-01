"""
Abre la ventana del vídeo con el esqueleto y, además, escribe en temp_output
(sustituyendo lo anterior en cada ejecución):

  acciones.json
  esqueleto.mp4
  superposicion.mp4

Uso:
  python visualize_sup.py /ruta/user_x/poses_full.npy
  python visualize_sup.py /ruta/user_x/poses_full.npy --video /ruta/clip.mp4

Espacio: pausa. Flechas: un frame. Q: salir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect_json import volcar_json
from har_acciones.detectar import acciones_en_frame, detectar, informe
from har_acciones.dibujar import (
    dibujar_esqueleto,
    guardar_temporal,
    lienzo_esqueleto,
    recolorear_esqueleto_activo,
    reproducir,
    unir_lista,
)
from har_acciones.pose import EJEMPLO_POSES, EJEMPLO_VIDEO, cargar_poses, fps_y_video, suavizar

ANCHO, ALTO = 960, 720


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ventana del vídeo y, en temp_output, el JSON, el esqueleto y la superposición."
    )
    parser.add_argument("poses", type=Path, nargs="?", default=EJEMPLO_POSES, help="poses_full.npy")
    parser.add_argument("--video", type=Path, default=None, help="clip.mp4. Si se omite, se busca junto a las poses.")
    args = parser.parse_args()

    if not args.poses.is_file():
        raise SystemExit(f"No está el npy: {args.poses}")

    poses = suavizar(cargar_poses(args.poses), ventana=3)
    fps, video = fps_y_video(args.poses, args.video)
    if video is None:
        video = EJEMPLO_VIDEO
    if not video.is_file():
        raise SystemExit(f"No está el vídeo: {video}")

    acciones = detectar(poses, fps)
    salida = ROOT / "temp_output"
    json_path = salida / "acciones.json"
    esqueleto_path = salida / "esqueleto.mp4"
    super_path = salida / "superposicion.mp4"

    volcar_json(json_path, args.poses, fps, video, len(poses), acciones)
    print(informe(acciones, fps))
    print()

    def construir_esqueleto(indice: int):
        actuales = acciones_en_frame(acciones, indice)
        lienzo, desplazado = lienzo_esqueleto(poses[indice], ANCHO, ALTO)
        recolorear_esqueleto_activo(lienzo, desplazado, actuales)
        return unir_lista(lienzo, acciones, indice, fps)

    print(f"JSON:           {json_path}")
    print(f"Esqueleto:      {esqueleto_path}")
    guardar_temporal(esqueleto_path, fps, len(poses), construir_esqueleto)

    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        raise SystemExit(f"No se ha podido abrir {video}")

    n = len(poses)
    leido = -1
    frame_actual = None

    def construir_super(indice: int):
        nonlocal leido, frame_actual
        if indice != leido + 1:
            captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
            leido = indice - 1
        if indice != leido:
            ok, frame = captura.read()
            if not ok:
                raise SystemExit(f"No se ha podido leer el frame {indice} de {video}")
            leido = indice
            frame_actual = frame
        vista = frame_actual.copy()
        actuales = acciones_en_frame(acciones, indice)
        dibujar_esqueleto(vista, poses[indice], actuales, radio=7, grosor=3)
        return unir_lista(vista, acciones, indice, fps)

    try:
        print(f"Superposición:  {super_path}")
        guardar_temporal(super_path, fps, n, construir_super)
        print("Espacio: pausa. Flechas: un frame. Q: salir.")
        reproducir("Vídeo + esqueleto", n, fps, construir_super)
    finally:
        captura.release()


if __name__ == "__main__":
    main()
