"""
Lee un poses_full.npy y escribe el JSON de gestos.
También deja una copia en temp_output/acciones.json, sustituida en cada ejecución.

Uso:
  python detect_json.py /ruta/user_x/poses_full.npy
  python detect_json.py /ruta/user_x/poses_full.npy -o /ruta/acciones.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_acciones.detectar import NO_OBSERVABLES, acciones_desactivadas, detectar, informe
from har_acciones.pose import EJEMPLO_POSES, NOMBRES, cargar_poses, fps_y_video, suavizar


def volcar_json(destino: Path, poses_path: Path, fps: float, video: Path | None, n_frames: int, acciones) -> dict:
    """Escribe el JSON de acciones en destino, sustituyendo el archivo si ya existe."""
    documento = {
        "poses": str(poses_path),
        "video": str(video) if video else None,
        "fps": fps,
        "n_frames": int(n_frames),
        "duracion_s": round(n_frames / fps, 2),
        "keypoints": list(NOMBRES),
        "nota": (
            "Cada acción es un gesto de brazos o de torso posterior a tener el producto "
            "encima. Con 8 puntos no se ve el objeto: el JSON marca el movimiento compatible "
            "con esconderlo, para comprobarlo en el vídeo."
        ),
        "no_observables_con_esta_pose": list(NO_OBSERVABLES),
        "acciones_desactivadas": acciones_desactivadas(),
        "acciones": [accion.a_dict(fps) for accion in acciones],
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(documento, ensure_ascii=False, indent=2), encoding="utf-8")
    return documento


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta gestos de esconder y escribe un JSON.")
    parser.add_argument("poses", type=Path, nargs="?", default=EJEMPLO_POSES, help="poses_full.npy")
    parser.add_argument("-o", "--salida", type=Path, default=None, help="JSON de salida. Por defecto, acciones.json junto al npy.")
    args = parser.parse_args()

    if not args.poses.is_file():
        raise SystemExit(f"No está el npy: {args.poses}")

    salida = args.salida if args.salida is not None else args.poses.with_name("acciones.json")
    crudas = cargar_poses(args.poses)
    poses = suavizar(crudas, ventana=3)
    fps, video = fps_y_video(args.poses)
    acciones = detectar(poses, fps)

    documento = volcar_json(salida, args.poses, fps, video, len(poses), acciones)
    temporal = ROOT / "temp_output" / "acciones.json"
    if temporal.resolve() != salida.resolve():
        volcar_json(temporal, args.poses, fps, video, len(poses), acciones)

    print(f"Poses:  {args.poses}")
    print(f"Vídeo:  {video}")
    print(f"Frames: {len(poses)}   fps: {fps}   duración: {documento['duracion_s']} s")
    apagadas = documento["acciones_desactivadas"]
    if apagadas:
        print("Apagadas en acciones_config.json: " + ", ".join(apagadas))
    print()
    print(informe(acciones, fps))
    print()
    print(f"JSON:      {salida}")
    print(f"Temporal:  {temporal}")


if __name__ == "__main__":
    main()
