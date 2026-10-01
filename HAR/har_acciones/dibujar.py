"""Dibujo del esqueleto y de los avisos de acción sobre un fotograma."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from har_acciones.detectar import Accion
from har_acciones.pose import HUESOS, NOMBRES, formatear_tiempo

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

COLOR_HUESO = (40, 210, 255)
COLOR_PUNTO = (0, 140, 255)
COLOR_ACTIVO = (40, 40, 255)
COLOR_TEXTO = (245, 245, 245)

COLORES_ACCION = {
    "acercar_al_pecho": (0, 140, 255),
    "pegar_al_torso": (0, 200, 255),
    "pasar_de_mano": (180, 40, 220),
    "tapar_con_antebrazo": (40, 40, 230),
    "una_mano_oculta": (220, 180, 40),
    "bajar_a_bolsillo": (200, 120, 20),
    "mano_en_bolsillo": (160, 80, 20),
    "subir_por_torso": (180, 60, 200),
    "brazos_cruzados": (40, 180, 40),
    "girar_torso": (200, 200, 200),
    "agacharse": (20, 120, 200),
    "mano_a_espalda": (180, 80, 160),
    "brazo_rigido": (40, 220, 80),
}

HUESOS_LADO = {
    "izquierdo": {(0, 2), (2, 4), (0, 6)},
    "derecho": {(1, 3), (3, 5), (1, 7)},
}


def _fuente(tamano: int, negrita: bool = False) -> ImageFont.ImageFont:
    ruta = FONT_BOLD if negrita else FONT_REGULAR
    return ImageFont.truetype(str(ruta), tamano)


def _pil(imagen_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))


def _bgr(imagen: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(imagen), cv2.COLOR_RGB2BGR)


def _xy(punto: np.ndarray, ancho: int, alto: int) -> tuple[int, int]:
    return int(round(punto[0] * ancho)), int(round(punto[1] * alto))


def lados_activos(acciones: list[Accion]) -> set[str]:
    lados = set()
    for accion in acciones:
        if accion.lado == "ambos":
            lados.update(("izquierdo", "derecho"))
        else:
            lados.add(accion.lado)
    return lados


def dibujar_esqueleto(
    imagen: np.ndarray,
    pose: np.ndarray,
    acciones: list[Accion] | None = None,
    radio: int = 6,
    grosor: int = 3,
) -> None:
    """Dibuja en coordenadas normalizadas sobre imagen BGR, in-place."""
    alto, ancho = imagen.shape[:2]
    activos = lados_activos(acciones or [])
    huesos_activos = set()
    for lado in activos:
        huesos_activos |= HUESOS_LADO.get(lado, set())

    for a, b in HUESOS:
        if not (np.isfinite(pose[a]).all() and np.isfinite(pose[b]).all()):
            continue
        color = COLOR_ACTIVO if (a, b) in huesos_activos or (b, a) in huesos_activos else COLOR_HUESO
        cv2.line(imagen, _xy(pose[a], ancho, alto), _xy(pose[b], ancho, alto), color, grosor, cv2.LINE_AA)

    for indice, punto in enumerate(pose):
        if not np.isfinite(punto).all():
            continue
        centro = _xy(punto, ancho, alto)
        cv2.circle(imagen, centro, radio, COLOR_PUNTO, -1, cv2.LINE_AA)
        cv2.circle(imagen, centro, radio, (255, 255, 255), 1, cv2.LINE_AA)


def _barra_tiempo(
    draw: ImageDraw.ImageDraw,
    acciones_todas: list[Accion],
    frame: int,
    n_frames: int,
    x0: int,
    x1: int,
    y: int,
    alto: int,
) -> None:
    draw.rectangle([x0, y, x1, y + alto], fill=(30, 30, 34))
    if n_frames <= 1:
        return
    ancho = x1 - x0
    for accion in acciones_todas:
        color = COLORES_ACCION.get(accion.id, (255, 180, 0))
        color_rgb = (color[2], color[1], color[0])
        a = x0 + int(ancho * accion.frame_inicio / n_frames)
        b = x0 + int(ancho * (accion.frame_fin + 1) / n_frames)
        draw.rectangle([a, y + 2, max(a + 2, b), y + alto - 2], fill=color_rgb)
    cursor = x0 + int(ancho * frame / n_frames)
    draw.line([(cursor, y), (cursor, y + alto)], fill=(255, 255, 255), width=2)


def componer(
    imagen: np.ndarray,
    pose: np.ndarray | None,
    acciones_frame: list[Accion],
    acciones_todas: list[Accion],
    frame: int,
    n_frames: int,
    fps: float,
    titulo: str,
) -> np.ndarray:
    """Añade esqueleto, el aviso de la acción y la línea de tiempo."""
    lienzo = imagen.copy()
    if acciones_frame:
        cv2.rectangle(lienzo, (3, 3), (imagen.shape[1] - 4, imagen.shape[0] - 4), (40, 40, 220), 6)
    if pose is not None:
        dibujar_esqueleto(lienzo, pose, acciones_frame)

    alto, ancho = lienzo.shape[:2]
    banda = 118
    franja = np.zeros((banda, ancho, 3), dtype=np.uint8)
    franja[:] = (16, 16, 20)
    if acciones_frame:
        franja[:, :10] = (40, 40, 220)
    salida = np.vstack([franja, lienzo])

    imagen_pil = _pil(salida)
    draw = ImageDraw.Draw(imagen_pil)
    fuente = _fuente(22, negrita=True)
    fuente_peq = _fuente(18)
    reloj = formatear_tiempo(frame / fps)
    draw.text((18, 10), f"{titulo}    {reloj}    frame {frame}", font=fuente, fill=(235, 235, 235))

    if acciones_frame:
        texto = "  ·  ".join(f"{a.nombre} ({a.lado})" for a in acciones_frame[:3])
        draw.text((18, 46), "ACCIÓN  " + texto, font=fuente_peq, fill=(255, 210, 80))
        if len(acciones_frame) > 3:
            extra = "  ·  ".join(f"{a.nombre} ({a.lado})" for a in acciones_frame[3:6])
            draw.text((18, 74), extra, font=fuente_peq, fill=(255, 210, 80))
    else:
        draw.text((18, 46), "Sin gesto de esconder", font=fuente_peq, fill=(170, 170, 170))

    y_barra = alto + banda - 22
    _barra_tiempo(draw, acciones_todas, frame, n_frames, 18, ancho - 18, y_barra, 14)
    return _bgr(imagen_pil)


def recorte_persona(imagen: np.ndarray, pose: np.ndarray, margen: int = 70) -> np.ndarray | None:
    alto, ancho = imagen.shape[:2]
    validos = pose[np.isfinite(pose).all(axis=1)]
    if len(validos) < 4:
        return None
    xs = validos[:, 0] * ancho
    ys = validos[:, 1] * alto
    x0 = max(0, int(xs.min()) - margen)
    x1 = min(ancho, int(xs.max()) + margen)
    y0 = max(0, int(ys.min()) - margen)
    y1 = min(alto, int(ys.max()) + margen)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return imagen[y0:y1, x0:x1]


def poner_recorte(imagen: np.ndarray, recorte: np.ndarray, ancho_caja: int = 280) -> None:
    if recorte is None or recorte.size == 0:
        return
    h, w = recorte.shape[:2]
    alto_caja = int(ancho_caja * h / w)
    caja = cv2.resize(recorte, (ancho_caja, alto_caja), interpolation=cv2.INTER_AREA)
    cv2.rectangle(caja, (0, 0), (ancho_caja - 1, alto_caja - 1), (255, 255, 255), 2)
    y0, x0 = 8, imagen.shape[1] - ancho_caja - 12
    if y0 + alto_caja > imagen.shape[0] or x0 < 0:
        return
    imagen[y0 : y0 + alto_caja, x0 : x0 + ancho_caja] = caja


def lienzo_esqueleto(pose: np.ndarray, ancho: int = 960, alto: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Fondo negro con el esqueleto ampliado al cuerpo de este frame."""
    imagen = np.zeros((alto, ancho, 3), dtype=np.uint8)
    imagen[:] = (18, 18, 22)
    validos = pose[np.isfinite(pose).all(axis=1)]
    if len(validos) < 2:
        return imagen, pose
    x0, y0 = validos.min(axis=0)
    x1, y1 = validos.max(axis=0)
    dx = max(float(x1 - x0), 0.05)
    dy = max(float(y1 - y0), 0.05)
    margen = 0.08
    x0 -= dx * margen
    y0 -= dy * margen
    dx *= 1 + 2 * margen
    dy *= 1 + 2 * margen
    escala = min((ancho - 40) / dx, (alto - 40) / dy)
    desplazado = pose.copy()
    desplazado[:, 0] = (pose[:, 0] - x0) * escala + 20
    desplazado[:, 1] = (pose[:, 1] - y0) * escala + 20
    for a, b in HUESOS:
        if not (np.isfinite(desplazado[a]).all() and np.isfinite(desplazado[b]).all()):
            continue
        p1 = (int(desplazado[a, 0]), int(desplazado[a, 1]))
        p2 = (int(desplazado[b, 0]), int(desplazado[b, 1]))
        cv2.line(imagen, p1, p2, COLOR_HUESO, 4, cv2.LINE_AA)
    for indice, punto in enumerate(desplazado):
        if not np.isfinite(punto).all():
            continue
        centro = (int(punto[0]), int(punto[1]))
        cv2.circle(imagen, centro, 8, COLOR_PUNTO, -1, cv2.LINE_AA)
        cv2.putText(
            imagen,
            NOMBRES[indice].replace("_", " "),
            (centro[0] + 10, centro[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return imagen, desplazado


def recolorear_esqueleto_activo(imagen: np.ndarray, desplazado: np.ndarray, acciones: list[Accion]) -> None:
    activos = lados_activos(acciones)
    huesos_activos = set()
    for lado in activos:
        huesos_activos |= HUESOS_LADO.get(lado, set())
    for a, b in HUESOS:
        if (a, b) not in huesos_activos and (b, a) not in huesos_activos:
            continue
        if not (np.isfinite(desplazado[a]).all() and np.isfinite(desplazado[b]).all()):
            continue
        p1 = (int(desplazado[a, 0]), int(desplazado[a, 1]))
        p2 = (int(desplazado[b, 0]), int(desplazado[b, 1]))
        cv2.line(imagen, p1, p2, COLOR_ACTIVO, 6, cv2.LINE_AA)


def _partir(draw: ImageDraw.ImageDraw, texto: str, fuente: ImageFont.ImageFont, ancho: int) -> list[str]:
    palabras = texto.split()
    if not palabras:
        return [""]
    lineas: list[str] = []
    actual = palabras[0]
    for palabra in palabras[1:]:
        prueba = f"{actual} {palabra}"
        if draw.textlength(prueba, font=fuente) <= ancho:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    lineas.append(actual)
    return lineas


def unir_lista(imagen: np.ndarray, acciones: list[Accion], frame: int, fps: float) -> np.ndarray:
    """Pone a la derecha la lista de acciones ya terminadas, con su intervalo."""
    alto_max = 720
    alto, ancho = imagen.shape[:2]
    if alto > alto_max:
        escala = alto_max / alto
        imagen = cv2.resize(imagen, (int(ancho * escala), alto_max), interpolation=cv2.INTER_AREA)
    en_curso = [a for a in acciones if a.frame_inicio <= frame <= a.frame_fin]
    if en_curso:
        h, w = imagen.shape[:2]
        cv2.rectangle(imagen, (3, 3), (w - 4, h - 4), (40, 40, 220), 4)

    reloj = formatear_tiempo(frame / fps)
    marca = _pil(imagen)
    dibujo = ImageDraw.Draw(marca)
    alto_img = imagen.shape[0]
    dibujo.rectangle([8, alto_img - 42, 250, alto_img - 8], fill=(0, 0, 0))
    dibujo.text((14, alto_img - 38), f"{reloj}   frame {frame}", font=_fuente(20, negrita=True), fill=(255, 255, 255))
    imagen = _bgr(marca)

    panel = _panel_acciones(acciones, frame, fps, imagen.shape[0], 480)
    return np.hstack([imagen, panel])


def _panel_acciones(acciones: list[Accion], frame: int, fps: float, alto: int, ancho: int) -> np.ndarray:
    en_curso = [a for a in acciones if a.frame_inicio <= frame < a.frame_fin]
    hechas = [a for a in acciones if frame >= a.frame_fin]
    hechas.sort(key=lambda a: (a.frame_fin, a.frame_inicio, a.nombre))

    lienzo = Image.new("RGB", (ancho, max(alto, 1)), (16, 16, 20))
    draw = ImageDraw.Draw(lienzo)
    titulo = _fuente(22, negrita=True)
    tiempo = _fuente(18, negrita=True)
    cuerpo = _fuente(17)
    pequeno = _fuente(15)
    margen = 16
    usable = ancho - 2 * margen

    draw.text((margen, 14), "Acciones", font=titulo, fill=(245, 245, 245))
    y = 52
    if en_curso:
        draw.text((margen, y), "En curso", font=pequeno, fill=(255, 196, 80))
        y += 24
        for accion in en_curso:
            for linea in _partir(draw, f"{accion.nombre} ({accion.lado})", cuerpo, usable):
                draw.text((margen, y), linea, font=cuerpo, fill=(230, 230, 230))
                y += 22
            y += 8
        draw.line([(margen, y), (ancho - margen, y)], fill=(55, 55, 62), width=1)
        y += 12

    if not hechas:
        for linea in _partir(draw, "Al terminar, cada acción aparece aquí con su intervalo.", pequeno, usable):
            draw.text((margen, y), linea, font=pequeno, fill=(150, 150, 150))
            y += 20
        return _bgr(lienzo)

    bloques: list[list[tuple[str, ImageFont.ImageFont, tuple[int, int, int]]]] = []
    for accion in hechas:
        datos = accion.a_dict(fps)
        lineas = [(f"{datos['t_inicio']} – {datos['t_fin']}", tiempo, (255, 210, 90))]
        for linea in _partir(draw, accion.nombre, cuerpo, usable):
            lineas.append((linea, cuerpo, (235, 235, 235)))
        lineas.append((accion.lado, pequeno, (170, 170, 170)))
        bloques.append(lineas)

    alturas = []
    for lineas in bloques:
        alto_bloque = 8 + sum(22 if fuente != pequeno else 20 for _, fuente, _ in lineas) + 10
        alturas.append(alto_bloque)
    total = sum(alturas)
    disponible = alto - y - 8
    desplazamiento = 0 if total <= disponible else total - disponible

    recorte = Image.new("RGB", (ancho, max(disponible, 1)), (16, 16, 20))
    dibujo = ImageDraw.Draw(recorte)
    cursor = -desplazamiento
    for lineas, alto_bloque in zip(bloques, alturas):
        if cursor + alto_bloque >= 0 and cursor < disponible:
            yy = cursor + 4
            for texto, fuente, color in lineas:
                if 0 <= yy < disponible - 4:
                    dibujo.text((margen, yy), texto, font=fuente, fill=color)
                yy += 22 if fuente != pequeno else 20
            linea_y = cursor + alto_bloque - 6
            if 0 <= linea_y < disponible:
                dibujo.line([(margen, linea_y), (ancho - margen, linea_y)], fill=(48, 48, 56), width=1)
        cursor += alto_bloque
    lienzo.paste(recorte, (0, y))
    return _bgr(lienzo)


def reproducir(titulo: str, n: int, fps: float, construir) -> None:
    """Abre una ventana. Espacio pausa, flechas un frame, Q sale. No guarda nada."""
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    indice = 0
    pausa = False
    abierta = False
    while True:
        if abierta and cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.imshow(titulo, construir(indice))
        abierta = True
        espera = 30 if pausa or indice >= n - 1 else max(1, int(round(1000 / fps)))
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord(" "):
            pausa = not pausa
            continue
        if tecla in (81, 2, ord("a")):
            indice = max(0, indice - 1)
            pausa = True
            continue
        if tecla in (83, 3, ord("d")):
            indice = min(n - 1, indice + 1)
            pausa = True
            continue
        if not pausa and tecla == 255 and indice < n - 1:
            indice += 1
    cv2.destroyAllWindows()


def guardar_temporal(path: Path, fps: float, n: int, construir) -> None:
    """Escribe el vídeo en path, sustituyendo el archivo si ya existía."""
    if n <= 0:
        return
    primero = construir(0)
    alto, ancho = primero.shape[:2]
    escritor = abrir_escritor(path, fps, ancho, alto)
    escritor.write(primero)
    for indice in range(1, n):
        escritor.write(construir(indice))
    escritor.release()


def abrir_escritor(path: Path, fps: float, ancho: int, alto: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    escritor = cv2.VideoWriter(str(path), fourcc, fps, (ancho, alto))
    if not escritor.isOpened():
        raise RuntimeError(f"No se ha podido crear el vídeo {path}")
    return escritor
