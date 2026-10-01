"""
Gestos de esconder a partir del esqueleto.

Solo se mira lo que ocurre después de tener el producto en la mano:
acercarlo al cuerpo, taparlo, bajarlo a la cadera o llevarlo pegado al andar.
Con 8 puntos no se ve el objeto: cada aviso es un gesto compatible con eso.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from har_acciones.pose import (
    CADERA_D,
    CADERA_I,
    HOMBRO_D,
    HOMBRO_I,
    LADO_BRAZO,
    formatear_tiempo,
)

CONFIG_ACCIONES = Path(__file__).resolve().parents[1] / "acciones_config.json"

# Estas de la lista no se pueden ver sin el objeto ni sin la bolsa.
NO_OBSERVABLES = (
    "Abrir la mochila, el bolso o el carro",
    "Meter el brazo en un bolso y sacarlo sin el objeto",
    "Tapar el producto con otro artículo",
)


@dataclass
class Accion:
    id: str
    nombre: str
    lado: str
    frame_inicio: int
    frame_fin: int
    detalle: str

    def a_dict(self, fps: float) -> dict:
        datos = asdict(self)
        datos["t_inicio"] = formatear_tiempo(self.frame_inicio / fps)
        datos["t_fin"] = formatear_tiempo((self.frame_fin + 1) / fps)
        datos["duracion_s"] = round((self.frame_fin - self.frame_inicio + 1) / fps, 2)
        return datos


def detectar(poses: np.ndarray, fps: float) -> list[Accion]:
    """poses suavizadas, shape (T, 8, 2)."""
    f = _rasgos(poses, fps)
    acciones: list[Accion] = []
    acciones += _acercar_al_torso(f)
    acciones += _antebrazo_pegado(f)
    acciones += _manos_juntas(f)
    acciones += _tapar_con_antebrazo(f)
    acciones += _una_mano_oculta(f)
    acciones += _bajar_a_cadera(f)
    acciones += _subir_por_el_torso(f)
    acciones += _brazos_cruzados(f)
    acciones += _girar_torso(f)
    acciones += _agacharse(f)
    acciones += _mano_a_la_espalda(f)
    acciones += _brazo_rigido(f)
    acciones = _fusionar(acciones, hueco=4)
    acciones += _estancia_tras_bajar(f, acciones)
    config = cargar_config_acciones()
    acciones = [accion for accion in acciones if accion_activa(accion.id, config)]
    acciones.sort(key=lambda a: (a.frame_inicio, a.frame_fin, a.id, a.lado))
    return acciones


def cargar_config_acciones(ruta: Path | None = None) -> dict[str, bool]:
    """id -> activa. Si el fichero no existe, todas quedan activas."""
    path = ruta or CONFIG_ACCIONES
    if not path.is_file():
        return {}
    datos = json.loads(path.read_text(encoding="utf-8"))
    activas: dict[str, bool] = {}
    for item in datos.get("acciones", []):
        if "id" in item:
            activas[item["id"]] = bool(item.get("activa", True))
    return activas


def accion_activa(accion_id: str, config: dict[str, bool] | None = None) -> bool:
    mapa = cargar_config_acciones() if config is None else config
    return mapa.get(accion_id, True)


def acciones_desactivadas() -> list[str]:
    return [accion_id for accion_id, activa in cargar_config_acciones().items() if not activa]


def acciones_en_frame(acciones: list[Accion], frame: int) -> list[Accion]:
    return [a for a in acciones if a.frame_inicio <= frame <= a.frame_fin]


def informe(acciones: list[Accion], fps: float) -> str:
    if not acciones:
        return "No se ha marcado ningún gesto de esconder en este clip."
    lineas = [f"{len(acciones)} gesto(s) de esconder:", ""]
    for accion in acciones:
        datos = accion.a_dict(fps)
        lineas.append(
            f"  {datos['t_inicio']} → {datos['t_fin']}   "
            f"{accion.nombre}   [{accion.lado}]"
        )
        lineas.append(f"      {accion.detalle}")
        lineas.append(
            f"      frames {accion.frame_inicio}–{accion.frame_fin}  "
            f"({datos['duracion_s']} s)"
        )
        lineas.append("")
    return "\n".join(lineas).rstrip()


def _rasgos(poses: np.ndarray, fps: float) -> dict:
    p = poses
    t = len(p)
    hombro_c = 0.5 * (p[:, HOMBRO_I] + p[:, HOMBRO_D])
    cadera_c = 0.5 * (p[:, CADERA_I] + p[:, CADERA_D])
    torso = np.linalg.norm(hombro_c - cadera_c, axis=1)
    ancho = np.linalg.norm(p[:, HOMBRO_I] - p[:, HOMBRO_D], axis=1)
    torso_ok = (torso > 0.06) & (torso < 0.55) & (cadera_c[:, 1] > hombro_c[:, 1] - 0.02)
    escala = np.maximum(torso, 1e-3)

    pecho = 0.50 * hombro_c + 0.50 * cadera_c
    abdomen = 0.28 * hombro_c + 0.72 * cadera_c

    velocidad = np.zeros(t)
    if t > 1:
        paso = np.linalg.norm(np.diff(cadera_c, axis=0), axis=1) * fps
        velocidad[1:] = paso / escala[1:]
        velocidad[0] = velocidad[1]

    brazos = {}
    for lado, (hombro, codo, muneca, cadera) in LADO_BRAZO.items():
        w = p[:, muneca]
        s = p[:, hombro]
        e = p[:, codo]
        hip = p[:, cadera]
        brazo_largo = np.linalg.norm(e - s, axis=1)
        ante = np.linalg.norm(w - e, axis=1)
        angulo = _angulo(s, e, w)
        # Un codo de menos de 40° es un brazo colapsado por el tracker, no un gesto.
        ok = torso_ok & (brazo_largo > 0.10 * escala) & (ante > 0.08 * escala) & (angulo > 40)
        alto_torso = np.maximum(hip[:, 1] - s[:, 1], 1e-3)
        # 0 = altura de la cadera, 1 = altura del hombro. Por encima del hombro > 1.
        altura = (hip[:, 1] - w[:, 1]) / alto_torso
        brazos[lado] = {
            "ok": ok,
            "angulo": angulo,
            "dist_pecho": np.linalg.norm(w - pecho, axis=1) / escala,
            "dist_abdomen": np.linalg.norm(w - abdomen, axis=1) / escala,
            "dist_cadera": np.linalg.norm(w - hip, axis=1) / escala,
            "dist_eje": _distancia_segmento(w, s, hip) / escala,
            "altura": altura,
            "cruza": (w[:, 0] - hombro_c[:, 0]) / np.maximum(ancho, 1e-3),
            "muneca": w,
            "cadera": hip,
        }
        if lado == "izquierdo":
            brazos[lado]["cruza_hacia_dentro"] = brazos[lado]["cruza"]
        else:
            brazos[lado]["cruza_hacia_dentro"] = -brazos[lado]["cruza"]

    munecas = np.linalg.norm(p[:, 4] - p[:, 5], axis=1) / escala
    return {
        "fps": fps,
        "n": t,
        "torso_ok": torso_ok,
        "escala": escala,
        "ancho": ancho,
        "ratio_frontal": ancho / escala,
        "velocidad": velocidad,
        "cadera_c": cadera_c,
        "hombro_c": hombro_c,
        "brazos": brazos,
        "sep_munecas": munecas,
        "poses": p,
    }


def _angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    cos = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _distancia_segmento(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.sum(ab * ab, axis=1) + 1e-8
    t = np.clip(np.sum((p - a) * ab, axis=1) / denom, 0.0, 1.0)
    proy = a + t[:, None] * ab
    return np.linalg.norm(p - proy, axis=1)


def _tramos(mascara: np.ndarray, minimo: int, hueco: int = 2) -> list[tuple[int, int]]:
    """Tramos inclusivos. Cierra huecos cortos y descarta los breves."""
    m = np.asarray(mascara, dtype=bool).copy()
    if hueco > 0 and m.any():
        i = 0
        n = len(m)
        while i < n:
            if m[i]:
                i += 1
                continue
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= hueco:
                m[i:j] = True
            i = j
    tramos = []
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        if (j - i) >= minimo:
            tramos.append((i, j - 1))
        i = j
    return tramos


def _fusionar(acciones: list[Accion], hueco: int) -> list[Accion]:
    grupos: dict[tuple[str, str], list[Accion]] = {}
    for accion in acciones:
        grupos.setdefault((accion.id, accion.lado), []).append(accion)
    salida: list[Accion] = []
    for grupo in grupos.values():
        grupo.sort(key=lambda a: a.frame_inicio)
        actual = grupo[0]
        for siguiente in grupo[1:]:
            if siguiente.frame_inicio <= actual.frame_fin + hueco:
                actual = Accion(
                    actual.id,
                    actual.nombre,
                    actual.lado,
                    actual.frame_inicio,
                    max(actual.frame_fin, siguiente.frame_fin),
                    actual.detalle,
                )
            else:
                salida.append(actual)
                actual = siguiente
        salida.append(actual)
    return salida


def _ventanas(n: int, ancho: int, paso: int = 1):
    tope = n - ancho
    i = 0
    while i <= tope:
        yield i, i + ancho - 1
        i += paso


def _fem(lado: str) -> str:
    """Concordancia: la muñeca izquierda / derecha."""
    return "izquierda" if lado == "izquierdo" else "derecha"


def _acercar_al_torso(f: dict) -> list[Accion]:
    """La muñeca baja hacia el abdomen. No cuenta el brazo estirado hacia el estante."""
    acciones = []
    ancho = max(6, int(round(0.70 * f["fps"])))
    for lado, brazo in f["brazos"].items():
        dist = brazo["dist_abdomen"]
        for i, j in _ventanas(f["n"], ancho, paso=2):
            if brazo["ok"][i : j + 1].mean() < 0.8:
                continue
            if brazo["altura"][i] < 0.45 or brazo["altura"][j] > 0.72:
                continue
            if brazo["altura"][j] > brazo["altura"][i] - 0.12:
                continue
            bajada = dist[i] - dist[j]
            if bajada < 0.22 or dist[j] > 0.48 or dist[i] < 0.60:
                continue
            if not (0.22 <= brazo["altura"][j] <= 0.72):
                continue
            acciones.append(
                Accion(
                    "acercar_al_pecho",
                    "Acercar el producto al pecho o al abdomen",
                    lado,
                    i,
                    j,
                    (
                        f"La muñeca {_fem(lado)} se acerca al abdomen: "
                        f"de {dist[i]:.2f} a {dist[j]:.2f} torsos de distancia."
                    ),
                )
            )
    return acciones


def _antebrazo_pegado(f: dict) -> list[Accion]:
    minimo = max(5, int(round(0.45 * f["fps"])))
    acciones = []
    for lado, brazo in f["brazos"].items():
        mascara = (
            brazo["ok"]
            & (brazo["angulo"] > 55)
            & (brazo["angulo"] < 125)
            & (brazo["dist_eje"] < 0.20)
            & (brazo["altura"] > 0.18)
            & (brazo["altura"] < 0.62)
        )
        for i, j in _tramos(mascara, minimo):
            acciones.append(
                Accion(
                    "pegar_al_torso",
                    "Pegar el antebrazo al torso",
                    lado,
                    i,
                    j,
                    (
                    f"El codo {lado} está doblado y la muñeca {_fem(lado)} "
                    f"va pegada al eje del torso."
                    ),
                )
            )
    return acciones


def _manos_juntas(f: dict) -> list[Accion]:
    """Las dos muñecas se juntan delante del abdomen: pasar el objeto de mano."""
    iz, de = f["brazos"]["izquierdo"], f["brazos"]["derecho"]
    minimo = max(6, int(round(0.50 * f["fps"])))
    angulo_menor = np.minimum(iz["angulo"], de["angulo"])
    mascara = (
        iz["ok"]
        & de["ok"]
        & (f["sep_munecas"] < 0.28)
        & (f["ratio_frontal"] > 0.20)
        & (angulo_menor > 50)
        & (angulo_menor < 145)
        & (iz["altura"] > 0.14)
        & (iz["altura"] < 0.62)
        & (de["altura"] > 0.14)
        & (de["altura"] < 0.62)
    )
    acciones = []
    for i, j in _tramos(mascara, minimo):
        acciones.append(
            Accion(
                "pasar_de_mano",
                "Pasar el objeto de una mano a la otra",
                "ambos",
                i,
                j,
                "Las dos muñecas se juntan delante del torso, con al menos un codo doblado.",
            )
        )
    return acciones


def _tapar_con_antebrazo(f: dict) -> list[Accion]:
    minimo = max(4, int(round(0.28 * f["fps"])))
    acciones = []
    for lado, brazo in f["brazos"].items():
        # Hace falta un torso algo de frente: en perfil puro la línea media no se ve.
        mascara = (
            brazo["ok"]
            & (f["ratio_frontal"] > 0.22)
            & (brazo["cruza_hacia_dentro"] > 0.45)
            & (brazo["angulo"] < 155)
            & (brazo["altura"] > 0.20)
            & (brazo["altura"] < 0.95)
        )
        for i, j in _tramos(mascara, minimo):
            acciones.append(
                Accion(
                    "tapar_con_antebrazo",
                    "Tapar el producto con el antebrazo",
                    lado,
                    i,
                    j,
                    f"La muñeca {_fem(lado)} cruza por delante de la línea media del pecho.",
                )
            )
    return acciones


def _una_mano_oculta(f: dict) -> list[Accion]:
    """Una mano sigue fuera, a media altura; la otra queda baja junto a la cadera."""
    minimo = max(4, int(round(0.35 * f["fps"])))
    acciones = []
    pares = (("izquierdo", "derecho"), ("derecho", "izquierdo"))
    for oculta, visible in pares:
        baja = f["brazos"][oculta]
        alta = f["brazos"][visible]
        mascara = (
            baja["ok"]
            & alta["ok"]
            & (baja["dist_cadera"] < 0.38)
            & (baja["altura"] < 0.35)
            & (alta["dist_eje"] > 0.55)
            & (alta["altura"] > 0.35)
            & (alta["altura"] < 0.90)
            & (alta["angulo"] > 140)
        )
        for i, j in _tramos(mascara, minimo):
            acciones.append(
                Accion(
                    "una_mano_oculta",
                    "Una mano a la vista y la otra oculta junto al cuerpo",
                    oculta,
                    i,
                    j,
                    (
                        f"La mano {_fem(oculta)} está junto a la cadera mientras la {_fem(visible)} "
                        f"permanece separada del torso, a media altura."
                    ),
                )
            )
    return acciones


def _bajar_a_cadera(f: dict) -> list[Accion]:
    """Baja desde arriba (acaba de coger) hasta la cadera. El brazo ya colgando no cuenta."""
    acciones = []
    ancho = max(8, int(round(0.85 * f["fps"])))
    for lado, brazo in f["brazos"].items():
        bajadas = []
        for i, j in _ventanas(f["n"], ancho, paso=2):
            if i < 3 or brazo["ok"][i : j + 1].mean() < 0.75:
                continue
            if brazo["altura"][i] < 0.75 or np.median(brazo["altura"][i - 3 : i + 1]) < 0.72:
                continue
            medio = (i + j) // 2
            if not (brazo["altura"][i] > brazo["altura"][medio] > brazo["altura"][j]):
                continue
            caida = brazo["altura"][i] - brazo["altura"][j]
            if caida < 0.40:
                continue
            if brazo["altura"][j] > 0.32 or brazo["dist_cadera"][j] > 0.48:
                continue
            bajadas.append((i, j, caida))
        for i, j, caida in bajadas:
            acciones.append(
                Accion(
                    "bajar_a_bolsillo",
                    "Bajar la mano hacia el bolsillo",
                    lado,
                    i,
                    j,
                    (
                        f"La mano {_fem(lado)} baja desde arriba hasta la cadera "
                        f"(la altura relativa cae {caida:.2f})."
                    ),
                )
            )
        acciones += _entrada_bolsillo(f, lado, brazo)
    return acciones


def _entrada_bolsillo(f: dict, lado: str, brazo: dict) -> list[Accion]:
    """La mano baja del pecho al bolsillo en poco tiempo y se queda ahí.

    El tracker suele colapsar el codo justo en la entrada, así que no se exige
    que todos los frames intermedios sean válidos.
    """
    fps = f["fps"]
    n = f["n"]
    corto = max(3, int(round(0.24 * fps)))
    # Hasta ~1 s: al meter la mano el antebrazo se encoge y esos frames no son válidos.
    largo = max(corto + 1, int(round(1.05 * fps)))
    estancia = max(4, int(round(0.40 * fps)))
    acciones = []
    altura = brazo["altura"]
    dist = brazo["dist_cadera"]
    ok = brazo["ok"]
    for j in range(largo, n):
        if not (ok[j] and altura[j] <= 0.22 and dist[j] <= 0.32):
            continue
        if altura[j - 1] <= 0.22 and dist[j - 1] <= 0.32:
            continue
        inicio = None
        for i in range(j - corto, max(2, j - largo) - 1, -1):
            if i < 3 or not ok[i] or altura[i] < 0.55:
                continue
            if float(np.nanmedian(altura[i - 3 : i + 1])) < 0.50:
                continue
            if altura[i] - altura[j] < 0.35:
                continue
            inicio = i
            break
        if inicio is None:
            continue
        fin = min(n - 1, j + estancia)
        bajos = sum(altura[k] < 0.30 and dist[k] < 0.42 for k in range(j, fin + 1))
        if bajos < estancia:
            continue
        caida = altura[inicio] - altura[j]
        acciones.append(
            Accion(
                "bajar_a_bolsillo",
                "Bajar la mano hacia el bolsillo",
                lado,
                inicio,
                j,
                (
                    f"La mano {_fem(lado)} pasa del pecho a la cadera "
                    f"y se queda ahí (la altura relativa cae {caida:.2f})."
                ),
            )
        )
    return acciones


def _estancia_tras_bajar(f: dict, acciones: list[Accion]) -> list[Accion]:
    """La mano se queda en la cadera solo justo después de haber bajado, y poco rato."""
    tope = max(6, int(round(0.70 * f["fps"])))
    minimo = max(4, int(round(0.35 * f["fps"])))
    salida = []
    for accion in acciones:
        if accion.id != "bajar_a_bolsillo":
            continue
        brazo = f["brazos"][accion.lado]
        j = accion.frame_fin
        k = j
        limite = min(f["n"] - 1, j + tope)
        while k < limite and brazo["ok"][k] and brazo["dist_cadera"][k] < 0.40 and brazo["altura"][k] < 0.32:
            k += 1
        if (k - j) >= minimo:
            salida.append(
                Accion(
                    "mano_en_bolsillo",
                    "Mantener la mano en el bolsillo o la cadera",
                    accion.lado,
                    j,
                    k,
                    f"Tras bajar, la muñeca {_fem(accion.lado)} se queda junto a la cadera.",
                )
            )
    return salida


def _subir_por_el_torso(f: dict) -> list[Accion]:
    """Sube pegada al cuerpo: el bajo de la ropa o la mano bajo la prenda. No es el brazo al estante."""
    acciones = []
    ancho = max(5, int(round(0.55 * f["fps"])))
    for lado, brazo in f["brazos"].items():
        for i, j in _ventanas(f["n"], ancho, paso=2):
            if not brazo["ok"][i : j + 1].mean() > 0.8:
                continue
            if brazo["altura"][i] > 0.40:
                continue
            subida = brazo["altura"][j] - brazo["altura"][i]
            if subida < 0.30 or not (0.45 <= brazo["altura"][j] <= 0.78):
                continue
            if np.median(brazo["dist_eje"][i : j + 1]) > 0.22:
                continue
            if np.median(brazo["angulo"][i : j + 1]) > 140:
                continue
            acciones.append(
                Accion(
                    "subir_por_torso",
                    "Subir la mano por el torso, contra el cuerpo",
                    lado,
                    i,
                    j,
                    (
                        f"La muñeca {_fem(lado)} sube desde la cadera hacia el pecho "
                        f"sin separarse del torso."
                    ),
                )
            )
    return acciones


def _brazos_cruzados(f: dict) -> list[Accion]:
    iz, de = f["brazos"]["izquierdo"], f["brazos"]["derecho"]
    minimo = max(4, int(round(0.30 * f["fps"])))
    mascara = (
        iz["ok"]
        & de["ok"]
        & (f["ratio_frontal"] > 0.20)
        & (iz["cruza_hacia_dentro"] > 0.25)
        & (de["cruza_hacia_dentro"] > 0.25)
        & (iz["angulo"] < 155)
        & (de["angulo"] < 155)
        & (iz["altura"] > 0.30)
        & (iz["altura"] < 1.05)
        & (de["altura"] > 0.30)
        & (de["altura"] < 1.05)
    )
    acciones = []
    for i, j in _tramos(mascara, minimo):
        acciones.append(
            Accion(
                "brazos_cruzados",
                "Cruzar los brazos sobre el pecho",
                "ambos",
                i,
                j,
                "Cada muñeca pasa al lado contrario del pecho, con los codos doblados.",
            )
        )
    return acciones


def _girar_torso(f: dict) -> list[Accion]:
    """El ancho de hombros se estrecha: el cuerpo se pone de lado y una mano va junto al torso."""
    ratio = f["ratio_frontal"].copy()
    ok = f["torso_ok"]
    acciones = []
    ancho = max(6, int(round(0.70 * f["fps"])))
    for i, j in _ventanas(f["n"], ancho, paso=2):
        if ok[i : j + 1].mean() < 0.8:
            continue
        if ratio[i] < 0.48 or ratio[j] > 0.22:
            continue
        if ratio[i] - ratio[j] < 0.26:
            continue
        despues = ratio[j : min(f["n"], j + 8)]
        if len(despues) < 5 or np.median(despues) > 0.26:
            continue
        mano_junto = False
        for brazo in f["brazos"].values():
            if brazo["ok"][i : j + 1].mean() > 0.6 and np.median(brazo["dist_eje"][i : j + 1]) < 0.35:
                mano_junto = True
        if not mano_junto:
            continue
        acciones.append(
            Accion(
                "girar_torso",
                "Girar el torso de lado para tapar las manos",
                "ambos",
                i,
                j,
                (
                    f"Los hombros pasan de verse de frente (ancho {ratio[i]:.2f}) "
                    f"a verse de lado (ancho {ratio[j]:.2f}), con una mano junto al cuerpo."
                ),
            )
        )
    return acciones


def _agacharse(f: dict) -> list[Accion]:
    """Las caderas bajan en la imagen y las dos manos quedan a la altura de las piernas."""
    y = f["cadera_c"][:, 1]
    iz, de = f["brazos"]["izquierdo"], f["brazos"]["derecho"]
    acciones = []
    ancho = max(5, int(round(0.70 * f["fps"])))
    for i, j in _ventanas(f["n"], ancho, paso=2):
        if f["torso_ok"][i : j + 1].mean() < 0.8:
            continue
        bajada = y[j] - y[i]
        if bajada < 0.045:
            continue
        manos_bajas = (
            iz["ok"][j]
            and de["ok"][j]
            and iz["altura"][j] < 0.25
            and de["altura"][j] < 0.25
        )
        if not manos_bajas:
            continue
        acciones.append(
            Accion(
                "agacharse",
                "Agacharse con las manos a la altura de las piernas",
                "ambos",
                i,
                j,
                "El cuerpo baja en la imagen y las dos muñecas quedan a la altura de las caderas o por debajo.",
            )
        )
    return acciones


def _mano_a_la_espalda(f: dict) -> list[Accion]:
    minimo = max(4, int(round(0.30 * f["fps"])))
    acciones = []
    pares = (
        ("izquierdo", "derecho"),
        ("derecho", "izquierdo"),
    )
    for lado, contrario in pares:
        brazo = f["brazos"][lado]
        cadera_contraria = f["brazos"][contrario]["cadera"]
        dist_contra = np.linalg.norm(brazo["muneca"] - cadera_contraria, axis=1) / f["escala"]
        mascara = (
            brazo["ok"]
            & (f["ratio_frontal"] > 0.28)
            & (dist_contra < brazo["dist_cadera"] - 0.15)
            & (dist_contra < 0.40)
            & (brazo["altura"] < 0.45)
            & (brazo["angulo"] < 150)
            & (brazo["angulo"] > 50)
        )
        for i, j in _tramos(mascara, minimo):
            acciones.append(
                Accion(
                    "mano_a_espalda",
                    "Llevar una mano a la espalda o a la cadera contraria",
                    lado,
                    i,
                    j,
                    f"La muñeca {_fem(lado)} queda más cerca de la cadera contraria que de la suya.",
                )
            )
    return acciones


def _brazo_rigido(f: dict) -> list[Accion]:
    """Camina y un brazo no se balancea: se queda a la misma distancia de su cadera."""
    ancho = max(8, int(round(1.0 * f["fps"])))
    acciones = []
    lados = ("izquierdo", "derecho")
    for i, j in _ventanas(f["n"], ancho, paso=3):
        if f["torso_ok"][i : j + 1].mean() < 0.85:
            continue
        if np.median(f["velocidad"][i : j + 1]) < 0.10:
            continue
        desvio = {}
        for lado in lados:
            brazo = f["brazos"][lado]
            if brazo["ok"][i : j + 1].mean() < 0.85:
                desvio[lado] = None
                continue
            rel = brazo["muneca"][i : j + 1] - brazo["cadera"][i : j + 1]
            rel = rel / f["escala"][i : j + 1, None]
            desvio[lado] = float(np.mean(np.linalg.norm(rel - rel.mean(axis=0), axis=1)))
        for lado, otro in (lados, lados[::-1]):
            if desvio[lado] is None or desvio[otro] is None:
                continue
            brazo = f["brazos"][lado]
            altura = float(np.median(brazo["altura"][i : j + 1]))
            pegado = float(np.median(brazo["dist_cadera"][i : j + 1])) < 0.40
            # Pegado al torso, no el brazo muerto colgando junto al muslo.
            en_torso = 0.18 <= altura <= 0.55
            if desvio[lado] < 0.05 and desvio[otro] > 0.12 and pegado and en_torso:
                acciones.append(
                    Accion(
                        "brazo_rigido",
                        "Caminar con un brazo pegado al cuerpo, sin balanceo",
                        lado,
                        i,
                        j,
                        (
                            f"Mientras el cuerpo se desplaza, el brazo {lado} apenas se mueve "
                            f"respecto a su cadera y el otro sí oscila."
                        ),
                    )
                )
    return acciones
