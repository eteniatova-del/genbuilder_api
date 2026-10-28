"""Проверки СП 4.13130.2013 п. 8.1.9, 8.1.10, 8.1.12, 8.1.15 (текст проверен по PDF, предоставленному автором).

Что норма требует, а что в модели допущение:
- 8.1.9: «В замкнутых и полузамкнутых дворах необходимо предусматривать проезды для пожарных автомобилей».
  Что считать замкнутым и полузамкнутым двором, в норме не определено — ДОПУЩЕНИЕ (см. COURT_*).
- 8.1.10: сквозные проезды (арки) шириной не менее 3,5 м и высотой не менее 4,5 м не более чем через 300 м
  (в реконструируемых районах при застройке по периметру — 180 м). Арки в модели не создаются, поэтому
  проверка только сообщает о непрерывном фронте длиннее 300 м. Что считать непрерывным фронтом — ДОПУЩЕНИЕ.
- 8.1.12: здание длиннее 100 м — сквозные проходы шириной не менее 1,2 м не реже чем через 100 м; не нужны, если
  водопровод с гидрантами предусмотрен с обеих продольных сторон. Гидранты в модели не заданы: проверка
  только предупреждает.
- 8.1.15: малоэтажная застройка Ф1.4 (до 3 этажей включительно) — подъезд пожарной техники к зданиям на
  расстояние не более 50 м.
"""
from math import hypot

from shapely import maximum_inscribed_circle
from shapely.geometry import Polygon
from shapely.ops import unary_union

ARCH_SPACING_M = 300.0           # п. 8.1.10
PASSAGE_SPACING_M = 100.0        # п. 8.1.12
LOW_RISE_REACH_M = 50.0          # п. 8.1.15
LOW_RISE_MAX_FLOORS = 3          # п. 8.1.15: Ф1.4, до 3 этажей включительно

COURT_CLOSING_M = 10.0           # ДОПУЩЕНИЕ: проём двора не шире 2·10 = 20 м
COURT_MIN_ADJ_SHARE = 0.7        # ДОПУЩЕНИЕ: не менее 70 % границы двора примыкает к зданиям (три стороны из четырёх)
COURT_MIN_DIAM_M = 3.5           # ДОПУЩЕНИЕ: в двор вписывается круг диаметром не меньше ширины пожарного проезда
FRONT_GAP_M = 3.5                # ДОПУЩЕНИЕ: здания ближе 3,5 м (ширина пожарного проезда) образуют один непрерывный фронт


def long_side(g) -> float:
    c = list(g.minimum_rotated_rectangle.exterior.coords)
    return max(hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]), hypot(c[2][0] - c[1][0], c[2][1] - c[1][1]))


def find_courtyards(footprints) -> list:
    """Замкнутые и полузамкнутые дворы: области, заполняемые замыканием контура застройки (допущение, см. модуль)."""
    if len(footprints) < 1:
        return []
    u = unary_union(list(footprints))
    closed = u.buffer(COURT_CLOSING_M).buffer(-COURT_CLOSING_M)
    holes = [Polygon(r) for pg in (u.geoms if hasattr(u, "geoms") else [u]) for r in pg.interiors]     # замкнутые дворы — отверстия контура
    filled = unary_union([closed] + holes)
    pockets = filled.difference(u)
    parts = list(pockets.geoms) if hasattr(pockets, "geoms") else [pockets]
    ub = u.buffer(0.05)
    out = []
    for pk in parts:
        if pk.is_empty or not isinstance(pk, Polygon):
            continue
        bounds = [pk.exterior] + list(pk.interiors)
        per = sum(b.length for b in bounds)
        if per <= 0:
            continue
        adj = sum(b.intersection(ub).length for b in bounds)
        openl = [b.difference(ub) for b in bounds]
        n_open = 0
        for o in openl:
            segs = list(o.geoms) if hasattr(o, "geoms") else [o]
            n_open += sum(1 for s in segs if s.length > 1.0)
        if n_open > 1 or adj / per < COURT_MIN_ADJ_SHARE:
            continue
        diam = 2.0 * maximum_inscribed_circle(pk, tolerance=0.5).length
        if diam >= COURT_MIN_DIAM_M:
            out.append(pk)
    return out


def continuous_fronts(footprints) -> list:
    """Группы зданий, между которыми нет прохода шириной пожарного проезда; возвращает (индексы, длина фронта, м)."""
    n = len(footprints)
    parent = list(range(n))

    def f(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if footprints[i].distance(footprints[j]) < FRONT_GAP_M:
                parent[f(i)] = f(j)
    groups = {}
    for i in range(n):
        groups.setdefault(f(i), []).append(i)
    return [(idx, long_side(unary_union([footprints[i] for i in idx]))) for idx in groups.values()]
