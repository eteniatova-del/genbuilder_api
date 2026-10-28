"""
Модуль внутриквартальных проездов (эскизный расчёт).

Вход: полигон квартала (метры, UTM), полигоны зданий, число этажей зданий (и, при наличии, название сервиса),
      при втором проходе — полигоны парковочных пятен, к которым нужно подвести проезд.
Выход: полигоны проезжей части (с шириной и классом), разворотные площадки, диагностика.

Критерии генерации (обоснование и источники: claude/критерии_генерации_проездов.md в проекте):
  К1 достижимость: к каждому зданию и пятну парковки подведён проезд (Chen, Song, Ortner 2024, требование «parcel reachability»);
  К2 связность: нет изолированных участков, сеть выходит на улицу (там же, «street connectivity»);
  К3 минимальная длина сети и К4 минимум развилок (там же): кратчайшие пути с поощрением совместного использования
     уже построенных участков (штраф за новую развилку);
  К5 прямоугольность: проезды ортогональны, развилки под 90°, число поворотов штрафуется (там же: junction angles close to 90°;
     прямоугольная сеть Штейнера, Hanan 1966); сетка ориентирована по преобладающему направлению зданий;
  К6 тупики: не более 150 м (СП 42 п. 8.49), в конце площадка 15 x 15 м без стоянки (СП 42 п. 8.50);
     тупик длиннее предела закрывается вторым выходом на улицу (петельная схема, МУ «Квартал» ВГАСУ 2022, раздел 3.2);
  К7 подъезд: внутренний край проезда 5-8 м (до 28 м) и 8-10 м (выше 28 м) от стены; ширина по классу (СП 42 табл. 8.9)
     и по высоте здания (СП 4.13130.2013 п. 8.1.4); подъезд с 1, 2 или 4 сторон по классу и высоте (п. 8.1.1); школы и детские сады:
     два проезда с противоположных сторон (СП 42 п. 6.4.49). Правило «шире 18 м» (МУ «Квартал» ВГАСУ 2022, вторичный источник) отключено.
ДОПУЩЕНИЯ (не нормы): высота этажа 3,0 м; положение края проезда в середине допустимого диапазона; штраф поворота 20 м,
  штраф развилки 10 м, коэффициент прохода по готовому проезду 0,15; шаг растра 2,5 м.
"""
from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, sin

import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import contains_xy
from shapely.affinity import rotate
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points, unary_union

EDU_SERVICES = ("школа", "детский сад")
_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))     # (dx, dy) в ячейках: +x, +y, -x, -y


@dataclass
class RoadParams:
    """Поля со ссылкой на пункт нормы — нормативные значения. Поля ASSUMPTION — внутренние константы метода (шаг растра, штрафы поиска пути, достройка съездов,
    скругления): наружу через API не выводятся, на выполнение нормативных требований не влияют, выполнение норм проверяется в layout_checks."""
    cell: float = 2.5                         # шаг растра, м (ASSUMPTION точности расчёта)
    floor_height: float = 3.0                 # ASSUMPTION: высота этажа, м
    main_width: float = 6.0                   # СП 42 табл. 8.9: 2 полосы x 3,0 м
    secondary_width: float = 3.5              # СП 42 табл. 8.9: 1 полоса x 3,5 м
    fire_width: tuple = ((13.0, 3.5), (46.0, 4.2), (10**9, 6.0))   # СП 4.13130.2013 п. 8.1.4 (текст проверен): по высоте здания, м
    wall_gap_low: tuple = (5.0, 8.0)          # внутренний край проезда до стены, здание до 28 м (СП 4.13130.2013 п. 8.1.6, текст проверен)
    wall_gap_high: tuple = (8.0, 10.0)        # то же, выше 28 м
    high_building_m: float = 28.0
    dead_end_max: float = 150.0               # СП 42 п. 8.49
    turnaround: float = 15.0                  # СП 4.13130.2013 п. 8.1.11 (текст проверен): 15 x 15 м; СП 42 п. 8.50
    close_dead_ends: bool = True              # длинные тупики закрываются вторым выходом на улицу; при False остаются промежуточные площадки (п. 8.1.11)
    stub_m: float = 5.0                       # ASSUMPTION: короткий отросток от сквозного проезда не считается тупиком (ни закрытия, ни площадки)
    turnaround_reach: float = 25.0            # ASSUMPTION: если у конца тупика места нет, площадка ищется вдоль проезда не дальше, м
    reuse_cost: float = 0.15                  # ASSUMPTION: относительная стоимость прохода по уже построенному проезду
    turn_penalty: float = 20.0                # ASSUMPTION: штраф за поворот, м эквивалентной длины (К5)
    junction_penalty: float = 10.0            # ASSUMPTION: штраф за новую развилку, м (К4)
    fire_sides_enabled: bool = True           # СП 4.13130.2013 п. 8.1.1 (текст проверен): подъезд вдоль стен по всей длине стороны
    fire_class_by_service: tuple = (("детский сад", "F1.1"), ("дошкольн", "F1.1"), ("школа", "F4.1"),
                                    ("общеобразовательн", "F4.1"), ("поликлиник", "F3.4"))   # 123-ФЗ ст. 32 (текст проверен)
    fire_two_sides_f13_m: float = 28.0        # п. 8.1.1 а): многоквартирные жилые Ф1.3 высотой 28 м и более: две продольные стороны
    fire_two_sides_public_m: float = 18.0     # п. 8.1.1 а): Ф1.2, Ф2.1, Ф2.2, Ф3, Ф4.2-4.4 высотой 18 м и более: две продольные стороны
    park_link_tol_m: float = 0.3              # зазор между пятном парковки и проездом, при превышении которого достраивается съезд, м
    park_link_max_m: float = 12.0             # ASSUMPTION: максимальная длина достраиваемого съезда к пятну, м
    fire_all_sides_area_m2: float = 10000.0  # п. 8.1.1: площадь застройки более 10 000 м2 — со всех сторон
    fire_all_sides_width_m: float = 100.0     # п. 8.1.1: ширина более 100 м — со всех сторон (ширина = меньшая сторона, допущение)
    corner_radius_main_m: float = 50.0        # НОРМА: наименьший радиус кривых в плане основного проезда, СП 42.13330.2026 табл. 8.9 (0 — без закруглений)
    corner_radius_secondary_m: float = 25.0   # НОРМА: то же для второстепенного проезда, табл. 8.9
    access_corner_radius_m: float = 6.0       # подъезд вдоль стен здания (п. 8.55 СП 42: допускается без закругления; СП 396 табл. 5.10 тип 3: R=6 м), радиус ограничен отступом от стены
    kerb_radius_m: float = 6.0                # ASSUMPTION: радиус закругления кромки проезжей части на примыканиях и углов разворотной площадки, м (СП 396.1325800 не получен; 0 — без закруглений)
    smooth_tol_m: float = 2.0                 # ASSUMPTION: допуск спрямления ступенчатой растровой линии, м (0 — не спрямлять)
    road_style: str = "minimal"               # minimal | loop: loop закрывает каждый тупик вторым выходом на улицу
    patch_reach: float = 12.0                 # ASSUMPTION: пятно парковки ближе этого расстояния к проезду отдельного заезда не получает
    edu_two_accesses: bool = True             # СП 42 п. 6.4.49 (условное требование; применяется всегда)


def smooth_polyline(pts, radius: float, tol: float, stats: list | None = None):
    """Спрямляет растровую «лесенку» (Дуглас-Пойкер, допуск tol) и скругляет повороты дугой радиуса radius.
    Радиус в углу уменьшается, если соседние отрезки короче 2·радиус (дуги не заходят друг на друга)."""
    ln = LineString(pts)
    if tol and tol > 0 and len(pts) > 2:
        ln = ln.simplify(tol, preserve_topology=False)
    c = [tuple(x) for x in ln.coords]
    if not radius or radius <= 0 or len(c) < 3:
        return c
    out = [c[0]]
    for k in range(1, len(c) - 1):
        a, b, d = np.array(c[k - 1]), np.array(c[k]), np.array(c[k + 1])
        v1, v2 = b - a, d - b
        l1, l2 = float(np.hypot(*v1)), float(np.hypot(*v2))
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        u1, u2 = v1 / l1, v2 / l2
        cosp = float(np.clip(u1 @ u2, -1.0, 1.0))
        phi = float(np.arccos(cosp))                      # угол поворота
        if phi < np.radians(5.0) or phi > np.radians(175.0):
            out.append(tuple(b))
            continue
        half1 = l1 / 2.0 if k > 1 else l1                 # на соседний угол отдаём не более половины отрезка
        half2 = l2 / 2.0 if k < len(c) - 2 else l2
        r = min(radius, min(half1, half2) / np.tan(phi / 2.0))
        if stats is not None:
            stats.append((float(radius), float(r)))
        t = r * np.tan(phi / 2.0)
        p1, p2 = b - u1 * t, b + u2 * t
        sgn = 1.0 if (u1[0] * u2[1] - u1[1] * u2[0]) > 0 else -1.0
        cen = p1 + sgn * np.array([-u1[1], u1[0]]) * r
        a0 = np.arctan2(*(p1 - cen)[::-1])
        a1 = np.arctan2(*(p2 - cen)[::-1])
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        n = max(3, int(abs(da) / np.radians(10)))
        for j in range(n + 1):
            ang = a0 + da * j / n
            out.append((float(cen[0] + r * np.cos(ang)), float(cen[1] + r * np.sin(ang))))
    out.append(c[-1])
    return out


def join_perpendicular(coords, ring, reach: float = 6.0, near: float = 3.0):
    """Подводящий проезд входит в кольцо подъезда под прямым углом: если конец линии лежит на кольце, хвост, идущий
    ближе reach к кольцу (вдоль него под малым углом, «двойная полоса»), заменяется прямым отрезком по нормали к кольцу."""
    from shapely.ops import substring
    ln = LineString(coords)
    out = ln
    for rev in (False, True):
        cur = LineString(list(out.coords)[::-1]) if rev else out
        end = Point(cur.coords[-1])
        if ring.distance(end) > near:
            continue
        L = cur.length
        s_ = 0.0
        cut = None
        while s_ < L:
            if ring.distance(cur.interpolate(L - s_)) > reach:
                cut = L - s_
                break
            s_ += 0.5
        if cut is None or cut < 1.0:
            continue
        head = substring(cur, 0.0, cut)
        pt = head.coords[-1]
        foot = nearest_points(ring, Point(pt))[0]
        new = list(head.coords) + [(foot.x, foot.y)]
        out = LineString(new[::-1] if rev else new)
    return list(out.coords)


def drop_parallel(coords, ring, near: float = 12.0, min_len: float = 15.0, step: float = 0.5):
    """Участки линии, идущие вдоль кольца подъезда ближе near на протяжении не менее min_len (лишняя «вторая полоса»),
    убираются; оставшиеся части выходят на кольцо коротким отрезком по нормали."""
    from shapely.ops import substring
    ln = LineString(coords)
    L = ln.length
    if L < min_len or ring is None or ring.is_empty:
        return [list(coords)]
    n_s = int(L / step) + 1
    ss = [min(k * step, L) for k in range(n_s + 1)]
    inside = [ring.distance(ln.interpolate(x)) < near for x in ss]
    cut, k = [], 0
    while k < len(ss):
        if inside[k]:
            k0 = k
            while k < len(ss) and inside[k]:
                k += 1
            if ss[k - 1] - ss[k0] >= min_len:
                cut.append((ss[k0], ss[k - 1]))
        else:
            k += 1
    ok_cut = []
    for a, b in cut:                                    # режем только то, что не рвёт связность: у конца линии, лежащего вне кольца, участок не удаляется
        if a <= 1e-6 and ring.distance(Point(ln.coords[0])) > 3.0:
            continue
        if b >= L - 1e-6 and ring.distance(Point(ln.coords[-1])) > 3.0:
            continue
        if a <= 1e-6 and b >= L - 1e-6:
            continue
        ok_cut.append((a, b))
    cut = ok_cut
    if not cut:
        return [list(coords)]
    pieces, pos = [], 0.0
    for a, b in cut + [(L, L)]:
        if a - pos > 2.0:
            seg = substring(ln, pos, a)
            c_ = list(seg.coords)
            if pos > 0.0:                                   # начало куска примыкает к удалённому участку: выход на кольцо
                fp = nearest_points(ring, Point(c_[0]))[0]
                c_ = [(fp.x, fp.y)] + c_
            if a < L:
                fp = nearest_points(ring, Point(c_[-1]))[0]
                c_ = c_ + [(fp.x, fp.y)]
            pieces.append(c_)
        pos = b
    return pieces or []


def fire_width_for(height_m: float, p: RoadParams) -> float:
    for lim, w in p.fire_width:
        if height_m <= lim:
            return w
    return p.fire_width[-1][1]


def _gap_mid(height_m: float, p: RoadParams) -> float:
    lo, hi = p.wall_gap_high if height_m > p.high_building_m else p.wall_gap_low
    return (lo + hi) / 2.0


def _target_slack(height_m: float, fw_m: float, p: RoadParams) -> float:
    """Продление полосы стороны за её концы, чтобы проезды соседних сторон здания смыкались в кольцо."""
    return _gap_mid(height_m, p) + fw_m / 2.0 + p.cell


def fire_class(service, p: RoadParams) -> str:
    """Класс функциональной пожарной опасности по названию сервиса (123-ФЗ ст. 32); без сервиса — Ф1.3 (многоквартирный дом)."""
    nm = (service or "").lower()
    if not nm:
        return "F1.3"
    for key, cls in p.fire_class_by_service:
        if key in nm:
            return cls
    return "F3"            # ASSUMPTION: прочее общественное здание — Ф3 (порог двух сторон 18 м)


def fire_sides_required(height_m: float, poly, service, p: RoadParams) -> int:
    """Число сторон, с которых нужен подъезд пожарных автомобилей по всей длине (СП 4.13130.2013 п. 8.1.1): 1, 2 или 4."""
    cls = fire_class(service, p)
    if cls in ("F1.1", "F4.1"):
        return 4
    if poly.area > p.fire_all_sides_area_m2 or _short_side(poly) > p.fire_all_sides_width_m:
        return 4
    thr = p.fire_two_sides_f13_m if cls == "F1.3" else p.fire_two_sides_public_m
    return 2 if height_m >= thr else 1


def _short_side(poly) -> float:
    xs, ys = poly.minimum_rotated_rectangle.exterior.coords.xy
    e = [hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2)]
    return min(e)


def dominant_angle(block, buildings) -> float:
    """Преобладающее направление (градусы, по модулю 90) длинных сторон зданий; без зданий — по кварталу."""
    sx = sy = 0.0
    for g in (buildings or [block]):
        xs, ys = g.minimum_rotated_rectangle.exterior.coords.xy
        e = [(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2)]
        dx, dy = max(e, key=lambda t: hypot(*t))
        a = atan2(dy, dx)
        w = g.area
        sx += w * cos(4 * a)
        sy += w * sin(4 * a)
    return degrees(atan2(sy, sx) / 4.0) if (sx or sy) else 0.0


def _transitions(idx, nn):
    """Рёбра графа состояний (ячейка, направление): прямо или поворот на 90°, разворот запрещён."""
    ny, nx = idx.shape
    src, dst, cs, cd, turn = [], [], [], [], []
    for d2, (dx, dy) in enumerate(_DIRS):
        ys0, ys1 = (slice(0, ny - dy), slice(dy, ny)) if dy >= 0 else (slice(-dy, ny), slice(0, ny + dy))
        xs0, xs1 = (slice(0, nx - dx), slice(dx, nx)) if dx >= 0 else (slice(-dx, nx), slice(0, nx + dx))
        a, b = idx[ys0, xs0], idx[ys1, xs1]
        m = (a >= 0) & (b >= 0)
        a, b = a[m], b[m]
        for d in ((d2) % 4, (d2 + 1) % 4, (d2 + 3) % 4):
            src.append(a * 4 + d)
            dst.append(b * 4 + d2)
            cs.append(a)
            cd.append(b)
            turn.append(np.full(len(a), 0 if d == d2 else 1))
    return (np.concatenate(src), np.concatenate(dst), np.concatenate(cs), np.concatenate(cd), np.concatenate(turn))


def generate_roads(block, buildings: list, floors: list, services: list | None = None,
                   p: RoadParams | None = None, patches: list | None = None) -> dict:
    p = p or RoadParams()
    n = len(buildings)
    diag = {"buildings": n, "unreachable": [], "dead_end_violations": [], "road_length_m": 0.0,
            "max_dead_end_m": 0.0, "turnaround_violations": [], "through_links_added": 0, "turnarounds": 0, "junctions": 0, "dead_ends": 0,
            "grid_angle_deg": 0.0, "assumptions": [
                "floor_height 3.0 m", "wall gap = middle of range", "turn penalty 20 m, junction penalty 10 m",
                "rectilinear raster network oriented to dominant building direction",
                "second street link added for dead ends longer than 150 m",
                "SP 4.13130.2013 (p. 8.1.1, 8.1.4, 8.1.6, 8.1.11) and SP 42 (p. 8.49, 8.50): text checked; methodical-guideline values (loop scheme) from a secondary source"]}
    if n == 0 or block is None or block.is_empty:
        return {"carriageways": [], "turnarounds": [], "diagnostics": diag}
    floors = [float(f) if f == f and f else 1.0 for f in floors]
    heights = [f * p.floor_height for f in floors]
    fw = [fire_width_for(h, p) for h in heights]
    patches_in = [g for g in (patches or []) if g is not None and not g.is_empty]

    # поворот в систему координат зданий: сетка ортогональна преобладающему направлению
    theta = dominant_angle(block, buildings)
    origin = block.centroid
    diag["grid_angle_deg"] = round(theta, 1)
    R = lambda g: rotate(g, -theta, origin=origin)
    Rinv = lambda g: rotate(g, theta, origin=origin)
    block_r = R(block)
    bld_r = [R(g) for g in buildings]
    pat_r = [R(g) for g in patches_in]
    bu = unary_union(bld_r)
    pu = unary_union(pat_r) if pat_r else None

    clearance = p.wall_gap_low[0] + max(w for _, w in p.fire_width) / 2.0
    minx, miny, maxx, maxy = block_r.bounds
    xs = np.arange(minx + p.cell / 2, maxx, p.cell)
    ys = np.arange(miny + p.cell / 2, maxy, p.cell)
    gx, gy = np.meshgrid(xs, ys)
    ny, nx = gx.shape
    half_w = max(w for _, w in p.fire_width) / 2.0 * 2 ** 0.5   # запас на выступ внешнего угла проезда при повороте на 90°
    tall = [g for g, h in zip(bld_r, heights) if h > p.high_building_m]
    low = [g for g, h in zip(bld_r, heights) if h <= p.high_building_m]
    walls = [g.buffer(p.wall_gap_low[0] + half_w) for g in low] + [g.buffer(p.wall_gap_high[0] + half_w) for g in tall]
    allowed_poly = block_r.difference(unary_union(walls))   # край проезда не ближе минимума для высоты здания (5 м, выше 28 м: 8 м)
    if pu is not None:
        allowed_poly = allowed_poly.difference(pu.buffer(p.cell / 2.0))
    allowed = contains_xy(allowed_poly, gx.ravel(), gy.ravel()).reshape(ny, nx)
    nn = int(allowed.sum())
    if nn == 0:
        diag["unreachable"] = list(range(n))
        return {"carriageways": [], "turnarounds": [], "diagnostics": diag}
    idx = -np.ones((ny, nx), dtype=np.int64)
    idx[allowed] = np.arange(nn)
    cell_xy = np.stack([gx[allowed], gy[allowed]], axis=1)
    pts_a = shapely.points(cell_xy[:, 0], cell_xy[:, 1])
    db_all = shapely.distance(block_r.boundary, pts_a)
    street_mask = db_all <= p.cell * 1.5
    street_nodes = np.where(street_mask)[0]
    if len(street_nodes) == 0:
        diag["unreachable"] = list(range(n))
        return {"carriageways": [], "turnarounds": [], "diagnostics": diag}
    street_set = set(street_nodes.tolist())

    # точки доступа
    dist_to_b = [shapely.distance(b, pts_a) for b in bld_r]
    accesses, widths_need = [], []
    for i in range(n):
        target = _gap_mid(heights[i], p) + fw[i] / 2.0
        cand = np.where(np.abs(dist_to_b[i] - target) <= p.cell * 1.2)[0]
        if len(cand) == 0:
            cand = np.where(dist_to_b[i] <= target + 3 * p.cell)[0]
        inner = cand[~street_mask[cand]]          # доступ не в полосе улицы: проезд у каждого дома нарисован
        if len(inner) > 0:
            cand = inner
        if len(cand) == 0:
            accesses.append([])
            continue
        a1 = int(cand[np.argmin(db_all[cand])])
        acc = [a1]
        nm = (services[i] or "").lower() if services else ""
        if (p.edu_two_accesses and any(s in nm for s in EDU_SERVICES)) and len(cand) > 1:
            far = np.hypot(cell_xy[cand, 0] - cell_xy[a1, 0], cell_xy[cand, 1] - cell_xy[a1, 1])
            acc.append(int(cand[np.argmax(far)]))
        accesses.append(acc)
    # парковочные пятна: подводится второстепенный проезд к краю пятна
    for k, pg in enumerate(pat_r):
        dp = shapely.distance(pg, pts_a)
        cand = np.where((dp >= p.cell / 2.0) & (dp <= p.cell * 2.2))[0]
        cand = cand[~street_mask[cand]] if len(cand) else cand
        fw.append(p.secondary_width)
        accesses.append([int(cand[np.argmin(db_all[cand])])] if len(cand) else [])

    # граф состояний (ячейка, направление)
    t_src, t_dst, t_cs, t_cd, t_turn = _transitions(idx, nn)
    step_len = np.full(len(t_src), p.cell)
    on_road = np.zeros(nn, dtype=bool)
    ns = nn * 4

    def solve(starts, targets_cells, extra_pen=None):
        w = step_len * np.where(on_road[t_cs] | on_road[t_cd], p.reuse_cost, 1.0)
        w = w + p.turn_penalty * t_turn + p.junction_penalty * (on_road[t_cd] & ~on_road[t_cs])
        if extra_pen is not None:
            w = w * np.where(extra_pen[t_cs] | extra_pen[t_cd], 40.0, 1.0)
        g = csr_matrix((w + 1e-6, (t_src, t_dst)), shape=(ns, ns))
        st = np.concatenate([[c * 4 + d for d in range(4)] for c in starts])
        d, pred, _src = dijkstra(g, directed=True, indices=st, min_only=True, return_predecessors=True)
        tg_states = np.concatenate([[c * 4 + k for k in range(4)] for c in targets_cells])
        dt = d[tg_states]
        if not np.isfinite(dt).any():
            return None
        s_end = int(tg_states[np.argmin(dt)])
        path_states = [s_end]
        while pred[path_states[-1]] >= 0:
            path_states.append(int(pred[path_states[-1]]))
        cells = []
        for s in path_states[::-1]:
            c = s // 4
            if not cells or cells[-1] != c:
                cells.append(c)
        return cells

    # подъезд пожарных автомобилей вдоль стен по всей длине сторон (СП 4.13130.2013 п. 8.1.1)
    fire_paths = [[] for _ in range(n)]
    nudge_map = {}
    fire_lines = []                              # (здание, точки идеального смещённого контура) — для рисования подъезда
    diag["fire_sides"] = []
    if p.fire_sides_enabled:
        from scipy.sparse.csgraph import connected_components

        from scipy.spatial import cKDTree
        tree = cKDTree(cell_xy)

        def ring_d(i):
            """Расстояние оси подъезда от стены: середина нормативного диапазона, но не ближе границы запретной зоны
            (она включает запас на угол проезда), иначе вдоль стены нет ни одной разрешённой ячейки."""
            lo_ = (p.wall_gap_high[0] if heights[i] > p.high_building_m else p.wall_gap_low[0]) + half_w + p.cell * 0.5
            return max(_gap_mid(heights[i], p) + fw[i] / 2.0, lo_)

        def ring_samples(i):
            """Точки контура здания, смещённого на расстояние подъезда (внешний контур), и ближайшие разрешённые ячейки."""
            target = ring_d(i)
            if p.smooth_tol_m and p.smooth_tol_m > 0:     # зубчатый контур здания не переносим на проезд: берём слегка раздутый, спрямлённый контур
                base = bld_r[i].buffer(p.cell, join_style=2).simplify(p.smooth_tol_m * 0.75)
                ring = base.buffer(max(target - p.cell, 0.0), join_style=2).exterior
            else:
                ring = bld_r[i].buffer(target, join_style=2).exterior
            n_s = max(8, int(ring.length / (p.cell * 0.5)))
            pts_ = np.array([ring.interpolate(k * ring.length / n_s).coords[0] for k in range(n_s)])
            dd, ii = tree.query(pts_)
            cells_ = np.where(dd <= p.cell * 1.0, ii, -1)
            return pts_, cells_, ring.length / n_s

        def bridge(a_c, b_c, mask):
            """Путь между ячейками внутри маски (полосы вдоль здания); None, если не связаны."""
            keep = mask[t_cs] & mask[t_cd]
            if not keep.any():
                return None
            w = step_len[keep] + p.turn_penalty * t_turn[keep]
            g = csr_matrix((w + 1e-6, (t_src[keep], t_dst[keep])), shape=(ns, ns))
            d_, pred_, _s = dijkstra(g, directed=True, indices=[a_c * 4 + k for k in range(4)], min_only=True,
                                     return_predecessors=True)
            tg_states = np.array([b_c * 4 + k for k in range(4)])
            dt = d_[tg_states]
            if not np.isfinite(dt).any():
                return None
            ps = [int(tg_states[np.argmin(dt)])]
            while pred_[ps[-1]] >= 0:
                ps.append(int(pred_[ps[-1]]))
            out_ = []
            for st_ in ps[::-1]:
                c_ = st_ // 4
                if not out_ or out_[-1] != c_:
                    out_.append(c_)
            return out_

        def side_path(i, edge, slack, samples):
            """Цепочка ячеек вдоль одной стороны здания i по смещённому контуру; (ячейки, охват 0..1)."""
            p0, u, length, nrm = edge
            pts_, cells_, step_ = samples
            rel = pts_ - p0
            t = rel @ u
            sd = rel @ nrm
            sel_ = np.where((sd > 0) & (t >= -slack) & (t <= length + slack))[0]
            if len(sel_) < 2:
                return None
            target = ring_d(i)
            band = np.abs(dist_to_b[i] - target) <= p.cell * 2.0
            seq, found, nz, runs, cur = [], 0, {}, [], []
            prev_k = None
            for k in sel_:
                if prev_k is not None and k != prev_k + 1 and len(cur) >= 2:    # пропуск точек контура: не соединяем напрямую
                    runs.append(cur)
                    cur = []
                elif prev_k is not None and k != prev_k + 1:
                    cur = []
                prev_k = k
                c_ = int(cells_[k])
                if c_ < 0:
                    if len(cur) >= 2:
                        runs.append(cur)
                    cur = []
                    continue
                cur.append(tuple(pts_[k]))
                found += 1
                nz.setdefault(c_, []).append(pts_[k])
                if not seq or seq[-1] != c_:
                    seq.append(c_)
            if len(cur) >= 2:
                runs.append(cur)
            if len(seq) < 2:
                return None
            chain = [seq[0]]
            for c_ in seq[1:]:
                a_c = chain[-1]
                if np.hypot(*(cell_xy[a_c] - cell_xy[c_])) <= p.cell * 1.01:
                    chain.append(c_)
                    continue
                br = bridge(a_c, c_, band)
                if br is None:
                    return None if len(chain) < 2 else (chain, round(found / len(sel_), 2), nz, runs)   # разрыв: оставляем пройденное
                chain.extend(br[1:])
            return chain, round(found / len(sel_), 2), nz, runs

        for i in range(n):
            if not accesses[i]:
                continue
            need = fire_sides_required(heights[i], bld_r[i], (services[i] if services else None), p)
            mrr = bld_r[i].minimum_rotated_rectangle
            xs_, ys_ = mrr.exterior.coords.xy
            pts = list(zip(xs_[:4], ys_[:4]))
            cen = np.array(bld_r[i].centroid.coords[0])
            edges = []
            for k in range(4):
                a_, b_ = np.array(pts[k]), np.array(pts[(k + 1) % 4])
                vec = b_ - a_
                ln = float(np.hypot(*vec))
                if ln < 1e-6:
                    continue
                u_ = vec / ln
                nr = np.array([u_[1], -u_[0]])
                if (((a_ + b_) / 2.0) - cen) @ nr < 0:
                    nr = -nr
                edges.append((a_, u_, ln, nr))
            edges.sort(key=lambda e: -e[2])
            slack_long = p.cell
            if need == 4:
                chosen = [(e, _target_slack(heights[i], fw[i], p)) for e in edges]
            elif need == 2:
                chosen = [(e, slack_long) for e in edges[:2]]
            else:
                chosen = None
            built, cover = [], []
            smp = ring_samples(i)
            if chosen is None:                       # одна продольная сторона: берём ту, где подъезд охватывает больше длины
                best_one = None
                for e in edges[:2]:
                    r_ = side_path(i, e, slack_long, smp)
                    if r_ is not None and (best_one is None or r_[1] > best_one[1]):
                        best_one = r_
                if best_one is not None:
                    built.append(best_one[0])
                    cover.append(best_one[1])
                    for c_, l_ in best_one[2].items():
                        nudge_map.setdefault(c_, []).extend(l_)
                    fire_lines.extend((i, rr) for rr in best_one[3])
            else:
                for e, sl in chosen:
                    r_ = side_path(i, e, sl, smp)
                    if r_ is not None:
                        built.append(r_[0])
                        cover.append(r_[1])
                        for c_, l_ in r_[2].items():
                            nudge_map.setdefault(c_, []).extend(l_)
                        fire_lines.extend((i, rr) for rr in r_[3])
            fire_paths[i] = built
            rec_ = {"building": i, "required": need, "built": len(built), "coverage": cover}
            if len(built) < need or any(c_ < 0.8 for c_ in cover):
                # причина неполного подъезда: соседнее здание ближе, чем нужно для проезда с отступами с двух сторон
                lim_ = ring_d(i) + (p.wall_gap_high[0] if heights[i] > p.high_building_m else p.wall_gap_low[0]) + half_w + 3.0
                near_ = sorted(((float(bld_r[i].distance(bld_r[j])), j) for j in range(n) if j != i))
                rec_["blocked_by"] = [{"building": j, "distance_m": round(d_, 1)} for d_, j in near_ if d_ < lim_][:3]
            diag["fire_sides"].append(rec_)

    bkey = lambda i: min([db_all[a] for a in accesses[i]] or [1e9])
    order = sorted(range(n), key=bkey)
    patch_order = sorted(range(n, len(accesses)), key=bkey)
    paths, served = {}, {}
    for i in order + patch_order:
        if i < n and fire_paths[i]:
            for k, sp in enumerate(fire_paths[i]):
                sp_arr = np.array(sp, dtype=np.int64)
                if not on_road[sp_arr].any():                      # сторона ещё не связана с сетью: проводим соединение
                    own = np.zeros(nn, dtype=bool)
                    own[sp_arr] = True
                    tg = np.where((on_road | street_mask) & ~own)[0]
                    conn = solve(list(sp), tg) if len(tg) else None
                    if conn is None:
                        diag["unreachable"].append(i)
                        continue
                    paths[(i, "c%d" % k)] = conn
                    for v in conn:
                        on_road[v] = True
                        served.setdefault(v, set()).add(i)
                paths[(i, "s%d" % k)] = list(sp)
                for v in sp:
                    on_road[v] = True
                    served.setdefault(v, set()).add(i)
            continue
        if i >= n and accesses[i] and on_road.any():
            # пятно парковки, уже находящееся рядом с проездом, отдельного заезда не получает
            dpr = float(np.min(np.hypot(cell_xy[on_road, 0] - cell_xy[accesses[i][0], 0],
                                        cell_xy[on_road, 1] - cell_xy[accesses[i][0], 1])))
            if dpr <= p.patch_reach:
                continue
        for k, a in enumerate(accesses[i]):
            tg = np.where(on_road | street_mask)[0]
            cells = solve([a], tg)
            if cells is None or len(cells) < 1:
                diag["unreachable"].append(i)
                continue
            paths[(i, k)] = cells
            for v in cells:
                on_road[v] = True
                served.setdefault(v, set()).add(i)

    seg_len = lambda u, v: float(np.hypot(*(cell_xy[u] - cell_xy[v])))

    def adjacency(all_paths):
        adj = {}
        for pth in all_paths:
            for u, v in zip(pth[:-1], pth[1:]):
                adj.setdefault(u, {})[v] = seg_len(u, v)
                adj.setdefault(v, {})[u] = seg_len(u, v)
        for v in list(adj):
            if v in street_set:
                adj.setdefault(-1, {})[v] = 0.0
                adj[v][-1] = 0.0
        return adj

    def dead_ends(adj):
        deg = {v: len(nb) for v, nb in adj.items()}
        removed, parent = set(), {}
        stack = [v for v, d in deg.items() if d == 1 and v != -1]      # улица (-1) — корень, её не «срезаем»: иначе тупик считается с конца улицы и длина равна 0
        leaves0 = list(stack)
        while stack:
            v = stack.pop()
            if v in removed:
                continue
            nb = [u for u in adj[v] if u not in removed]
            if len(nb) != 1:
                continue
            u = nb[0]
            removed.add(v)
            parent[v] = (u, adj[v][u])
            deg[u] -= 1
            if deg[u] == 1 and u != -1:          # улицу (-1) не «срезаем» ни с какой стороны
                stack.append(u)
        out = []
        for lf in leaves0:
            chain, total, cur = [lf], 0.0, lf
            while cur in parent:
                u, w_ = parent[cur]
                total += w_
                chain.append(u)
                cur = u
            out.append((lf, total, chain))
        return out

    # К6: тупики длиннее предела закрываем вторым выходом на улицу
    extra = []
    for _round in range(6):
        all_paths = list(paths.values()) + extra
        limit = 0.0 if p.road_style == "loop" else p.dead_end_max
        def _patch_only(ch):
            who_ = set().union(*[served.get(v, set()) for v in ch]) if ch else set()
            return bool(who_) and min(who_) >= n and p.road_style == "loop"
        bad = [x for x in dead_ends(adjacency(all_paths)) if x[1] > max(limit, p.stub_m) and not _patch_only(x[2])] if p.close_dead_ends else []
        if not bad:
            break
        bad.sort(key=lambda x: -x[1])
        changed = False
        for lf, total, chain in bad[:12]:
            acc_len, mid = 0.0, chain[0]
            for a_, b_ in zip(chain[:-1], chain[1:]):
                acc_len += seg_len(a_, b_)
                if acc_len >= total / 2 and p.road_style != "loop":
                    mid = b_            # minimal: выход от середины тупика; loop: от самого конца, чтобы тупика не осталось
                    break
            pen = np.zeros(nn, dtype=bool)
            pen[np.array(chain, dtype=np.int64)] = True
            end_street = chain[-1] if chain[-1] in street_set else None
            tg = [t for t in street_nodes.tolist()
                  if end_street is None or np.hypot(*(cell_xy[t] - cell_xy[end_street])) > 40.0]
            if not tg:
                continue
            cells = solve([mid], np.array(tg, dtype=np.int64), extra_pen=pen)
            if cells is None or len(cells) < 2:
                continue
            extra.append(cells)
            who = set(served.get(mid, set())) or set(served.get(lf, set()))
            for v in cells:
                on_road[v] = True
                served.setdefault(v, set()).update(who)
            changed = True
        if not changed:
            break
    all_paths = list(paths.values()) + extra
    fire_keys = {k for k in paths if k[0] < n and isinstance(k[1], str) and k[1].startswith("s")}
    skip_draw = {id(paths[k]) for k in fire_keys} if (p.fire_sides_enabled and fire_lines) else set()

    def node_width(v):
        s = served.get(v, set())
        cls = p.main_width if len(s) >= 2 else p.secondary_width
        return max([cls] + [fw[i] for i in s])

    draw_xy = cell_xy.copy()
    if p.fire_sides_enabled:
        for c_, pl_ in nudge_map.items():
            draw_xy[c_] = np.mean(pl_, axis=0)
    # полилинии одинаковой ширины -> полигоны (прямые углы: квадратные торцы, митра)
    lines_by_w, seen, total_len = {}, set(), 0.0
    curve_stats: list = []
    walls_u = unary_union(walls) if walls else None

    def _radius_for(w_):
        return p.corner_radius_main_m if w_ >= p.main_width else p.corner_radius_secondary_m

    fire_smooth = [(i_, smooth_polyline(pts_, p.access_corner_radius_m, p.smooth_tol_m)) for i_, pts_ in fire_lines] if skip_draw else []
    ring_all = unary_union([LineString(c_) for _, c_ in fire_smooth]) if fire_smooth else None

    def _emit(run_, w_):
        # радиус по СП 42 табл. 8.9; если дуга заходит в нормативный отступ от стен (СП 4.13130 п. 8.1.6), радиус уменьшается
        base_ = LineString(run_)
        viol0_ = base_.intersection(walls_u).length if walls_u is not None else 0.0
        cc, st_ = None, []
        r_req = _radius_for(w_)
        for r_try in dict.fromkeys([r_req, r_req * 0.7, r_req * 0.5, 15.0, 10.0, p.access_corner_radius_m, 0.0]):
            if r_try > r_req:
                continue
            st_ = []
            cc = smooth_polyline(run_, r_try, p.smooth_tol_m, st_)
            if walls_u is None or LineString(cc).intersection(walls_u).length <= viol0_ + 0.5:
                break
        curve_stats.extend((r_req, ra) for _, ra in st_)
        parts_ = [cc]
        if ring_all is not None and not ring_all.is_empty and p.fire_sides_enabled:
            parts_ = [join_perpendicular(x, ring_all) for x in drop_parallel(cc, ring_all)]
        for x in parts_:
            if len(x) >= 2:
                lines_by_w.setdefault(w_, []).append(LineString(x))

    for path in all_paths:
        run, run_w = [], None
        drawn = id(path) not in skip_draw
        for u, v in zip(path[:-1], path[1:]):
            w = max(node_width(u), node_width(v))
            e = (min(u, v), max(u, v))
            if e not in seen:
                seen.add(e)
                total_len += seg_len(u, v)
            if not drawn:
                continue
            if run_w is not None and w != run_w:
                _emit(run, run_w)
                run = [run[-1]]
            if not run:
                run = [tuple(draw_xy[u])]
            run.append(tuple(draw_xy[v]))
            run_w = w
        if len(run) >= 2:
            _emit(run, run_w)
    for i_, cc_ in fire_smooth:                       # подъезд вдоль стен рисуем по ровному смещённому контуру, а не по ячейкам растра
        lines_by_w.setdefault(fw[i_], []).append(LineString(cc_))
    carriageways = []
    road_polys = []
    for w, segs in sorted(lines_by_w.items()):
        poly = unary_union([sg.buffer(w / 2.0, cap_style=3, join_style=2) for sg in segs]).difference(bu)
        if pu is not None:
            poly = poly.difference(pu)
        road_polys.append(poly)
        carriageways.append({"width": w, "class": "main" if w >= p.main_width else "secondary", "geometry": Rinv(poly)})
    diag["road_length_m"] = round(total_len, 1)
    if curve_stats:
        short = [(rq, ra) for rq, ra in curve_stats if ra < rq - 0.5]
        diag["curves"] = len(curve_stats)
        diag["curves_below_norm"] = len(short)
        diag["curve_radius_min_m"] = round(min(ra for _, ra in curve_stats), 1)

    # разворотные площадки в тупиках: накрывают конец тупика, не ближе 5 м к стенам, без стоянок
    final_adj = adjacency(all_paths)
    final = dead_ends(final_adj)
    h = p.turnaround / 2.0
    cands = sorted(((dx, dy) for dx in np.arange(-h, h + 1e-6, p.cell) for dy in np.arange(-h, h + 1e-6, p.cell)),
                   key=lambda t: hypot(*t))
    reach = max(p.turnaround_reach, h)
    cands_far = sorted(((dx, dy) for dx in np.arange(-reach, reach + 1e-6, p.cell) for dy in np.arange(-reach, reach + 1e-6, p.cell)),
                       key=lambda t: hypot(*t))
    road_u = unary_union(road_polys) if road_polys else Point(0, 0).buffer(0)
    turn, turn_r = [], []
    for lf, total, chain in final:
        x, y = cell_xy[lf]
        who_ = set().union(*[served.get(v, set()) for v in chain]) if chain else set()
        if total <= p.stub_m:
            continue          # короткий отросток у сквозного проезда: разворот выполняется на самом проезде
        if who_ and min(who_) >= n and total <= 50.0:
            continue          # ASSUMPTION: короткий заезд только к парковке (не пожарный проезд) без разворотной площадки
        def _fits(c_):
            if bu.distance(c_) < p.wall_gap_low[0] - 1e-6:
                return False
            if pu is not None and pu.intersection(c_).area > 1e-6:
                return False
            return block_r.contains(c_)

        sq = None
        for dx, dy in cands:                       # 1) площадка накрывает конец тупика
            c_ = box(x + dx - h, y + dy - h, x + dx + h, y + dy + h)
            if _fits(c_):
                sq = c_
                break
        if sq is None:                             # 2) площадка примыкает к проезду недалеко от конца тупика
            near = road_u.intersection(Point(x, y).buffer(p.turnaround_reach))
            for dx, dy in cands_far:
                c_ = box(x + dx - h, y + dy - h, x + dx + h, y + dy + h)
                if _fits(c_) and c_.intersects(near):
                    sq = c_
                    break
        if sq is None:
            sq = box(x - h, y - h, x + h, y + h).difference(bu)
            if pu is not None:
                sq = sq.difference(pu)
            diag["turnaround_violations"].append(
                {"leaf_xy": [round(float(x), 1), round(float(y), 1)], "wall_gap_m": round(float(bu.distance(sq)), 1)})
        if p.kerb_radius_m and p.kerb_radius_m > 0 and not sq.is_empty:      # углы площадки скругляются (выпуклые углы)
            rr = min(p.kerb_radius_m, h / 2.0)
            sq_r = sq.buffer(-rr, join_style=1).buffer(rr, join_style=1)
            if not sq_r.is_empty and sq_r.area > 0.8 * sq.area:
                sq = sq_r.union(sq.intersection(road_u.buffer(1.0)))      # у въезда на площадку угол не срезаем: проезд не должен оторваться
        turn_r.append(sq)
        turn.append(Rinv(sq))
        if total > p.dead_end_max:
            # СП 4.13130 п. 8.1.11: на проезде длиннее 150 м предусматриваются ещё площадки не реже чем через 150 м
            missing = 0
            for k_ in range(1, int(total // p.dead_end_max) + 1):
                if k_ * p.dead_end_max >= total:
                    break
                acc_, cell_ = 0.0, None
                for a_, b_ in zip(chain[:-1], chain[1:]):
                    acc_ += seg_len(a_, b_)
                    if acc_ >= k_ * p.dead_end_max - p.cell:
                        cell_ = b_
                        break
                placed_ = False
                if cell_ is not None:
                    x2, y2 = cell_xy[cell_]
                    near2 = road_u.intersection(Point(x2, y2).buffer(p.turnaround_reach))
                    for dx, dy in cands_far:
                        c_ = box(x2 + dx - h, y2 + dy - h, x2 + dx + h, y2 + dy + h)
                        if _fits(c_) and c_.intersects(near2) and all(c_.distance(t_) > 0 for t_ in turn_r):
                            turn_r.append(c_)
                            turn.append(Rinv(c_))
                            placed_ = True
                            break
                if not placed_:
                    missing += 1
            diag["intermediate_turnarounds"] = diag.get("intermediate_turnarounds", 0) + int(total // p.dead_end_max) - missing
            if missing:
                diag["dead_end_violations"].append({"leaf_xy": [round(float(x), 1), round(float(y), 1)], "dead_end_m": round(total, 1), "missing_intermediate_turnarounds": missing})
    if p.kerb_radius_m and p.kerb_radius_m > 0 and (road_polys or turn_r):
        # закругление кромки на примыканиях: «замыкание» объединённой сети радиусом r; берутся только небольшие
        # заполнения во внутренних углах (не мосты между параллельными проездами) и не внутри зданий и парковок
        r_k = p.kerb_radius_m
        allr = unary_union(list(road_polys) + turn_r)
        closed = allr.buffer(r_k, join_style=1).buffer(-r_k, join_style=1)
        walls_min = unary_union([g.buffer(p.wall_gap_low[0]) for g in low] + [g.buffer(p.wall_gap_high[0]) for g in tall])
        fill = closed.difference(allr).difference(walls_min)      # заполнение не ближе нормативного отступа до стен
        if pu is not None:
            fill = fill.difference(pu)
        pieces = [g for g in (fill.geoms if hasattr(fill, "geoms") else [fill]) if not g.is_empty and 0.5 < g.area <= 1.2 * r_k * r_k and g.distance(allr) < 0.05]   # только примыкающие к сети (без оторванных обрезков)
        diag["kerb_fillets"] = len(pieces)
        for g in pieces:
            k_best = min(range(len(carriageways)), key=lambda k: carriageways[k]['geometry'].distance(Rinv(g))) if carriageways else None
            if k_best is None:
                continue
            carriageways[k_best]["geometry"] = carriageways[k_best]["geometry"].union(Rinv(g))
    # парковочные пятна: подводящий проезд строится до ячейки растра и может не доходить до кромки пятна;
    # достраивается короткий съезд (ширина второстепенного проезда) от ближайшей точки сети до кромки пятна
    diag["parking_links"], diag["parking_unlinked"] = 0, []
    if pat_r and carriageways:
        net_r = unary_union([R(c["geometry"]) for c in carriageways] + turn_r)
        low_u = unary_union(low) if low else None
        tall_u = unary_union(tall) if tall else None
        for k_, pg_ in enumerate(pat_r):
            d_ = pg_.distance(net_r)
            if d_ <= p.park_link_tol_m:
                continue
            if d_ > p.park_link_max_m:
                diag["parking_unlinked"].append({"patch": k_, "gap_m": round(float(d_), 1)})
                continue
            # кандидаты: точки кромки пятна с шагом 2,5 м -> ближайшая точка сети; берётся ближайший допустимый съезд
            ring_ = pg_.exterior
            cand_ = []
            for t_ in np.arange(0.0, ring_.length, 2.5):
                a_ = ring_.interpolate(float(t_))
                b_ = nearest_points(net_r, a_)[0]
                dd_ = a_.distance(b_)
                if 0.05 < dd_ <= p.park_link_max_m:
                    cand_.append((dd_, a_, b_))
            cand_.sort(key=lambda t: t[0])
            lk = None
            for dd_, a_, b_ in cand_[:40]:
                ux, uy = (a_.x - b_.x) / dd_, (a_.y - b_.y) / dd_
                lk_ = LineString([(b_.x - ux * 0.5, b_.y - uy * 0.5), (a_.x, a_.y)]).buffer(p.secondary_width / 2.0, cap_style=2).difference(pg_)   # торец съезда не заходит в своё пятно
                lk_n = lk_.difference(net_r)
                if lk_.intersection(bu).area > 1e-6:
                    continue
                if (low_u is not None and lk_n.distance(low_u) < p.wall_gap_low[0] - 1e-6) or \
                        (tall_u is not None and lk_n.distance(tall_u) < p.wall_gap_high[0] - 1e-6):
                    continue          # съезд не ближе нормативного отступа до стены (5 м; выше 28 м — 8 м)
                if pu is not None and lk_.difference(pg_.buffer(0.05)).intersection(pu).area > 1e-6:
                    continue
                lk = lk_
                break
            if lk is None:
                diag["parking_unlinked"].append({"patch": k_, "gap_m": round(float(d_), 1)})
                continue
            carriageways.append({"width": p.secondary_width, "class": "secondary", "geometry": Rinv(lk)})   # перекрывается с сетью на 0,5 м: съезд не отрывается
            net_r = net_r.union(lk)
            diag["parking_links"] += 1
    # оторванные обрезки (остатки после вычитания зданий и парковок) площадью меньше 3 м² убираются: проезд из них не состоит
    all_geoms = [c["geometry"] for c in carriageways] + list(turn)
    for k_, c_ in enumerate(carriageways):
        parts_ = list(c_["geometry"].geoms) if hasattr(c_["geometry"], "geoms") else [c_["geometry"]]
        keep_ = []
        for g_ in parts_:
            if g_.area < 3.0 and not any(o_ is not c_["geometry"] and o_.distance(g_) < 0.05 for o_ in all_geoms) \
                    and not any(o_.distance(g_) < 0.05 for o_ in parts_ if o_ is not g_):
                diag.setdefault("slivers_removed", 0)
                diag["slivers_removed"] += 1
                continue
            keep_.append(g_)
        if len(keep_) != len(parts_):
            c_["geometry"] = unary_union(keep_) if keep_ else c_["geometry"].difference(c_["geometry"])
    diag["max_dead_end_m"] = round(max([t for _, t, _ in final] or [0.0]), 1)
    diag["through_links_added"] = len(extra)
    diag["turnarounds"] = len(turn)
    diag["dead_ends"] = len(final)
    diag["junctions"] = sum(1 for v, nb in final_adj.items() if v != -1 and len([u for u in nb if u != -1]) >= 3)
    diag["unreachable"] = sorted(set(diag["unreachable"]))
    return {"carriageways": carriageways, "turnarounds": turn, "diagnostics": diag}
