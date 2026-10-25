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
     и по высоте здания (СП 4.13130, вторичный источник); школы и детские сады: два проезда с противоположных сторон
     (СП 42 п. 6.4.49); здания шире 18 м: подъезд с двух сторон (МУ «Квартал» ВГАСУ 2022, раздел 11, вторичный источник).
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
    cell: float = 2.5                         # шаг растра, м (ASSUMPTION точности расчёта)
    floor_height: float = 3.0                 # ASSUMPTION: высота этажа, м
    main_width: float = 6.0                   # СП 42 табл. 8.9: 2 полосы x 3,0 м
    secondary_width: float = 3.5              # СП 42 табл. 8.9: 1 полоса x 3,5 м
    fire_width: tuple = ((13.0, 3.5), (46.0, 4.2), (10**9, 6.0))   # СП 4.13130 (вторичный): по высоте здания, м
    wall_gap_low: tuple = (5.0, 8.0)          # внутренний край проезда до стены, здание до 28 м (вторичный источник)
    wall_gap_high: tuple = (8.0, 10.0)        # то же, выше 28 м
    high_building_m: float = 28.0
    dead_end_max: float = 150.0               # СП 42 п. 8.49
    turnaround: float = 15.0                  # СП 4.13130 (вторичный): 15 x 15 м; СП 42 п. 8.50
    reuse_cost: float = 0.15                  # ASSUMPTION: относительная стоимость прохода по уже построенному проезду
    turn_penalty: float = 20.0                # ASSUMPTION: штраф за поворот, м эквивалентной длины (К5)
    junction_penalty: float = 10.0            # ASSUMPTION: штраф за новую развилку, м (К4)
    wide_building_m: float = 18.0             # МУ «Квартал» ВГАСУ (вторичный): шире 18 м — подъезд с двух сторон
    road_style: str = "minimal"               # minimal | loop: loop закрывает каждый тупик вторым выходом на улицу
    patch_reach: float = 12.0                 # ASSUMPTION: пятно парковки ближе этого расстояния к проезду отдельного заезда не получает
    edu_two_accesses: bool = True             # СП 42 п. 6.4.49 (условное требование; применяется всегда)


def fire_width_for(height_m: float, p: RoadParams) -> float:
    for lim, w in p.fire_width:
        if height_m <= lim:
            return w
    return p.fire_width[-1][1]


def _gap_mid(height_m: float, p: RoadParams) -> float:
    lo, hi = p.wall_gap_high if height_m > p.high_building_m else p.wall_gap_low
    return (lo + hi) / 2.0


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
            "max_dead_end_m": 0.0, "through_links_added": 0, "turnarounds": 0, "junctions": 0, "dead_ends": 0,
            "grid_angle_deg": 0.0, "assumptions": [
                "floor_height 3.0 m", "wall gap = middle of range", "turn penalty 20 m, junction penalty 10 m",
                "rectilinear raster network oriented to dominant building direction",
                "second street link added for dead ends longer than 150 m",
                "SP 4.13130 and methodical-guideline values from secondary sources"]}
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
    allowed_poly = block_r.difference(bu.buffer(clearance))
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
        wide = _short_side(bld_r[i]) > p.wide_building_m
        if ((p.edu_two_accesses and any(s in nm for s in EDU_SERVICES)) or wide) and len(cand) > 1:
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

    bkey = lambda i: min([db_all[a] for a in accesses[i]] or [1e9])
    order = sorted(range(n), key=bkey)
    patch_order = sorted(range(n, len(accesses)), key=bkey)
    paths, served = {}, {}
    for i in order + patch_order:
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
        stack = [v for v, d in deg.items() if d == 1]
        leaves0 = [v for v in stack if v != -1]
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
            if deg[u] == 1:
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
        bad = [x for x in dead_ends(adjacency(all_paths)) if x[1] > limit and not _patch_only(x[2])]
        if not bad:
            break
        bad.sort(key=lambda x: -x[1])
        changed = False
        for lf, total, chain in bad[:12]:
            acc_len, mid = 0.0, chain[0]
            for a_, b_ in zip(chain[:-1], chain[1:]):
                acc_len += seg_len(a_, b_)
                if acc_len >= total / 2:
                    mid = b_
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

    def node_width(v):
        s = served.get(v, set())
        cls = p.main_width if len(s) >= 2 else p.secondary_width
        return max([cls] + [fw[i] for i in s])

    # полилинии одинаковой ширины -> полигоны (прямые углы: квадратные торцы, митра)
    lines_by_w, seen, total_len = {}, set(), 0.0
    for path in all_paths:
        run, run_w = [], None
        for u, v in zip(path[:-1], path[1:]):
            w = max(node_width(u), node_width(v))
            e = (min(u, v), max(u, v))
            if e not in seen:
                seen.add(e)
                total_len += seg_len(u, v)
            if run_w is not None and w != run_w:
                lines_by_w.setdefault(run_w, []).append(LineString(run))
                run = [run[-1]]
            if not run:
                run = [tuple(cell_xy[u])]
            run.append(tuple(cell_xy[v]))
            run_w = w
        if len(run) >= 2:
            lines_by_w.setdefault(run_w, []).append(LineString(run))
    carriageways = []
    for w, segs in sorted(lines_by_w.items()):
        poly = unary_union([sg.buffer(w / 2.0, cap_style=3, join_style=2) for sg in segs]).difference(bu)
        if pu is not None:
            poly = poly.difference(pu)
        carriageways.append({"width": w, "class": "main" if w >= p.main_width else "secondary", "geometry": Rinv(poly)})
    diag["road_length_m"] = round(total_len, 1)

    # разворотные площадки в тупиках: накрывают конец тупика, не ближе 5 м к стенам, без стоянок
    final_adj = adjacency(all_paths)
    final = dead_ends(final_adj)
    h = p.turnaround / 2.0
    cands = sorted(((dx, dy) for dx in np.arange(-h, h + 1e-6, p.cell) for dy in np.arange(-h, h + 1e-6, p.cell)),
                   key=lambda t: hypot(*t))
    turn = []
    for lf, total, chain in final:
        x, y = cell_xy[lf]
        who_ = set().union(*[served.get(v, set()) for v in chain]) if chain else set()
        if who_ and min(who_) >= n and total <= 50.0:
            continue          # ASSUMPTION: короткий заезд только к парковке (не пожарный проезд) без разворотной площадки
        sq = None
        for dx, dy in cands:
            c_ = box(x + dx - h, y + dy - h, x + dx + h, y + dy + h)
            if bu.distance(c_) < p.wall_gap_low[0] - 1e-6:
                continue
            if pu is not None and pu.intersection(c_).area > 1e-6:
                continue
            if not block_r.contains(c_):
                continue
            sq = c_
            break
        if sq is None:
            sq = box(x - h, y - h, x + h, y + h).difference(bu)
            if pu is not None:
                sq = sq.difference(pu)
        turn.append(Rinv(sq))
        if total > p.dead_end_max:
            diag["dead_end_violations"].append({"leaf_xy": [round(float(x), 1), round(float(y), 1)], "dead_end_m": round(total, 1)})
    diag["max_dead_end_m"] = round(max([t for _, t, _ in final] or [0.0]), 1)
    diag["through_links_added"] = len(extra)
    diag["turnarounds"] = len(turn)
    diag["dead_ends"] = len(final)
    diag["junctions"] = sum(1 for v, nb in final_adj.items() if v != -1 and len([u for u in nb if u != -1]) >= 3)
    diag["unreachable"] = sorted(set(diag["unreachable"]))
    return {"carriageways": carriageways, "turnarounds": turn, "diagnostics": diag}
