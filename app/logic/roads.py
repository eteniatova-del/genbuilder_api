"""
Модуль внутриквартальных проездов (эскизный расчёт).

Вход: полигон квартала (метры, UTM), полигоны зданий, число этажей зданий (и, при наличии, название сервиса).
Выход: полигоны проезжей части (с шириной и классом), разворотные площадки, диагностика.

Правила (источники и статусы: claude/нормы_проезды_для_генерации.md в проекте):
  СП 42.13330 (текст проверен): табл. 8.9 (основной проезд 2 полосы по 3,0 м, второстепенный 1 полоса 3,5 м),
      п. 8.48 (проезды обеспечивают подъезд к зданиям внутри кварталов), п. 8.49 (тупик не более 150 м),
      п. 8.50 (разворотная площадка в конце тупика, стоянка на ней не допускается),
      п. 6.4.49 (к объектам образования не менее двух проездов с противоположных сторон).
  СП 4.13130.2013 с Изм. 3 (ВТОРИЧНЫЕ источники, сверить): ширина проезда 3,5 / 4,2 / 6,0 м при высоте здания
      до 13 / 13-46 / более 46 м; расстояние от внутреннего края проезда до стены 5-8 м (до 28 м) и 8-10 м (выше 28 м);
      разворотная площадка не менее 15 x 15 м.
ДОПУЩЕНИЯ (не нормы): высота этажа 3,0 м (в проекте не задана); положение внутреннего края проезда внутри
  допустимого диапазона (середина); сеть строится по растру кратчайшими путями от границы квартала
  (граница квартала = оси улиц) с поощрением совместного использования участков; кольцевые связи строятся только как второй выход на улицу для тупиков длиннее 150 м.
"""
from dataclasses import dataclass, field
from math import hypot

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union


EDU_SERVICES = ("школа", "детский сад")


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
    edu_two_accesses: bool = True             # СП 42 п. 6.4.49 (условное требование; применяется всегда)


def fire_width_for(height_m: float, p: RoadParams) -> float:
    for lim, w in p.fire_width:
        if height_m <= lim:
            return w
    return p.fire_width[-1][1]


def _gap_mid(height_m: float, p: RoadParams) -> float:
    lo, hi = p.wall_gap_high if height_m > p.high_building_m else p.wall_gap_low
    return (lo + hi) / 2.0


def _grid(block, p):
    minx, miny, maxx, maxy = block.bounds
    xs = np.arange(minx + p.cell / 2, maxx, p.cell)
    ys = np.arange(miny + p.cell / 2, maxy, p.cell)
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, gx, gy


def generate_roads(block: Polygon, buildings: list, floors: list, services: list | None = None,
                   p: RoadParams | None = None) -> dict:
    p = p or RoadParams()
    n = len(buildings)
    diag = {"buildings": n, "unreachable": [], "dead_end_violations": [], "road_length_m": 0.0,
            "max_dead_end_m": 0.0, "through_links_added": 0, "turnarounds": 0, "assumptions": [
                "floor_height 3.0 m", "wall gap = middle of range", "raster shortest-path network; second street link added for dead ends longer than 150 m",
                "SP 4.13130 values from secondary sources"]}
    if n == 0 or block is None or block.is_empty:
        return {"carriageways": [], "turnarounds": [], "diagnostics": diag}
    floors = [float(f) if f == f and f else 1.0 for f in floors]
    heights = [f * p.floor_height for f in floors]
    fw = [fire_width_for(h, p) for h in heights]
    bu = unary_union(buildings)
    # центр проезда не ближе 5 м (край) + половина самой широкой пожарной ширины 6 м
    clearance = p.wall_gap_low[0] + max(w for _, w in p.fire_width) / 2.0
    xs, ys, gx, gy = _grid(block, p)
    ny, nx = gx.shape
    allowed_poly = block.difference(bu.buffer(clearance))
    allowed = contains_xy(allowed_poly, gx.ravel(), gy.ravel()).reshape(ny, nx)
    idx = -np.ones((ny, nx), dtype=np.int64)
    idx[allowed] = np.arange(allowed.sum())
    nn = int(allowed.sum())
    if nn == 0:
        diag["unreachable"] = list(range(n))
        return {"carriageways": [], "turnarounds": [], "diagnostics": diag}
    # граница квартала (улица) — источники
    bnd = block.boundary
    import shapely
    sdist = shapely.distance
    dist_bnd = sdist(bnd, shapely.points(gx.ravel(), gy.ravel())).reshape(ny, nx)
    is_street = allowed & (dist_bnd <= p.cell * 1.5)
    # рёбра (8 соседей)
    ii, jj, ww = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ys0 = slice(0, ny - dy)
        ys1 = slice(dy, ny)
        if dx >= 0:
            xs0, xs1 = slice(0, nx - dx), slice(dx, nx)
        else:
            xs0, xs1 = slice(-dx, nx), slice(0, nx + dx)
        a = idx[ys0, xs0]
        b = idx[ys1, xs1]
        m = (a >= 0) & (b >= 0)
        ii.append(a[m]); jj.append(b[m]); ww.append(np.full(m.sum(), p.cell * hypot(dx, dy)))
    ii = np.concatenate(ii); jj = np.concatenate(jj); ww = np.concatenate(ww)
    cell_xy = np.stack([gx[allowed], gy[allowed]], axis=1)
    street_nodes = idx[is_street]
    street_nodes = street_nodes[street_nodes >= 0]
    on_road = np.zeros(nn, dtype=bool)
    if len(street_nodes) == 0:
        diag["unreachable"] = list(range(n))
        return {"carriageways": [], "turnarounds": [], "diagnostics": diag}

    # точки доступа: здание -> список узлов
    pts_a = shapely.points(cell_xy[:, 0], cell_xy[:, 1])
    dist_to_b = [sdist(b, pts_a) for b in buildings]
    db_all = dist_bnd[allowed]
    accesses = []
    for i in range(n):
        target = _gap_mid(heights[i], p) + fw[i] / 2.0
        band = np.abs(dist_to_b[i] - target) <= p.cell * 1.2
        cand = np.where(band)[0]
        if len(cand) == 0:
            cand = np.where(dist_to_b[i] <= target + 3 * p.cell)[0]
        if len(cand) == 0:
            accesses.append([])
            continue
        # ближайший к улице
        db = db_all[cand]
        a1 = int(cand[np.argmin(db)])
        acc = [a1]
        nm = (services[i] or "").lower() if services else ""
        if p.edu_two_accesses and any(s in nm for s in EDU_SERVICES) and len(cand) > 1:
            far = np.hypot(cell_xy[cand, 0] - cell_xy[a1, 0], cell_xy[cand, 1] - cell_xy[a1, 1])
            acc.append(int(cand[np.argmax(far)]))
        accesses.append(acc)

    def graph():
        w = ww * np.where(on_road[ii] | on_road[jj], p.reuse_cost, 1.0)
        return csr_matrix((w, (ii, jj)), shape=(nn, nn))

    order = sorted(range(n), key=lambda i: min([db_all[a] for a in accesses[i]] or [1e9]))
    paths = {}     # (i, k) -> список узлов от доступа до цели
    served = {}    # узел -> множество зданий
    for i in order:
        for k, a in enumerate(accesses[i]):
            g = graph()
            d, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
            targets = np.where(on_road | np.isin(np.arange(nn), street_nodes))[0]
            if len(targets) == 0:
                continue
            dt = d[targets]
            if not np.isfinite(dt).any():
                diag["unreachable"].append(i)
                continue
            t = int(targets[np.argmin(dt)])
            path = [t]
            while path[-1] != a:
                path.append(int(pred[path[-1]]))
                if path[-1] < 0:
                    break
            if path[-1] != a:
                diag["unreachable"].append(i)
                continue
            path = path[::-1]    # от доступа к цели
            paths[(i, k)] = path
            for v in path:
                on_road[v] = True
                served.setdefault(v, set()).add(i)

    street_set = set(street_nodes.tolist())
    seg_len = lambda u, v: float(np.hypot(*(cell_xy[u] - cell_xy[v])))

    def adjacency(all_paths):
        adj = {}
        for pth in all_paths:
            for u, v in zip(pth[:-1], pth[1:]):
                adj.setdefault(u, {})[v] = seg_len(u, v)
                adj.setdefault(v, {})[u] = seg_len(u, v)
        for v in list(adj):
            if v in street_set:       # улица — единый узел S (-1): сквозной проезд = два выхода на улицу
                adj.setdefault(-1, {})[v] = 0.0
                adj[v][-1] = 0.0
        return adj

    def dead_ends(adj):
        """Тупиковые ветви: послойно срезаем узлы степени 1; длина ветви листа до ядра (узел, оставшийся после срезки)."""
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
                u, w = parent[cur]
                total += w
                chain.append(u)
                cur = u
            out.append((lf, total, chain))
        return out

    extra = []
    for _round in range(4):
        all_paths = list(paths.values()) + extra
        res_de = dead_ends(adjacency(all_paths))
        bad = [x for x in res_de if x[1] > p.dead_end_max]
        if not bad:
            break
        bad.sort(key=lambda x: -x[1])
        changed = False
        for lf, total, chain in bad[:6]:
            # узел примерно на середине ветви: от него прокладываем второй выход на улицу в обход самой ветви
            acc, mid = 0.0, chain[0]
            for a, b in zip(chain[:-1], chain[1:]):
                acc += seg_len(a, b)
                if acc >= total / 2:
                    mid = b
                    break
            wts = ww * np.where(on_road[ii] | on_road[jj], p.reuse_cost, 1.0)
            pen = np.isin(ii, chain) | np.isin(jj, chain)
            wts = np.where(pen, wts * 40.0, wts)
            g = csr_matrix((wts, (ii, jj)), shape=(nn, nn))
            d, pred = dijkstra(g, directed=False, indices=mid, return_predecessors=True)
            # цель: узел улицы, не ближайший к текущему выходу ветви
            end_street = chain[-1] if chain[-1] in street_set else None
            tg = [t for t in street_nodes.tolist() if end_street is None or np.hypot(*(cell_xy[t] - cell_xy[end_street])) > 40.0]
            tg = np.array(tg, dtype=np.int64)
            if len(tg) == 0 or not np.isfinite(d[tg]).any():
                continue
            t = int(tg[np.argmin(d[tg])])
            path = [t]
            while path[-1] != mid and pred[path[-1]] >= 0:
                path.append(int(pred[path[-1]]))
            if path[-1] != mid:
                continue
            path = path[::-1]
            extra.append(path)
            who = set(served.get(mid, set())) or set(served.get(lf, set()))
            for v in path:
                on_road[v] = True
                served.setdefault(v, set()).update(who)
            changed = True
        if not changed:
            break

    all_paths = list(paths.values()) + extra

    # ширины по узлам: класс по числу обслуживаемых зданий, не меньше пожарной ширины
    def node_width(v):
        sset = served.get(v, set())
        cls = p.main_width if len(sset) >= 2 else p.secondary_width
        return max([cls] + [fw[i] for i in sset])

    lines_by_w = {}
    seen, total_len = set(), 0.0
    for path in all_paths:
        for u, v in zip(path[:-1], path[1:]):
            w = max(node_width(u), node_width(v))
            lines_by_w.setdefault(w, []).append(LineString([cell_xy[u], cell_xy[v]]))
            e = (min(u, v), max(u, v))
            if e not in seen:
                seen.add(e)
                total_len += seg_len(u, v)
    carriageways = []
    for w, segs in sorted(lines_by_w.items()):
        poly = unary_union([sg.buffer(w / 2.0, cap_style=1, join_style=1) for sg in segs]).difference(bu)
        carriageways.append({"width": w, "class": "main" if w >= p.main_width else "secondary", "geometry": poly})
    diag["road_length_m"] = round(total_len, 1)

    final = dead_ends(adjacency(all_paths))
    turn = []
    h = p.turnaround / 2.0
    for lf, total, chain in final:
        x, y = cell_xy[lf]
        turn.append(box(x - h, y - h, x + h, y + h).difference(bu))
        if total > p.dead_end_max:
            diag["dead_end_violations"].append({"leaf_xy": [round(float(x), 1), round(float(y), 1)], "dead_end_m": round(total, 1)})
    diag["max_dead_end_m"] = round(max([t for _, t, _ in final] or [0.0]), 1)
    diag["through_links_added"] = len(extra)
    diag["turnarounds"] = len(turn)
    diag["unreachable"] = sorted(set(diag["unreachable"]))
    diag.pop("max_route_to_street_m", None)
    return {"carriageways": carriageways, "turnarounds": turn, "diagnostics": diag}
