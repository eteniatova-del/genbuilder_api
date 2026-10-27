"""
Экспорт результата генерации в слои GeoJSON и скрипт открытия в QGIS на заданном масштабе.

Использование:
    python scripts/export_qgis_project.py result.json out_dir --scale 1000 [--crown 15]

result.json — ответ генерации с полями generated_buildings, generated_greenery, generated_parking, generated_roads.
В out_dir создаются слои *.geojson и open_in_qgis.py. В QGIS: «Плагины» -> «Консоль Python» -> «Показать редактор» ->
«Открыть скрипт» -> open_in_qgis.py -> «Запустить скрипт». Проект откроется на масштабе 1:<scale>.

Масштаб — только значение из ряда, взятого из ГОСТ 21.508-2020 (п. 4.21: основные 1:500 и 1:1000, допускаемый 1:2000 для планов
благоустройства) и СП 47.13330.2016 (прил. Б: 1:5000 для проекта планировки). Текст норм сверен по сайтам-зеркалам, не по первоисточнику.
"""
import argparse
import json
import sys
from pathlib import Path

SCALES = (500, 1000, 2000, 5000)
LAYERS = {
    "generated_buildings": ("buildings", "#d99a6c"),
    "generated_greenery": ("greenery", "#cfe8c0"),
    "generated_parking": ("parking", "#8fb4e8"),
    "generated_roads": ("roads", "#555555"),
}

QGIS_TEMPLATE = '''# Скрипт для консоли Python в QGIS (создан export_qgis_project.py). Масштаб 1:{scale}.
from qgis.core import (QgsProject, QgsVectorLayer, QgsCoordinateReferenceSystem, QgsFillSymbol, QgsMarkerSymbol, QgsUnitTypes,
                       QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsProperty, QgsSymbolLayer)
from qgis.utils import iface
import os

D = r"{out_dir}"
SCALE = {scale}
CROWN = {crown}
project = QgsProject.instance()
project.setCrs(QgsCoordinateReferenceSystem("EPSG:{crs}"))  # метрическая СК: масштаб и размер кроны в метрах без искажения
layers = []
for name, color in {layers}:
    path = os.path.join(D, name + ".geojson")
    if not os.path.exists(path):
        continue
    lyr = QgsVectorLayer(path, name, "ogr")
    if not lyr.isValid():
        continue
    if name == "greenery":
        lyr.setSubsetString("kind = 'lawn'")
        lyr.renderer().setSymbol(QgsFillSymbol.createSimple({{"color": color, "outline_style": "no"}}))
        trees = QgsVectorLayer(path, "trees", "ogr")
        trees.setSubsetString("kind = 'tree'")
        sym = QgsMarkerSymbol.createSimple({{"name": "circle", "color": "#4f9a3a", "outline_color": "#2f6a22"}})
        sym.setSizeUnit(QgsUnitTypes.RenderMapUnits)
        sym.setSize(CROWN)
        sym.setOpacity(0.55)
        trees.renderer().setSymbol(sym)
        project.addMapLayer(trees)
        layers.append(trees)
        shr = QgsVectorLayer(path, "shrubs", "ogr")
        shr.setSubsetString("kind = 'shrub'")
        ssym = QgsMarkerSymbol.createSimple({{"name": "circle", "color": "#7fb04a", "outline_color": "#4d7a26"}})
        ssym.setSizeUnit(QgsUnitTypes.RenderMapUnits)
        ssym.setSize(1.5)
        ssym.symbolLayer(0).setDataDefinedProperty(QgsSymbolLayer.PropertySize, QgsProperty.fromField("crown_d_m"))
        shr.renderer().setSymbol(ssym)
        project.addMapLayer(shr)
        layers.append(shr)
    elif name == "buildings" and lyr.fields().indexFromName("parking_class") >= 0:
        # дома с подземным паркингом окрашены иначе (атрибуты parking_class, underground_parking_places)
        def fsym(c):
            return QgsFillSymbol.createSimple({{"color": c, "outline_color": "#333333", "outline_width": "0.1"}})
        cats = [QgsRendererCategory("underground", fsym("#a0522d"), "подземная парковка"),
                QgsRendererCategory("mixed", fsym("#cd853f"), "подземная и наземная"),
                QgsRendererCategory("surface", fsym("#e9c9a0"), "только наземная"),
                QgsRendererCategory("", fsym(color), "прочие (сервисы и др.)")]
        lyr.setRenderer(QgsCategorizedSymbolRenderer("parking_class", cats))
    else:
        lyr.renderer().setSymbol(QgsFillSymbol.createSimple({{"color": color, "outline_color": "#333333", "outline_width": "0.1"}}))
    project.addMapLayer(lyr)
    layers.append(lyr)
# охват по зданиям; масштаб экранный, зависит от DPI монитора
ext = None
blayer = None
for l in layers:
    if l.name() == "buildings":
        blayer = l
        ext = l.extent()
if ext is not None:
    iface.mapCanvas().setExtent(iface.mapCanvas().mapSettings().layerExtentToOutputExtent(blayer, ext))
iface.mapCanvas().zoomScale(SCALE)
iface.mapCanvas().refresh()
print("Открыто на масштабе 1:%d" % SCALE)
'''


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("out_dir")
    ap.add_argument("--scale", type=int, default=1000)
    ap.add_argument("--crs", type=int, default=32636, help="EPSG метрической СК карты (32636 — UTM 36N, Санкт-Петербург)")
    ap.add_argument("--crown", type=float, default=15.0, help="диаметр кроны, м (размер маркера дерева в единицах карты)")
    a = ap.parse_args(argv)
    if a.scale not in SCALES:
        sys.exit("Масштаб должен быть одним из: " + ", ".join("1:%d" % s for s in SCALES))
    data = json.load(open(a.result, encoding="utf-8"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    used = []
    for key, (name, color) in LAYERS.items():
        fc = data.get(key)
        if fc and fc.get("features"):
            (out / (name + ".geojson")).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
            used.append((name, color))
    (out / "open_in_qgis.py").write_text(
        QGIS_TEMPLATE.format(scale=a.scale, crown=a.crown, crs=a.crs, out_dir=str(out.resolve()), layers=repr(used)), encoding="utf-8")
    print("Слои:", ", ".join(n for n, _ in used), "| масштаб 1:%d" % a.scale)


if __name__ == "__main__":
    main()
