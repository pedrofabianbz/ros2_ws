from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

Rect = Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


@dataclass
class StaticWorld:
    static_rects: List[Rect]


def _parse_pose(text: str | None):
    # "x y z roll pitch yaw"
    if not text:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    vals = [float(v) for v in text.split()]
    while len(vals) < 6:
        vals.append(0.0)
    return vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]


def _is_true(text: str | None) -> bool:
    return bool(text) and text.strip().lower() in ("true", "1")


def load_static_rects_from_sdf(sdf_path: str | Path) -> StaticWorld:
    sdf_path = Path(sdf_path).expanduser().resolve()
    root = ET.parse(sdf_path).getroot()

    rects: List[Rect] = []

    # Solo modelos estáticos (paredes + cajas que marcaste <static>true</static>)
    for model in root.findall(".//world//model"):
        if not _is_true(model.findtext("static")):
            continue

        mx, my, _, _, _, myaw = _parse_pose(model.findtext("pose"))

        # Links directos del modelo (NO usar ".//link" para evitar arrastrar cosas raras)
        for link in model.findall("link"):
            lx, ly, _, _, _, lyaw = _parse_pose(link.findtext("pose"))

            # Collisions directos del link (tampoco usar ".//collision")
            for coll in link.findall("collision"):
                cx, cy, _, _, _, cyaw = _parse_pose(coll.findtext("pose"))

                size_el = coll.find("geometry/box/size")
                if size_el is None or not size_el.text:
                    continue

                sx, sy, _ = [float(v) for v in size_el.text.split()]

                # Para tu caso: axis-aligned (yaw ignorado).
                # Si luego metes yaw != 0, se puede extender a OBB/AABB inflado.
                x = mx + lx + cx
                y = my + ly + cy

                rects.append((x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2))

    return StaticWorld(static_rects=rects)
