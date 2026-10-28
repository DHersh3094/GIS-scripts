#!/usr/bin/env python3
"""Rasterize the vector geometry (LineStrings / Polygons) in a KMZ into an MBTiles file.

Usage: kmz2mbtiles.py input.kmz output.mbtiles [maxzoom] [opacity]
  maxzoom  default 16
  opacity  0..1, default 0.5
"""
import io
import math
import os
import re
import sqlite3
import sys
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

TILE = 256
SS = 2  # supersampling factor for antialiasing
MINZOOM = 0
DEFAULT_LINE = ((255, 255, 255, 255), 1.0)


def strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def kml_color(text, default):
    """KML colors are aabbggrr hex."""
    if not text:
        return default
    t = text.strip().lstrip("#")
    if len(t) != 8:
        return default
    a, b, g, r = (int(t[i:i + 2], 16) for i in (0, 2, 4, 6))
    return (r, g, b, a)


def parse_style(style_el):
    """Return dict with line color/width and poly fill info from a <Style> element."""
    s = {}
    ls = style_el.find("LineStyle")
    if ls is not None:
        c = ls.findtext("color")
        if c:
            s["line_color"] = kml_color(c, DEFAULT_LINE[0])
        w = ls.findtext("width")
        if w:
            s["line_width"] = float(w)
    ps = style_el.find("PolyStyle")
    if ps is not None:
        c = ps.findtext("color")
        if c:
            s["poly_color"] = kml_color(c, (255, 255, 255, 255))
        f = ps.findtext("fill")
        if f is not None:
            s["poly_fill"] = f.strip() != "0"
        o = ps.findtext("outline")
        if o is not None:
            s["poly_outline"] = o.strip() != "0"
    return s


def collect_styles(root):
    styles = {}
    for st in root.iter("Style"):
        sid = st.get("id")
        if sid:
            styles[sid] = parse_style(st)
    stylemaps = {}
    for sm in root.iter("StyleMap"):
        sid = sm.get("id")
        if not sid:
            continue
        for pair in sm.findall("Pair"):
            if (pair.findtext("key") or "").strip() == "normal":
                stylemaps[sid] = (pair.findtext("styleUrl") or "").strip()
    return styles, stylemaps


def resolve_style(url, styles, stylemaps, depth=0):
    if not url or depth > 5:
        return {}
    sid = url.split("#")[-1]
    if sid in styles:
        return styles[sid]
    if sid in stylemaps:
        return resolve_style(stylemaps[sid], styles, stylemaps, depth + 1)
    return {}


def parse_coords(text):
    pts = []
    for tok in (text or "").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return pts


def extract_features(root):
    """Yield (kind, coords_list, style) where kind is 'line' or 'poly'."""
    styles, stylemaps = collect_styles(root)
    feats = []
    for pm in root.iter("Placemark"):
        style = dict(resolve_style(pm.findtext("styleUrl"), styles, stylemaps))
        inline = pm.find("Style")
        if inline is not None:
            style.update(parse_style(inline))
        for ls in pm.iter("LineString"):
            pts = parse_coords(ls.findtext("coordinates"))
            if len(pts) >= 2:
                feats.append(("line", [pts], style))
        for poly in pm.iter("Polygon"):
            rings = []
            for path in ("outerBoundaryIs/LinearRing", "innerBoundaryIs/LinearRing"):
                for lr in poly.findall(path):
                    pts = parse_coords(lr.findtext("coordinates"))
                    if len(pts) >= 3:
                        rings.append(pts)
            if rings:
                feats.append(("poly", rings, style))
    return feats


def lonlat_to_px(lon, lat, z):
    """Global pixel coords at zoom z (Web Mercator)."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = TILE * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def render_zoom(feats, z, opacity):
    """Return dict {(tx, ty): png_bytes} for one zoom level."""
    # Project everything to global pixel space at this zoom and compute per-feature bbox
    proj = []
    for kind, rings, style in feats:
        prings = [[lonlat_to_px(lon, lat, z) for lon, lat in r] for r in rings]
        width = style.get("line_width", DEFAULT_LINE[1])
        pad = width / 2 + 1
        xs = [p[0] for r in prings for p in r]
        ys = [p[1] for r in prings for p in r]
        bbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        proj.append((kind, prings, style, bbox))

    # Bucket features into the tiles their bbox touches
    max_t = 2 ** z - 1
    buckets = {}
    for i, (_, _, _, (x0, y0, x1, y1)) in enumerate(proj):
        for tx in range(max(0, int(x0 // TILE)), min(max_t, int(x1 // TILE)) + 1):
            for ty in range(max(0, int(y0 // TILE)), min(max_t, int(y1 // TILE)) + 1):
                buckets.setdefault((tx, ty), []).append(i)

    out = {}
    size = TILE * SS
    for (tx, ty), idxs in buckets.items():
        ox, oy = tx * TILE, ty * TILE
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for i in idxs:
            kind, prings, style, _ = proj[i]
            color = style.get("line_color", DEFAULT_LINE[0])
            width = max(1, round(style.get("line_width", DEFAULT_LINE[1]) * SS))
            local = [[((x - ox) * SS, (y - oy) * SS) for x, y in r] for r in prings]
            if kind == "poly":
                if style.get("poly_fill", True) and style.get("poly_color", (0, 0, 0, 0))[3] > 0:
                    draw.polygon(local[0], fill=style["poly_color"])
                    for hole in local[1:]:
                        draw.polygon(hole, fill=(0, 0, 0, 0))
                if not style.get("poly_outline", True):
                    continue
                local = [r + [r[0]] for r in local]
            if color[3] == 0:
                continue
            for r in local:
                draw.line(r, fill=color, width=width, joint="curve")
        if img.getbbox() is None:
            continue
        img = img.resize((TILE, TILE), Image.LANCZOS)
        if opacity < 1.0:
            a = img.getchannel("A").point(lambda v: int(v * opacity))
            img.putalpha(a)
        if img.getchannel("A").getbbox() is None:
            continue
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)
        out[(tx, ty)] = buf.getvalue()
    return out


def read_kml(kmz_path):
    with zipfile.ZipFile(kmz_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
        if not names:
            sys.exit(f"No .kml found inside {kmz_path}")
        name = "doc.kml" if "doc.kml" in names else names[0]
        data = zf.read(name)
    try:
        return strip_ns(ET.fromstring(data))
    except ET.ParseError:
        # Google Earth exports sometimes use undeclared prefixes (e.g. xsi:); drop prefixes and retry
        text = data.decode("utf-8", errors="replace")
        text = re.sub(r"\s[\w.-]+:[\w.-]+=\"[^\"]*\"", "", text)  # prefixed attributes
        text = re.sub(r"<(/?)[\w.-]+:", r"<\1", text)  # prefixed tags
        text = re.sub(r"\sxmlns(:[\w.-]+)?=\"[^\"]*\"", "", text)
        return strip_ns(ET.fromstring(text))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    kmz_path, out_path = sys.argv[1], sys.argv[2]
    maxzoom = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    opacity = float(sys.argv[4]) if len(sys.argv) > 4 else 1

    root = read_kml(kmz_path)
    feats = extract_features(root)
    if not feats:
        sys.exit("No LineString/Polygon geometry found in KMZ")
    lons = [p[0] for _, rings, _ in feats for r in rings for p in r]
    lats = [p[1] for _, rings, _ in feats for r in rings for p in r]
    print(f"{len(feats)} features, bounds {min(lons):.5f},{min(lats):.5f},{max(lons):.5f},{max(lats):.5f}")

    if os.path.exists(out_path):
        os.remove(out_path)
    db = sqlite3.connect(out_path)
    db.executescript("""
        CREATE TABLE metadata (name TEXT, value TEXT);
        CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB);
        CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row);
    """)
    name = os.path.splitext(os.path.basename(out_path))[0]
    meta = {
        "name": name,
        "type": "overlay",
        "version": "1",
        "description": f"Rasterized KMZ vectors from {os.path.basename(kmz_path)}",
        "format": "png",
        "minzoom": str(MINZOOM),
        "maxzoom": str(maxzoom),
        "bounds": f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}",
        "center": f"{(min(lons) + max(lons)) / 2},{(min(lats) + max(lats)) / 2},{min(maxzoom, 12)}",
    }
    db.executemany("INSERT INTO metadata VALUES (?, ?)", meta.items())

    total = 0
    for z in range(MINZOOM, maxzoom + 1):
        tiles = render_zoom(feats, z, opacity)
        flip = 2 ** z - 1
        db.executemany(
            "INSERT INTO tiles VALUES (?, ?, ?, ?)",
            ((z, tx, flip - ty, data) for (tx, ty), data in tiles.items()),  # TMS y
        )
        db.commit()
        total += len(tiles)
        print(f"zoom {z:2d}: {len(tiles)} tiles")
    db.close()
    print(f"Wrote {total} tiles to {out_path}")


if __name__ == "__main__":
    main()
