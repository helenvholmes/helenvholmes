"""Render a looping ASCII koi pond for a GitHub profile README.

Each koi swims a slow closed loop around the pond. Its spine follows the
path, its tail undulates, and every frame the body is rasterized onto a
grid of square keycap tiles. Tiles at the edge of a fish fade with how
much of the fish covers them, so the motion reads as smooth even though
everything snaps to the grid. Ripples spread as rings of / \\ | - glyphs.

GitHub READMEs can't run JavaScript, but they play animated images, so
the frames are encoded as an animated WebP. The loop is seamless: every
path, tail beat and ripple repeats exactly every LOOP seconds.

Usage: python3 koi/generate.py   (needs Pillow, numpy and ffmpeg)
Writes koi/pond-light.webp and koi/pond-dark.webp.
"""
import math
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE = 12                # tile size in layout px (square)
COLS, ROWS = 74, 26
W, H = COLS * TILE, ROWS * TILE
SCALE = 1.5              # render scale; 1.5x stays crisp without huge files
LOOP = 20.0              # seconds; the animation repeats exactly
FPS = 20
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

THEMES = {
    "light": {
        "bg": "#f3f2ef",
        "ripple": "#b4b2ac",
        "eye": "#2b2a27",
        # fill colors for body tiles, light to dark; glyph ink is derived
        "grey": ["#d6d4cf", "#b9b7b2", "#9a9894", "#7c7a76", "#64625f"],
        "orange": ["#ebc29d", "#e6a26a", "#e08440", "#cf6a28", "#b4561e"],
        "edge": ["#e6e4e0", "#dddbd6"],
    },
    "dark": {
        "bg": "#191919",
        "ripple": "#55534f",
        "eye": "#f3f1ea",
        "grey": ["#3a3936", "#4e4d49", "#64625e", "#7f7d78", "#9b9994"],
        "orange": ["#6b4227", "#8f532a", "#b8642a", "#d8762f", "#e8925a"],
        "edge": ["#2a2927", "#323130"],
    },
}

# name, body length px, pattern seed, path (cx, cy, ax, ay, fx, fy, phase),
# tail beats per loop, depth (1 = surface, lower = deeper and fainter)
# fx and fy are path cycles per loop; fx=1, fy=2 traces a figure eight.
FISH = [
    ("deep-a", 150, 11, (300, 210, 210, 70, 1, 2, 0.15), 22, 0.45),
    ("deep-b", 140, 12, (640, 120, 170, 60, -1, 2, 0.60), 20, 0.45),
    ("a", 190, 1, (450, 160, 330, 95, 1, 2, 0.00), 26, 1.0),
    ("b", 175, 2, (430, 170, 300, 85, -1, 2, 0.50), 24, 1.0),
    ("c", 165, 3, (250, 150, 160, 80, 1, 1, 0.30), 21, 1.0),
    ("d", 170, 4, (660, 175, 170, 90, -1, 1, 0.80), 23, 1.0),
]

# ripples: x, y, start time, lifetime seconds, max radius px
RIPPLES = [
    (150, 70, 1.0, 5.0, 70), (760, 260, 6.0, 5.0, 80),
    (520, 60, 11.0, 5.0, 60), (330, 280, 15.5, 5.0, 75),
]


def head(path, t):
    cx, cy, ax, ay, fx, fy, ph = path
    a = 2 * math.pi * (t / LOOP + ph)
    return np.array([cx + ax * math.sin(fx * a), cy + ay * math.sin(fy * a)])


def path_speed(path):
    """Average speed along the path, px/s."""
    n = 400
    pts = np.array([head(path, LOOP * i / n) for i in range(n + 1)])
    return np.hypot(*np.diff(pts, axis=0).T).sum() / LOOP


def spine(f, t):
    _, length, _, path, beats, _ = f
    v = path_speed(path)
    n = 28
    pts = np.array([head(path, t - (j / (n - 1)) * length / v) for j in range(n)])
    out = pts.copy()
    for j in range(n):
        u = j / (n - 1)
        a, b = pts[max(j - 1, 0)], pts[min(j + 1, n - 1)]
        tx, ty = b - a
        norm = math.hypot(tx, ty) or 1
        nx, ny = -ty / norm, tx / norm
        amp = 1 + 10 * u ** 2
        off = amp * math.sin(2 * math.pi * beats * t / LOOP - u * 4.5)
        out[j] += (nx * off, ny * off)
    return out


def halfwidth(u, length):
    """Body half-width at u (0 = nose, 1 = tail tip) for a fish of this length."""
    m = 0.2 * length
    k0 = np.clip(u / 0.22, 0, 1)
    nose = m * np.sqrt(k0 * (2 - k0))
    k = np.clip((u - 0.22) / 0.62, 0, 1)
    body = m * (1 - 0.86 * k ** 1.15)
    tail = m * 0.14 + (u - 0.84) / 0.16 * m * 0.55
    return np.where(u < 0.22, nose, np.where(u < 0.84, body, tail))


def hsh(*parts):
    return zlib.crc32(":".join(map(str, parts)).encode())


def pattern(seed):
    """Orange patches in fish-local (u, v) space."""
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0.05, 0.7), rng.uniform(-0.7, 0.7), rng.uniform(0.09, 0.16), rng.uniform(0.4, 0.9))
            for _ in range(rng.integers(2, 4))]


SUB = 3


def rasterize(f, t):
    """Return tiles for one fish, (col, row, kind, shade, glyph, coverage), and its eye positions."""
    name, length, seed, path, beats, depth = f
    pts = spine(f, t)
    blobs = pattern(seed)
    pad = 0.3 * length
    x0, x1 = int(max(0, (pts[:, 0].min() - pad) // TILE)), int(min(COLS - 1, (pts[:, 0].max() + pad) // TILE))
    y0, y1 = int(max(0, (pts[:, 1].min() - pad) // TILE)), int(min(ROWS - 1, (pts[:, 1].max() + pad) // TILE))
    if x0 > x1 or y0 > y1:
        return [], []
    cs, rs = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    cs, rs = cs.ravel(), rs.ravel()
    offs = (np.arange(SUB) + 0.5) / SUB
    ox, oy = np.meshgrid(offs, offs)
    q = np.stack([(cs[:, None] + ox.ravel()) * TILE, (rs[:, None] + oy.ravel()) * TILE], -1).reshape(-1, 2)
    a, b = pts[:-1], pts[1:]
    seg = b - a
    seglen2 = np.maximum((seg ** 2).sum(1), 1e-6)
    rel = q[:, None, :] - a[None]
    w = np.clip((rel * seg[None]).sum(-1) / seglen2, 0, 1)
    proj = a[None] + seg[None] * w[..., None]
    dist = np.hypot(*(q[:, None, :] - proj).transpose(2, 0, 1))
    k = dist.argmin(1)
    ar = np.arange(len(q))
    dd, u = dist[ar, k], (k + w[ar, k]) / len(seg)
    side = np.sign(seg[k][:, 0] * rel[ar, k][:, 1] - seg[k][:, 1] * rel[ar, k][:, 0])
    hw = halfwidth(u, length)
    body = dd <= hw
    fin = (~body) & (u > 0.2) & (u < 0.34) & (dd < hw + 0.07 * length)
    n = SUB * SUB
    cover = (body | fin).reshape(-1, n).mean(1)
    mid = n // 2
    uc, dc, sc, hc = (x.reshape(-1, n)[:, mid] for x in (u, dd, side, hw))
    finc = fin.reshape(-1, n)[:, mid]
    tiles = []
    for i in np.nonzero(cover > 0.15)[0]:
        ui, vi = float(uc[i]), float(np.clip(sc[i] * dc[i] / max(hc[i], 1.0), -1.5, 1.5))
        edge = bool(finc[i]) or abs(vi) > 0.86 or ui > 0.9
        h = hsh(seed, int(ui * 30), int(vi * 4 + 8))
        if edge:
            kind, shade = "edge", h % 2
            glyph = "." if h % 3 else "S"
        else:
            orange = any(((ui - bu) / ru) ** 2 + ((vi - bv) / rv) ** 2 < 1 for bu, bv, ru, rv in blobs)
            kind = "orange" if orange else "grey"
            # mottled: darker toward the spine, varied per tile
            shade = min(4, max(0, int((1 - abs(vi)) * 3.2) + (h % 3) - 1))
            glyph = "XS%"[h % 3]
        tiles.append((int(cs[i]), int(rs[i]), kind, shade, glyph, float(cover[i])))
    # eyes: either side of the head
    nose, behind = pts[0], pts[3]
    d = nose - behind
    d = d / (np.linalg.norm(d) or 1)
    nrm = np.array([-d[1], d[0]])
    back = nose - d * 0.11 * length
    eyes = [back + nrm * 0.12 * length, back - nrm * 0.12 * length]
    return tiles, eyes


def ripple_glyphs(t):
    out = []
    for x, y, start, life, rmax in RIPPLES:
        age = (t - start) % LOOP
        if age > life:
            continue
        p = age / life
        for ring in (1.0, 0.62):
            r = rmax * p * ring
            if r < 8:
                continue
            alpha = (1 - p) * (1 if ring == 1 else 0.7)
            steps = max(8, int(2 * math.pi * r / TILE))
            seen = set()
            for s in range(steps):
                ang = 2 * math.pi * s / steps
                cx, cy = x + r * math.cos(ang), y + r * math.sin(ang) * 0.8
                c, rr = int(cx // TILE), int(cy // TILE)
                if (c, rr) in seen or not (0 <= c < COLS and 0 <= rr < ROWS):
                    continue
                seen.add((c, rr))
                # the ring's tangent decides the glyph; screen y grows downward
                tang = (math.degrees(ang) + 90) % 180
                if tang < 22.5 or tang > 157.5:
                    g = "-"
                elif 67.5 < tang < 112.5:
                    g = "|"
                else:
                    g = "\\" if tang < 90 else "/"
                out.append((c, rr, g, alpha))
    return out


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(c1, c2, k):
    return tuple(int(a + (b - a) * k) for a, b in zip(c1, c2))


def make_renderer(theme):
    th = THEMES[theme]
    S = SCALE
    ts = int(round(TILE * S))
    font = ImageFont.truetype(FONT, int(round(9 * S)))
    bg = hex_rgb(th["bg"])
    cache = {}

    def tile(kind, shade, glyph, cov, depth):
        lvl = max(1, min(8, round(cov * 8)))
        key = (kind, shade, glyph, lvl, depth)
        if key in cache:
            return cache[key]
        fill = hex_rgb(th[kind][shade])
        # glyph ink: a step darker on light themes, lighter on dark
        ink = mix(fill, (0, 0, 0) if theme == "light" else (255, 255, 255), 0.28)
        alpha = int(255 * (lvl / 8) ** 0.9 * depth)
        im = Image.new("RGBA", (ts, ts), (0, 0, 0, 0))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, ts - 2, ts - 2], fill=fill + (alpha,))
        dr.text((ts / 2 - 0.5, ts / 2), glyph, font=font, fill=ink + (int(alpha * 0.9),), anchor="mm")
        cache[key] = im
        return im

    def glyph_img(g, color, alpha):
        key = ("g", g, color, alpha)
        if key not in cache:
            im = Image.new("RGBA", (ts, ts), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((ts / 2, ts / 2), g, font=font, fill=hex_rgb(color) + (int(255 * alpha),), anchor="mm")
            cache[key] = im
        return cache[key]

    def frame(t):
        im = Image.new("RGBA", (int(W * S), int(H * S)), bg + (255,))
        for c, r, g, a in ripple_glyphs(t):
            im.alpha_composite(glyph_img(g, th["ripple"], round(a, 1)), (c * ts, r * ts))
        for f in sorted(FISH, key=lambda f: f[5]):
            tiles, eyes = rasterize(f, t)
            depth = f[5]
            for c, r, kind, shade, glyph, cov in tiles:
                im.alpha_composite(tile(kind, shade, glyph, cov, depth), (c * ts, r * ts))
            if depth == 1:
                for e in eyes:
                    c, r = int(e[0] // TILE), int(e[1] // TILE)
                    if 0 <= c < COLS and 0 <= r < ROWS:
                        im.alpha_composite(glyph_img("o", th["eye"], 1.0), (c * ts, r * ts))
        return im.convert("RGB")

    return frame


def render(theme, out_path):
    frame = make_renderer(theme)
    tmp = Path(tempfile.mkdtemp())
    for i in range(int(LOOP * FPS)):
        frame(i / FPS).save(tmp / f"{i:04d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "%04d.png"),
         "-c:v", "libwebp_anim", "-lossless", "0", "-q:v", "60", "-compression_level", "6",
         "-pix_fmt", "yuv420p", "-loop", "0", str(out_path)],
        check=True,
    )
    shutil.rmtree(tmp)


if __name__ == "__main__":
    import sys
    here = Path(__file__).parent
    if "--still" in sys.argv:  # quick preview frames
        for theme in THEMES:
            fr = make_renderer(theme)
            for t in (0, 1, 5):
                fr(t).save(f"/tmp/koi-{theme}-{t}.png")
        sys.exit()
    for theme in THEMES:
        out = here / f"pond-{theme}.webp"
        render(theme, out)
        print(theme, f"{out.stat().st_size / 1024:.0f} KB")
