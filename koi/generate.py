"""Render a looping ASCII koi pond for a GitHub profile README.

Each koi is simulated with a spine that follows its swim path and a tail
that undulates. Every frame, the bodies are rasterized onto a character
grid of keycap tiles, which is what gives the smooth-but-gridded look.
Frames are encoded as an animated WebP (GitHub READMEs can't run
JavaScript, but they play animated images).

The loop is seamless: every fish's speed, wobble and tail beat are
chosen so they all repeat exactly every LOOP seconds.

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

COLS, ROWS = 110, 24
CW, CH = 8, 10          # cell size in layout px
SCALE = 1.5              # 1.5x keeps it reasonably crisp without huge files
W, H = COLS * CW, ROWS * CH
LOOP = 8.0              # seconds; the animation repeats exactly
FPS = 30
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

THEMES = {
    "light": {
        "bg": (255, 255, 255, 255),
        "dot": (217, 214, 207, 255),
        "white": ("#eeeae2", "#9b968c"),
        "red": ("#e0623a", "#7a2610"),
        "black": ("#3a3836", "#bdb8ae"),
        "gold": ("#e2ad3c", "#6e4c08"),
        "edge": ("#e6e2da", "#b7b1a6"),
        "eye": ("#2a2826", "#eeeae2"),
    },
    "dark": {
        "bg": (13, 17, 23, 255),
        "dot": (44, 47, 51, 255),
        "white": ("#d8d4cb", "#6b665d"),
        "red": ("#d9572f", "#ffd2bf"),
        "black": ("#26252a", "#a9a49b"),
        "gold": ("#d9a032", "#fff0c4"),
        "edge": ("#5d5a55", "#c9c4ba"),
        "eye": ("#111111", "#d8d4cb"),
    },
}

DENSE, MID, LIGHT = "%#XZ", "xz*+", ":+·"

# variety, body length px, row (center y in cells), direction, speed px/s,
# wraps per loop, tail beats per loop, y-wobble cycles per loop, wobble rows, start phase
FISH = [
    ("kohaku",   230, 4.8,   1, 150, 1, 14, 1, 1.5, 0.10),
    ("showa",    200, 18.6, -1, 170, 1, 16, 2, 1.0, 0.55),
    ("yamabuki", 170, 11.4, -1, 140, 1, 12, 1, 2.0, 0.85),
    ("sanke",    210, 10.2,  1, 185, 1, 17, 1, 1.5, 0.62),
    ("kohaku",   150, 15.6, -1, 240, 1, 19, 1, 1.5, 0.30),
]


def halfwidth(u):
    """Body half-height in px at u (0 = nose, 1 = tail tip)."""
    if u < 0.14:
        return 7 + 21 * math.sin(u / 0.14 * math.pi / 2) ** 0.7
    if u < 0.78:
        k = (u - 0.14) / 0.64
        return 28 - 21 * k ** 1.3
    # tail fin flares out
    return 7 + (u - 0.78) / 0.22 * 20


def blobs_for(variety, seed):
    rng = np.random.default_rng(seed)
    out = []
    if variety in ("kohaku", "showa", "sanke"):
        for _ in range(3):
            out.append(("red", rng.uniform(0.08, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(0.08, 0.13), rng.uniform(0.45, 0.8)))
    if variety in ("showa", "sanke"):
        for _ in range(4 if variety == "showa" else 2):
            out.append(("black", rng.uniform(0.15, 0.65), rng.uniform(-0.8, 0.8), rng.uniform(0.04, 0.08), rng.uniform(0.3, 0.55)))
    return out


def color_at(variety, blobs, u, v):
    if variety == "yamabuki":
        return "gold"
    c = "white"
    for name, bu, bv, ru, rv in blobs:
        if ((u - bu) / ru) ** 2 + ((v - bv) / rv) ** 2 < 1:
            c = name
    return c


def glyph_for(seed, u, side, depth):
    h = zlib.crc32(f"{seed}:{int(u * 24)}:{side}".encode())
    pool = DENSE if depth > 0.6 else MID if depth > 0.3 else LIGHT
    return pool[h % len(pool)]


def spine(f, t):
    """Points along the spine (nose first), in unwrapped px coordinates."""
    variety, length, row, d, speed, wraps, beats, wob, wob_rows, phase = f
    span = speed * LOOP / wraps
    n = 26
    pts = []
    for j in range(n):
        u = j / (n - 1)
        tt = t - u * length / speed            # where the head was a moment ago
        x = (phase * span + d * speed * tt)
        y = row * CH + wob_rows * 12 * math.sin(2 * math.pi * wob * tt / LOOP + phase * 6)
        pts.append([x, y])
    pts = np.array(pts)
    # lateral undulation, stronger toward the tail
    for j in range(n):
        u = j / (n - 1)
        a, b = pts[max(j - 1, 0)], pts[min(j + 1, n - 1)]
        tx, ty = b - a
        norm = math.hypot(tx, ty) or 1
        nx, ny = -ty / norm, tx / norm
        amp = 1.5 + 9 * u ** 2
        off = amp * math.sin(2 * math.pi * beats * t / LOOP - u * 4.2)
        pts[j] += (nx * off, ny * off)
    return pts, span


def halfwidth_v(u):
    """Vectorized halfwidth for numpy arrays."""
    head = 7 + 21 * np.sin(np.clip(u / 0.14, 0, 1) * np.pi / 2) ** 0.7
    k = np.clip((u - 0.14) / 0.64, 0, 1)
    body = 28 - 21 * k ** 1.3
    tail = 7 + (u - 0.78) / 0.22 * 20
    return np.where(u < 0.14, head, np.where(u < 0.78, body, tail))


SUB = 3  # supersamples per cell edge, for partial coverage at the fish's edges


def rasterize(f, idx, t, layer):
    """Append (col, row, colorkey, glyph, coverage) tiles for one fish."""
    variety, length = f[0], f[1]
    pts, span = spine(f, t)
    blobs = blobs_for(variety, 40 + idx)
    base = pts[:, 0].min()
    shift = -math.floor((base + length) / span) * span
    for copy in (shift, shift + span, shift - span):
        p = pts + (copy, 0)
        if p[:, 0].max() < -40 or p[:, 0].min() > W + 40:
            continue
        x0, x1 = int(max(0, (p[:, 0].min() - 40) // CW)), int(min(COLS - 1, (p[:, 0].max() + 40) // CW))
        y0, y1 = int(max(0, (p[:, 1].min() - 40) // CH)), int(min(ROWS - 1, (p[:, 1].max() + 40) // CH))
        cs, rs = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        cs, rs = cs.ravel(), rs.ravel()
        offs = (np.arange(SUB) + 0.5) / SUB
        ox, oy = np.meshgrid(offs, offs)
        # sample points: (cells, SUB*SUB, 2)
        qx = (cs[:, None] + ox.ravel()[None]) * CW
        qy = (rs[:, None] + oy.ravel()[None]) * CH
        q = np.stack([qx, qy], -1).reshape(-1, 2)
        a, b = p[:-1], p[1:]
        seg = b - a
        seglen2 = (seg ** 2).sum(1)
        rel = q[:, None, :] - a[None]
        w = np.clip((rel * seg[None]).sum(-1) / seglen2, 0, 1)
        proj = a[None] + seg[None] * w[..., None]
        dist = np.hypot(*(q[:, None, :] - proj).transpose(2, 0, 1))
        k = dist.argmin(1)
        ar = np.arange(len(q))
        dd = dist[ar, k]
        u = (k + w[ar, k]) / len(seg)
        relk = rel[ar, k]
        segk = seg[k]
        side = np.where(segk[:, 0] * relk[:, 1] - segk[:, 1] * relk[:, 0] > 0, 1, -1)
        hw = halfwidth_v(u)
        body = dd <= hw
        fin = (~body) & (u > 0.24) & (u < 0.38) & (dd < hw + 11)
        fork = (u > 0.9) & (dd < 6)
        inside = (body | fin) & ~fork
        n = SUB * SUB
        inside = inside.reshape(-1, n)
        cover = inside.mean(1)
        mid = n // 2  # center sample decides color and glyph
        uc, ddc, sc, hwc = (arr.reshape(-1, n)[:, mid] for arr in (u, dd, side, hw))
        finc = fin.reshape(-1, n)[:, mid]
        for i in np.nonzero(cover > 0)[0]:
            ui, di, si, hi = float(uc[i]), float(ddc[i]), int(sc[i]), float(hwc[i])
            depth = max(0.0, 1 - di / hi)
            if finc[i] or ui > 0.78 or depth < 0.22:
                ck = "edge"
            else:
                ck = color_at(variety, blobs, ui, si * min(di / hi, 1))
            g = glyph_for(idx, ui, si, depth)
            if 0.06 < ui < 0.11 and 0.35 < di / hi < 0.75 and si < 0:
                ck, g = "eye", "o"
            layer.append((int(cs[i]), int(rs[i]), ck, g, float(cover[i])))


def tile_cache(theme):
    font = ImageFont.truetype(FONT, 7 * SCALE)
    cache = {}

    def get(ck, g, cov):
        level = max(1, min(6, round(cov * 6)))
        key = (ck, g, level)
        if key not in cache:
            fill, ink = THEMES[theme][ck]
            im = Image.new("RGBA", (int(CW * SCALE), int(CH * SCALE)), (0, 0, 0, 0))
            dr = ImageDraw.Draw(im)
            dr.rounded_rectangle([1, 1, int(CW * SCALE) - 2, int(CH * SCALE) - 2], radius=int(2 * SCALE), fill=fill)
            dr.text((CW * SCALE / 2, CH * SCALE / 2 + 1), g, font=font, fill=ink, anchor="mm")
            if level < 6:
                a = im.getchannel("A").point(lambda v: int(v * (level / 6) ** 0.8))
                im.putalpha(a)
            cache[key] = im
        return cache[key]

    return get


def background(theme):
    im = Image.new("RGBA", (int(W * SCALE), int(H * SCALE)), THEMES[theme]["bg"])
    dr = ImageDraw.Draw(im)
    rad = 0.6 * SCALE
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                cx, cy = (c * CW + CW / 2) * SCALE, (r * CH + CH / 2) * SCALE
                dr.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=THEMES[theme]["dot"])
    return im


def render(theme, out_path):
    bg = background(theme)
    tiles = tile_cache(theme)
    n = int(LOOP * FPS)
    tmp = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / FPS
        frame = bg.copy()
        for idx, f in enumerate(FISH):
            layer = []
            rasterize(f, idx, t, layer)
            for c, r, ck, g, cov in layer:
                frame.alpha_composite(tiles(ck, g, cov), (int(c * CW * SCALE), int(r * CH * SCALE)))
        frame.save(tmp / f"{i:04d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "%04d.png"),
         "-c:v", "libwebp_anim", "-lossless", "0", "-q:v", "55", "-compression_level", "6", "-pix_fmt", "yuv420p", "-loop", "0", str(out_path)],
        check=True,
    )
    shutil.rmtree(tmp)


if __name__ == "__main__":
    here = Path(__file__).parent
    for theme in THEMES:
        out = here / f"pond-{theme}.webp"
        render(theme, out)
        print(theme, f"{out.stat().st_size / 1024:.0f} KB")
