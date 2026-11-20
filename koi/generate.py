"""Generate an animated ASCII koi pond as SVG for a GitHub profile README.

GitHub strips JavaScript from READMEs, but CSS animations inside an SVG
referenced with <img> still play. Each koi is a grid of keycap tiles;
movement uses steps() timing so the fish hop cell to cell like a
character grid instead of gliding.

Usage: python3 koi/generate.py   (writes koi/pond-light.svg and koi/pond-dark.svg)
"""
import math
import random
from pathlib import Path

COLS, ROWS = 88, 16
CW, CH = 10, 12  # cell size in px
W, H = COLS * CW, ROWS * CH

THEMES = {
    "light": {
        "dot": "#d9d6cf",
        "ripple": "#b9c3c7",
        "white": ("#eeeae2", "#9b968c"),
        "red": ("#e0623a", "#7a2610"),
        "black": ("#3a3836", "#bdb8ae"),
        "gold": ("#e2ad3c", "#6e4c08"),
        "edge": ("#e6e2da", "#b7b1a6"),
    },
    "dark": {
        "dot": "#2c2f33",
        "ripple": "#3b4a52",
        "white": ("#d8d4cb", "#6b665d"),
        "red": ("#d9572f", "#ffd2bf"),
        "black": ("#26252a", "#a9a49b"),
        "gold": ("#d9a032", "#fff0c4"),
        "edge": ("#5d5a55", "#c9c4ba"),
    },
}

# Density glyphs, dense in the middle of the body, light at the edges.
DENSE = "%#XZ"
MID = "x*+z"
LIGHT = ":·+"


def body_halfwidth(t):
    """Half-height of the fish body at position t (0 = tail tip, 1 = nose)."""
    if t < 0.18:  # tail fin: flares out
        return 0.6 + (0.18 - t) * 9
    if t < 0.35:  # caudal peduncle, the narrow waist
        return 0.6 + (t - 0.18) * 4
    u = (t - 0.35) / 0.65
    return 0.9 + 1.6 * math.sin(min(u * 1.25, 1) * math.pi / 2) * (1 - max(0, u - 0.75) * 2.4)


def make_fish(length, variety, rng, wiggle):
    """Return {(col,row): (colorkey, glyph)} for a right-facing fish.

    wiggle shifts the tail rows up or down a little to make a swim frame.
    """
    cells = {}
    cy = 3
    for c in range(length):
        t = c / (length - 1)
        hw = body_halfwidth(t)
        # the rear third bends with the wiggle
        bend = wiggle * max(0.0, (0.4 - t)) * 2.6
        is_tail = t < 0.18
        for r in range(-4, 5):
            d = abs(r - bend)
            if d > hw + 0.25:
                continue
            if is_tail and d < 0.5 and t < 0.1:
                continue  # fork in the tail fin
            edge = d > hw - 0.7
            glyph = rng.choice(LIGHT if edge else (MID if d > hw * 0.45 else DENSE))
            if is_tail:
                glyph = rng.choice(LIGHT + "x")
            cells[(c, cy + r)] = [("edge" if (edge or is_tail) else "white"), glyph]
    # eye
    eye = (length - 3, cy - 1)
    if eye in cells:
        cells[eye] = ["black", "o"]
    # pectoral fins
    fc = int(length * 0.62)
    for dr in (-1, 1):
        r = cy + dr * round(body_halfwidth(fc / (length - 1)) + 0.4)
        cells.setdefault((fc, r), ["edge", ":"])
        cells[(fc - 1, r + dr)] = ["edge", "·"]
        cells[(fc - 2, r + dr)] = ["edge", ":"]
    paint(cells, variety, length, cy, rng)
    return {k: tuple(v) for k, v in cells.items()}


def paint(cells, variety, length, cy, rng):
    body = [k for k, v in cells.items() if v[0] == "white"]
    if variety == "yamabuki":
        for k in body:
            cells[k][0] = "gold"
        return
    blobs = []
    if variety in ("kohaku", "showa", "sanke"):
        for _ in range(3):
            blobs.append(("red", rng.uniform(0.35, 0.95) * length, cy + rng.uniform(-1.2, 1.2), rng.uniform(1.6, 2.8)))
    if variety in ("showa", "sanke"):
        n = 4 if variety == "showa" else 2
        for _ in range(n):
            blobs.append(("black", rng.uniform(0.3, 0.9) * length, cy + rng.uniform(-1.5, 1.5), rng.uniform(1.0, 1.9)))
    for k in body:
        for color, bx, by, rad in blobs:
            if (k[0] - bx) ** 2 / 2.2 + (k[1] - by) ** 2 < rad ** 2:
                cells[k][0] = color


def tile(x, y, colorkey, glyph, theme):
    fill, ink = THEMES[theme][colorkey]
    g = {"<": "&lt;", "&": "&amp;"}.get(glyph, glyph)
    return (
        f'<rect x="{x * CW + 0.5}" y="{y * CH + 0.5}" width="{CW - 1}" height="{CH - 1}" rx="1.5" fill="{fill}"/>'
        f'<text x="{x * CW + CW / 2}" y="{y * CH + CH - 3}" fill="{ink}">{g}</text>'
    )


def build(theme):
    rng = random.Random(7)
    t = THEMES[theme]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="ASCII koi swimming in a pond">',
        "<style>",
        "text{font:600 8px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;text-anchor:middle}",
        ".f{animation-iteration-count:infinite}",
        ".swim{animation-name:swim;animation-iteration-count:infinite}",
        ".swimL{animation-name:swimL;animation-iteration-count:infinite}",
        ".bob{animation-name:bob;animation-iteration-count:infinite}",
        ".fa{animation:fa .9s steps(1) infinite}",
        ".fb{animation:fb .9s steps(1) infinite}",
        ".fc{animation:fc .9s steps(1) infinite}",
        "@keyframes fa{0%{opacity:1}25%,100%{opacity:0}}",
        "@keyframes fb{0%{opacity:0}25%{opacity:1}50%{opacity:0}75%{opacity:1}100%{opacity:0}}",
        "@keyframes fc{0%,50%{opacity:0}50.01%{opacity:1}75%,100%{opacity:0}}",
        ".rp{animation:rp 6s steps(1) infinite;opacity:0}",
        "@keyframes rp{0%{opacity:0}40%{opacity:.9}55%{opacity:.5}70%,100%{opacity:0}}",
        "@media (prefers-reduced-motion:reduce){*{animation:none!important}.fa{opacity:1}.fb,.fc{opacity:0}}",
    ]

    fishes = [
        # variety, length, row, direction, seconds per crossing, start offset (0-1), bob rows
        ("kohaku", 18, 1, 1, 34, 0.15, 2),
        ("showa", 15, 7, -1, 29, 0.55, 2),
        ("yamabuki", 13, 3, -1, 41, 0.05, 3),
        ("sanke", 16, 8, 1, 38, 0.7, 1),
        ("kohaku", 11, 4, 1, 26, 0.45, 2),
    ]
    for i, (_, length, _, d, _, _, bob) in enumerate(fishes):
        travel = (COLS + length + 4) * CW
        out.append(
            f"@keyframes s{i}{{from{{transform:translateX({-length * CW if d > 0 else W}px)}}"
            f"to{{transform:translateX({W if d > 0 else -length * CW}px)}}}}"
        )
        out.append(
            f"@keyframes b{i}{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY({bob * CH}px)}}}}"
        )
    out.append("</style>")

    # static water: a faint dotted grid
    dots = []
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                dots.append(f"M{c * CW + CW / 2},{r * CH + CH / 2}h.01")
    out.append(f'<path d="{"".join(dots)}" stroke="{t["dot"]}" stroke-width="1.6" stroke-linecap="round"/>')

    # a few ripples that come and go
    for k in range(9):
        c, r = rng.randrange(2, COLS - 2), rng.randrange(1, ROWS - 1)
        delay = rng.uniform(0, 6)
        out.append(
            f'<text class="rp" style="animation-delay:-{delay:.2f}s" x="{c * CW + CW / 2}" y="{r * CH + CH - 3}" fill="{t["ripple"]}">~</text>'
        )

    for i, (variety, length, row, d, secs, offset, bob) in enumerate(fishes):
        frng = random.Random(100 + i)
        steps_x = COLS + length
        frames = []
        for wig in (-1, 0, 1):
            frames.append(make_fish(length, variety, random.Random(100 + i), wig))
        # keep patterns stable across frames: frames share a seed
        out.append(
            f'<g style="animation:s{i} {secs}s steps({steps_x}) infinite;animation-delay:-{offset * secs:.2f}s">'
        )
        out.append(
            f'<g style="animation:b{i} {secs / 2.3:.2f}s steps({bob * 2}) infinite;animation-delay:-{frng.uniform(0, 5):.2f}s">'
        )
        for cls, cells in zip(("fa", "fb", "fc"), frames):
            parts = []
            for (c, r), (ck, glyph) in cells.items():
                x = c if d > 0 else (length - 1 - c)
                parts.append(tile(x, r + row - 1, ck, glyph, theme))
            out.append(f'<g class="{cls}" style="animation-delay:-{i * 0.17:.2f}s">{"".join(parts)}</g>')
        out.append("</g></g>")

    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    here = Path(__file__).parent
    for theme in THEMES:
        svg = build(theme)
        (here / f"pond-{theme}.svg").write_text(svg)
        print(theme, f"{len(svg) / 1024:.0f} KB")
