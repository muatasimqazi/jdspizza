"""3840x1080 two-column menu images in the approved style:
food on the left half (left TV), name / price / short line on the right half (right TV).

usage: python3 wide.py OUT_DIR [slug ...]
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402
from items import ITEMS  # noqa: E402
from panels import LINE, best_wrap  # noqa: E402

W, H = 3840, 1080
HALF = W // 2
SEAM_L, SEAM_R = render.SEAM_L, render.SEAM_R  # center 10%: 1728..2112, background only
MX, MY = round(W * 0.05), round(H * 0.08)  # 192 x 86

CREAM, DESC, ACCENT = render.CREAM, render.DESC, render.ACCENT
WDTH = 92
NAME_SIZE = 150
PRICE_SIZE = 290
LINE_SIZE = 54

TEXT_CX = HALF + HALF // 2  # 2880, centered on the right TV
TEXT_W = 1400  # 2180 .. 3580


def font(size):
    return render.bricolage(size, 800, WDTH)


def tw(f, s):
    b = f.getbbox(s, anchor="ls")
    return b[2] - b[0]


def cap_h(f):
    return -f.getbbox("H", anchor="ls")[1]


def name_lines(name):
    words = name.split()
    for size in [NAME_SIZE] + list(range(NAME_SIZE - 6, 89, -6)):
        f = font(size)
        for k in range(1, min(3, len(words)) + 1):
            w, lines = best_wrap(words, f, TEXT_W, k)
            if w <= TEXT_W:
                return lines, size
    raise ValueError(name)


def price_parts(price, size):
    big, small = font(size), font(int(size * 0.5))
    s, parts = price, []
    if s.startswith("$"):
        parts.append(("$", small))
        s = s[1:]
    tail = ""
    if s.endswith("+"):
        tail, s = "+", s[:-1]
    parts.append((s, big))
    if tail:
        parts.append((tail, small))
    return parts, big


def draw_price(d, price, base_y):
    parts, big = price_parts(price, PRICE_SIZE)
    gap = int(PRICE_SIZE * 0.03)
    total = sum(tw(f, t) for t, f in parts) + gap * (len(parts) - 1)
    x = TEXT_CX - total / 2
    ch = cap_h(big)
    for t, f in parts:
        if f is big:
            d.text((x, base_y), t, font=f, fill=ACCENT + (255,), anchor="ls")
        else:
            top = -f.getbbox(t, anchor="ls")[1]
            d.text((x, base_y - ch + top), t, font=f, fill=ACCENT + (255,), anchor="ls")
        x += tw(f, t) + gap


def draw_text(canvas, name, price, line):
    lines, nsize = name_lines(name)
    nf, pf = font(nsize), font(PRICE_SIZE)
    lf = render.plex(LINE_SIZE, 500)
    if line:
        assert tw(lf, line) <= TEXT_W, line

    ys, y = [], cap_h(nf)
    for _ in lines:
        ys.append(y)
        y += int(nsize * 1.02)
    p_base = ys[-1] + 80 + cap_h(pf)
    bottom = p_base
    l_base = None
    if line:
        l_base = p_base + 84 + cap_h(lf)
        bottom = l_base
    off = (H - bottom) / 2 - 6

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for ln, by in zip(lines, ys):
        d.text((TEXT_CX, by + off), ln, font=nf, fill=CREAM + (255,), anchor="ms")
    draw_price(d, price, p_base + off)
    if line:
        d.text((TEXT_CX, l_base + off), line, font=lf, fill=DESC + (255,), anchor="ms")

    bb = layer.getbbox()
    assert bb[0] >= SEAM_R + 40 and bb[2] <= W - MX, (name, bb)
    assert bb[1] >= MY and bb[3] <= H - MY, (name, bb)

    a = layer.getchannel("A")
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    m = Image.new("L", canvas.size, 0)
    m.paste(a, (0, 6))
    sh.putalpha(m.filter(ImageFilter.GaussianBlur(12)).point(lambda v: int(v * 0.6)))
    canvas.alpha_composite(sh)
    canvas.alpha_composite(layer)
    return bb


def render_item(item, out_dir, bg):
    slug, photo, name, _desc, price = item
    canvas = bg.copy()
    if photo in render.PHOTO_MODE:
        render.place_photo(canvas, photo)
    else:
        render.place_cutout(canvas, photo)  # asserts the food stays left of the seam zone
    draw_text(canvas, name, price, LINE.get(slug, ""))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{slug}.jpg")
    canvas.convert("RGB").save(path, quality=92, subsampling=0, optimize=True)
    return path


if __name__ == "__main__":
    out = sys.argv[1]
    want = set(sys.argv[2:])
    bg = render.background()
    for it in ITEMS:
        if not want or it[0] in want:
            print(render_item(it, out, bg), flush=True)
