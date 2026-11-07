"""Render 1920x1080 menu panels: food fills two-thirds, name / price / short line on the other third.

Two mirrored variants per item so any item can go on either TV:
  left-tv   text in the outer-left third, food in the right two-thirds (toward the center)
  right-tv  food in the left two-thirds (toward the center), text in the outer-right third

usage: python3 panels.py OUT_DIR [slug ...]
"""
import itertools
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from items import ITEMS  # noqa: E402
from render import SRC, PHOTO_MODE, CREAM, DESC, ACCENT, bricolage, plex, get_cutout, grade  # noqa: E402

W, H = 1920, 1080
MX, MY = round(W * 0.08), round(H * 0.08)  # 8% text margins: 154 x 86
THIRD = W // 3  # 640
GAP = 10
WDTH = 85  # slightly condensed: the text column is narrow

# Short line under the price (Toast description trimmed, or piece count). Empty = none.
LINE = {
    "01-mutton-kebab-pizza": "Mutton kabab, red pepper",
    "02-meaty-premier-pizza": "Beef, pepperoni, olives",
    "03-pizza-de-pepperoni": "Beef pepperoni, olive oil",
    "04-exotic-tropical-hawaiian-pizza": "Beef, pineapple, jalapeño",
    "05-chicken-kebab-pizza": "Chicken kabab, peppers",
    "06-polo-fajita-pizza": "Chicken fajita, peppers",
    "07-barbecued-chicken-pizza": "BBQ chicken, peppers, onion",
    "08-veg-supreme-pizza": "Peppers, mushrooms, olives",
    "09-tandoori-paneer-pizza": "Tandoori paneer, peppers",
    "10-jds-seafood-pizza": "Prawns, peppers, onion",
    "11-margherita-pizza": "Thin base, cheese, olive oil",
    "12-jds-italian-pizza": "Thin base, beef, chicken",
    "14-garlic-bread": "4 pieces",
    "15-garlic-bread-with-cheese": "4 pieces",
    "16-cheesy-mozzarella-sticks": "6 pieces",
    "19-spicy-potato-wedges": "Oven baked",
    "20-crispy-onion-rings": "8 pieces",
    "21-marinated-chicken-wings": "6 pieces",
    "22-crispy-chicken-strips": "6 pieces",
    "23-chicken-nuggets": "8 pieces",
}


def layout(side):
    if side == "left-tv":
        return dict(food_cx=1280, text_l=MX, text_r=THIRD - GAP)
    return dict(food_cx=640, text_l=2 * THIRD + GAP, text_r=W - MX)


def font(size):
    return bricolage(size, 800, WDTH)


def tw(f, s):
    b = f.getbbox(s, anchor="ls")
    return b[2] - b[0]


def cap_h(f):
    return -f.getbbox("H", anchor="ls")[1]


# ---------------------------------------------------------------- background
def background(food_cx, text_cx, seed=11):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.zeros((H, W, 3), np.float32)
    base[:] = (22, 20, 18)
    lo = rng.normal(0, 1, (27, 48)).astype(np.float32)
    lo = np.array(Image.fromarray(lo).resize((W, H), Image.BICUBIC))
    lo = ndimage.gaussian_filter(lo, 30)
    lo = lo / (np.abs(lo).max() + 1e-6)
    base += lo[..., None] * 5.0

    def pool(cx, cy, rx, ry, k, color):
        d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
        return (np.exp(-d * 1.6) * k)[..., None] * np.array(color, np.float32)

    base += pool(food_cx, 520, 900, 700, 1.0, (70, 40, 16))
    base += pool(text_cx, 540, 520, 700, 0.3, (52, 32, 14))
    vx = np.minimum(x, W - 1 - x) / 380.0
    vy = np.minimum(y, H - 1 - y) / 300.0
    v = np.clip(np.minimum(vx, vy), 0, 1)
    base *= (0.55 + 0.45 * (1 - (1 - v) ** 2))[..., None]
    base += rng.normal(0, 2.2, (H, W, 1)).astype(np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


# ---------------------------------------------------------------- food
def place_cutout(canvas, photo, cx):
    food = grade(get_cutout(photo))
    box_w, box_h = 1140, 880
    s = min(box_w / food.width, box_h / food.height)
    food = food.resize((round(food.width * s), round(food.height * s)), Image.LANCZOS)
    food = food.filter(ImageFilter.UnsharpMask(radius=2, percent=40, threshold=2))
    x0, y0 = cx - food.width // 2, 548 - food.height // 2
    a = food.getchannel("A")
    for blur, dy, op in ((44, 34, 0.75), (13, 11, 0.55)):
        sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        m = Image.new("L", canvas.size, 0)
        m.paste(a, (x0, y0 + dy))
        sh.putalpha(m.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: int(v * op)))
        canvas.alpha_composite(sh)
    canvas.alpha_composite(food, (x0, y0))
    return (x0, x0 + food.width)


def place_photo(canvas, photo, side):
    im = grade(Image.open(os.path.join(SRC, photo)).convert("RGBA"))
    rw, rh = 2 * THIRD, H
    s = max(rw / im.width, rh / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - rw) // 2, (im.height - rh) // 2
    im = im.crop((l, t, l + rw, t + rh))
    xs = np.arange(rw, dtype=np.float32)
    ys = np.arange(rh, dtype=np.float32)
    toward_text = (rw - xs) if side == "right-tv" else xs  # distance to the text side
    fx = np.clip(toward_text / 380.0, 0, 1) ** 1.6
    fy = np.clip(np.minimum(ys, rh - 1 - ys) / 120.0, 0, 1)
    im.putalpha(Image.fromarray(((fx[None, :] * fy[:, None]) * 255).astype(np.uint8), "L"))
    x0 = 0 if side == "right-tv" else W - rw
    canvas.alpha_composite(im, (x0, 0))
    return (x0, x0 + rw)


# ---------------------------------------------------------------- text
def best_wrap(words, f, max_w, k):
    best = None
    for cuts in itertools.combinations(range(1, len(words)), k - 1):
        idx = (0,) + cuts + (len(words),)
        lines = [" ".join(words[idx[i]:idx[i + 1]]) for i in range(k)]
        w = max(tw(f, ln) for ln in lines)
        if best is None or w < best[0]:
            best = (w, lines)
    return best


NAME_SIZE = 92


def name_lines(name, max_w):
    """One name size everywhere; wrap to <= 3 balanced lines; shrink only if that's impossible."""
    words = name.split()
    for size in [NAME_SIZE] + list(range(NAME_SIZE - 4, 59, -4)):
        f = font(size)
        for k in range(1, min(3, len(words)) + 1):
            w, lines = best_wrap(words, f, max_w, k)
            if w <= max_w:
                return lines, size
    raise ValueError(name)


PRICE_SIZE = 184  # one size for every panel so prices read consistently


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


def price_width(price, size):
    parts, _ = price_parts(price, size)
    gap = int(size * 0.03)
    return sum(tw(f, t) for t, f in parts) + gap * (len(parts) - 1)


def draw_price(d, price, cx, base_y, size):
    parts, big = price_parts(price, size)
    gap = int(size * 0.03)
    x = cx - price_width(price, size) / 2
    ch = cap_h(big)
    for t, f in parts:
        if f is big:
            d.text((x, base_y), t, font=f, fill=ACCENT + (255,), anchor="ls")
        else:
            top = -f.getbbox(t, anchor="ls")[1]
            d.text((x, base_y - ch + top), t, font=f, fill=ACCENT + (255,), anchor="ls")
        x += tw(f, t) + gap


def line_layout(text, max_w):
    if not text:
        return [], None
    for size in (40, 38, 36):
        f = plex(size, 500)
        if tw(f, text) <= max_w:
            return [text], f
    raise ValueError(f"short line too long for one line: {text!r}")


def draw_text(canvas, name, price, line, text_l, text_r):
    max_w = text_r - text_l
    cx = (text_l + text_r) / 2
    lines, nsize = name_lines(name, max_w)
    nf = font(nsize)
    assert price_width(price, PRICE_SIZE) <= max_w, (name, price)
    pf = font(PRICE_SIZE)
    sub, lf = line_layout(line, max_w)

    ys, y = [], cap_h(nf)
    for _ in lines:
        ys.append(y)
        y += int(nsize * 1.02)
    y = ys[-1] + 58 + cap_h(pf)
    p_base = y
    sub_ys = []
    if sub:
        y += 62 + cap_h(lf)
        for _ in sub:
            sub_ys.append(y)
            y += int(lf.size * 1.3)
        bottom = sub_ys[-1]
    else:
        bottom = p_base
    off = (H - bottom) / 2 - 4

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for ln, by in zip(lines, ys):
        d.text((cx, by + off), ln, font=nf, fill=CREAM + (255,), anchor="ms")
    draw_price(d, price, cx, p_base + off, PRICE_SIZE)
    for ln, by in zip(sub, sub_ys):
        d.text((cx, by + off), ln, font=lf, fill=DESC + (255,), anchor="ms")

    bb = layer.getbbox()
    assert bb[0] >= text_l - 1 and bb[2] <= text_r + 1, (name, bb)
    assert bb[0] >= MX and bb[2] <= W - MX and bb[1] >= MY and bb[3] <= H - MY, (name, bb)

    a = layer.getchannel("A")
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    m = Image.new("L", canvas.size, 0)
    m.paste(a, (0, 5))
    sh.putalpha(m.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.6)))
    canvas.alpha_composite(sh)
    canvas.alpha_composite(layer)
    return bb


def render(item, side, out_dir):
    slug, photo, name, _desc, price = item
    L = layout(side)
    text_cx = (L["text_l"] + L["text_r"]) / 2
    canvas = background(L["food_cx"], text_cx)
    if photo in PHOTO_MODE:
        fx = place_photo(canvas, photo, side)
    else:
        fx = place_cutout(canvas, photo, L["food_cx"])
    bb = draw_text(canvas, name, price, LINE.get(slug, ""), L["text_l"], L["text_r"])
    if photo not in PHOTO_MODE:  # food and text never overlap
        assert fx[1] <= bb[0] or fx[0] >= bb[2], (slug, side, fx, bb)
    os.makedirs(os.path.join(out_dir, side), exist_ok=True)
    path = os.path.join(out_dir, side, f"{slug}.jpg")
    canvas.convert("RGB").save(path, quality=92, subsampling=0, optimize=True)
    return path


if __name__ == "__main__":
    out = sys.argv[1]
    want = set(sys.argv[2:])
    for it in ITEMS:
        if want and it[0] not in want:
            continue
        for side in ("left-tv", "right-tv"):
            print(render(it, side, out), flush=True)
