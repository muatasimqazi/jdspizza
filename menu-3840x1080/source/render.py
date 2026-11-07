"""Render 3840x1080 split menu slides: food on the left TV, text on the right TV.

usage: python3 render.py OUT_DIR [slug ...]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from items import ITEMS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/mnt/user-data/uploads/jdspizza/images"
CUT_DIR = os.path.join(HERE, "cut")
W, H = 3840, 1080
HALF = W // 2
MX, MY = int(W * 0.05), int(H * 0.05)  # 5% safe margins: 192 x 54
SEAM_L, SEAM_R = HALF - int(W * 0.05), HALF + int(W * 0.05)  # center 10%: 1728..2112

# Photos that already sit on a dark background and don't cut out cleanly.
PHOTO_MODE = {"pepperoni-pizza.jpg"}

CREAM = (255, 244, 226)
DESC = (233, 218, 196)
ACCENT = (255, 174, 43)

FONT_DIR = os.path.join(HERE, "fonts")


def bricolage(size, wght=800, wdth=100):
    f = ImageFont.truetype(os.path.join(FONT_DIR, "Bricolage.ttf"), size)
    f.set_variation_by_axes([min(96, max(12, size)), wght, wdth])
    return f


def plex(size, wght=500):
    f = ImageFont.truetype(os.path.join(FONT_DIR, "PlexSans.ttf"), size)
    f.set_variation_by_axes([wght, 100])
    return f


# ---------------------------------------------------------------- background
def background(seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.zeros((H, W, 3), np.float32)
    base[:] = (22, 20, 18)  # warm charcoal

    # low-frequency mottling (slate / charcoal)
    lo = rng.normal(0, 1, (27, 96)).astype(np.float32)
    lo = np.array(Image.fromarray(lo).resize((W, H), Image.BICUBIC))
    lo = ndimage.gaussian_filter(lo, 30)
    lo = lo / (np.abs(lo).max() + 1e-6)
    base += lo[..., None] * 5.0

    # warm light pooled behind the food, a softer pool behind the text
    def pool(cx, cy, rx, ry, strength, color):
        d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
        g = np.exp(-d * 1.6) * strength
        return g[..., None] * np.array(color, np.float32)

    base += pool(960, 520, 1150, 760, 1.0, (70, 40, 16))
    base += pool(2880, 540, 1300, 800, 0.35, (52, 32, 14))

    # vignette toward the outer edges only
    vx = np.minimum(x, W - 1 - x) / 520.0
    vy = np.minimum(y, H - 1 - y) / 300.0
    v = np.clip(np.minimum(vx, vy), 0, 1)
    v = 0.55 + 0.45 * (1 - (1 - v) ** 2)
    base *= v[..., None]

    # fine grain
    base += rng.normal(0, 2.2, (H, W, 1)).astype(np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


# ---------------------------------------------------------------- food
def get_cutout(photo):
    stem = photo.rsplit(".", 1)[0].strip()
    path = os.path.join(CUT_DIR, stem + ".png")
    if not os.path.exists(path):
        from rembg import new_session, remove

        global _SESSION
        try:
            _SESSION
        except NameError:
            _SESSION = new_session("isnet-general-use")
        im = Image.open(os.path.join(SRC, photo)).convert("RGB")
        remove(im, session=_SESSION).save(path)
    im = Image.open(path).convert("RGBA")
    a = np.array(im)[..., 3].astype(np.float32)

    # subtract the side sauce bowl (segmented separately with SAM, visible pixels only)
    bowl_path = os.path.join(HERE, "bowl", stem + ".png")
    if os.path.exists(bowl_path):
        bowl = np.array(Image.open(bowl_path).convert("L")) > 128
        # hull of the bowl (covers the sauce surface SAM sometimes misses), grown to take its shadow
        from scipy.spatial import ConvexHull

        pts = np.argwhere(bowl)[:, ::-1]
        hull = pts[ConvexHull(pts).vertices]
        hm = Image.new("L", (bowl.shape[1], bowl.shape[0]), 0)
        ImageDraw.Draw(hm).polygon([tuple(map(int, p)) for p in hull], fill=255)
        zone = ndimage.binary_dilation(np.array(hm) > 0, iterations=70)
        # food in front of the bowl is fully opaque in the cutout; the bowl ghost is not
        a[zone & (a < 240)] = 0
        a[ndimage.binary_dilation(bowl, iterations=3)] = 0

    a[a < 16] = 0
    # keep the largest blob (drops leftover bowl shadow, stray crumbs)
    lab, n = ndimage.label(a > 40)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(a), lab, range(1, n + 1))
        keep = lab == (int(np.argmax(sizes)) + 1)
        keep = ndimage.binary_dilation(keep, iterations=4)
        a = a * keep

    # where the photo crops the food at the frame edge, fade it out instead of a hard cut
    h, w = a.shape
    edges = {
        "top": (a[0] > 40).sum(),
        "bottom": (a[-1] > 40).sum(),
        "left": (a[:, 0] > 40).sum(),
        "right": (a[:, -1] > 40).sum(),
    }
    cropped = {k for k, v in edges.items() if v > 20}
    if cropped:
        f = 150.0 if len(cropped) >= 2 else 90.0
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        ramp = np.ones_like(a)
        if "top" in cropped:
            ramp *= np.clip(yy / f, 0, 1)
        if "bottom" in cropped:
            ramp *= np.clip((h - 1 - yy) / f, 0, 1)
        if "left" in cropped:
            ramp *= np.clip(xx / f, 0, 1)
        if "right" in cropped:
            ramp *= np.clip((w - 1 - xx) / f, 0, 1)
        a = a * (ramp ** 1.4)

    arr = np.array(im)
    arr[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, "RGBA")
    bb = Image.fromarray((arr[..., 3] > 8).astype(np.uint8) * 255).getbbox()
    return im.crop(bb)


def grade(im):
    """Gentle pop: a touch more contrast and saturation, slightly warm."""
    rgb, a = im.convert("RGB"), im.getchannel("A")
    rgb = ImageEnhance.Contrast(rgb).enhance(1.06)
    rgb = ImageEnhance.Color(rgb).enhance(1.08)
    r, g, b = rgb.split()
    r = r.point(lambda v: min(255, int(v * 1.02)))
    b = b.point(lambda v: int(v * 0.97))
    out = Image.merge("RGB", (r, g, b)).convert("RGBA")
    out.putalpha(a)
    return out


def place_cutout(canvas, photo):
    food = grade(get_cutout(photo))
    box_w, box_h = 1440, 900
    s = min(box_w / food.width, box_h / food.height)
    food = food.resize((round(food.width * s), round(food.height * s)), Image.LANCZOS)
    food = food.filter(ImageFilter.UnsharpMask(radius=2, percent=40, threshold=2))
    cx, cy = 960, 548
    x0, y0 = cx - food.width // 2, cy - food.height // 2
    assert x0 >= MX - 1 and x0 + food.width <= SEAM_L - 20, (photo, x0, food.width)
    assert y0 >= MY - 1 and y0 + food.height <= H - MY + 1, (photo, y0, food.height)

    a = food.getchannel("A")
    for blur, dy, op in ((48, 38, 0.75), (14, 12, 0.55)):
        sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        m = Image.new("L", canvas.size, 0)
        m.paste(a, (x0, y0 + dy))
        m = m.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: int(v * op))
        sh.putalpha(m)
        canvas.alpha_composite(sh)
    canvas.alpha_composite(food, (x0, y0))


def place_photo(canvas, photo):
    """Photo already on a dark ground: fill the left half and feather it out before the seam."""
    im = grade(Image.open(os.path.join(SRC, photo)).convert("RGBA"))
    region_w, region_h = SEAM_L - 40, H
    s = max(region_w / im.width, region_h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    left = (im.width - region_w) // 2
    top = (im.height - region_h) // 2
    im = im.crop((left, top, left + region_w, top + region_h))
    xs = np.arange(region_w, dtype=np.float32)
    fade_r = np.clip((region_w - xs) / 520.0, 0, 1) ** 1.6
    fade_l = np.clip(xs / 160.0, 0, 1)
    ys = np.arange(region_h, dtype=np.float32)
    fade_y = np.clip(np.minimum(ys, region_h - 1 - ys) / 140.0, 0, 1)
    mask = (fade_r * fade_l)[None, :] * fade_y[:, None]
    im.putalpha(Image.fromarray((mask * 255).astype(np.uint8), "L"))
    canvas.alpha_composite(im, (0, 0))


# ---------------------------------------------------------------- text
TEXT_CX = HALF + HALF // 2  # 2880
TEXT_MAX_W = 1440


def cap_h(font):
    return -font.getbbox("H", anchor="ls")[1]


def tw(font, s):
    b = font.getbbox(s, anchor="ls")
    return b[2] - b[0]


def wrap_name(name, size):
    f = bricolage(size)
    if tw(f, name) <= TEXT_MAX_W:
        return [name]
    words = name.split()
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        w = max(tw(f, a), tw(f, b))
        if best is None or w < best[0]:
            best = (w, [a, b])
    return best[1]


def name_lines(name):
    # one line if it fits at a reasonable size, otherwise two balanced lines
    for size in range(168, 135, -4):
        if tw(bricolage(size), name) <= TEXT_MAX_W:
            return [name], size
    size = 168
    while True:
        lines = wrap_name(name, size)
        f = bricolage(size)
        if all(tw(f, ln) <= TEXT_MAX_W for ln in lines):
            return lines, size
        size -= 4


def draw_price(layer, price, base_y, size):
    """'$' and '+' set smaller and raised, digits full size; centered at TEXT_CX."""
    d = ImageDraw.Draw(layer)
    big = bricolage(size)
    small = bricolage(int(size * 0.52))
    parts = []
    s = price
    if s.startswith("$"):
        parts.append(("$", small))
        s = s[1:]
    tail = ""
    if s.endswith("+"):
        tail, s = "+", s[:-1]
    parts.append((s, big))
    if tail:
        parts.append((tail, small))
    gap = int(size * 0.03)
    widths = [tw(f, t) for t, f in parts]
    total = sum(widths) + gap * (len(parts) - 1)
    x = TEXT_CX - total / 2
    ch_big = cap_h(big)
    for (t, f), w in zip(parts, widths):
        if f is big:
            d.text((x, base_y), t, font=f, fill=ACCENT + (255,), anchor="ls")
        else:
            # align the top of the small glyph with the cap height of the digits
            top = -f.getbbox(t, anchor="ls")[1]
            d.text((x, base_y - ch_big + top), t, font=f, fill=ACCENT + (255,), anchor="ls")
        x += w + gap
    return ch_big


def draw_text(canvas, name, desc, price):
    lines, nsize = name_lines(name)
    nf = bricolage(nsize)
    df = plex(64, 500)
    price_size = 300
    pf = bricolage(price_size)

    n_cap = cap_h(nf)
    n_lead = int(nsize * 1.04)
    d_cap = cap_h(df)
    p_cap = cap_h(pf)
    gap_nd, gap_dp, gap_np = 64, 92, 104

    # stack by baselines
    ys = []
    y = n_cap
    for i in range(len(lines)):
        ys.append(y)
        y += n_lead
    y = ys[-1]
    if desc:
        y += gap_nd + d_cap
        d_base = y
        y += gap_dp + p_cap
    else:
        d_base = None
        y += gap_np + p_cap
    p_base = y
    total = p_base
    off = (H - total) / 2 - 6
    assert off >= MY, (name, off)

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for ln, by in zip(lines, ys):
        d.text((TEXT_CX, by + off), ln, font=nf, fill=CREAM + (255,), anchor="ms")
    if desc:
        d.text((TEXT_CX, d_base + off), desc, font=df, fill=DESC + (255,), anchor="ms")
    draw_price(layer, price, p_base + off, price_size)

    bbox = layer.getbbox()
    assert bbox[0] >= SEAM_R + 20 and bbox[2] <= W - MX, (name, bbox)
    assert bbox[1] >= MY and bbox[3] <= H - MY, (name, bbox)

    # subtle shadow
    a = layer.getchannel("A")
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    m = Image.new("L", canvas.size, 0)
    m.paste(a, (0, 6))
    m = m.filter(ImageFilter.GaussianBlur(12)).point(lambda v: int(v * 0.6))
    sh.putalpha(m)
    canvas.alpha_composite(sh)
    canvas.alpha_composite(layer)
    return bbox


def render(item, out_dir, bg):
    slug, photo, name, desc, price = item
    canvas = bg.copy()
    if photo in PHOTO_MODE:
        place_photo(canvas, photo)
    else:
        place_cutout(canvas, photo)
    draw_text(canvas, name, desc, price)
    path = os.path.join(out_dir, slug + ".jpg")
    canvas.convert("RGB").save(path, quality=92, subsampling=0, optimize=True)
    return path


if __name__ == "__main__":
    out = sys.argv[1]
    want = set(sys.argv[2:])
    os.makedirs(out, exist_ok=True)
    os.makedirs(CUT_DIR, exist_ok=True)
    bg = background()
    for it in ITEMS:
        if want and it[0] not in want:
            continue
        print(render(it, out, bg), flush=True)
