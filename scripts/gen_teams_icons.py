"""Generate Microsoft Teams app icons for Cyprus AI.

Creates:
  color.png   192x192  (filled, transparent background)
  outline.png 32x32    (transparent, white outline)
into the teams_app/ folder.
"""

import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "teams_app")

RED = (224, 0, 0, 255)
WHITE = (255, 255, 255, 255)


def _font(size: int):
    candidates = [
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _rounded_rect(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def _draw_canvas(size: int, background, letter_color, stroke=None, top=False):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = max(int(size * 0.06), 2)
    _rounded_rect(d, [pad, pad, size - pad, size - pad], int(size * 0.20), background)

    if stroke:
        d.rounded_rectangle(
            [pad, pad, size - pad, size - pad],
            radius=int(size * 0.20),
            outline=stroke,
            width=max(int(size * 0.02), 1),
        )

    font = _font(int(size * 0.52))
    letter = "C"
    bbox = d.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # Vertical optical centring: nudge slightly up by default
    y_off = int(size * 0.02 if top else size * 0.10)
    x = (size - w) / 2 - bbox[0]
    y = (size - h) / 2 - bbox[1] - y_off
    d.text((x, y), letter, font=font, fill=letter_color)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)

    # color.png — 192x192, red rounded square with white "C"
    color = _draw_canvas(192, RED, WHITE)
    color.save(os.path.join(OUT, "color.png"))

    # outline.png — 32x32, transparent bg, red filled square + white outline letter.
    # Microsoft requires outline icon to read clearly on light/dark backgrounds;
    # a common choice is a solid colour tile with a white glyph.
    outline = _draw_canvas(32, RED, WHITE)
    outline.save(os.path.join(OUT, "outline.png"))

    print(f"Wrote {OUT}/color.png (192x192) and {OUT}/outline.png (32x32)")


if __name__ == "__main__":
    main()
