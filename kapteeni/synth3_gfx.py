"""synth3 graphics core: styles, canvases, drawing helpers, row builders.

Shared by every synth3 family generator. The anti-narrow-template lesson
from synth2, applied to pixels: each render randomizes palette, font
faces, margins, and alignment within a strict readability envelope, and
every row's gold comes from the facts dict that drove the render.
"""

from __future__ import annotations

import random
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(f"{FONT_DIR}/{name}", size)


FACES = {  # (heading, body, small) font files
    "sans": ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "DejaVuSans.ttf"),
    "serif": ("DejaVuSerif-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSerif.ttf"),
    "mono": ("DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSansMono.ttf"),
}

# readable background/ink pairs; accents chosen for contrast on both
PALETTES = [
    ("#ffffff", "#101010"), ("#f7f4ec", "#26221a"), ("#eef2f6", "#131a22"),
    ("#fff8f0", "#1e1b16"), ("#f2f7f2", "#14200f"),
]
ACCENTS = ["#7a1f1f", "#1f4d7a", "#1f7a3f", "#6b3fa0", "#b06000", "#0f6070"]
DARK = [("#14161c", "#f0f0f0"), ("#1c1610", "#f5eede"), ("#0f1c14", "#e8f2e8")]
SIZE = 640


class Style:
    """One render's cosmetic identity."""

    def __init__(self, rng: random.Random, allow_dark: bool = True):
        if allow_dark and rng.random() < 0.2:
            self.bg, self.ink = rng.choice(DARK)
            self.dark = True
        else:
            self.bg, self.ink = rng.choice(PALETTES)
            self.dark = False
        self.accent = rng.choice(ACCENTS)
        face = FACES[rng.choice(list(FACES))]
        hs = rng.randint(44, 54)
        bs = rng.randint(26, 32)
        ss = rng.randint(20, 24)
        self.head = _font(face[0], hs)
        self.body = _font(face[1], bs)
        self.small = _font(face[2], ss)
        self.margin = rng.choice([36, 52, 68])
        self.centered_titles = rng.random() < 0.5

    def dim(self) -> str:  # secondary ink
        return self.accent if self.dark else "#888888"


def canvas(style: Style, title: str | None = None):
    img = Image.new("RGB", (SIZE, SIZE), style.bg)
    d = ImageDraw.Draw(img)
    if title:
        x = SIZE // 2 if style.centered_titles else style.margin
        d.text((x, 40), title, font=style.head, fill=style.accent,
               anchor="ma" if style.centered_titles else "la")
    return img, d


def leaders(d, x0, x1, y, style: Style):
    d.line((x0, y + 28, x1, y + 28), fill=style.dim(), width=2)


def checkbox(d, x, y, style: Style, checked: bool, s: int = 38):
    d.rectangle((x, y, x + s, y + s), outline=style.ink, width=3)
    if checked:
        d.line((x + 7, y + s * 0.55, x + s * 0.42, y + s - 6),
               fill=style.ink, width=5)
        d.line((x + s * 0.42, y + s - 6, x + s - 5, y + 5),
               fill=style.ink, width=5)


def star(d, x, y, style: Style, s: int = 34):
    """A check-like mark for attendance grids."""
    d.line((x, y + s * 0.5, x + s * 0.4, y + s), fill=style.accent, width=5)
    d.line((x + s * 0.4, y + s, x + s, y), fill=style.accent, width=5)


# ------------------------------------------------------------- row builders

def noul_row(rid, image, state_text, q, crit_true, crit_false, label, fam):
    assert label in (0, 1)
    return {"row_id": rid, "primitive": "noul", "image": image,
            "state_text": state_text, "instructions": q,
            "criteria": {"true": crit_true, "false": crit_false},
            "label": label, "meta": {"family": fam}}


def choice_row(rid, image, state_text, q, options, label, fam):
    assert len(set(options)) == len(options)
    assert 0 <= label < len(options)
    return {"row_id": rid, "primitive": "choice", "image": image,
            "state_text": state_text, "instructions": q, "criteria": None,
            "options": options, "label": label, "meta": {"family": fam}}


def score_row(rid, image, state_text, q, levels, label, fam):
    assert 0 <= label < len(levels)
    return {"row_id": rid, "primitive": "score", "image": image,
            "state_text": state_text, "instructions": q, "criteria": None,
            "levels": levels, "label": label, "meta": {"family": fam}}


def validate_rows(rows) -> None:
    for r in rows:
        assert r["meta"]["family"]
        assert r["image"].endswith(".png")
        if r["primitive"] == "noul":
            assert set(r["criteria"]) == {"true", "false"}
        elif r["primitive"] == "choice":
            assert r["options"] and r["label"] < len(r["options"])
        else:
            assert r["levels"] and r["label"] < len(r["levels"])