"""Rough text-overflow check for the generated deck (no renderer available).

    python3 check_layout.py CDNA3_GEMM_FlyDSL.pptx
Estimates wrapped line count per paragraph (CJK char ~ 1.0 em, ASCII ~ 0.55 em)
and flags text boxes whose estimated height runs past their box or the slide.
"""
import math
import sys

from pptx import Presentation
from pptx.util import Emu

EMU_PER_PT = 12700


def est_height(tf, width_pt):
    h = 0.0
    for p in tf.paragraphs:
        text = "".join(r.text for r in p.runs)
        size = max((r.font.size.pt for r in p.runs if r.font.size), default=18)
        w = sum(size if ord(ch) > 0x2E80 else size * 0.55 for ch in text)
        lines = max(1, math.ceil(w / max(1, width_pt - 14)))
        h += lines * size * 1.2 + 4
    return h


prs = Presentation(sys.argv[1])
SH = prs.slide_height / EMU_PER_PT
bad = 0
for i, s in enumerate(prs.slides, 1):
    for shp in s.shapes:
        if shp.shape_type == 13:  # picture
            if (shp.top + shp.height) / EMU_PER_PT > SH + 1 or (shp.left + shp.width) > prs.slide_width:
                bad += 1
                print(f"slide {i}: picture runs off the slide")
            continue
        if not shp.has_text_frame or not shp.text_frame.text.strip():
            continue
        w = shp.width / EMU_PER_PT
        top = shp.top / EMU_PER_PT
        h = est_height(shp.text_frame, w)
        box = shp.height / EMU_PER_PT
        if top + h > SH - 5 or h > box * 1.15 + 20:
            bad += 1
            print(f"slide {i}: '{shp.text_frame.text[:40]}...' est {h:.0f}pt, box {box:.0f}pt, "
                  f"bottom {top + h:.0f}/{SH:.0f}pt")
print("flagged", bad)
