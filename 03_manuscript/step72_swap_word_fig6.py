# -*- coding: utf-8 -*-
"""
step72_swap_word_fig6.py — put the step65-fixed Figure 6 composite into the
manuscript.

Root cause of "Figure 6F still not aligned with E": step65 rebuilt
Fig8_SHAP_stability (1x2 -> 2x1) and Figure6_SHAP.png, and the deck was
updated, but the manuscript still embedded the pre-step65 composite
(5696 x 7636, horizontal F panel with a large white gap under it).

Fix
* replace the image part at paragraph 72 with the current
  Figure6_SHAP.png (5644 x 7577, F flush with E, bottoms within 20 px
  at 400 dpi ~= 1.6 pt on the page);
* keep the displayed width (454 pt) and recompute height from the true
  aspect so nothing is stretched;
* sync the _v90 mirror copy.

Outputs : manuscript .docx (main + _v90 sync copy)
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import docx
from docx.oxml.ns import qn
from PIL import Image

PROJ = Path(r"D:/BJND-4002-11项目/BJND-4002-11")
MAIN = PROJ / "NOAF_stacking_manuscript_EndNote_revised.docx"
MIRROR = PROJ / "NOAF_stacking_manuscript_EndNote_revised_v90.docx"
PNG = Path(r"<LOCAL_ROOT>/WorkBuddy/2026-09-09-17-02-31"
           r"/output/rev_noaf/v90_primary/figures/Figure6_SHAP.png")
TMP = PROJ / "_tmp_fig6_fixed.docx"

PARA_IDX = 72
DISPLAY_W_PT = 454.0


def swap(path: Path) -> None:
    d = docx.Document(str(path))
    para = d.paragraphs[PARA_IDX]
    blips = para._p.findall(".//" + qn("a:blip"))
    assert len(blips) == 1, f"expected 1 blip in para {PARA_IDX}, got {len(blips)}"
    blip = blips[0]
    old_rid = blip.get(qn("r:embed"))
    old_part = d.part.rels[old_rid].target_part
    old_im = Image.open(__import__("io").BytesIO(old_part.blob))
    print(f"old embed: rId={old_rid} {old_im.size} "
          f"{len(old_part.blob)//1024} KB")

    w_px, h_px = Image.open(PNG).size
    aspect = w_px / h_px
    w_emu = int(DISPLAY_W_PT / 72 * 914400)
    h_emu = int(DISPLAY_W_PT / aspect / 72 * 914400)

    rId, _img = d.part.get_or_add_image(str(PNG))
    blip.set(qn("r:embed"), rId)

    for tag, val in ((qn("wp:extent"), (w_emu, h_emu)),):
        ext = para._p.findall(".//" + tag)
        if ext:
            ext[0].set("cx", str(val[0]))
            ext[0].set("cy", str(val[1]))
    for ext in para._p.findall(".//" + qn("a:ext")):
        ext.set("cx", str(w_emu))
        ext.set("cy", str(h_emu))

    # drop the old rel if no other blip uses it
    used = set()
    for p in d.paragraphs:
        for b in p._p.findall(".//" + qn("a:blip")):
            used.add(b.get(qn("r:embed")))
    for hdr in d.sections:
        for cont in (hdr.header, hdr.footer):
            for p in cont.paragraphs:
                for b in p._p.findall(".//" + qn("a:blip")):
                    used.add(b.get(qn("r:embed")))
    if old_rid not in used and old_rid != rId:
        d.part.drop_rel(old_rid)
        print(f"dropped orphan rel {old_rid}")

    d.save(str(TMP))
    print(f"new embed: rId={rId} {w_px}x{h_px}  "
          f"display {DISPLAY_W_PT:.0f}x{DISPLAY_W_PT/aspect:.1f} pt")


swap(MAIN)
for i in range(8):
    try:
        shutil.copyfile(TMP, MAIN)
        print("[OK] main overwritten")
        break
    except PermissionError:
        print("main locked, retry", i)
        time.sleep(3)
else:
    raise SystemExit("[FAIL] main still locked")

for i in range(8):
    try:
        shutil.copyfile(TMP, MIRROR)
        print("[OK] _v90 mirror synced")
        break
    except PermissionError:
        print("mirror locked, retry", i)
        time.sleep(3)
else:
    print("[WARN] mirror still locked — sync later")
TMP.unlink(missing_ok=True)

# quick verification
d = docx.Document(str(MAIN))
bl = d.paragraphs[PARA_IDX]._p.findall(".//" + qn("a:blip"))[0]
part = d.part.rels[bl.get(qn("r:embed"))].target_part
import hashlib
expect = hashlib.md5(PNG.read_bytes()).hexdigest()
got = hashlib.md5(part.blob).hexdigest()
print("verify md5:", "OK" if got == expect else "MISMATCH")
print("EndNote fields:",
      sum(1 for p in d.paragraphs
          for f in p._p.findall(qn("w:fldChar"))
          if f.get(qn("w:fldCharType")) == "begin"), "begins")
