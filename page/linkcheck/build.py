# Builds index.html (E2E Link Check page) from the counts of e2e_linkacc.py (2026-10-16).
N = 4185
rows = [
    ("A", "Oracle: annotated part boxes + annotated text boxes", "92.9 %", [("right", 3888), ("notright", 297)]),
    ("B", "Detector part boxes + annotated text boxes", "87.5 %", [("right", 2991), ("wrong", 142), ("none", 286), ("ocr", 3), ("det", 763)]),
    ("C", "Detector part boxes + all OCR boxes (the Claude-read version)", "52.2 %", [("right", 1635), ("wrong", 518), ("none", 982), ("ocr", 287), ("det", 763)]),
    ("D", "Detector part boxes + OCR boxes that PARSeq reads as a designator", "84.9 %", [("right", 2065), ("wrong", 90), ("none", 277), ("ocr", 990), ("det", 763)]),
]
X0, BW, RH, TOP = 8, 664, 74, 8
svg = [f'<svg viewBox="0 0 680 {TOP+len(rows)*RH+4}" role="img" aria-label="Link accuracy of the same linker on 4,185 labelled parts: 92.9 % with annotated boxes, 87.5 % of found parts with detector boxes, 52.2 % with all OCR boxes, 84.9 % with designator-only OCR boxes.">']
for r, (key, label, acc, segs) in enumerate(rows):
    y = TOP + r*RH
    svg.append(f'<text x="{X0}" y="{y+14}" class="rk">{key}</text><text x="{X0+18}" y="{y+14}" class="rl">{label}</text>')
    svg.append(f'<text x="{X0+BW}" y="{y+14}" class="ra" text-anchor="end">{acc}</text>')
    x = X0
    for kind, n in segs:
        w = BW*n/N
        svg.append(f'<rect x="{x:.1f}" y="{y+22}" width="{max(w,0.8):.1f}" height="26" class="s-{kind}"><title>{kind}: {n:,}</title></rect>')
        if w > 34: svg.append(f'<text x="{x+w/2:.1f}" y="{y+39.5}" text-anchor="middle" class="sv sv-{kind}">{n:,}</text>')
        x += w
svg.append('</svg>')
SVG = "\n".join(svg)
html = open("template.html").read().replace("@@SVG@@", SVG)
open("index.html", "w").write(html)
