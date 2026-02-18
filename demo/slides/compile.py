#!/usr/bin/env python3
"""
Compile individual slide HTML files into a single presentation.html.

Usage:
    python demo/slides/compile.py

Reads:
    _base.css, _engine.js, slide-*.html (sorted by filename)

Outputs:
    demo/presentation.html
"""

import glob
import os

SLIDES_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(os.path.dirname(SLIDES_DIR), "presentation.html")


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    # Read shared assets
    css = read_file(os.path.join(SLIDES_DIR, "_base.css"))
    js = read_file(os.path.join(SLIDES_DIR, "_engine.js"))

    # Glob slide files in order
    slide_files = sorted(glob.glob(os.path.join(SLIDES_DIR, "slide-*.html")))
    total = len(slide_files)

    if total == 0:
        print("ERROR: No slide-*.html files found in", SLIDES_DIR)
        return

    # Read all slide contents
    slides_html = []
    for sf in slide_files:
        content = read_file(sf)
        slides_html.append(f"<!-- {os.path.basename(sf)} -->\n{content}")

    all_slides = "\n\n".join(slides_html)

    # Build the final HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PR&amp;DW Grievance Portal — Demo Presentation</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" rel="stylesheet">
<style>
{css}
</style>
</head>
<body>

<!-- ===== SLIDE DECK (1920x1080) ===== -->
<div class="deck" id="deck">

<!-- ===== PROGRESS BAR ===== -->
<div class="progress-track"><div class="progress-fill" id="progressFill" style="width:{100/total:.2f}%"></div></div>

{all_slides}

<!-- ===== FOOTER BAR ===== -->
<div class="footer-bar">
  <div class="govt">
    <span class="icon filled">account_balance</span>
    Panchayati Raj &amp; Drinking Water Department &middot; Government of Odisha
  </div>
  <div class="counter" id="counter">1 / {total}</div>
</div>

<!-- ===== NAVIGATION ===== -->
<div class="nav-hint">
  <button class="nav-btn" id="btnPrev" title="Previous (Left Arrow)"><span class="icon">chevron_left</span></button>
  <button class="nav-btn" id="btnNext" title="Next (Right Arrow)"><span class="icon">chevron_right</span></button>
</div>

</div><!-- /deck -->

<!-- ===== PRESENTATION ENGINE ===== -->
<script>
{js}
</script>
</body>
</html>"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Compiled {total} slides -> {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE):,} bytes")


if __name__ == "__main__":
    main()
