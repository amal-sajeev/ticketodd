#!/usr/bin/env python3
"""Capture screenshots of presentation slides for visual inspection."""
import asyncio
import os
import sys

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing playwright...")
    os.system(f"{sys.executable} -m pip install playwright -q")
    os.system(f"{sys.executable} -m playwright install chromium")
    from playwright.async_api import async_playwright

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "slide_screenshots")
# Use file URL for reliability (works without server)
URL = "file:///" + os.path.join(os.path.dirname(__file__), "presentation.html").replace("\\", "/")


async def capture_slides(start=1, count=6):
    """Capture slides starting at 'start' (1-based), for 'count' slides."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        try:
            await page.goto(URL, wait_until="networkidle", timeout=10000)
        except Exception as e:
            print(f"ERROR: Could not connect to {URL}. Is the server running?")
            print(f"Start with: cd demo && python -m uvicorn ticketer:app --reload")
            print(f"Details: {e}")
            await browser.close()
            return []

        # Advance to start slide (press right N-1 times from slide 1)
        # Wait 600ms after each advance to allow 550ms slide transition to complete
        for _ in range(start - 1):
            await page.keyboard.press("ArrowRight")
            await page.wait_for_timeout(600)

        paths = []
        for i in range(count):
            await page.wait_for_timeout(800)  # Let animations settle
            slide_num = start + i
            path = os.path.join(OUTPUT_DIR, f"slide-{slide_num:02d}.png")
            await page.screenshot(path=path)
            paths.append(path)
            if i < count - 1:
                await page.keyboard.press("ArrowRight")
                await page.wait_for_timeout(600)
        await browser.close()
        return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="First slide (1-based)")
    parser.add_argument("--count", type=int, default=6, help="Number of slides")
    args = parser.parse_args()
    paths = asyncio.run(capture_slides(args.start, args.count))
    if paths:
        print("Captured slides:")
        for p in paths:
            print(f"  {p}")
