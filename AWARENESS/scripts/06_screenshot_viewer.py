"""Headless screenshots of outputs/ghost_viewer.html in the 4 camera modes.

Uses Playwright driving an installed Chromium-based browser (Brave by default,
no browser download needed). WebGL runs on SwiftShader.

Usage: .venv/bin/python scripts/06_screenshot_viewer.py [clip_index] [frame_fraction]
Writes outputs/media/screenshots/clip{i}_{camera}.png (git-ignored).
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

AW = Path(__file__).resolve().parents[1]
BROWSER = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
clip_i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
out_dir = AW / "outputs" / "media" / "screenshots"


async def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=BROWSER, headless=True,
                                    args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = await b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("pageerror", lambda e: print("JS error:", e))
        await pg.goto((AW / "outputs" / "ghost_viewer.html").as_uri())
        await pg.wait_for_timeout(3000)
        await pg.evaluate(f"loadClip({clip_i}); setFrame(Math.floor(clip.frames.length*{frac}))")
        await pg.wait_for_timeout(800)
        for cam in ["broadcast", "top", "follow", "pov"]:
            await pg.click(f"button[data-cam={cam}]")
            await pg.wait_for_timeout(800)
            await pg.screenshot(path=str(out_dir / f"clip{clip_i}_{cam}.png"))
        await b.close()
    print("screenshots ->", out_dir)

asyncio.run(main())
