"""Render README visuals from the bundled synthetic study.

Run: uv run --extra visuals python scripts/render_readme_assets.py
The generated images are checked in; Pillow is not a runtime dependency.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

from traceai.engine import Experiment

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
INK = "#07131e"
PANEL = "#0c1e2c"
LINE = "#244053"
WHITE = "#f3f7f8"
MUTED = "#9eb4bf"
CYAN = "#58dbc9"
AMBER = "#ffc16a"
BLUE = "#7aa8f7"


def study():
    with tempfile.TemporaryDirectory(prefix="traceai-readme-") as directory:
        experiment = Experiment.from_file(
            ROOT / "examples" / "controlled-study.yaml",
            project_dir=Path(directory) / ".traceai",
        )
        experiment.run()
        report = experiment.report()
    if report.config.target.runtime != "mock" or len(report.evidence) != 48:
        raise RuntimeError("The README demo must use the 48-record synthetic fixture")
    checkpoints = [item.id for item in report.config.checkpoints]
    values = {
        probe: [
            next(
                item.score
                for item in report.observations
                if item.probe == probe and item.checkpoint == checkpoint
            )
            for checkpoint in checkpoints
        ]
        for probe in report.config.probes
    }
    return checkpoints, values, report


def svg_points(values: list[float], xs: list[int], top: int, bottom: int) -> str:
    return " ".join(
        f"{x},{bottom - round((bottom - top) * score)}" for x, score in zip(xs, values, strict=True)
    )


def render_hero(checkpoints, values, count):
    xs = [866, 1005, 1144, 1283]
    reward = values["reward_hacking"]
    spec = values["specification_gaming"]
    grid = "".join(
        f'<path d="M{x} 0V480" stroke="#152b3a" stroke-width="1"/>' for x in range(0, 1401, 40)
    ) + "".join(
        f'<path d="M0 {y}H1400" stroke="#152b3a" stroke-width="1"/>' for y in range(0, 481, 40)
    )
    reward_dots = "".join(
        f'<circle cx="{x}" cy="{354 - round(190 * score)}" r="6" fill="{AMBER}" '
        f'stroke="{PANEL}" stroke-width="3"/>'
        for x, score in zip(xs, reward, strict=True)
    )
    spec_dots = "".join(
        f'<circle cx="{x}" cy="{354 - round(190 * score)}" r="6" fill="{CYAN}" '
        f'stroke="{PANEL}" stroke-width="3"/>'
        for x, score in zip(xs, spec, strict=True)
    )
    labels = "".join(
        f'<text x="{x}" y="390" text-anchor="middle" class="mono dim">{escape(cp)}</text>'
        for x, cp in zip(xs, checkpoints, strict=True)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="480" viewBox="0 0 1400 480" role="img" aria-labelledby="title desc">
<title id="title">TraceAI — make behavioral change visible</title>
<desc id="desc">A synthetic example with two failure-rate trajectories across four checkpoints. All values come from the bundled mock study.</desc>
<defs><clipPath id="round"><rect width="1400" height="480" rx="28"/></clipPath></defs>
<style>.sans{{font-family:Arial,Helvetica,sans-serif}}.mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}.dim{{fill:{MUTED};font-size:15px}}.tiny{{font-size:12px;letter-spacing:2px;font-weight:bold}}.axis{{stroke:{LINE};stroke-width:1}}</style>
<g clip-path="url(#round)"><rect width="1400" height="480" fill="{INK}"/>{grid}
<path d="M75 69h38l19 19-19 19H75L56 88z" fill="none" stroke="{CYAN}" stroke-width="3"/>
<path d="M76 88h36" stroke="{CYAN}" stroke-width="3" stroke-linecap="round"/>
<text x="157" y="95" class="mono tiny" fill="{CYAN}">TRACEAI / LOCAL RESEARCH INSTRUMENT</text>
<text x="76" y="211" class="sans" font-size="65" font-weight="700" fill="{WHITE}">Behavior has</text>
<text x="76" y="283" class="sans" font-size="65" font-weight="700" fill="{WHITE}">a trajectory<tspan fill="{CYAN}">.</tspan></text>
<text x="79" y="332" class="sans" font-size="22" fill="{MUTED}">Follow the evidence behind every checkpoint.</text>
<path d="M78 382H682" stroke="{LINE}" stroke-width="1"/>
<text x="78" y="415" class="mono tiny" fill="{CYAN}">01  OBSERVE</text>
<text x="282" y="415" class="mono tiny" fill="{WHITE}">02  COMPARE</text>
<text x="485" y="415" class="mono tiny" fill="{AMBER}">03  EXPLAIN</text>
<rect x="763" y="58" width="579" height="362" rx="19" fill="{PANEL}" stroke="{LINE}"/>
<circle cx="792" cy="88" r="4" fill="{CYAN}"/><text x="810" y="93" class="mono tiny" fill="{WHITE}">CHECKPOINT SIGNALS</text>
<rect x="1130" y="72" width="185" height="31" rx="15" fill="#173542"/>
<text x="1222" y="92" text-anchor="middle" class="mono" font-size="11" fill="{CYAN}">SYNTHETIC EXAMPLE</text>
<path d="M839 165H1298M839 260H1298M839 354H1298" class="axis"/>
<text x="814" y="170" text-anchor="end" class="mono dim">1.0</text>
<text x="814" y="265" text-anchor="end" class="mono dim">.5</text>
<text x="814" y="359" text-anchor="end" class="mono dim">0</text>
<polyline points="{svg_points(reward, xs, 164, 354)}" fill="none" stroke="{AMBER}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
<polyline points="{svg_points(spec, xs, 164, 354)}" fill="none" stroke="{CYAN}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
{reward_dots}{spec_dots}{labels}
<path d="M945 117h20" stroke="{AMBER}" stroke-width="4" stroke-linecap="round"/>
<text x="973" y="122" class="mono" font-size="13" fill="{WHITE}">reward</text>
<path d="M1087 117h20" stroke="{CYAN}" stroke-width="4" stroke-linecap="round"/>
<text x="1115" y="122" class="mono" font-size="13" fill="{WHITE}">specification</text>
<text x="765" y="452" class="mono" font-size="12" fill="{MUTED}">BUNDLED MOCK STUDY  /  {count} RAW RECORDS  /  NO MODEL CLAIMS</text>
</g></svg>'''
    (ASSETS / "traceai-hero.svg").write_text(svg, encoding="utf-8")


def render_mobile_hero(checkpoints, values, count):
    xs = [139, 292, 445, 598]
    top, bottom = 440, 642
    reward = values["reward_hacking"]
    spec = values["specification_gaming"]
    lines = "".join(
        f'<path d="M{x} 0V760" stroke="#152b3a" stroke-width="1"/>' for x in range(0, 721, 40)
    ) + "".join(
        f'<path d="M0 {y}H720" stroke="#152b3a" stroke-width="1"/>' for y in range(0, 761, 40)
    )
    dots = "".join(
        f'<circle cx="{x}" cy="{bottom - round((bottom - top) * score)}" '
        f'r="7" fill="{color}" stroke="{PANEL}" stroke-width="3"/>'
        for color, series in ((AMBER, reward), (CYAN, spec))
        for x, score in zip(xs, series, strict=True)
    )
    labels = "".join(
        f'<text x="{x}" y="679" text-anchor="middle" class="mono" font-size="16" '
        f'fill="{MUTED}">{escape(cp)}</text>'
        for x, cp in zip(xs, checkpoints, strict=True)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="760" viewBox="0 0 720 760" role="img" aria-labelledby="title desc">
<title id="title">TraceAI — make behavioral change visible</title>
<desc id="desc">Mobile cover with synthetic failure-rate trajectories across four checkpoints from the bundled mock study.</desc>
<defs><clipPath id="round"><rect width="720" height="760" rx="28"/></clipPath></defs>
<style>.sans{{font-family:Arial,Helvetica,sans-serif}}.mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}.axis{{stroke:{LINE};stroke-width:1}}</style>
<g clip-path="url(#round)"><rect width="720" height="760" fill="{INK}"/>{lines}
<path d="M53 61h34l17 17-17 17H53L36 78z" fill="none" stroke="{CYAN}" stroke-width="3"/>
<path d="M55 78h30" stroke="{CYAN}" stroke-width="3" stroke-linecap="round"/>
<text x="126" y="85" class="mono" font-size="15" font-weight="bold" letter-spacing="1" fill="{CYAN}">TRACEAI / LOCAL RESEARCH</text>
<text x="40" y="180" class="sans" font-size="65" font-weight="700" fill="{WHITE}">Behavior has</text>
<text x="40" y="251" class="sans" font-size="65" font-weight="700" fill="{WHITE}">a trajectory<tspan fill="{CYAN}">.</tspan></text>
<text x="43" y="303" class="sans" font-size="24" fill="{MUTED}">Follow the evidence behind every checkpoint.</text>
<rect x="34" y="344" width="652" height="373" rx="20" fill="{PANEL}" stroke="{LINE}"/>
<circle cx="62" cy="378" r="5" fill="{CYAN}"/>
<text x="81" y="384" class="mono" font-size="17" font-weight="bold" fill="{WHITE}">CHECKPOINT SIGNALS</text>
<text x="65" y="414" class="mono" font-size="12" fill="{CYAN}">SYNTHETIC EXAMPLE · {count} RAW RECORDS</text>
<path d="M117 440H636M117 541H636M117 642H636" class="axis"/>
<text x="104" y="445" text-anchor="end" class="mono" font-size="14" fill="{MUTED}">1.0</text>
<text x="104" y="546" text-anchor="end" class="mono" font-size="14" fill="{MUTED}">.5</text>
<text x="104" y="647" text-anchor="end" class="mono" font-size="14" fill="{MUTED}">0</text>
<polyline points="{svg_points(reward, xs, top, bottom)}" fill="none" stroke="{AMBER}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
<polyline points="{svg_points(spec, xs, top, bottom)}" fill="none" stroke="{CYAN}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
{dots}{labels}
<path d="M79 697h24" stroke="{AMBER}" stroke-width="4" stroke-linecap="round"/>
<text x="112" y="702" class="mono" font-size="14" fill="{WHITE}">reward</text>
<path d="M329 697h24" stroke="{CYAN}" stroke-width="4" stroke-linecap="round"/>
<text x="362" y="702" class="mono" font-size="14" fill="{WHITE}">specification</text>
<text x="39" y="746" class="mono" font-size="13" fill="{MUTED}">OBSERVE  /  COMPARE  /  EXPLAIN</text>
</g></svg>'''
    (ASSETS / "traceai-hero-mobile.svg").write_text(svg, encoding="utf-8")


def find_font(mono: bool, size: int):
    choices = (
        [
            "/System/Library/Fonts/SFNSMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
        if mono
        else ["/System/Library/Fonts/SFNS.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in choices:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def render_frame(checkpoints, values, visible: float):
    scale = 2
    width, height = 1000, 522
    im = Image.new("RGB", (width * scale, height * scale), INK)
    d = ImageDraw.Draw(im)

    def rect(xy, fill, radius=0, outline=None, width=1):
        box = tuple(round(value * scale) for value in xy)
        if radius:
            d.rounded_rectangle(
                box, radius=radius * scale, fill=fill, outline=outline, width=width * scale
            )
        else:
            d.rectangle(box, fill=fill, outline=outline, width=width * scale)

    def text(xy, string, color, size=14, mono=False, anchor=None):
        d.text(
            (xy[0] * scale, xy[1] * scale),
            string,
            font=find_font(mono, size * scale),
            fill=color,
            anchor=anchor,
        )

    def line(points, fill, width=1):
        d.line(
            [(round(x * scale), round(y * scale)) for x, y in points],
            fill=fill,
            width=width * scale,
            joint="curve",
        )

    for x in range(0, width, 40):
        line([(x, 0), (x, height)], "#112838")
    for y in range(0, height, 40):
        line([(0, y), (width, y)], "#112838")
    rect((28, 26, 972, 496), PANEL, 17, LINE)
    rect((29, 27, 971, 72), "#102536", 16)
    rect((29, 58, 971, 73), "#102536")
    for index, color in enumerate(("#ff777a", AMBER, CYAN)):
        d.ellipse(
            (
                round((52 + index * 20) * scale),
                47 * scale,
                round((62 + index * 20) * scale),
                57 * scale,
            ),
            fill=color,
        )
    text((133, 47), "TRACEAI   /   EXPERIMENT VIEW", CYAN, 13, True)
    text((809, 48), "SYNTHETIC DEMO", MUTED, 11, True)
    text((62, 110), "controlled-behavior-study", WHITE, 23)
    text((63, 148), "mock runtime  ·  4 checkpoints  ·  2 probes", MUTED, 15)
    line([(62, 186), (938, 186)], LINE)
    text((63, 211), "CHECKPOINTS", CYAN, 12, True)
    for index, cp in enumerate(checkpoints):
        x = 67 + index * 127
        completed = visible >= index + 0.98
        active = index < visible < index + 1
        color = CYAN if completed else AMBER if active else "#486374"
        rect(
            (x, 238, x + 106, 287),
            "#143141" if completed else "#132a39",
            9,
            color if (completed or active) else LINE,
        )
        text((x + 14, 249), cp.zfill(2), WHITE if completed or active else MUTED, 19, True)
        text((x + 69, 255), "✓" if completed else "●" if active else "·", color, 16)
    text((63, 326), "EVIDENCE", CYAN, 12, True)
    completed_count = min(4, int(visible + 0.02))
    count = completed_count * 12
    text((63, 361), f"{count:02d}", WHITE, 39, True)
    text((130, 378), "raw records persisted", MUTED, 14)
    text((63, 440), "Output-only measurements. Inspect every case.", MUTED, 13)
    line([(568, 211), (568, 455)], LINE)
    text((582, 212), "FAILURE RATE BY CHECKPOINT", CYAN, 12, True)
    xs = [618, 720, 822, 923]
    top, bottom = 258, 426
    for value in (0, 0.5, 1):
        y = bottom - round((bottom - top) * value)
        line([(603, y), (936, y)], LINE)
        text((589, y), f"{value:.1f}", MUTED, 10, True, "rm")
    for x, cp in zip(xs, checkpoints, strict=True):
        text((x, 438), cp, MUTED, 11, True, "mt")
    for probe, color in (("reward_hacking", AMBER), ("specification_gaming", CYAN)):
        scores = values[probe]
        seen = max(0, min(4, math.ceil(visible)))
        points = [(xs[i], bottom - round((bottom - top) * scores[i])) for i in range(seen)]
        if points:
            if len(points) > 1:
                index = len(points) - 1
                if visible < index + 1:
                    previous = points[-2]
                    final = points[-1]
                    part = max(0.0, min(1.0, visible - index))
                    points[-1] = (
                        previous[0] + (final[0] - previous[0]) * part,
                        previous[1] + (final[1] - previous[1]) * part,
                    )
                line(points, color, 4)
            for x, y in points[:-1]:
                d.ellipse(
                    ((x - 5) * scale, (y - 5) * scale, (x + 5) * scale, (y + 5) * scale), fill=color
                )
            x, y = points[-1]
            d.ellipse(
                ((x - 6) * scale, (y - 6) * scale, (x + 6) * scale, (y + 6) * scale), fill=color
            )
    line([(623, 468), (646, 468)], AMBER, 4)
    text((655, 468), "reward", WHITE, 11, True, "lm")
    line([(765, 468), (788, 468)], CYAN, 4)
    text((797, 468), "specification", WHITE, 11, True, "lm")
    return im.resize((width, height), Image.Resampling.LANCZOS)


def main():
    ASSETS.mkdir(exist_ok=True)
    checkpoints, values, report = study()
    render_hero(checkpoints, values, len(report.evidence))
    render_mobile_hero(checkpoints, values, len(report.evidence))
    timeline = (
        [4.0] * 5
        + [0.15] * 3
        + [0.25, 0.45, 0.7, 1.0]
        + [1.0] * 2
        + [1.2, 1.45, 1.7, 2.0]
        + [2.0] * 2
        + [2.2, 2.45, 2.7, 3.0]
        + [3.0] * 2
        + [3.2, 3.45, 3.7, 4.0]
        + [4.0] * 10
    )
    frames = [render_frame(checkpoints, values, position) for position in timeline]
    frames[-1].save(ASSETS / "traceai-demo-poster.png", optimize=True)
    palette = [frame.quantize(colors=96, method=Image.Quantize.FASTOCTREE) for frame in frames]
    palette[0].save(
        ASSETS / "traceai-trajectory.gif",
        save_all=True,
        append_images=palette[1:],
        duration=115,
        loop=0,
        optimize=True,
        disposal=2,
    )
    for name in (
        "traceai-hero.svg",
        "traceai-hero-mobile.svg",
        "traceai-trajectory.gif",
        "traceai-demo-poster.png",
    ):
        file = ASSETS / name
        print(f"{file.relative_to(ROOT)}  {file.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
