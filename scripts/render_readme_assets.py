"""Render README visuals from the bundled synthetic study.

Run: uv run --extra visuals python scripts/render_readme_assets.py
The generated images are checked in; Pillow is not a runtime dependency.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont, ImageOps

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


def cover_font(size: int, *, bold: bool = False, mono: bool = False):
    if mono:
        candidates = [
            ("/System/Library/Fonts/SFNSMono.ttf", 0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
        ]
    elif bold:
        candidates = [
            ("/System/Library/Fonts/Avenir Next.ttc", 0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
        ]
    else:
        candidates = [
            ("/System/Library/Fonts/Avenir Next.ttc", 5),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
        ]
    for path, index in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size, index=index)
    return ImageFont.load_default(size)


def draw_cover_mark(draw: ImageDraw.ImageDraw, x: int, y: int, size: int):
    half = size // 2
    draw.polygon(
        [(x + half, y), (x + size, y + half), (x + half, y + size), (x, y + half)],
        outline=CYAN,
        width=4,
    )
    draw.line((x + size // 4, y + half, x + size * 3 // 4, y + half), fill=CYAN, width=4)


def render_covers():
    source = Image.open(ASSETS / "traceai-instrument-background.jpg").convert("RGB")
    if source.size != (2172, 724):
        raise RuntimeError("README cover artwork must be 2172×724")

    desktop = source.copy()
    d = ImageDraw.Draw(desktop)
    draw_cover_mark(d, 125, 92, 52)
    d.text((207, 96), "TRACEAI", font=cover_font(42, bold=True), fill=WHITE)
    d.text((129, 195), "AI BEHAVIOR OBSERVATORY", font=cover_font(25, mono=True), fill=CYAN)
    title = cover_font(112, bold=True)
    d.text((118, 261), "Behavior has", font=title, fill=WHITE)
    second = "a trajectory"
    d.text((118, 381), second, font=title, fill=WHITE)
    second_end = d.textbbox((118, 381), second, font=title)[2]
    d.text((second_end + 1, 381), ".", font=title, fill=AMBER)
    d.text(
        (126, 545),
        "Follow the evidence behind every checkpoint.",
        font=cover_font(35),
        fill="#c3d0d5",
    )
    d.line((128, 635, 1000, 635), fill="#2a4b57", width=2)
    d.text(
        (128, 656),
        "01  OBSERVE      02  COMPARE      03  EXPLAIN",
        font=cover_font(25, mono=True),
        fill=CYAN,
    )
    desktop.save(ASSETS / "traceai-cover-v2.jpg", quality=93, subsampling=0, optimize=True)

    mobile = Image.new("RGB", (720, 900), INK)
    art = ImageOps.fit(
        source.crop((950, 0, 2172, 724)), (720, 465), method=Image.Resampling.LANCZOS
    )
    fade = Image.new("L", art.size, 255)
    fade_data = ImageDraw.Draw(fade)
    for y in range(355, 465):
        opacity = round(255 * (465 - y) / 110)
        fade_data.line((0, y, 720, y), fill=opacity)
    mobile.paste(art, (0, 0), fade)
    m = ImageDraw.Draw(mobile)
    m.rounded_rectangle((30, 28, 289, 110), radius=18, fill=INK, outline="#2a4b57", width=2)
    draw_cover_mark(m, 47, 45, 43)
    m.text((113, 47), "TRACEAI", font=cover_font(36, bold=True), fill=WHITE)
    m.text((49, 458), "AI BEHAVIOR OBSERVATORY", font=cover_font(20, mono=True), fill=CYAN)
    mobile_title = cover_font(76, bold=True)
    m.text((42, 510), "Behavior has", font=mobile_title, fill=WHITE)
    second = "a trajectory"
    m.text((42, 591), second, font=mobile_title, fill=WHITE)
    second_end = m.textbbox((42, 591), second, font=mobile_title)[2]
    m.text((second_end + 1, 591), ".", font=mobile_title, fill=AMBER)
    m.text(
        (48, 714),
        "Evidence behind every checkpoint.",
        font=cover_font(29),
        fill="#c3d0d5",
    )
    m.line((49, 791, 669, 791), fill="#2a4b57", width=2)
    m.text(
        (49, 815),
        "OBSERVE  /  COMPARE  /  EXPLAIN",
        font=cover_font(22, mono=True),
        fill=CYAN,
    )
    mobile.save(ASSETS / "traceai-cover-mobile-v2.jpg", quality=92, subsampling=0, optimize=True)


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


def render_pipeline():
    stages = [
        ("01", "DESIGN", "Cases · seeds", "Versioned study"),
        ("02", "EXECUTE", "Local runtime", "Model checkpoint"),
        ("03", "MEASURE", "Raw response", "Deterministic rule"),
        ("04", "TRACE", "Failure rates", "Bootstrap intervals"),
        ("05", "INSPECT", "CLI · dashboard", "Evidence export"),
    ]
    cards = []
    for index, (number, title, first, second) in enumerate(stages):
        x = 52 + index * 266
        accent = AMBER if index == 3 else CYAN
        cards.append(
            f'<g><rect x="{x}" y="139" width="236" height="201" rx="16" '
            f'fill="{PANEL}" stroke="{LINE}"/>'
            f'<path d="M{x + 22} 181h30" stroke="{accent}" stroke-width="3"/>'
            f'<text x="{x + 62}" y="187" class="mono tag" fill="{accent}">{number}</text>'
            f'<text x="{x + 22}" y="235" class="sans title" fill="{WHITE}">{title}</text>'
            f'<text x="{x + 22}" y="278" class="sans body" fill="{MUTED}">{escape(first)}</text>'
            f'<text x="{x + 22}" y="308" class="sans body" fill="{MUTED}">{escape(second)}</text>'
            "</g>"
        )
    arrows = "".join(
        f'<path d="M{288 + index * 266} 239h28m-8-7 8 7-8 7" '
        f'stroke="{CYAN}" stroke-width="2" fill="none"/>'
        for index in range(4)
    )
    desktop = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="430" viewBox="0 0 1400 430" role="img" aria-labelledby="title desc">
<title id="title">TraceAI evidence pipeline</title>
<desc id="desc">Design a versioned study, execute a local model checkpoint, measure raw responses with deterministic rules, trace rates and intervals, and inspect evidence in the CLI or dashboard.</desc>
<style>.sans{{font-family:Arial,Helvetica,sans-serif}}.mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}.tag{{font-size:17px;letter-spacing:2px}}.title{{font-size:26px;font-weight:700;letter-spacing:1px}}.body{{font-size:19px}}</style>
<rect width="1400" height="430" rx="24" fill="{INK}"/>
<path d="M0 81h1400M0 373h1400" stroke="{LINE}"/>
<text x="52" y="54" class="mono" font-size="17" fill="{CYAN}">TRACEAI / EVIDENCE CHAIN</text>
<text x="1348" y="54" text-anchor="end" class="mono" font-size="15" fill="{MUTED}">FROM CASE TO EVIDENCE</text>
{"".join(cards)}{arrows}
<text x="52" y="405" class="mono" font-size="15" fill="{MUTED}">CASE  →  PROMPT  →  RESPONSE  →  RULE  →  SCORE</text>
<text x="1348" y="405" text-anchor="end" class="mono" font-size="15" fill="{MUTED}">LOCAL FIRST · RAW EVIDENCE SAVED</text>
</svg>'''
    (ASSETS / "traceai-pipeline.svg").write_text(desktop, encoding="utf-8")

    mobile_cards = []
    for index, (number, title, first, second) in enumerate(stages):
        y = 121 + index * 151
        accent = AMBER if index == 3 else CYAN
        mobile_cards.append(
            f'<g><rect x="36" y="{y}" width="648" height="126" rx="16" '
            f'fill="{PANEL}" stroke="{LINE}"/>'
            f'<text x="65" y="{y + 49}" class="mono" font-size="23" fill="{accent}">{number}</text>'
            f'<text x="123" y="{y + 50}" class="sans" font-size="30" font-weight="700" fill="{WHITE}">{title}</text>'
            f'<text x="123" y="{y + 91}" class="sans" font-size="23" fill="{MUTED}">{escape(first)}  ·  {escape(second)}</text>'
            "</g>"
        )
    mobile_arrows = "".join(
        f'<path d="M360 {247 + index * 151}v22m-7-8 7 8 7-8" '
        f'stroke="{CYAN}" stroke-width="2" fill="none"/>'
        for index in range(4)
    )
    mobile = f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="930" viewBox="0 0 720 930" role="img" aria-labelledby="title desc">
<title id="title">TraceAI evidence pipeline</title>
<desc id="desc">Five stages from study design through local execution, measurement, trajectory analysis, and evidence inspection.</desc>
<style>.sans{{font-family:Arial,Helvetica,sans-serif}}.mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}</style>
<rect width="720" height="930" rx="24" fill="{INK}"/>
<text x="36" y="55" class="mono" font-size="21" fill="{CYAN}">TRACEAI / EVIDENCE CHAIN</text>
<text x="36" y="88" class="mono" font-size="16" fill="{MUTED}">FROM CASE TO EVIDENCE</text>
{"".join(mobile_cards)}{mobile_arrows}
<path d="M36 888h648" stroke="{LINE}"/>
<text x="36" y="911" class="mono" font-size="16" fill="{MUTED}">LOCAL FIRST  ·  RAW EVIDENCE SAVED</text>
</svg>'''
    (ASSETS / "traceai-pipeline-mobile.svg").write_text(mobile, encoding="utf-8")


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
    render_covers()
    render_hero(checkpoints, values, len(report.evidence))
    render_mobile_hero(checkpoints, values, len(report.evidence))
    render_pipeline()
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
        "traceai-cover-v2.jpg",
        "traceai-cover-mobile-v2.jpg",
        "traceai-hero.svg",
        "traceai-hero-mobile.svg",
        "traceai-pipeline.svg",
        "traceai-pipeline-mobile.svg",
        "traceai-trajectory.gif",
        "traceai-demo-poster.png",
    ):
        file = ASSETS / name
        print(f"{file.relative_to(ROOT)}  {file.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
