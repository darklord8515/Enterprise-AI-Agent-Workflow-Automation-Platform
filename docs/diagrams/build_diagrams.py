"""Generate the README diagrams as SVG (source of truth) and, if Playwright is available, 2x PNGs.

    python docs/diagrams/build_diagrams.py            # SVG only
    python docs/diagrams/build_diagrams.py --png      # also render PNGs (needs `pip install playwright`
                                                      # and Chrome/Edge; uses channel="msedge" by default)

The visual language matches the web UI: warm neutrals, ink-blue accent, IBM Plex.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent

INK = "#1b1a17"
TEXT2 = "#55514a"
TEXT3 = "#8b867c"
BORDER = "#d9d5cc"
SURFACE = "#ffffff"
SURFACE2 = "#faf9f6"
BG = "#f5f4f0"
ACCENT = "#2b4c7e"
ACCENT_WEAK = "#e9eef6"
MET, MET_BG = "#1e7148", "#e7f2eb"
GAP, GAP_BG = "#a8521c", "#f8ece2"
WARN, WARN_BG = "#8a6100", "#fbf3dc"
SLATE, SLATE_BG = "#5a6273", "#eceef2"
FONT = "'IBM Plex Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'IBM Plex Mono', Consolas, monospace"

BADGES = {
    "llm": ("LLM · structured output", ACCENT, ACCENT_WEAK),
    "search": ("Azure AI Search", "#2f6f73", "#e4f1f1"),
    "code": ("deterministic code", SLATE, SLATE_BG),
    "azure": ("Azure", ACCENT, ACCENT_WEAK),
}


class Svg:
    def __init__(self, w: int, h: int, title: str, subtitle: str) -> None:
        self.w, self.h = w, h
        self.parts: list[str] = []
        self.title, self.subtitle = title, subtitle

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=13, weight=400, fill=INK, anchor="start", mono=False, extra=""):
        fam = MONO if mono else FONT
        self.add(
            f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" {extra}>{escape(s)}</text>'
        )

    def box(self, x, y, w, h, title, lines=(), badge=None, *, fill=SURFACE, stroke=BORDER, dashed=False,
            accent=None, title_size=15):
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.3"{dash}/>')
        if accent:
            self.add(f'<rect x="{x + 6}" y="{y + 16}" width="3" height="{h - 32}" rx="1.5" fill="{accent}"/>')
        ty = y + 28
        self.text(x + 18, ty, title, size=title_size, weight=600)
        for i, line in enumerate(lines):
            mono = line.startswith("`") and line.endswith("`")
            self.text(x + 18, ty + 22 + i * 19, line.strip("`"), size=12.5, fill=TEXT2, mono=mono)
        if badge:
            self.badge(x + 18, y + h - 32, badge)

    def badge(self, x, y, kind):
        label, fg, bg = BADGES[kind]
        w = 16 + len(label) * 6.0
        self.add(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="20" rx="10" fill="{bg}" stroke="{fg}" stroke-opacity="0.35"/>')
        self.text(x + w / 2, y + 14, label, size=11, weight=500, fill=fg, anchor="middle")

    def pill(self, cx, cy, label, fill=INK, fg="#fff"):
        w = 18 + len(label) * 8
        self.add(f'<rect x="{cx - w / 2}" y="{cy - 15}" width="{w}" height="30" rx="15" fill="{fill}"/>')
        self.text(cx, cy + 5, label, size=12, weight=500, fill=fg, anchor="middle", mono=True)

    def arrow(self, pts, label=None, label_at=None, color=TEXT3, dashed=False, label_anchor="middle", label_color=None):
        d = "M" + " L".join(f"{x} {y}" for x, y in pts)
        dash = ' stroke-dasharray="5 5"' if dashed else ""
        marker = "url(#arrow-accent)" if color == ACCENT else "url(#arrow)"
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round"{dash} marker-end="{marker}"/>')
        if label:
            lx, ly = label_at or pts[len(pts) // 2]
            w = 12 + len(label) * 6.4
            bx = lx - w / 2 if label_anchor == "middle" else lx
            self.add(f'<rect x="{bx:.0f}" y="{ly - 11}" width="{w:.0f}" height="20" rx="5" fill="{SURFACE}"/>')
            tx = lx if label_anchor == "middle" else lx + 6
            self.text(tx, ly + 3, label, size=11.5, fill=label_color or TEXT2, anchor=label_anchor, mono=True)

    def render(self) -> str:
        head = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{TEXT3}"/></marker>
  <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{ACCENT}"/></marker>
</defs>
<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}" rx="18" fill="{BG}" stroke="{BORDER}"/>
"""
        title = (
            f'<text x="40" y="52" font-family="{FONT}" font-size="22" font-weight="600" fill="{INK}">{escape(self.title)}</text>'
            f'<text x="40" y="78" font-family="{FONT}" font-size="13.5" fill="{TEXT2}">{escape(self.subtitle)}</text>'
        )
        return head + title + "\n".join(self.parts) + "\n</svg>\n"


# ---------------------------------------------------------------------------------------- 1. architecture


def architecture() -> Svg:
    s = Svg(1320, 700, "System architecture",
            "FastAPI service, one LangGraph workflow, provider interfaces with Azure and mock implementations.")

    # Clients
    s.text(40, 128, "CLIENTS", size=11, weight=500, fill=TEXT3, mono=True)
    s.box(40, 140, 210, 86, "Web UI", ["Single page served at /", "upload · run · review"])
    s.box(40, 246, 210, 86, "API clients", ["curl · scripts", "Swagger UI at /docs"])

    # FastAPI
    s.text(310, 128, "API LAYER", size=11, weight=500, fill=TEXT3, mono=True)
    s.box(310, 140, 250, 210, "FastAPI", [
        "X-API-Key → principal",
        "Pydantic request validation",
        "Upload allow-list, 5 MB cap,",
        "  magic-byte & UTF-8 checks",
        "Owner-scoped documents & tasks",
        "Typed JSON errors",
        "Structured JSON logs",
    ], accent=ACCENT)
    s.box(310, 380, 250, 90, "Document registry", ["`data/uploads/ + documents.json`", "server-generated ids"], fill=SURFACE2)
    s.arrow([(435, 350), (435, 378)])

    s.arrow([(250, 183), (308, 183)])
    s.arrow([(250, 289), (308, 289)])

    # Services
    s.text(660, 128, "SERVICES", size=11, weight=500, fill=TEXT3, mono=True)
    s.box(660, 140, 270, 106, "Ingestion", ["parse → chunk → embed → index", "txt · md · pdf, page / section kept", "injection heuristics flag chunks"])
    s.box(660, 266, 270, 74, "Task service", ["async run · poll GET /tasks/{id}"])
    s.box(660, 370, 270, 182, "LangGraph workflow", [
        "Planner",
        "Retriever",
        "Analyst",
        "Evidence reviewer",
        "Router  (bounded revisions)",
        "Report generator",
    ], accent=ACCENT)
    s.text(830, 415, "typed state", size=11, fill=TEXT3, mono=True)
    s.text(830, 432, "trace + usage", size=11, fill=TEXT3, mono=True)
    s.arrow([(560, 190), (658, 190)], "ingest", (609, 172))
    s.arrow([(560, 303), (658, 303)], "analyze", (609, 285))
    s.arrow([(795, 340), (795, 368)])

    # Provider bus
    bus_x = 965
    s.add(f'<line x1="{bus_x}" y1="175" x2="{bus_x}" y2="616" stroke="{ACCENT}" stroke-width="2" stroke-opacity="0.35"/>')
    s.text(bus_x - 10, 327, "provider interfaces", size=11, fill=ACCENT, mono=True, anchor="middle",
           extra=f'transform="rotate(-90 {bus_x - 10} 327)"')
    s.arrow([(930, 193), (bus_x, 193)])
    s.arrow([(930, 461), (bus_x, 461)])

    # Azure group
    s.add(f'<rect x="1000" y="112" width="290" height="398" rx="14" fill="{ACCENT_WEAK}" fill-opacity="0.55" stroke="{ACCENT}" stroke-opacity="0.35"/>')
    s.text(1018, 134, "PROVIDER_MODE=azure", size=11, weight=500, fill=ACCENT, mono=True)
    s.box(1016, 148, 258, 98, "Azure OpenAI · chat", ["`gpt-5-mini`", "structured outputs (Pydantic)"])
    s.box(1016, 262, 258, 80, "Azure OpenAI · embeddings", ["`text-embedding-3-small · 1536-d`"])
    s.box(1016, 358, 258, 136, "Azure AI Search", ["HNSW cosine + BM25", "hybrid ranking (RRF)", "OData filters:", "`owner · supplier · category`"])
    s.arrow([(bus_x, 197), (1014, 197)], color=ACCENT)
    s.arrow([(bus_x, 302), (1014, 302)], color=ACCENT)
    s.arrow([(bus_x, 426), (1014, 426)], color=ACCENT)

    s.box(1000, 530, 290, 96, "Mock mode  (local · CI)", ["heuristic analyst · hashed embeddings", "in-memory BM25 + cosine (RRF)"],
          dashed=True, fill=SURFACE2)
    s.arrow([(bus_x, 580), (998, 580)], dashed=True)

    s.text(40, 662, "No silent fallback: in azure mode every component is a real Azure client and failures surface as typed errors "
                    "(auth, rate limit, deployment not found, index schema mismatch).", size=12.5, fill=TEXT2)
    return s


# ---------------------------------------------------------------------------------------- 2. workflow


def workflow() -> Svg:
    s = Svg(1400, 690, "LangGraph workflow",
            "One explicit StateGraph. Each node is a single-responsibility step, not an autonomous agent.")
    y, h, w = 250, 128, 172
    cy = y + h / 2

    s.pill(68, cy, "START")
    nodes = [
        (120, "Planner", ["request → checks", "explicit or from policy"], "llm"),
        (322, "Retriever", ["hybrid search per check", "owner + supplier filter"], "search"),
        (524, "Analyst", ["status, verbatim quotes,", "gaps & follow-ups"], "llm"),
        (726, "Evidence reviewer", ["citations retrieved?", "quote verbatim? supplier?"], "code"),
    ]
    for x, title, lines, badge in nodes:
        s.box(x, y, w, h, title, lines, badge, accent=ACCENT if badge == "code" else None)
    for (x1, *_), (x2, *_) in zip(nodes, nodes[1:], strict=False):
        s.arrow([(x1 + w, cy), (x2 - 2, cy)])
    s.arrow([(100, cy), (118, cy)])

    # Router diamond
    rx, r = 1010, 62
    s.add(f'<path d="M{rx} {cy - r} L{rx + r} {cy} L{rx} {cy + r} L{rx - r} {cy} z" fill="{SURFACE}" stroke="{ACCENT}" stroke-width="1.6"/>')
    s.text(rx, cy - 2, "Router", size=15, weight=600, anchor="middle")
    s.text(rx, cy + 16, "route?", size=11.5, fill=TEXT3, anchor="middle", mono=True)
    s.arrow([(726 + w, cy), (rx - r - 2, cy)])

    # Report + END
    s.box(1160, y, 172, h, "Report generator", ["findings, citations,", "warnings, trace, usage"], "code")
    s.arrow([(rx + r, cy), (1158, cy)], "all supported", (1110, cy - 16), color=MET, label_color=MET)
    s.pill(1246, 470, "END")
    s.arrow([(1246, y + h), (1246, 453)])

    # Revision loop
    top = 168
    s.arrow([(rx, cy - r), (rx, top), (408, top), (408, y - 2)], color=GAP)
    s.add(f'<rect x="512" y="{top - 30}" width="390" height="22" rx="6" fill="{GAP_BG}"/>')
    s.text(707, top - 14, "revise: unsupported & revisions < MAX_REVISIONS (default 1)", size=11.5, fill=GAP, anchor="middle", mono=True)
    s.text(420, top + 20, "broader query, 2× top-k,", size=11.5, fill=TEXT3, mono=True)
    s.text(420, top + 36, "reviewer feedback to analyst", size=11.5, fill=TEXT3, mono=True)

    # Insufficient branch
    s.box(910, 460, 200, 112, "Insufficient-evidence", ["downgrade unverifiable", "findings, keep notes"], "code", fill=SURFACE2)
    s.arrow([(rx, cy + r), (rx, 458)], "still unsupported", (rx, 420), color=SLATE, label_color=SLATE)
    s.arrow([(1110, 516), (1190, 516), (1190, y + h + 2)], color=SLATE)

    # State legend
    s.add(f'<rect x="40" y="602" width="1320" height="56" rx="12" fill="{SURFACE}" stroke="{BORDER}"/>')
    s.text(60, 626, "WorkflowState (TypedDict)", size=12.5, weight=600)
    s.text(60, 645, "task · supplier_id · owner · checks · evidence{} · assessments{} · findings{} · review_issues · targets · "
                    "revision_count · route · warnings[+] · trace[+] · usage(+)", size=11.5, fill=TEXT2, mono=True)
    s.text(1340, 626, "recursion_limit = 40 (hard loop guard)", size=11.5, fill=TEXT3, anchor="end", mono=True)
    return s


# ---------------------------------------------------------------------------------------- 3. verification


def verification() -> Svg:
    s = Svg(1320, 640, "Evidence verification",
            "The reviewer is code, not a prompt. A conclusion survives only if its evidence can be checked against what was retrieved.")
    y, h, w = 150, 104, 220
    cy = y + h / 2
    s.box(40, y, 170, h, "Model citation", ["`chunk_id + quote`", "from the analyst"], fill=SURFACE2)
    checks = [
        (270, "1  Retrieved?", ["chunk was returned for", "this requirement"]),
        (530, "2  Verbatim?", ["quote found in chunk text", "(whitespace/case normalised)"]),
        (790, "3  Supplier source?", ["policy text cannot prove", "supplier compliance"]),
    ]
    prev = 210
    for x, title, lines in checks:
        s.box(x, y, w, h, title, lines, accent=ACCENT)
        s.arrow([(prev, cy), (x - 2, cy)], "yes" if prev != 210 else None, ((prev + x) / 2, cy - 14), color=MET if prev != 210 else TEXT3, label_color=MET)
        s.arrow([(x + w / 2, y + h), (x + w / 2, 318)], "no", (x + w / 2, 288), color=GAP, label_color=GAP)
        prev = x + w
    s.box(1060, y, 220, h, "Valid evidence", ["shown with file, section,", "page and chunk id"], fill=MET_BG, stroke=MET)
    s.arrow([(prev, cy), (1058, cy)], "yes", ((prev + 1060) / 2, cy - 14), color=MET, label_color=MET)

    s.add(f'<rect x="270" y="320" width="740" height="56" rx="12" fill="{GAP_BG}" stroke="{GAP}" stroke-opacity="0.45"/>')
    s.text(290, 346, "Citation removed", size=14, weight=600, fill=GAP)
    s.text(290, 365, "and a reviewer note is attached to the finding, so the removal is visible in the report", size=12.5, fill=TEXT2)

    s.add(f'<rect x="1060" y="270" width="220" height="70" rx="12" fill="{WARN_BG}" stroke="{WARN}" stroke-opacity="0.45"/>')
    s.text(1076, 294, "Flagged chunk cited?", size=13, weight=600, fill=WARN)
    s.text(1076, 313, "kept, with a \"verify manually\"", size=12, fill=TEXT2)
    s.text(1076, 330, "note (possible injection)", size=12, fill=TEXT2)
    s.arrow([(1170, y + h), (1170, 268)], dashed=True)

    # Finding-level rule
    s.text(40, 432, "THEN, PER FINDING", size=11, weight=500, fill=TEXT3, mono=True)
    fy, fh = 448, 120
    s.box(40, fy, 260, fh, "Status met or gap…", ["…with zero valid supplier", "citations is unsupported"], accent=ACCENT)
    s.box(380, fy, 280, fh, "Bounded revision", ["re-retrieve with a broader query,", "re-analyse with reviewer feedback", "(MAX_REVISIONS, default 1)"])
    s.box(800, fy, 480, fh, "insufficient_evidence", ["Still unsupported after the last revision: reported as insufficient",
                                                         "evidence with the reasons. Absence of contrary information is",
                                                         "never treated as compliance."], fill=SLATE_BG, stroke=SLATE)
    s.arrow([(300, fy + fh / 2), (378, fy + fh / 2)], color=GAP)
    s.arrow([(660, fy + fh / 2), (798, fy + fh / 2)], "still unsupported", (729, fy + fh / 2 - 16), color=SLATE, label_color=SLATE)
    s.text(40, 606, "Tested with scripted model outputs: fabricated chunk ids, paraphrased quotes and policy-as-proof are all caught "
                    "(tests/test_workflow.py).", size=12.5, fill=TEXT2)
    return s


DIAGRAMS = {"architecture": architecture, "workflow": workflow, "verification": verification}


def render_png(names: list[str]) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(device_scale_factor=2)
        for name in names:
            svg = (OUT / f"{name}.svg").read_text(encoding="utf-8")
            page.set_content(
                "<html><head><link href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
                "&family=IBM+Plex+Sans:wght@400;500;600&display=block' rel='stylesheet'>"
                "<style>body{margin:0;background:transparent}</style></head><body>" + svg + "</body></html>"
            )
            page.wait_for_load_state("networkidle")
            page.evaluate("document.fonts.ready")
            page.locator("svg").first.screenshot(path=str(OUT / f"{name}.png"), omit_background=True)
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    for name, build in DIAGRAMS.items():
        (OUT / f"{name}.svg").write_text(build().render(), encoding="utf-8")
        print("wrote", f"{name}.svg")
    if args.png:
        render_png(list(DIAGRAMS))
        print("rendered PNGs")


if __name__ == "__main__":
    main()
