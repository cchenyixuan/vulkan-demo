"""Export a dossier HTML (docs/n56_scaling/dossier_v3.html) to
  <name>_standalone.html  — figures embedded as data URIs (single file, opens anywhere)
  <name>.md               — Markdown (headings, paragraphs, lists, tables, figure captions) for a Claude project
  <name>_bundle.zip       — the .md, the standalone .html and fig/*.png
usage: export_dossier.py docs/n56_scaling/dossier_v3.html"""
import base64, html, pathlib, re, sys, zipfile
from html.parser import HTMLParser

src = pathlib.Path(sys.argv[1]); docs = src.parent
text = src.read_text(encoding="utf-8")

# ---------------- standalone html: inline fig/*.png
def inline(m):
    p = docs / m.group(1)
    if not p.exists():
        return m.group(0)
    return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'
standalone = re.sub(r'src="fig/([^"]+)"', inline, text)
out_html = docs / f"{src.stem}_standalone.html"
out_html.write_text("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'></head><body>" + standalone + "</body></html>", encoding="utf-8")


# ---------------- markdown
class MD(HTMLParser):
    def __init__(self):
        super().__init__(); self.out = []; self.stack = []; self.cell = None; self.row = None; self.rows = []; self.in_thead = False
        self.skip = 0; self.list_depth = 0; self.fig_src = None; self.buf = None
    def start(self, tag): self.stack.append(tag)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "link":          # void element: no end tag, just ignore it
            return
        if tag in ("style", "nav"):
            self.skip += 1; return
        if self.skip:
            return
        if self.cell is not None:      # inline markup inside a table cell goes into the cell
            if tag in ("b", "strong"): self.cell.append("**")
            elif tag == "code": self.cell.append("`")
            elif tag == "sub": self.cell.append("_")
            return
        if tag == "h1": self.out.append("\n# ")
        elif tag == "h2": self.out.append("\n\n## ")
        elif tag == "h3": self.out.append("\n\n### ")
        elif tag == "p": self.out.append("\n\n")
        elif tag == "ul": self.list_depth += 1
        elif tag == "li": self.out.append("\n" + "  " * (self.list_depth - 1) + "- ")
        elif tag in ("b", "strong"): self.out.append("**")
        elif tag in ("em", "i"): self.out.append("*")
        elif tag == "code": self.out.append("`")
        elif tag == "br": self.out.append("  \n")
        elif tag == "table": self.rows = []; self.in_thead = False
        elif tag == "thead": self.in_thead = True
        elif tag == "tr": self.row = []
        elif tag in ("td", "th"): self.cell = []
        elif tag == "img": self.fig_src = a.get("src", "").replace("fig/", "")
        elif tag == "figcaption": self.out.append("\n\n")
        elif tag == "dt": self.out.append("\n- **")
        elif tag == "dd": self.out.append(" ")
        elif tag == "div" and "tile" in (a.get("class") or ""): self.out.append("\n- ")
        elif tag == "span" and "chip" in (a.get("class") or ""): self.out.append(" [")
        elif tag == "sub": self.out.append("_")
    def handle_endtag(self, tag):
        if tag in ("style", "nav"):
            self.skip -= 1; return
        if self.skip:
            return
        if self.cell is not None and tag not in ("td", "th"):
            if tag in ("b", "strong"): self.cell.append("**")
            elif tag == "code": self.cell.append("`")
            return
        if tag in ("h1", "h2", "h3"): self.out.append("\n")
        elif tag == "ul": self.list_depth -= 1; self.out.append("\n")
        elif tag in ("b", "strong"): self.out.append("**")
        elif tag in ("em", "i"): self.out.append("*")
        elif tag == "code": self.out.append("`")
        elif tag in ("td", "th"):
            self.row.append(" ".join("".join(self.cell).split())); self.cell = None
        elif tag == "tr":
            self.rows.append((self.in_thead, self.row)); self.row = None
        elif tag == "thead": self.in_thead = False
        elif tag == "table":
            if not self.rows:
                return
            width = max(len(r) for _, r in self.rows)
            lines = ["\n"]
            header_done = False
            for is_head, r in self.rows:
                r = r + [""] * (width - len(r))
                lines.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
                if is_head and not header_done:
                    lines.append("|" + "---|" * width); header_done = True
            if not header_done:
                lines.insert(2, "|" + "---|" * width)
            self.out.append("\n".join(lines) + "\n")
        elif tag == "figure":
            if self.fig_src:
                self.out.append(f"\n\n*Figure file: fig/{self.fig_src}*\n"); self.fig_src = None
        elif tag == "dt": self.out.append("**")
        elif tag == "span": self.out.append("]")
        elif tag == "dl": self.out.append("\n")
    def handle_data(self, data):
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        else:
            self.out.append(data)

parser = MD(); parser.feed(text)
md = html.unescape("".join(parser.out))
md = re.sub(r"[ \t]+\n", "\n", md); md = re.sub(r"\n{3,}", "\n\n", md); md = re.sub(r"[ \t]{2,}", " ", md)
md = "<!-- exported from docs/n56_scaling/dossier_v3.html; figures in the fig/ folder of the bundle -->\n" + md.strip() + "\n"
out_md = docs / f"{src.stem}.md"; out_md.write_text(md, encoding="utf-8")

# ---------------- zip bundle
out_zip = docs / f"{src.stem}_bundle.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_md, out_md.name); z.write(out_html, out_html.name)
    for m in sorted(set(re.findall(r'src="fig/([^"]+)"', text))):
        p = docs / m
        if p.exists():
            z.write(p, f"fig/{m}")
print("wrote", out_html, out_html.stat().st_size, "B;", out_md, out_md.stat().st_size, "B;", out_zip, out_zip.stat().st_size, "B")
