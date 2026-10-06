#!/usr/bin/env python3
"""Render the homepage, paper summaries and full research logs."""

from __future__ import annotations

import argparse
from datetime import date
from html import escape
import hashlib
from pathlib import Path
import posixpath
import re
import tomllib
from urllib.parse import urlsplit

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
STYLE_VERSION = hashlib.sha256((SITE / "notebook.css").read_bytes()).hexdigest()[:12]
PAPERS = tomllib.loads((ROOT / "studies/registry.toml").read_text())["studies"]
GITHUB = "https://github.com/fjoad/atk-evidence"
PUBLIC = "https://fjoad.github.io/atk-evidence/"
SUMMARY_SECTIONS = ["What the paper claims", "Why we doubted it", "What we did", "What we found", "Limits"]
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
ENTRY_PREFIX = re.compile(
    rf"^(?:\d{{1,2}}(?:[–-]\d{{1,2}})? (?:{MONTHS})(?: and the later arithmetic checks)?|Later checks) — "
)
HASH_FORWARD = '''<script>
window.addEventListener("load", function () {
  if (!location.hash) return;
  var id = location.hash.slice(1);
  try { id = decodeURIComponent(id); } catch (_) {}
  if (!document.getElementById(id)) {
    location.replace("journal/" + location.hash);
  }
});
</script>'''


def section_title(heading: str) -> str:
    """Dates remain in source records and legacy anchors, not the reading flow."""
    return ENTRY_PREFIX.sub("", heading)


def markdown_parser() -> MarkdownIt:
    return MarkdownIt("commonmark", {"html": False}).enable("table")


def rewrite_links(tokens, source_path: Path, output: Path, *, own_journal: Path | None = None) -> None:
    for token in tokens:
        for child in token.children or []:
            if child.type not in {"link_open", "image"}:
                continue
            attribute = "src" if child.type == "image" else "href"
            href = child.attrGet(attribute) or ""
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (source_path.parent / parsed.path).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                raise ValueError(f"{source_path.relative_to(ROOT)}: Missing or outside-repository source: {href}")
            if child.type == "link_open" and target == own_journal:
                public = "journal/"
            elif target.is_relative_to(SITE):
                public = posixpath.relpath(target.as_posix(), output.parent.as_posix())
                if target.is_dir():
                    public += "/"
            elif child.type == "image":
                public = "https://raw.githubusercontent.com/fjoad/atk-evidence/main/" + target.relative_to(ROOT).as_posix()
            else:
                kind = "tree" if target.is_dir() else "blob"
                public = f"{GITHUB}/{kind}/main/" + target.relative_to(ROOT).as_posix()
            child.attrSet(attribute, public + ("#" + parsed.fragment if parsed.fragment else ""))


def render_body(markdown: MarkdownIt, tokens) -> str:
    body = markdown.renderer.render(tokens, markdown.options, {})
    body = body.replace("<table>", '<div class="table-scroll" role="region" aria-label="Comparison table" tabindex="0"><table>')
    return body.replace("</table>", "</table></div>")


def page_html(*, source_path: Path, output: Path, title: str, description: str,
              page_class: str, body: str, header: str, footer: str, script: str = "") -> str:
    page_title = escape(title)
    description = escape(description)
    stylesheet = posixpath.relpath((SITE / "notebook.css").as_posix(), output.parent.as_posix())
    relative_url = output.parent.relative_to(SITE).as_posix()
    public_url = PUBLIC + (relative_url + "/" if relative_url != "." else "")
    header_html = f'<header class="site-header">\n  {header}\n</header>\n' if header else ""
    return f'''<!doctype html>
<!-- Generated from {source_path.relative_to(ROOT)} and studies/registry.toml.
     Run .venv/bin/python scripts/render_journals.py to rebuild. -->
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <meta name="description" content="{description}">
  <meta property="og:title" content="{page_title}">
  <meta property="og:description" content="{description}">
  <meta property="og:type" content="{'website' if page_class == 'home' else 'article'}">
  <meta property="og:url" content="{escape(public_url)}">
  <meta name="twitter:title" content="{page_title}">
  <meta name="twitter:description" content="{description}">
  <link rel="stylesheet" href="{stylesheet}?v={STYLE_VERSION}">
</head>
<body>
<a class="skip-link" href="#main">Skip to content</a>
{header_html}<main class="{page_class}" id="main">
{body}
  <footer>{footer}</footer>
</main>
{script + chr(10) if script else ''}</body>
</html>
'''


def inline_text(token) -> str:
    """Plain text for titles and descriptions, without Markdown formatting."""
    return "".join(" " if child.type in {"softbreak", "hardbreak"} else child.content
                   for child in token.children or [] if not child.type.endswith(("_open", "_close")))


def render_index() -> str:
    source_path = ROOT / "docs/website-home.md"
    output = SITE / "index.html"
    markdown = markdown_parser()
    tokens = markdown.parse(source_path.read_text(encoding="utf-8"))
    description = next(inline_text(tokens[i + 1]) for i, token in enumerate(tokens)
                       if token.type == "paragraph_open")
    rewrite_links(tokens, source_path, output)
    body = render_body(markdown, tokens)
    body += '<h2>Papers</h2>\n<ul class="cards" aria-label="Papers">\n'
    for paper in PAPERS:
        surname = paper["authors"][0].split()[-1]
        author = surname + (" et al." if len(paper["authors"]) > 1 else "")
        venue = re.split(r"\d", paper["venue"], maxsplit=1)[0].rstrip(" ,")
        meta = f"{author} · {venue} · {paper['year']}"
        href = escape(f"papers/{paper['id']}/")
        body += f'''  <li class="card">
    <h3><a href="{href}">{escape(paper['title'])}</a></h3>
    <p class="meta">{escape(meta)}</p>
    <p class="status">{escape(paper['card_status'])}</p>
    <p class="summary">{escape(paper['card_summary'])}</p>
    <a class="read" href="{href}">Read the study</a>
  </li>
'''
    body += "</ul>\n"
    return page_html(
        source_path=source_path, output=output, title="ATK Evidence", description=description,
        page_class="home", body=body, header="",
        footer='A maintainer of this project co-authored the water-network paper. '
               f'<a href="{GITHUB}">Code and records on GitHub</a>',
    )


def render_summary(study_id: str) -> str:
    paper = next(paper for paper in PAPERS if paper["id"] == study_id)
    source_path = ROOT / paper["path"] / "README.md"
    output = ROOT / paper["page"]
    markdown = markdown_parser()
    tokens = markdown.parse(source_path.read_text(encoding="utf-8"))
    headings = [(i, token) for i, token in enumerate(tokens) if token.type == "heading_open"]
    titles = [inline_text(tokens[i + 1]) for i, token in headings if token.tag == "h1"]
    sections = [inline_text(tokens[i + 1]) for i, token in headings if token.tag == "h2"]
    if len(titles) != 1 or sections not in (SUMMARY_SECTIONS, SUMMARY_SECTIONS + ["Read more"]):
        raise ValueError(f"{source_path.relative_to(ROOT)}: Summary needs exactly one # title and ## sections in order: "
                         + ", ".join(SUMMARY_SECTIONS) + "; optionally Read more last")
    heading_ids = {"main"}
    for i, token in headings:
        slug = re.sub(r"[^a-z0-9]+", "-", tokens[i + 1].content.lower()).strip("-")
        if not slug or slug in heading_ids:
            raise ValueError(f"{source_path.relative_to(ROOT)}: Summary needs distinct headings: {tokens[i + 1].content}")
        heading_ids.add(slug)
        token.attrSet("id", slug)
    rewrite_links(tokens, source_path, output, own_journal=source_path.with_name("RESEARCH_LOG.md"))
    first_section = next(i for i, token in headings if token.tag == "h2")
    body = render_body(markdown, tokens[:first_section])
    body += '<p class="log-link"><a href="journal/">Read the full research log</a></p>\n'
    body += render_body(markdown, tokens[first_section:])
    return page_html(
        source_path=source_path, output=output, title=titles[0] + " — ATK Evidence",
        description=paper["card_summary"], page_class="summary", body=body,
        header='<a class="brand" href="../../">← All papers</a>',
        footer=f'<a href="../../">All papers</a><a href="{GITHUB}/blob/main/{source_path.relative_to(ROOT)}">Page source</a>',
        script=HASH_FORWARD,
    )


def render(study_id: str) -> str:
    """Render the full journal; retain the historical Python entry point."""
    paper = next(paper for paper in PAPERS if paper["id"] == study_id)
    source_path = ROOT / paper["path"] / "RESEARCH_LOG.md"
    output = (ROOT / paper["page"]).parent / "journal/index.html"
    markdown = markdown_parser()
    source = source_path.read_text(encoding="utf-8")
    update_line = re.search(r"^Updated: (\d{4}-\d{2}-\d{2})$", source, re.MULTILINE)
    if update_line is None:
        raise ValueError(f"{source_path.relative_to(ROOT)}: Journal needs an Updated: YYYY-MM-DD line")
    date.fromisoformat(update_line.group(1))  # Validate internal provenance only.
    tokens = markdown.parse(source.replace(update_line.group(0), "", 1))
    contents: list[tuple[str, str]] = []
    heading_ids: set[str] = set()
    title = ""
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            heading = tokens[index + 1].content
            slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            if not slug or slug in heading_ids:
                raise ValueError(f"Journal needs distinct, stable headings: {heading}")
            heading_ids.add(slug)
            token.attrSet("id", slug)
            display_heading = section_title(heading)
            if display_heading != heading:
                tokens[index + 1].content = display_heading
                tokens[index + 1].children = markdown.parseInline(display_heading)[0].children
            if token.tag == "h1":
                if title:
                    raise ValueError("Journal must have exactly one title")
                title = heading
            elif token.tag == "h2":
                contents.append((slug, display_heading))
    if not title:
        raise ValueError("Journal needs a title")
    if not contents or contents[-1][1] != "Current conclusion":
        raise ValueError("Each notebook must end with Current conclusion")
    rewrite_links(tokens, source_path, output)
    body = render_body(markdown, tokens)
    navigation = '<details class="contents"><summary>Contents of this notebook</summary><nav aria-label="Notebook contents"><ol>'
    navigation += "".join(
        f'<li><a href="#{slug}">{escape(heading)}</a></li>'
        for slug, heading in contents
    )
    navigation += "</ol></nav></details>"
    first_entry = body.index('<h2 ')
    body = body[:first_entry] + navigation + body[first_entry:]
    return page_html(
        source_path=source_path, output=output, title=title + " — full research log — ATK Evidence",
        description=f"Reproduction notes for {paper['title']}: starting hypotheses, experiments, corrections and current conclusions.",
        page_class="journal", body=body,
        header='<a class="brand" href="../">← Summary</a> <a class="brand" href="../../../">All papers</a>',
        footer=f'<a href="../../../">All papers</a><a href="{GITHUB}/blob/main/{source_path.relative_to(ROOT)}">Notebook source</a>',
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that the site matches its source without writing")
    args = parser.parse_args()
    # Validate all sources before changing any published page.
    pages = [(SITE / "index.html", render_index())]
    for paper in PAPERS:
        output = ROOT / paper["page"]
        pages.extend([(output, render_summary(paper["id"])),
                      (output.parent / "journal/index.html", render(paper["id"]))])
    for output, rendered in pages:
        if args.check:
            if not output.exists() or output.read_text(encoding="utf-8") != rendered:
                print(f"Page missing or stale: {output.relative_to(ROOT)}; run scripts/render_journals.py")
                return 1
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            print(output.relative_to(ROOT))
    if args.check:
        print("Homepage, summaries and journals match their sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
