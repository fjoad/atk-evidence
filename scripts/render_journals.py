#!/usr/bin/env python3
"""Render registered paper notebooks from their editable research logs."""

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
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
ENTRY_PREFIX = re.compile(
    rf"^(?:\d{{1,2}}(?:[–-]\d{{1,2}})? (?:{MONTHS})(?: and the later arithmetic checks)?|Later checks) — "
)


def section_title(heading: str) -> str:
    """Dates remain in source records and legacy anchors, not the reading flow."""
    return ENTRY_PREFIX.sub("", heading)


def render_index() -> str:
    """Keep the authored index; refresh only its stylesheet cache key."""
    source = (SITE / "index.html").read_text(encoding="utf-8")
    rendered, count = re.subn(
        r'href="notebook\.css(?:\?v=[0-9a-f]+)?"',
        f'href="notebook.css?v={STYLE_VERSION}"', source,
    )
    if count != 1:
        raise ValueError("Index needs exactly one notebook stylesheet")
    return rendered


def render(study_id: str) -> str:
    paper = next(paper for paper in PAPERS if paper["id"] == study_id)
    source_path = ROOT / paper["path"] / "RESEARCH_LOG.md"
    output = ROOT / paper["page"]
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    source = source_path.read_text(encoding="utf-8")
    update_line = re.search(r"^Updated: (\d{4}-\d{2}-\d{2})$", source, re.MULTILINE)
    if update_line is None:
        raise ValueError("Journal needs an Updated: YYYY-MM-DD line")
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
        for child in token.children or []:
            if child.type != "link_open":
                continue
            href = child.attrGet("href") or ""
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (source_path.parent / parsed.path).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                raise ValueError(f"Missing or outside-repository journal source: {href}")
            if target.is_relative_to(SITE):
                public = posixpath.relpath(target.as_posix(), output.parent.as_posix())
                if target.is_dir():
                    public += "/"
            else:
                kind = "tree" if target.is_dir() else "blob"
                public = f"https://github.com/fjoad/atk-evidence/{kind}/main/" + target.relative_to(ROOT).as_posix()
            child.attrSet("href", public + ("#" + parsed.fragment if parsed.fragment else ""))
    if not title:
        raise ValueError("Journal needs a title")
    body = markdown.renderer.render(tokens, markdown.options, {})
    if not contents or contents[-1][1] != "Current conclusion":
        raise ValueError("Each notebook must end with Current conclusion")
    body = body.replace("<table>", '<div class="table-scroll" role="region" aria-label="Comparison table" tabindex="0"><table>')
    body = body.replace("</table>", "</table></div>")
    navigation = '<details class="contents"><summary>Contents of this notebook</summary><nav aria-label="Notebook contents"><ol>'
    navigation += "".join(
        f'<li><a href="#{slug}">{escape(heading)}</a></li>'
        for slug, heading in contents
    )
    navigation += "</ol></nav></details>"
    first_entry = body.index('<h2 ')
    body = body[:first_entry] + navigation + body[first_entry:]
    page_title = escape(title + " — Paper reproduction notes")
    description = escape(f"Reproduction notes for {paper['title']}: starting hypotheses, experiments, corrections and current conclusions.")
    return f'''<!doctype html>
<!-- Generated from {source_path.relative_to(ROOT)}.
     Edit the journal, then run .venv/bin/python scripts/render_journals.py. -->
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <meta name="description" content="{description}">
  <meta property="og:title" content="{page_title}">
  <meta property="og:description" content="{description}">
  <meta property="og:type" content="article">
  <meta property="og:url" content="https://fjoad.github.io/atk-evidence/papers/{study_id}/">
  <meta name="twitter:title" content="{page_title}">
  <meta name="twitter:description" content="{description}">
  <link rel="stylesheet" href="../../notebook.css?v={STYLE_VERSION}">
</head>
<body>
<a class="skip-link" href="#main">Skip to content</a>
<header class="site-header">
  <a class="brand" href="../../">← All papers</a>
</header>
<main class="journal" id="main">
{body}
  <footer><a href="../../">All papers</a><a href="https://github.com/fjoad/atk-evidence/blob/main/{source_path.relative_to(ROOT)}">Notebook source</a></footer>
</main>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that the site matches its source without writing")
    args = parser.parse_args()
    index = SITE / "index.html"
    rendered_index = render_index()
    if args.check and index.read_text(encoding="utf-8") != rendered_index:
        print("Index stylesheet version is stale; run scripts/render_journals.py")
        return 1
    if not args.check:
        index.write_text(rendered_index, encoding="utf-8")
    for paper in PAPERS:
        output = ROOT / paper["page"]
        rendered = render(paper["id"])
        if args.check:
            if not output.exists() or output.read_text(encoding="utf-8") != rendered:
                print(f"Notebook missing or stale: {paper['id']}; run scripts/render_journals.py")
                return 1
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            print(output.relative_to(ROOT))
    if args.check:
        print("All registered notebooks match their research logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
