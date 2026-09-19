#!/usr/bin/env python3
"""Render the one editable Paper 3 research journal into the static site."""

from __future__ import annotations

import argparse
from datetime import date
from html import escape
from pathlib import Path
import re
from urllib.parse import urlsplit

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "studies/takiddin-2021-robust-poisoning/RESEARCH_LOG.md"
OUTPUT = ROOT / "site/papers/takiddin-2021-robust-poisoning/index.html"
DESCRIPTION = (
    "A research journal rebuilding an electricity-theft poisoning experiment: "
    "what we noticed, the checks we chose, and what remains to be tested."
)


def render() -> str:
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    source = SOURCE.read_text(encoding="utf-8")
    update_line = re.search(r"^Updated: (\d{4}-\d{2}-\d{2})$", source, re.MULTILINE)
    if update_line is None:
        raise ValueError("Journal needs an Updated: YYYY-MM-DD line")
    updated = date.fromisoformat(update_line.group(1))
    date_label = f"{updated.day} {updated:%B %Y}"
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
            if token.tag == "h1":
                if title:
                    raise ValueError("Journal must have exactly one title")
                title = heading
            elif token.tag == "h2":
                contents.append((slug, heading))
        for child in token.children or []:
            if child.type != "link_open":
                continue
            href = child.attrGet("href") or ""
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (SOURCE.parent / parsed.path).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                raise ValueError(f"Missing or outside-repository journal source: {href}")
            public = "https://github.com/fjoad/atk-evidence/blob/main/" + target.relative_to(ROOT).as_posix()
            child.attrSet("href", public + ("#" + parsed.fragment if parsed.fragment else ""))
    if not title:
        raise ValueError("Journal needs a title")
    body = markdown.renderer.render(tokens, markdown.options, {})
    first_paragraph = body.index("</h1>") + len("</h1>")
    navigation = '<nav class="contents" aria-label="Journal entries"><p>Follow the investigation</p><ol>'
    navigation += "".join(
        f'<li><a href="#{slug}">{escape(heading)}</a></li>'
        for slug, heading in contents
    )
    navigation += "</ol></nav>"
    # Leave the opening explanation and stage visible before the entry list.
    first_entry = body.index('<h2 ', first_paragraph)
    body = body[:first_entry] + navigation + body[first_entry:]
    page_title = escape(title + " — Research journal · ATK Evidence")
    description = escape(DESCRIPTION)
    return f'''<!doctype html>
<!-- Generated from studies/takiddin-2021-robust-poisoning/RESEARCH_LOG.md.
     Edit the journal, then run .venv/bin/python scripts/render_robust_journal.py. -->
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <meta name="description" content="{description}">
  <meta property="og:title" content="{page_title}">
  <meta property="og:description" content="{description}">
  <meta property="og:type" content="article">
  <meta property="og:url" content="https://fjoad.github.io/atk-evidence/papers/takiddin-2021-robust-poisoning/">
  <meta name="twitter:title" content="{page_title}">
  <meta name="twitter:description" content="{description}">
  <link rel="stylesheet" href="../../styles.css">
</head>
<body>
<a class="skip-link" href="#main">Skip to content</a>
<header class="site-header">
  <a class="brand" href="../../">ATK Evidence</a>
  <nav aria-label="Main navigation"><a href="../../#studies">All studies</a><a href="#inspect-the-record">Evidence &amp; sources</a></nav>
</header>
<main class="journal" id="main">
  <div class="eyebrow">Study 3 · Research journal · Updated {date_label}</div>
{body}
  <footer><p>This is a working investigation. Dated entries preserve our questions, evidence, and corrections.</p><p><a href="../../">Back to all studies</a></p></footer>
</main>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that the site matches its source without writing")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            print("Journal page is missing or stale; run scripts/render_robust_journal.py")
            return 1
        print("Journal page matches RESEARCH_LOG.md")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
