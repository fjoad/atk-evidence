"""Check the open-ended paper index and the boundaries of its claims."""

from html import escape
import importlib.util
import json
from pathlib import Path
import re
import tomllib
import tempfile
import unittest
from unittest.mock import patch

from tests.test_public_report import Page

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PAPERS = tomllib.loads((ROOT / "studies/registry.toml").read_text())["studies"]
spec = importlib.util.spec_from_file_location("notebook_renderer", ROOT / "scripts/render_journals.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class NotebookSiteTests(unittest.TestCase):
    def test_visible_pages_read_as_a_continuation_not_a_dated_diary(self):
        pages = [SITE / "index.html", *[ROOT / paper["page"] for paper in PAPERS]]
        calendar_date = rf"\b\d{{1,2}}(?:[–-]\d{{1,2}})? (?:{renderer.MONTHS})\b"
        for path in pages:
            with self.subTest(page=path):
                visible = Page(path).text
                self.assertNotRegex(visible, calendar_date)
                self.assertNotIn("Notebook updated", visible)
                self.assertNotIn("Notes updated", visible)
        for heading, expected in (
            ("20 September — The printed loss loses the label", "The printed loss loses the label"),
            ("30–31 August — Completing a declared FC-SAE reproduction", "Completing a declared FC-SAE reproduction"),
            ("24 July and the later arithmetic checks — Do the printed metrics fit?", "Do the printed metrics fit?"),
            ("Later checks — The graph does help", "The graph does help"),
        ):
            self.assertEqual(renderer.section_title(heading), expected)
        poisoning = (SITE / "papers/takiddin-2021-robust-poisoning/journal/index.html").read_text()
        self.assertIn('id="20-september-the-printed-loss-loses-the-label"', poisoning)
        self.assertIn('href="#20-september-the-printed-loss-loses-the-label"', poisoning)
        self.assertIn('>The printed loss loses the label</h2>', poisoning)

    def test_home_is_an_open_paper_index_not_a_result_report(self):
        path = SITE / "index.html"
        content = path.read_text()
        page = Page(path)
        links = [link for link in page.links if link.startswith("papers/")]
        self.assertEqual(links, [f"papers/{paper['id']}/" for paper in PAPERS for _ in range(2)])
        for paper in PAPERS:
            for field in ("title", "year", "card_status", "card_summary"):
                self.assertIn(escape(str(paper[field])), content)
            author = paper["authors"][0].split()[-1] + (" et al." if len(paper["authors"]) > 1 else "")
            venue = re.split(r"\d", paper["venue"], maxsplit=1)[0].rstrip(" ,")
            self.assertIn(escape(f"{author} · {venue} · {paper['year']}"), content)
        self.assertIn("<h1>ATK Evidence</h1>", content)
        self.assertIn('<main class="home" id="main">', content)
        self.assertIn('<ul class="cards" aria-label="Papers">', content)
        self.assertEqual(content.count('<li class="card">'), len(PAPERS))
        self.assertNotRegex(content, r"Paper \d+ of \d+")
        self.assertEqual(content, renderer.render_index())
        self.assertIn(f'href="notebook.css?v={renderer.STYLE_VERSION}"', content)
        self.assertIn("https://github.com/fjoad/atk-evidence/blob/main/docs/APPROACH.md", page.links)
        self.assertIn("A maintainer of this project co-authored the water-network paper.", page.text)
        self.assertIn("Code and records on GitHub", page.text)
        self.assertIn("<title>ATK Evidence</title>", content)
        for key in ("og:title", "twitter:title"):
            self.assertEqual(page.metadata[key], "ATK Evidence")
        description = " ".join((ROOT / "docs/website-home.md").read_text().split("\n\n")[1].split())
        for key in ("description", "og:description", "twitter:description"):
            self.assertEqual(page.metadata[key], description)

    def test_summaries_render_the_readme_with_links_to_the_journal_and_index(self):
        expected = ["What the paper claims", "Why we doubted it", "What we did", "What we found", "Limits"]
        for paper in PAPERS:
            with self.subTest(paper=paper["id"]):
                content = (ROOT / paper["page"]).read_text()
                page = Page(ROOT / paper["page"])
                self.assertEqual(content, renderer.render_summary(paper["id"]))
                headings = re.findall(r'<h2 id="[^"]+">(.*?)</h2>', content)
                self.assertIn(headings, (expected, expected + ["Read more"]))
                self.assertIn('<main class="summary" id="main">', content)
                self.assertIn(f'href="../../notebook.css?v={renderer.STYLE_VERSION}"', content)
                header = re.search(r'<header class="site-header">.*?</header>', content, re.S).group()
                self.assertEqual(header.count("<a "), 1)
                self.assertIn('href="../../">← All papers</a>', header)
                log_link = '<p class="log-link"><a href="journal/">Read the full research log</a></p>'
                self.assertIn(log_link + '\n<h2 ', content)
                self.assertIn(renderer.HASH_FORWARD, content)
                self.assertIn('window.addEventListener("load"', content)
                self.assertIn('if (!location.hash) return;', content)
                self.assertIn('if (!document.getElementById(id))', content)
                self.assertIn('location.replace("journal/" + location.hash)', content)
                self.assertIn(f'{renderer.GITHUB}/blob/main/{paper["path"]}/README.md', page.links)
                self.assertEqual(page.metadata["description"], paper["card_summary"])
                self.assertEqual(page.metadata["og:url"], f'{renderer.PUBLIC}papers/{paper["id"]}/')
                title = (ROOT / paper["path"] / "README.md").read_text().splitlines()[0][2:]
                self.assertIn(f'<title>{escape(title)} — ATK Evidence</title>', content)
                if "<table>" in content:
                    self.assertIn('tabindex="0"><table>', content)

    def test_notebooks_link_to_the_summary_and_index_without_a_fixed_switcher(self):
        for paper in PAPERS:
            with self.subTest(paper=paper["id"]):
                path = (ROOT / paper["page"]).parent / "journal/index.html"
                content = path.read_text()
                self.assertEqual(content, renderer.render(paper["id"]))
                self.assertIn(f'href="../../../notebook.css?v={renderer.STYLE_VERSION}"', content)
                self.assertNotIn("paper-tabs", content)
                self.assertNotIn("Choose a paper", content)
                self.assertNotIn("All three papers", content)
                self.assertNotRegex(content, r"Paper \d+ of \d+")
                header = re.search(r'<header class="site-header">.*?</header>', content, re.S).group()
                self.assertEqual(header.count("<a "), 2)
                self.assertIn('href="../">← Summary</a>', header)
                self.assertIn('href="../../../">All papers</a>', header)
                headings = re.findall(r'<h2 id="[^"]+">(.*?)</h2>', content)
                self.assertEqual(headings[:2], ["What the paper claims", "Starting hypothesis"])
                self.assertEqual(headings[-1], "Current conclusion")
                self.assertIn('<details class="contents">', content)
                self.assertIn('class="table-scroll"', content)
                self.assertIn(' — full research log — ATK Evidence</title>', content)
                self.assertEqual(Page(path).metadata["og:url"], f'{renderer.PUBLIC}papers/{paper["id"]}/journal/')

    def test_an_additional_registry_entry_does_not_change_existing_navigation(self):
        before = {paper["id"]: (renderer.render_summary(paper["id"]), renderer.render(paper["id"]))
                  for paper in PAPERS}
        future_paper = {**PAPERS[0], "id": "future-study", "sequence": 99}
        with patch.object(renderer, "PAPERS", [*PAPERS, future_paper]):
            for study_id, pages in before.items():
                self.assertEqual((renderer.render_summary(study_id), renderer.render(study_id)), pages)
            self.assertIn('href="papers/future-study/"', renderer.render_index())

    def test_earlier_studies_are_not_presented_as_fresh_experiments(self):
        for study_id in ("atk-2022-deep-autoencoder", "tlstgt-2025-water"):
            directory = SITE / "papers" / study_id
            self.assertIn("First attempt", (directory / "index.html").read_text())
            content = (directory / "journal/index.html").read_text()
            self.assertIn("awaiting reassessment", content)
            self.assertIn("earlier workflow", content)
            self.assertIn('href="../earlier-notes.html"', content)
            self.assertTrue((directory / "earlier-notes.html").is_file())
        water = (SITE / "papers/tlstgt-2025-water/journal/index.html").read_text()
        for text in ("co-author", "provisional", "retracted", "229 selected", "20 of 27",
                     "not an unconditional detection ceiling", "The topology is not decorative"):
            self.assertIn(text, water)

    def test_summary_structure_rejects_missing_repeated_or_reordered_headings(self):
        paper = PAPERS[0]
        sections = ["What the paper claims", "Why we doubted it", "What we did", "What we found", "Limits"]
        source = "# Title\n\nIntro.\n\n" + "\n\n".join("## " + heading for heading in sections)
        for invalid in (source.replace("# Title", "Intro"), source + "\n\n# Second title",
                        source.replace("## Limits", "## Unexpected"), source + "\n\n## Limits",
                        source.replace("## What we did", "## TEMP").replace("## Limits", "## What we did").replace("## TEMP", "## Limits"),
                        source.replace("## What we found", "## Read more\n\n## What we found")):
            with self.subTest(source=invalid), patch.object(Path, "read_text", return_value=invalid):
                with self.assertRaisesRegex(ValueError, f"{paper['path']}/README.md"):
                    renderer.render_summary(paper["id"])
        for valid in (source, source + "\n\n## Read more"):
            with self.subTest(source=valid), patch.object(Path, "read_text", return_value=valid):
                self.assertIn('<h1 id="title">Title</h1>', renderer.render_summary(paper["id"]))

    def test_journal_validation_preserves_update_conclusion_and_unique_heading_rules(self):
        source = "# Journal\n\nUpdated: 2026-10-06\n\n## What the paper claims\n\n## Current conclusion"
        invalid_sources = [source.replace("Updated:", "Date:"), source.replace("2026-10-06", "2026-13-06"),
                           source.replace("# Journal", "Intro"), source + "\n\n# Second title",
                           source + "\n\n## A later section", source + "\n\n### What the paper claims"]
        for invalid in invalid_sources:
            with self.subTest(source=invalid), patch.object(Path, "read_text", return_value=invalid):
                with self.assertRaises(ValueError):
                    renderer.render(PAPERS[0]["id"])

    def test_home_escapes_every_registry_text_field(self):
        special = '<sample & "quoted">'
        paper = {**PAPERS[0], "id": special, "title": special, "authors": ["Author " + special],
                 "venue": special, "year": special, "card_status": special, "card_summary": special}
        with patch.object(renderer, "PAPERS", [paper]):
            content = renderer.render_index()
        self.assertNotIn(special, content)
        self.assertIn(f'href="papers/{escape(special)}/"', content)
        for tag, field in (("h3", "title"), ("status", "card_status"), ("summary", "card_summary")):
            self.assertIn(escape(paper[field]), content)
        self.assertIn(escape(paper["authors"][0].split()[-1] + " · " + special + " · " + special), content)

    def test_link_rewriting_preserves_site_repository_image_and_missing_target_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            site = root / "site"
            study = root / "studies/example"
            study.mkdir(parents=True)
            (site / "assets").mkdir(parents=True)
            for path in (study / "README.md", study / "RESEARCH_LOG.md", study / "record.md",
                         study / "image.png", site / "assets/plot.png"):
                path.touch()
            source = ("[own](RESEARCH_LOG.md#entry) [record](record.md) [directory](.) "
                      "[site](../../site/assets/) ![plot](../../site/assets/plot.png) "
                      "![raw](image.png) [external](https://example.com/) [fragment](#main)")
            markdown = renderer.markdown_parser()
            with patch.object(renderer, "ROOT", root), patch.object(renderer, "SITE", site):
                for suffix, asset_prefix in (("index.html", "../../"), ("journal/index.html", "../../../")):
                    tokens = markdown.parse(source)
                    renderer.rewrite_links(tokens, study / "README.md", site / "papers/example" / suffix,
                                           own_journal=study / "RESEARCH_LOG.md" if suffix == "index.html" else None)
                    body = renderer.render_body(markdown, tokens)
                    own = "journal/#entry" if suffix == "index.html" else renderer.GITHUB + "/blob/main/studies/example/RESEARCH_LOG.md#entry"
                    for href in (own, renderer.GITHUB + "/blob/main/studies/example/record.md",
                                 renderer.GITHUB + "/tree/main/studies/example", asset_prefix + "assets/",
                                 "https://example.com/", "#main"):
                        self.assertIn(f'href="{href}"', body)
                    self.assertIn(f'src="{asset_prefix}assets/plot.png"', body)
                    self.assertIn('src="https://raw.githubusercontent.com/fjoad/atk-evidence/main/studies/example/image.png"', body)
                for target in ("missing.md", "../../../outside.md"):
                    with self.subTest(target=target), self.assertRaisesRegex(ValueError, "studies/example/README.md"):
                        renderer.rewrite_links(markdown.parse(f"[bad]({target})"), study / "README.md", site / "index.html")

    def test_poisoning_math_findings_include_assumptions_and_the_averaging_correction(self):
        content = (SITE / "papers/takiddin-2021-robust-poisoning/journal/index.html").read_text()
        for text in ("73.1959–73.2959%", "72.65–72.75%", "39.12–40.54%", "47.37–57.89%",
                     "42.11–43.31%", "44.00–44.82%", "Table IV explicitly averages",
                     "not an unconditional", "single-evaluation interpretation", "log(1−p)",
                     "without reconstruction", "claimed mechanism remains unresolved"):
            self.assertIn(text, content)
        self.assertIn("20% row is not excluded", content)
        self.assertIn("independently averaged metrics", content)

    def test_recent_control_summary_matches_preserved_results(self):
        base = ROOT / "studies/takiddin-2021-robust-poisoning"
        result = json.loads((base / "results/poison_balance_20260923/result.json").read_text())
        content = (SITE / "papers/takiddin-2021-robust-poisoning/journal/index.html").read_text()
        for arm in ("B_p30", "D_p30"):
            for metric in ("DR", "FA", "AUC"):
                self.assertIn(f"{result['comparison'][arm]['primary'][metric]:.2f}%", content)
        self.assertIn("The proposed sequential ensemble has now been tested", content)
        self.assertIn("No complete published result pattern has been reproduced", content)

    def test_completed_neural_pairs_match_their_saved_research_metrics(self):
        base = ROOT / "studies/takiddin-2021-robust-poisoning/results"
        content = (SITE / "papers/takiddin-2021-robust-poisoning/journal/index.html").read_text()
        for record in ("gru_completion_20260924", "sequential_pilot_20261001"):
            for case in ("p00", "p30"):
                result = json.loads((base / record / f"{case}_result.json").read_text())
                self.assertTrue(result["neural"]["training_complete"])
                self.assertEqual(result["neural"]["epochs_completed"], 50)
                for metric in ("DR", "FA", "AUC"):
                    self.assertIn(f"{result['analysis']['primary'][metric]:.2f}%", content)
        conclusion = content.split('id="current-conclusion"', 1)[1]
        self.assertIn("remains untested on electricity data", conclusion)
        self.assertIn("fourth planned case was not run", conclusion)
        self.assertNotIn("ensemble has not been tested", conclusion)

    def test_published_figures_are_unchanged_copies_with_local_image_links(self):
        study = "takiddin-2021-robust-poisoning"
        content = (SITE / "papers" / study / "journal/index.html").read_text()
        for record, filename in (
            ("sequential_gates_20261002", "gate-dynamics.png"),
            ("sequential_cells_20261002", "learning-curves.png"),
        ):
            source = ROOT / "studies" / study / "results" / record / filename
            published = SITE / "papers" / study / "figures" / filename
            self.assertEqual(source.read_bytes(), published.read_bytes())
            self.assertIn(f'<img src="../figures/{filename}" alt="', content)
            self.assertIn(f'href="../figures/{filename}"', content)


if __name__ == "__main__":
    unittest.main()
