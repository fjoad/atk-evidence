"""Check the open-ended paper index and the boundaries of its claims."""

import importlib.util
import json
from pathlib import Path
import re
import tomllib
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
        poisoning = (SITE / "papers/takiddin-2021-robust-poisoning/index.html").read_text()
        self.assertIn('id="20-september-the-printed-loss-loses-the-label"', poisoning)
        self.assertIn('href="#20-september-the-printed-loss-loses-the-label"', poisoning)
        self.assertIn('>The printed loss loses the label</h2>', poisoning)

    def test_home_is_an_open_paper_index_not_a_result_report(self):
        content = (SITE / "index.html").read_text()
        links = [link for link in Page(SITE / "index.html").links if link.startswith("papers/")]
        self.assertEqual(links, [f"papers/{paper['id']}/" for paper in PAPERS])
        for paper in PAPERS:
            self.assertIn(paper["title"], content)
            self.assertIn(str(paper["year"]), content)
        self.assertNotIn("<table", content)
        self.assertNotIn("<article", content)
        self.assertNotIn("class=\"note", content)
        self.assertNotIn("50.93%", content)
        self.assertIn("Paper reproduction notes", content)
        self.assertNotIn("Three attempted", content)
        self.assertNotIn("three papers", content)
        self.assertIn('aria-label="Papers"', content)
        self.assertNotIn('class="number"', content)
        self.assertEqual(content, renderer.render_index())
        self.assertIn(f'href="notebook.css?v={renderer.STYLE_VERSION}"', content)

    def test_notebooks_link_to_the_index_without_a_fixed_switcher(self):
        for paper in PAPERS:
            with self.subTest(paper=paper["id"]):
                content = (ROOT / paper["page"]).read_text()
                self.assertEqual(content, renderer.render(paper["id"]))
                self.assertIn(f'href="../../notebook.css?v={renderer.STYLE_VERSION}"', content)
                self.assertNotIn("paper-tabs", content)
                self.assertNotIn("Choose a paper", content)
                self.assertNotIn("All three papers", content)
                self.assertNotRegex(content, r"Paper \d+ of \d+")
                header = re.search(r'<header class="site-header">.*?</header>', content, re.S).group()
                self.assertEqual(header.count("<a "), 1)
                self.assertIn('href="../../">← All papers</a>', header)
                headings = re.findall(r'<h2 id="[^"]+">(.*?)</h2>', content)
                self.assertEqual(headings[:2], ["What the paper claims", "Starting hypothesis"])
                self.assertEqual(headings[-1], "Current conclusion")
                self.assertIn('<details class="contents">', content)
                self.assertIn('class="table-scroll"', content)

    def test_an_additional_registry_entry_does_not_change_existing_navigation(self):
        paper = PAPERS[0]
        before = renderer.render(paper["id"])
        future_paper = {**paper, "id": "future-study", "sequence": 99}
        with patch.object(renderer, "PAPERS", [*PAPERS, future_paper]):
            self.assertEqual(renderer.render(paper["id"]), before)

    def test_earlier_studies_are_not_presented_as_fresh_experiments(self):
        for paper in PAPERS[:2]:
            content = (ROOT / paper["page"]).read_text()
            self.assertIn("awaiting reassessment", content)
            self.assertIn("earlier workflow", content)
            self.assertIn('href="earlier-notes.html"', content)
            self.assertTrue((ROOT / paper["page"]).with_name("earlier-notes.html").is_file())
        water = (SITE / "papers/tlstgt-2025-water/index.html").read_text()
        for text in ("co-author", "provisional", "retracted", "229 selected", "20 of 27",
                     "not an unconditional detection ceiling", "The topology is not decorative"):
            self.assertIn(text, water)

    def test_poisoning_math_findings_include_assumptions_and_the_averaging_correction(self):
        content = (SITE / "papers/takiddin-2021-robust-poisoning/index.html").read_text()
        for text in ("73.1959–73.2959%", "72.65–72.75%", "39.12–40.54%", "47.37–57.89%",
                     "42.11–43.31%", "44.00–44.82%", "Table IV explicitly averages",
                     "not an unconditional", "single-evaluation interpretation", "log(1−p)",
                     "without reconstruction", "still untested"):
            self.assertIn(text, content)
        self.assertIn("20% row is not excluded", content)
        self.assertIn("independently averaged metrics", content)

    def test_recent_control_summary_matches_preserved_results(self):
        base = ROOT / "studies/takiddin-2021-robust-poisoning"
        result = json.loads((base / "results/poison_balance_20260923/result.json").read_text())
        content = (SITE / "papers/takiddin-2021-robust-poisoning/index.html").read_text()
        for arm in ("B_p30", "D_p30"):
            for metric in ("DR", "FA", "AUC"):
                self.assertIn(f"{result['comparison'][arm]['primary'][metric]:.2f}%", content)
        self.assertIn("main proposed sequential ensemble has not been tested", content)
        self.assertIn("No complete published result pattern has been reproduced", content)


if __name__ == "__main__":
    unittest.main()
