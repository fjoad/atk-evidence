"""Check the three-paper notebook design and the boundaries of its claims."""

import importlib.util
import json
from pathlib import Path
import re
import tomllib
import unittest

from tests.test_public_report import Page

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PAPERS = tomllib.loads((ROOT / "studies/registry.toml").read_text())["studies"]
spec = importlib.util.spec_from_file_location("notebook_renderer", ROOT / "scripts/render_journals.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class NotebookSiteTests(unittest.TestCase):
    def test_home_is_a_three_paper_index_not_a_result_report(self):
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

    def test_all_notebooks_are_generated_have_three_tabs_and_end_in_conclusion(self):
        for paper in PAPERS:
            with self.subTest(paper=paper["id"]):
                content = (ROOT / paper["page"]).read_text()
                self.assertEqual(content, renderer.render(paper["id"]))
                self.assertIn('href="../../notebook.css"', content)
                tabs = re.search(r'<nav class="paper-tabs".*?</nav>', content).group()
                self.assertEqual(tabs.count("<a "), 3)
                self.assertEqual(tabs.count('aria-current="page"'), 1)
                self.assertIn(f'href="../{paper["id"]}/" aria-current="page"', tabs)
                headings = re.findall(r'<h2 id="[^"]+">(.*?)</h2>', content)
                self.assertEqual(headings[:2], ["What the paper claims", "Starting hypothesis"])
                self.assertEqual(headings[-1], "Current conclusion")
                self.assertIn('<details class="contents">', content)
                self.assertIn('class="table-scroll"', content)

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
