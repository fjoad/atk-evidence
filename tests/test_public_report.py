"""Static checks for the public account; no models, network, or browser."""

from __future__ import annotations

import json
import re
import subprocess
import unittest
import xml.etree.ElementTree as ET
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
STUDY = ROOT / "studies/atk-2022-deep-autoencoder"
RECORDS = STUDY / "results/clean_reader_anchor_20260831"
REPORT = SITE / "papers/atk-2022-deep-autoencoder/reproduction/index.html"


class Page(HTMLParser):
    def __init__(self, path: Path):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.metadata = {}
        self.metrics = {}
        self.h1_count = 0
        self.row = None
        self.cells = []
        self.cell = None
        self.feed(path.read_text())
        self.close()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == "h1":
            self.h1_count += 1
        if tag == "meta":
            self.metadata[attrs.get("property", attrs.get("name"))] = attrs.get("content")
        if tag == "tr" and "data-metric" in attrs:
            self.row = attrs["data-metric"]
            self.cells = []
        if self.row and tag in ("th", "td"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.cells.append("".join(self.cell).strip())
            self.cell = None
        if tag == "tr" and self.row:
            self.metrics[self.row] = self.cells
            self.row = None


@lru_cache(maxsize=None)
def git_file(revision: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True
    )


class PublicReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((RECORDS / "result.json").read_text())
        cls.audit = json.loads((RECORDS / "clean_reader_anchor_audit.json").read_text())
        cls.pages = {path: Page(path) for path in SITE.rglob("*.html")}
        cls.report_text = REPORT.read_text()

    def test_complete_current_result_matches_saved_metrics(self):
        rows = self.pages[REPORT].metrics
        self.assertEqual(set(rows), {"DR", "FA", "SP", "PR", "ACC", "F1", "AUC"})
        for page in self.pages.values():
            for metric, cells in page.metrics.items():
                self.assertEqual(cells[1], f"{self.result['reported_table_3'][metric]:.2f}")
                self.assertEqual(cells[2], f"{self.result['metrics'][metric]:.2f}")

    def test_readme_has_complete_current_comparison(self):
        readme = (ROOT / "README.md").read_text()
        for metric in ("DR", "FA", "SP", "PR", "ACC", "F1", "AUC"):
            pair = f"{self.result['reported_table_3'][metric]:.2f}% | {self.result['metrics'][metric]:.2f}%"
            self.assertIn(pair, readme)

    def test_static_links_and_fragments_resolve(self):
        for source, page in self.pages.items():
            for link in page.links:
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc:
                    continue
                target = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source
                if target.is_dir():
                    target /= "index.html"
                with self.subTest(source=source, link=link):
                    self.assertTrue(target.is_relative_to(SITE))
                    self.assertTrue(target.is_file())
                    if parsed.fragment and target.suffix == ".html":
                        self.assertIn(unquote(parsed.fragment), self.pages[target].ids)

    def test_source_links_name_existing_files_and_valid_lines(self):
        prefix = "/fjoad/atk-evidence/blob/"
        for source, page in self.pages.items():
            for link in page.links:
                parsed = urlsplit(link)
                if parsed.netloc != "github.com" or not parsed.path.startswith(prefix):
                    continue
                revision, path = unquote(parsed.path[len(prefix):]).split("/", 1)
                with self.subTest(source=source, link=link):
                    content = (ROOT / path).read_text() if revision == "main" else git_file(revision, path)
                    if parsed.fragment.startswith("L"):
                        match = re.fullmatch(r"L(\d+)(?:-L(\d+))?", parsed.fragment)
                        self.assertIsNotNone(match)
                        start, end = int(match[1]), int(match[2] or match[1])
                        self.assertTrue(1 <= start <= end <= len(content.splitlines()))

    def test_pages_have_unique_ids_and_record_specific_metadata(self):
        for source, page in self.pages.items():
            with self.subTest(source=source):
                self.assertEqual(page.h1_count, 1)
                self.assertEqual(len(page.ids), len(set(page.ids)))
                for key in ("description", "og:title", "og:description", "twitter:title", "twitter:description"):
                    self.assertTrue(page.metadata.get(key))
                if source != SITE / "index.html":
                    self.assertNotIn("og:image", page.metadata)
                    self.assertNotIn("twitter:image", page.metadata)

    def test_model_diagram_matches_recorded_layer_order(self):
        svg = ET.parse(REPORT.parent / "model.svg")
        namespace = {"s": "http://www.w3.org/2000/svg"}
        widths = [int(node.text) for node in svg.findall(".//s:text", namespace)
                  if node.text and node.text.isdigit()]
        self.assertEqual(widths, [48, 400, 300, 200, 100, 100, 200, 300, 400, 48])
        self.assertIn("450,448", self.report_text)
        self.assertIn(self.result["git_commit"], self.report_text)

    def test_diagnostic_numbers_match_records(self):
        scores = self.audit["score_audit"]
        self.assertIn(f"{scores['reported_direction']['ACC']:.5f}%", self.report_text)
        self.assertIn(f"{scores['reversed_direction_control']['ACC']:.2f}%", self.report_text)
        for baseline in ("zero_reconstruction_full_test", "softmax_projection_floor_full_test"):
            for metric in ("ACC", "AUC"):
                self.assertIn(f"{self.result['baselines'][baseline][metric]:.2f}%", self.report_text)
        for count in ("TP", "TN", "FP", "FN"):
            self.assertIn(f"{self.result['metrics'][count]:,}", self.report_text)

    def test_historical_pages_and_results_are_distinguished(self):
        previous = (SITE / "papers/atk-2022-deep-autoencoder/index.html").read_text()
        water = (SITE / "papers/tlstgt-2025-water/index.html").read_text()
        self.assertIn("This is the earlier account, not the current reproduction", previous)
        self.assertIn("26.18%", previous)
        self.assertIn("58.22%", previous)
        self.assertIn("229 selected test sizes", water)
        self.assertIn("20 of 27", water)
        self.assertIn("not an unconditional detection ceiling", water)
        self.assertIn("not been run", self.report_text)
        self.assertIn("There are no seed-level confidence intervals here", self.report_text)
        for name in ("atk-2022-deep-autoencoder", "tlstgt-2025-water"):
            self.assertTrue((SITE / f"reports/{name}.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
