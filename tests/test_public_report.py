"""Static checks for the public account; no models, network, or browser."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import unittest
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
STUDY = ROOT / "studies/atk-2022-deep-autoencoder"
RECORDS = STUDY / "results/clean_reader_anchor_20260831"
DIAGNOSTICS = STUDY / "results/post_anchor_20260831"
ASSUMPTIONS = STUDY / "results/source_assumption_20260831"
SIGMOID_RANGE = STUDY / "results/sigmoid_sanity_20260831"
SIGMOID_FIT = STUDY / "results/sigmoid_fit_20260831"
PAPER_TIME = STUDY / "results/lstm_sae_paper_time_20260901.json"
REPORT = SITE / "papers/atk-2022-deep-autoencoder/reproduction/index.html"
REPORT_SOURCE = ROOT / "reports/atk-2022-deep-autoencoder/main.tex"


class Page(HTMLParser):
    def __init__(self, path: Path):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.metadata = {}
        self.metrics = {}
        self.bounds = {}
        self.controls = {}
        self.gains = {}
        self.scalings = {}
        self.ranges = {}
        self.fits = {}
        self.lstm = {}
        self.h1_count = 0
        self.row = None
        self.row_group = None
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
        if tag == "tr":
            groups = {"metric": self.metrics, "bound": self.bounds,
                      "control": self.controls, "gain": self.gains,
                      "scaling": self.scalings, "range": self.ranges,
                      "fit": self.fits, "lstm": self.lstm}
            for name, group in groups.items():
                if f"data-{name}" in attrs:
                    self.row = attrs[f"data-{name}"]
                    self.row_group = group
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
            self.row_group[self.row] = self.cells
            self.row = None
            self.row_group = None


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
        cls.diagnostics = json.loads((DIAGNOSTICS / "full/diagnostics.json").read_text())
        cls.control = json.loads((DIAGNOSTICS / "energy_band_control.json").read_text())
        cls.assumptions = json.loads((ASSUMPTIONS / "full.json").read_text())
        cls.sigmoid_range = json.loads((SIGMOID_RANGE / "full.json").read_text())
        cls.sigmoid_fit = json.loads((SIGMOID_FIT / "small.json").read_text())
        cls.paper_time = json.loads(PAPER_TIME.read_text())

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

    def test_paper_time_lstm_result_matches_audited_record(self):
        rows = self.pages[REPORT].lstm
        expected = {"DR", "FA", "SP", "PR", "ACC", "F1", "AUC"}
        self.assertEqual(set(rows), expected)
        reported = self.paper_time["paper_claim"]
        observed = self.paper_time["observed"]
        targets = {"DR": 85, "FA": 13, "SP": 87, "PR": 85,
                   "ACC": 86, "F1": 85, "AUC": 82}
        for metric in expected:
            self.assertEqual(rows[metric][1], f"{targets[metric]:.2f}")
            self.assertEqual(rows[metric][2], f"{observed['printed_cutoff'][metric]:.2f}")
        self.assertEqual(reported["training_minutes_full_iset"], 183)
        self.assertFalse(observed["reported_corner_reached_at_any_cutoff"])
        self.assertIn("7,036,998", self.report_text)
        self.assertIn("23.02%", self.report_text)
        self.assertIn("23.98 hours", self.report_text)
        self.assertIn("1.16 times faster", self.report_text)
        self.assertIn("unlimited-time impossibility", self.report_text)

    def test_current_entry_points_publish_paper_time_lstm_finding(self):
        home = (SITE / "index.html").read_text()
        readme = (ROOT / "README.md").read_text()
        legacy = (SITE / "papers/atk-2022-deep-autoencoder/index.html").read_text()
        for text in (home, readme):
            for value in ("16.62", "31.91", "23.02", "23.98"):
                self.assertIn(value, text)
            self.assertIn("unlimited-time impossibility", text)
            self.assertIn("183 minutes", text)
        self.assertIn("paper-time LSTM-SAE", legacy)
        self.assertIn("FC-SAE scientific report through 1 September", legacy)

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
        self.assertIn("have <strong>not been run</strong>", self.report_text)
        self.assertIn("There are no seed-level confidence intervals here", self.report_text)
        for name in ("atk-2022-deep-autoencoder", "tlstgt-2025-water"):
            self.assertTrue((SITE / f"reports/{name}.pdf").is_file())

    def test_output_domain_bounds_are_rounded_up_from_saved_results(self):
        bound = self.diagnostics["bounds"]["full"]["printed"]
        values = {"ACC": bound["max_ACC"], "AUC": bound["max_AUC"],
                  "DR15": bound["at_FA_cap"]["15.0"]["max_DR"],
                  "DR155": bound["at_FA_cap"]["15.5"]["max_DR"]}
        rows = self.pages[REPORT].bounds
        self.assertEqual(set(rows), set(values))
        for key, value in values.items():
            self.assertEqual(rows[key][2], f"{math.ceil(value * 100) / 100:.2f}%")
            self.assertGreaterEqual(float(rows[key][2].rstrip("%")), value)
        self.assertFalse(bound["target_pair_not_excluded"])
        self.assertFalse(bound["rounded_target_pair_not_excluded"])
        self.assertIn("Limits are rounded upward", self.report_text)

    def test_useful_work_gains_and_intervals_match_customer_records(self):
        comparisons = self.diagnostics["customer_statistics"]["comparisons"]
        rows = self.pages[REPORT].gains
        self.assertEqual(set(rows), set(comparisons))
        for key, comparison in comparisons.items():
            low, high = comparison["original_ACC_gain_95CI_pp"]
            self.assertEqual(rows[key][1], f"{comparison['original_ACC_gain_pp']:+.3f}")
            self.assertEqual(rows[key][2], f"[{low:.3f}, {high:.3f}]")
        self.assertEqual(self.diagnostics["customer_statistics"]["resamples"], 2000)
        self.assertIn("assume exchangeable customer clusters", self.report_text)
        self.assertIn("not a matched trained-versus-untrained causal test", self.report_text)

    def test_adaptive_control_table_matches_its_separate_record(self):
        rows = self.pages[REPORT].controls
        values = self.control["pair_weighted_within_bin_AUC"]
        self.assertEqual(set(rows), set(values))
        for key, value in values.items():
            self.assertEqual(rows[key][1], f"{value:.2f}%")
        self.assertEqual(self.control["rows"], 70000)
        self.assertEqual(self.control["sample_source_days"], 10000)
        self.assertIn("This statistic has no confidence interval here", self.report_text)
        self.assertIn("not an independent confirmation", self.report_text)

    def test_public_diagnostic_figures_preserve_scientific_exports(self):
        for name in ("output-domain-envelope.svg", "useful-work-by-attack.svg"):
            saved = DIAGNOSTICS / "full" / name
            public = REPORT.parent / name
            self.assertEqual(public.read_bytes(), saved.read_bytes())
            self.assertEqual(ET.parse(public).getroot().tag, "{http://www.w3.org/2000/svg}svg")

    def test_diagnostic_provenance_and_original_result_are_unchanged(self):
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        execution = json.loads((DIAGNOSTICS / "execution.json").read_text())
        for stage in ("pilot", "full"):
            self.assertEqual(sha(DIAGNOSTICS / stage / "diagnostics.json"),
                             execution[f"{stage}_sha256"])
        self.assertEqual(sha(DIAGNOSTICS / "energy_band_control.json"),
                         execution["adaptive_control"]["result_sha256"])
        for record, script, contract in (
            (self.diagnostics, "post_anchor_diagnostics.py", "POST_ANCHOR_DIAGNOSTICS.md"),
            (self.control, "energy_band_control.py", "ENERGY_BAND_CONTROL.md"),
        ):
            self.assertEqual(sha(STUDY / "checks" / script), record["script_sha256"])
            self.assertEqual(sha(STUDY / contract), record["contract_sha256"])
            self.assertEqual(sha(RECORDS / "result.json"), record["source_result_sha256"])

    def test_bound_scope_remains_visible_in_all_current_accounts(self):
        for path in (ROOT / "README.md", SITE / "index.html", REPORT):
            content = path.read_text()
            with self.subTest(path=path):
                self.assertIn("50.93%", content)
                self.assertIn("9.25%", content)
                self.assertIn("Softmax", content)
                self.assertIn("prepared", content)
        self.assertIn("not certified interval arithmetic", self.report_text)
        self.assertIn("not to another dataset, normalization, output domain, or score", self.report_text)

    def test_normalization_detection_limits_match_all_three_frozen_cases(self):
        rows = self.pages[REPORT].scalings
        expected = {"joint_feature_softmax", "joint_scalar_softmax", "separate_class_feature_softmax"}
        self.assertEqual(set(rows), expected)
        for key in expected:
            value = self.assumptions["branches"][key]["printed"]["at_printed_threshold"]["max_DR"]
            self.assertEqual(rows[key][1], f"{math.ceil(value * 100) / 100:.2f}%")
        self.assertIn("4,504,602", self.report_text)
        self.assertIn("does not exhaust every reasonable normalization", self.report_text)

    def test_sigmoid_controls_preserve_direction_and_original_row_scope(self):
        rows = self.pages[REPORT].ranges
        self.assertEqual(set(rows), {"softmax_printed", "sigmoid_printed", "sigmoid_reversed"})
        for key, branch, direction in (
            ("softmax_printed", "joint_feature_softmax", "printed"),
            ("sigmoid_printed", "joint_feature_sigmoid_control", "printed"),
            ("sigmoid_reversed", "joint_feature_sigmoid_control", "reversed_control"),
        ):
            result = self.assumptions["branches"][branch][direction]
            value = result["at_FA_cap"]["15.0"]["max_DR"]
            self.assertEqual(rows[key][2], f"{math.ceil(value * 100) / 100:.2f}%")
            if direction == "printed":
                value = result["at_printed_threshold"]["max_DR"]
                self.assertEqual(rows[key][1], f"{math.ceil(value * 100) / 100:.2f}%")
        self.assertIn("No synthetic benign rows are included", self.report_text)
        self.assertIn("first calculation did not train a Sigmoid model", self.report_text)
        self.assertIn("A permitted target is not a reproduced target", self.report_text)

    def test_complete_sigmoid_range_and_fitted_scores_are_current(self):
        complete = self.sigmoid_range["views"]["full"]["bounds"]
        printed = complete["printed"]
        reversed_control = complete["reversed_control"]
        for value in (
            printed["at_FA_cap"]["15.0"]["max_DR"],
            reversed_control["at_FA_cap"]["15.0"]["max_DR"],
            printed["at_printed_threshold"]["min_FA"],
        ):
            self.assertIn(f"{math.ceil(value * 100) / 100:.2f}%", self.report_text)
        self.assertTrue(printed["target_pair_not_excluded"])
        self.assertTrue(reversed_control["target_pair_not_excluded"])

        rows = self.pages[REPORT].fits
        expected = {
            "softmax_printed": ("softmax", "printed"),
            "sigmoid_printed": ("sigmoid", "printed"),
            "softmax_reversed": ("softmax", "reversed_control"),
            "sigmoid_reversed": ("sigmoid", "reversed_control"),
        }
        self.assertEqual(set(rows), set(expected))
        for key, (head, direction) in expected.items():
            result = self.sigmoid_fit["models"][head]["selected"]["sampled_prepared"][direction]["all_cutoffs_diagnostic"]
            values = (result["at_FA_cap"]["15.0"]["max_DR"],
                      result["at_FA_cap"]["15.5"]["max_DR"],
                      result["max_ACC"], result["max_AUC"])
            displayed = [f"{Decimal(str(value)).quantize(Decimal('0.00001'), rounding=ROUND_HALF_UP)}%"
                         for value in values]
            self.assertEqual(rows[key][1:], displayed)
            self.assertFalse(result["target_pair_not_excluded"])
            self.assertFalse(result["rounded_target_pair_not_excluded"])
        sigmoid = self.sigmoid_fit["models"]["sigmoid"]
        self.assertEqual(sigmoid["selected_epoch"], 10)
        self.assertLess(sigmoid["epochs"][-1]["val_loss"], sigmoid["epochs"][0]["val_loss"])
        self.assertIn("not a universal Sigmoid impossibility proof", self.report_text)

    def test_all_current_entry_points_name_the_fitted_sigmoid_scope(self):
        for path in (ROOT / "README.md", SITE / "index.html", REPORT):
            content = path.read_text()
            with self.subTest(path=path):
                self.assertIn("9.75%", content)
                self.assertIn("25.39%", content)
                self.assertIn("still improving", content)

    def test_downloadable_report_source_is_current_and_bounded(self):
        source = REPORT_SOURCE.read_text()
        for value in ("40.18", "50.93", "9.74935", "25.39063"):
            self.assertIn(value, source)
        self.assertIn("not an all-Sigmoid", source)
        self.assertIn("Four proposed Table III models", source)
        self.assertIn("No current", source)
        self.assertIn("fabrication or author intent", source)

    def test_plain_language_questions_precede_the_relevant_checks(self):
        question = "even if we could choose the most favorable allowed output separately for every attack, could enough attacks cross the detection threshold?"
        self.assertIn(question, self.report_text)
        self.assertLess(self.report_text.index(question), self.report_text.index('id="bound-table"'))
        self.assertIn('id="source-assumptions"', self.report_text)
        self.assertIn('id="sigmoid-range"', self.report_text)
        self.assertIn("Escaping a bound means", self.report_text)


if __name__ == "__main__":
    unittest.main()
