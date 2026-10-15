"""Verify the paper uses saved run data and covers every TeX graphics reference."""
from pathlib import Path
import tempfile
import unittest

from reproduction.common import ROOT, write_json
from reproduction.paper import prepare_paper


class PaperBuildTests(unittest.TestCase):
    def test_stages_sources_and_maps_saved_measurements_to_tex_assets(self):
        with tempfile.TemporaryDirectory(prefix="paper build ") as tmp:
            results = Path(tmp) / "saved run"
            output = Path(tmp) / "compiled paper"
            write_json(results / "run.json", {"config": {"max_plot_bytes": 4500}})
            for i, (suite, dataset, status) in enumerate([
                    ("synthetic", "normal", "ok"), ("real", "books", "ok"),
                    ("merge", "normal", "ok"), ("real", "hepmass", "failed")]):
                write_json(results / "records" / f"{i}.json", dict(
                    task=dict(suite=suite, dataset=dataset, sketch="splinesketch", variant=None,
                              parameter=25, requested_queries=100, nominal_n=2000),
                    status=status, serialized_bytes=400, actual_n=2000, actual_queries=100,
                    average_error=0.01, maximum_error=0.02, update_or_merge_us=0.1, query_us=0.2))
            source_before = (ROOT / "paper/splineSketchCameraReady.tex").read_bytes()
            report = prepare_paper(results, output)
            self.assertEqual((output / "splineSketchCameraReady.tex").read_bytes(), source_before)
            self.assertEqual((ROOT / "paper/splineSketchCameraReady.tex").read_bytes(), source_before)
            self.assertTrue((output / "literature.bib").is_file())
            self.assertTrue((output / "buckets-spline.pdf").is_file())
            self.assertTrue((output / "buckets-joinSplit.pdf").is_file())
            self.assertTrue((output / "plots/legend.png").is_file())
            self.assertEqual(report["failed_measurements"], 1)
            self.assertIn("hepmass_average_nolegend.pdf", report["missing_panels"])
            for name, suite, dataset, metric in [
                    ("normal_logN8_average_nolegend.pdf", "synthetic", "normal", "average_error"),
                    ("sosd_books_max_nolegend.pdf", "real", "books", "maximum_error"),
                    ("normal_logN8_mergingM10000_update_time_nolegend.pdf", "merge", "normal", "update_or_merge_us")]:
                panel = report["panels"][name]
                self.assertFalse(panel["missing"])
                self.assertEqual(panel["spec"]["suite"], suite)
                self.assertEqual(panel["spec"]["dataset"], dataset)
                self.assertEqual(panel["spec"]["metric"], metric)
                self.assertTrue((output / "plots" / name).read_bytes().startswith(b"%PDF"))

    def test_missing_run_does_not_create_build_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "build"
            with self.assertRaisesRegex(FileNotFoundError, "No saved run"):
                prepare_paper(Path(tmp) / "missing", output)
            self.assertFalse(output.exists())

    def test_rejects_build_directory_inside_sources(self):
        with self.assertRaisesRegex(ValueError, "outside the paper source"):
            prepare_paper(ROOT / "results/smoke", ROOT / "paper/build")


if __name__ == "__main__":
    unittest.main()
