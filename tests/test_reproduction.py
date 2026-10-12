"""Regression tests for experimental semantics, cache safety and Java integration."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reproduction.config import plan, settings
from reproduction.data import generate, prepare_data, prepare_queries, read_real, select_queries, write_values
from reproduction.java import check_java, compile_java, run_java
from reproduction.plots import curve_points


class PythonCompatibilityTests(unittest.TestCase):
    def test_setup_accepts_old_and_new_interpreters(self):
        from unittest.mock import patch
        from reproduction.cli import setup
        for version in ((3, 8), (3, 9), (3, 13), (3, 14), (3, 15)):
            with self.subTest(version=version), \
                    patch("reproduction.cli.sys.version_info", version), \
                    patch("reproduction.cli.subprocess.run") as run, \
                    patch("builtins.print"):
                setup()
                self.assertEqual(run.call_count, 4)
        with patch("reproduction.cli.sys.version_info", (3, 7)), \
                patch("reproduction.cli.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "3.8 or newer"):
                setup()
            run.assert_not_called()


class WorkloadTests(unittest.TestCase):
    def test_historical_heavy_workload_keeps_extra_normal_segment(self):
        data = generate("normal_and_distinct_42", 1000, 123)
        self.assertEqual(len(data), 1500)
        self.assertTrue(np.all(np.isin(data[1000:], np.arange(-21, 21))))

    def test_seeded_workloads_do_not_depend_on_global_rng(self):
        reference = generate("signed_loguniform", 200, 456)
        np.random.seed(999)
        np.random.random(10000)
        np.testing.assert_array_equal(reference, generate("signed_loguniform", 200, 456))

    def test_query_ranks_include_duplicates(self):
        queries, truth = select_queries(np.array([1, 1, 2, 2, 2, 3]), 3)
        np.testing.assert_array_equal(queries, [1, 2, 2])
        np.testing.assert_array_equal(truth, [2, 5, 5])
        queries, truth = select_queries(np.array([1, 1, 2]), 100)
        self.assertEqual(len(queries), 3)
        np.testing.assert_array_equal(truth, [2, 2, 3])

    def test_cache_rejects_corruption_and_separates_query_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, data = prepare_data("normal", 200, 42, tmp, tmp, None)
            q100, _ = prepare_queries(directory, data, 100)
            q50, _ = prepare_queries(directory, data, 50)
            self.assertEqual(q100["actual_queries"], 100)
            self.assertEqual(q50["actual_queries"], 50)
            with (directory / "data.txt").open("a") as stream:
                stream.write("123\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_data("normal", 200, 42, tmp, tmp, None)

    def test_power_missing_values_preserve_source_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "household_power_consumption/household_power_consumption.txt"
            path.parent.mkdir()
            path.write_text("Date;Time;Global_active_power\na;b;1.5\na;b;?\na;b;\na;b;2.5\n")
            data, stats = read_real("power", tmp, 42, limit=20)
            np.testing.assert_array_equal(data, [1.5, 2.5])
            self.assertEqual(stats["rows"], 4)
            self.assertEqual(stats["missing"], 2)

    def test_books_header_and_little_endian_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SOSDdata/books_800M_uint64"
            path.parent.mkdir()
            np.array([4, 5, 10, 15, 20], dtype="<u8").tofile(path)
            data, _ = read_real("books", tmp, 42, limit=4)
            np.testing.assert_array_equal(np.sort(data), [5, 10, 15, 20])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "header"):
                read_real("books", tmp, 42, limit=4)

    def test_paper_merge_plan_and_timing_baseline_parameters(self):
        cfg = settings("paper")
        tasks = plan(cfg, ["merge", "update"])
        self.assertEqual({t["dataset"] for t in tasks if t["suite"] == "merge"}, {"normal", "hepmass"})
        self.assertTrue(all(t["parts"] == 10000 for t in tasks if t["suite"] == "merge"))
        self.assertFalse(any(t["sketch"] == "gk" and t["suite"] == "merge" for t in tasks))
        self.assertEqual({t["parameter"] for t in tasks if t["sketch"] == "moments" and t["suite"] == "update"}, {15})

    def test_plot_sort_keeps_size_and_error_paired(self):
        records = [dict(task=dict(suite="synthetic", dataset="normal", sketch="tdigest", variant=None,
            parameter=k, requested_queries=100, nominal_n=2000), status="ok", serialized_bytes=x,
            average_error=y) for k, x, y in [(10, 300, 0.3), (20, 200, 0.2)]]
        curves = curve_points(records, dict(suite="synthetic", dataset="normal", x="serialized_bytes",
                                           metric="average_error"), 4500)
        self.assertEqual(curves["t-digest"], [(200, 0.2, 0), (300, 0.3, 0)])

    def test_merge_partition_mean_uses_actual_merge_count(self):
        from unittest.mock import patch
        from reproduction.runner import measure
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "raw").mkdir()
            task = dict(repeat=0, sketch="splinesketch", requested_queries=2, parameter=25,
                        parts=10, params=None, nominal_n=1000)
            cfg = dict(seed=42, java_heap="1g", java_timeout=120)
            data = dict(key="fixture", actual_n=1500, files={"data.txt": "fixture"})
            queries = dict(actual_queries=2)
            with patch("reproduction.runner.run_java", return_value=(np.array([1, 20]), [400, 9000, 2000], {})):
                result = measure(task, cfg, out, data, queries, np.array([0, 10]), out)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["update_or_merge_us"], 1)
            self.assertAlmostEqual(result["average_error"], 5.5 / 1500)

    def test_resume_skips_verified_results_and_rejects_changed_configuration(self):
        from unittest.mock import patch
        from reproduction.runner import execute
        cfg = settings("smoke")
        cfg.update(sizes=[25], moment_k=[7], kll_k=[8], sketches=["splinesketch"], n=100, queries=10)
        tasks = plan(cfg, ["synthetic"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            output = path / "out"
            with patch("reproduction.runner.run_java", return_value=(np.arange(10), [400, 1000, 1000], {})) as java:
                first = execute(cfg, tasks, output, path, path / "cache", {"test": "JDK17"})
                self.assertEqual(java.call_count, 1)
                self.assertEqual(first["counts"]["ok"], 1)
                second = execute(cfg, tasks, output, path, path / "cache", {"test": "JDK17"})
                self.assertEqual(java.call_count, 1)
                self.assertEqual(second["counts"]["pending"], 0)
                with self.assertRaisesRegex(ValueError, "different configuration"):
                    execute(dict(cfg, seed=123), tasks, output, path, path / "cache", {"test": "JDK17"})

    def test_provenance_without_git_for_container_or_archive(self):
        from unittest.mock import patch
        from reproduction.runner import provenance
        with patch("reproduction.runner.shutil.which", return_value=None):
            data = provenance()
        self.assertIsNone(data["git_commit"])
        self.assertIsNone(data["git_dirty"])
        self.assertIn("java/other_sketches/KLLProgram.java", data["source_hashes"])


class JavaVersionTests(unittest.TestCase):
    def test_accepts_17_and_newer_with_different_compiler_version(self):
        from unittest.mock import patch
        for runtime, compiler in ((17, 17), (21, 21), (25, 25), (17, 25)):
            with self.subTest(runtime=runtime, compiler=compiler), \
                    patch("reproduction.java.shutil.which", side_effect=lambda name: name), \
                    patch("reproduction.java.java_version", side_effect=[
                        f'openjdk version "{runtime}" 2026-01-01', f'javac {compiler}.0.1']):
                self.assertIn("java", check_java())

    def test_rejects_missing_old_and_unrecognized_commands(self):
        from unittest.mock import patch
        with patch("reproduction.java.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Both java and javac"):
                check_java()
        for runtime, compiler in (('java version "1.8.0"', 'javac 17'),
                                  ('openjdk 21.0.1', 'javac 16.0.2'),
                                  ('unknown runtime', 'javac 21')):
            with self.subTest(runtime=runtime, compiler=compiler), \
                    patch("reproduction.java.shutil.which", side_effect=lambda name: name), \
                    patch("reproduction.java.java_version", side_effect=[runtime, compiler]):
                with self.assertRaises(RuntimeError):
                    check_java()


class JavaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            check_java()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error))
        compile_java()

    def test_wrappers_accept_paths_with_spaces_and_return_valid_metrics(self):
        with tempfile.TemporaryDirectory(prefix="sketch test ") as tmp:
            data = Path(tmp) / "input values.txt"
            queries = Path(tmp) / "query values.txt"
            write_values(data, np.random.RandomState(42).normal(size=501))
            write_values(queries, np.linspace(-2, 2, 20))
            for sketch in ("splinesketch", "splinesketch_mg", "kll", "moments", "tdigest", "gk", "ddsketch", "adjustable"):
                with self.subTest(sketch=sketch):
                    params = [2, 3, 1.5, 1e-8, 0.01, 1.25, 3] if sketch == "adjustable" else None
                    ranks, metrics, _ = run_java(sketch, data, queries, 501, 15, params=params)
                    self.assertEqual(len(ranks), 20)
                    self.assertGreater(metrics[0], 0)
                    self.assertTrue(np.all((ranks >= 0) & (ranks <= 501)))

    def test_kll_same_seed_repeats_exact_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, queries = Path(tmp) / "data.txt", Path(tmp) / "queries.txt"
            write_values(data, np.random.RandomState(1).normal(size=2000))
            write_values(queries, np.linspace(-2, 2, 100))
            a, _, _ = run_java("kll", data, queries, 2000, 8, seed=42)
            b, _, _ = run_java("kll", data, queries, 2000, 8, seed=42)
            np.testing.assert_array_equal(a, b)

    def test_moment_merge_keeps_nondivisible_remainder(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, queries = Path(tmp) / "data.txt", Path(tmp) / "queries.txt"
            values = np.concatenate([np.arange(500) / 500, np.full(3, 20)])
            write_values(data, values)
            write_values(queries, [1, 20])
            merged, _, _ = run_java("moments", data, queries, len(values), 7, parts=10)
            # The final outliers must affect the estimated distribution.
            self.assertLess(merged[0], len(values) - 1)

    def test_unsupported_merge_is_rejected(self):
        for sketch in ("gk", "splinesketch_mg"):
            with self.subTest(sketch=sketch), self.assertRaisesRegex(ValueError, "merge is unavailable"):
                run_java(sketch, "unused", "unused", 1000, 100, parts=10)


if __name__ == "__main__":
    unittest.main()
