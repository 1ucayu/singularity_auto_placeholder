import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from singularity_placeholder import profile, session


class ProfileValidationTests(unittest.TestCase):
    def test_defaults_are_portable_and_independent(self):
        first = profile.normalize({})
        self.assertIsNone(first["gpu_count"])
        self.assertIsNone(first["gpu_model"])
        self.assertIsNone(first["blob_root"])
        self.assertEqual(first["blob_prefix"], "")
        self.assertIsNone(first["tunnel_name"])
        self.assertFalse(first["tunnel"])
        first["aml"]["input_name"] = "changed"
        self.assertEqual(profile.normalize({})["aml"]["input_name"], "storage")

    def test_partial_config_keeps_defaults_and_normalizes_selectors(self):
        config = profile.normalize({"gpu_model": "H100", "gpu_count": 2, "gpus": "GPU-aaa, GPU-bbb", "aml": {"keep_alive_seconds": 3600}})
        self.assertEqual(config["gpus"], "GPU-aaa,GPU-bbb")
        self.assertEqual(config["poll_seconds"], 1)
        self.assertEqual(config["aml"]["keep_alive_seconds"], 3600)
        self.assertEqual(config["aml"]["clone_retry_seconds"], 30)

    def test_invalid_profiles_fail_offline(self):
        cases = [
            [], {"schema_version": 2}, {"schema_version": True},
            {"gpu_cout": 4}, {"aml": {"keep_alive_second": 60}}, {"aml": None},
            {"gpu_count": True}, {"gpu_count": 0}, {"gpu_count": -1},
            {"gpu_count": "4"}, {"gpu_count": 2.5}, {"tunnel": "false"},
            {"poll_seconds": 0}, {"idle_seconds": -1}, {"poll_seconds": float("nan")},
            {"retry_seconds": float("inf")}, {"worker_stop_seconds": 0},
            {"max_utilization": 101}, {"max_utilization": 0.1}, {"max_memory_mib": -1},
            {"matrix_size": 100}, {"matrix_size": 8193}, {"matrix_size": True},
            {"gpus": "GPU-a,GPU-a"}, {"gpus": "0,00"}, {"gpus": "GPU-a,"},
            {"gpus": "MIG-a"}, {"gpus": []},
            {"blob_root": "relative"}, {"local_root": "/local/../blob"},
            {"blob_prefix": "/absolute"}, {"blob_prefix": "../escape"},
            {"blob_root": "/blob", "local_root": "/blob/cache"},
            {"blob_root": "/local/blob", "local_root": "/local"},
            {"tunnel_name": "space here"}, {"tunnel_name": "a" * 21},
            {"blob_root": "${{inputs.blob}}"}, {"blob_prefix": "line\nbreak"},
            {"aml": {"input_name": "foo} }"}}, {"aml": {"datastore_uri": "https://blob"}},
            {"aml": {"repository": "https://token@github.com/org/repo"}},
            {"aml": {"repository": "https://github.com/org/repo?token=secret"}},
            {"aml": {"ref": "--orphan"}}, {"aml": {"ref": "$(touch marker)"}},
            {"aml": {"keep_alive_seconds": 0}}, {"aml": {"clone_retry_seconds": True}},
        ]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                profile.normalize(config)

    def test_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "profile.json"
            for text in ('{"gpu_count":2,"gpu_count":8}', '{"aml":{"ref":"main","ref":"old"}}'):
                config.write_text(text)
                with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                    profile.load(config)

    def test_renderer_supports_aml_mount_exact_path_or_no_storage(self):
        for config in (
            {"aml": {"datastore_uri": "azureml://datastores/shared/paths/project/", "input_name": "data"}},
            {"blob_root": "/already/mounted/project"},
            {},
        ):
            with self.subTest(config=config):
                command, inputs = profile.render_aml(config)
                result = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(bool(inputs), "aml" in config)
                if inputs:
                    self.assertEqual(inputs["data"]["mode"], "rw_mount")
                    self.assertIn("${{inputs.data}}", command)
                else:
                    self.assertNotIn("${{inputs.", command)
        for config in (
            {"blob_prefix": "project"},
            {"blob_root": "/mount", "aml": {"datastore_uri": "azureml://datastores/shared/paths/"}},
        ):
            with self.assertRaises(ValueError):
                profile.render_aml(config)

    def test_cli_writes_reviewable_artifacts_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "profile.json"
            config.write_text(json.dumps({"gpu_count": 3, "aml": {"keep_alive_seconds": 7200}}))
            output = root / "generated"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(profile.main(["render-aml", "--config", str(config), "--output-dir", str(output)]), 0)
            self.assertEqual({path.name for path in output.iterdir()}, {"command.sh", "aml-inputs.json", "resolved-profile.json"})
            self.assertIn("sleep 7200", (output / "command.sh").read_text())
            self.assertEqual(json.loads((output / "resolved-profile.json").read_text())["gpu_count"], 3)
            before = (output / "command.sh").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(profile.main(["render-aml", "--config", str(config), "--output-dir", str(output)]), 1)
            self.assertEqual((output / "command.sh").read_bytes(), before)

    def test_checked_in_example_matches_renderer(self):
        repo = Path(__file__).resolve().parents[1]
        command, _ = profile.render_aml(profile.load(repo / "configs/example.json"))
        documented = (repo / "docs/aml-command.sh").read_text()
        self.assertEqual(documented.split("\n", 1)[1], command)


class SessionProfileTests(unittest.TestCase):
    def test_cli_overrides_json_and_explicit_false_and_auto_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"gpu_count": 8, "gpu_model": "H100", "tunnel": True, "blob_root": "/blob", "blob_prefix": "project", "matrix_size": 1024}))
            args, dry_run = session.parse_options(["--config", str(config), "--gpu-count", "auto", "--no-tunnel", "--no-blob", "--matrix-size", "2048"])
            self.assertIsNone(args.gpu_count)
            self.assertFalse(args.tunnel)
            self.assertIsNone(args.blob_root)
            self.assertEqual(args.blob_prefix, "")
            self.assertEqual(args.gpu_model, "H100")
            self.assertEqual(args.matrix_size, 2048)
            self.assertFalse(dry_run)

    def test_dry_run_has_no_filesystem_service_or_tunnel_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "must-not-exist"
            with patch.object(session, "prepare_paths") as paths, patch.object(session, "run_services") as services, contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(session.main(["--local-root", str(local), "--tunnel", "--dry-run"]), 0)
            paths.assert_not_called()
            services.assert_not_called()
            result = json.loads(stdout.getvalue())
            self.assertIsNone(result["tunnel_name"])
            self.assertTrue(result["tunnel"])
            self.assertFalse(local.exists())

    def test_invalid_threshold_rejected_before_creating_paths(self):
        for argv in (["--poll-seconds", "nan"], ["--max-utilization", "101"], ["--gpu-count", "0"]):
            with patch.object(session, "prepare_paths") as paths, self.assertRaises(ValueError):
                session.main(argv)
            paths.assert_not_called()

    def test_aml_profile_requires_runtime_mount_and_allows_injected_override(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"aml": {"datastore_uri": "azureml://datastores/project/paths/"}}))
            with self.assertRaisesRegex(ValueError, "actual AML mount"):
                session.parse_options(["--config", str(config)])
            args, _ = session.parse_options(["--config", str(config), "--blob-root", "/actual/mount"])
            self.assertEqual(args.blob_root, "/actual/mount")
            self.assertIsNone(args.aml["datastore_uri"])
            profile.render_aml(vars(args))  # The resolved runtime profile is reusable.
            args, _ = session.parse_options(["--config", str(config), "--no-blob"])
            self.assertIsNone(args.aml["datastore_uri"])

    def test_blob_root_already_final_is_not_extended(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = session.parse_options(["--blob-root", str(root / "project"), "--local-root", str(root / "local")])
            paths = session.prepare_paths(args)
            self.assertEqual(paths["mount"], paths["persistent"])

    def test_local_only_session_and_parameters_reach_independent_services(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "local"
            with patch.object(session, "run_services") as run, patch.object(session, "storage_loop") as storage, patch.object(session, "log"):
                self.assertEqual(session.main(["--local-root", str(local), "--tunnel", "--gpu-count", "2", "--matrix-size", "2048", "--worker-retry-seconds", "23"]), 0)
            storage.assert_not_called()
            services = run.call_args.args[0]
            self.assertEqual([service.name for service in services], ["tunnel", "GPU supervisor"])
            gpu_command = services[1].command
            self.assertEqual(gpu_command[gpu_command.index("--gpu-count") + 1], "2")
            self.assertEqual(gpu_command[gpu_command.index("--matrix-size") + 1], "2048")
            self.assertEqual(gpu_command[gpu_command.index("--worker-retry-seconds") + 1], "23.0")
            self.assertEqual(services[0].env["BLOB_ROOT"], "")
            saved = json.loads((local / "profile.json").read_text())
            self.assertRegex(saved["tunnel_name"], r"^aml-[a-f0-9]{12}$")
            self.assertEqual(saved["gpu_count"], 2)
            self.assertIsNone(saved["blob_root"])
            self.assertEqual((local / "profile.json").stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
