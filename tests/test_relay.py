import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from singularity_placeholder import profile, relay, session


def example():
    return {
        "relay": {
            "enabled": True,
            "host": "192.0.2.10",
            "user": "jumper",
            "identity_file": "/tmp/relay-secrets/key",
            "known_hosts_file": "/tmp/relay-secrets/known_hosts",
            "forwards": [{"remote_socket": "/tmp/relay-fixture/api.sock", "local_port": 30000}],
        }
    }


class RelayProfileTests(unittest.TestCase):
    def test_existing_profiles_leave_relay_disabled_and_defaults_independent(self):
        first = profile.normalize({})
        self.assertFalse(first["relay"]["enabled"])
        first["relay"]["forwards"].append({"remote_socket": "/tmp/test.sock", "local_port": 2})
        self.assertEqual(profile.normalize({})["relay"]["forwards"], [])
        enabled = profile.normalize(example())
        self.assertEqual(enabled["relay"]["server_alive_interval"], 30)

    def test_rejects_invalid_or_externally_bound_forwards_and_credential_content(self):
        invalid = [
            {"enabled": "true"}, {"host": None}, {"user": "user@host"},
            {"host": "-oProxyCommand=x"}, {"host": "host other"}, {"port": 65536},
            {"connect_timeout": 0}, {"server_alive_interval": False},
            {"identity_file": "relative"}, {"identity_file": "/tmp/../blob/key"},
            {"identity_file": "/tmp/${KEY}"}, {"known_hosts_file": "/tmp/a b"},
            {"private_key": "credentials cannot be stored in a profile"},
            {"forwards": []}, {"forwards": {}},
            {"forwards": [{"remote_socket": "/tmp/relay-fixture/api.sock"}]},
            {"forwards": [{"remote_socket": "/tmp/relay-fixture/api.sock", "local_port": 0}]},
            {"forwards": [{"remote_socket": "/tmp/relay-fixture/api.sock", "local_port": 65536}]},
            {"forwards": [{"remote_port": True, "local_port": 30000}]},
            {"forwards": [{"remote_socket": "/tmp/relay-fixture/api.sock", "local_port": 30000, "bind_host": "0.0.0.0"}]},
            {"forwards": [{"remote_socket": "/tmp/relay-fixture/api.sock", "local_port": 30000}] * 2},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                config = example()
                config["relay"].update(overrides)
                profile.normalize(config)
        with self.assertRaisesRegex(ValueError, "Blob"):
            profile.normalize({**example(), "blob_root": "/tmp/relay-secrets"})

    def test_ssh_command_uses_remote_socket_and_local_loopback_and_requires_known_host(self):
        config = example()
        config["relay"]["forwards"].append({"remote_socket": "/tmp/relay-fixture/ssh.sock", "local_port": 22})
        command = relay.forward_command(config, Path("/tmp/control.sock"))
        self.assertEqual(command[:5], ["ssh", "-F", "/dev/null", "-T", "-o"])
        self.assertIn("/tmp/relay-fixture/api.sock:127.0.0.1:30000", command)
        self.assertIn("/tmp/relay-fixture/ssh.sock:127.0.0.1:22", command)
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertIn("ExitOnForwardFailure=yes", command)
        self.assertIn("BatchMode=yes", command)
        self.assertIn("IdentityAgent=none", command)
        self.assertIn("GlobalKnownHostsFile=/dev/null", command)
        self.assertNotIn("-f", command)
        self.assertEqual(command[-2:], ["--", "jumper@192.0.2.10"])

    def test_deployment_profile_keeps_eight_gpus_and_exact_storage_namespace(self):
        repo = Path(__file__).resolve().parents[1]
        config = profile.load(repo / "configs/sandbox-gpu-backend.json")
        self.assertEqual(config["gpu_count"], 8)
        self.assertEqual(config["blob_prefix"], "lucayu/sglang")
        self.assertFalse(config["tunnel"])
        self.assertEqual(config["relay"]["forwards"], [{"remote_socket": "/home/jumper/.sglang-relay/api.sock", "local_port": 30000}])
        command, inputs = profile.render_aml(config)
        self.assertEqual(inputs["lucayu"], {
            "type": "uri_folder", "path": "azureml://datastores/zhiyuhe/paths/", "mode": "rw_mount",
        })
        self.assertIn("sleep 8553600", command)

    def test_checked_in_deployment_artifacts_match_pinned_profile(self):
        repo = Path(__file__).resolve().parents[1]
        config = profile.load(repo / "configs/sandbox-gpu-backend.json")
        self.assertRegex(config["aml"]["ref"], r"^[a-f0-9]{40}$")
        command, inputs = profile.render_aml(config)
        generated = repo / "docs/sandbox-gpu-backend"
        self.assertEqual((generated / "command.sh").read_text(), command)
        self.assertEqual(json.loads((generated / "aml-inputs.json").read_text()), inputs)
        self.assertEqual(json.loads((generated / "resolved-profile.json").read_text()), config)


class RelayRuntimeTests(unittest.TestCase):
    def prepared(self, root):
        config = profile.normalize(example())
        for name in ("identity_file", "known_hosts_file"):
            path = root / name
            path.write_text("test fixture, not a real credential")
            path.chmod(0o600)
            config["relay"][name] = str(path)
        return config

    def test_credentials_are_checked_without_reading_key_contents(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            config = self.prepared(Path(directory))
            with patch.object(relay, "local_filesystem"), \
                    patch.object(Path, "read_text", side_effect=AssertionError("must not read credential content")), \
                    patch.object(Path, "read_bytes", side_effect=AssertionError("must not read credential content")):
                relay.check_credentials(config)

    def test_rejects_world_readable_key_missing_credentials_and_blob_symlink(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            config = self.prepared(root)
            key = Path(config["relay"]["identity_file"])
            key.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "permissions"):
                relay.check_credentials(config)
            key.unlink()
            with self.assertRaises(FileNotFoundError):
                relay.check_credentials(config)
            blob = root / "blob"
            blob.mkdir()
            (blob / "key").write_text("fixture")
            (blob / "key").chmod(0o600)
            key.symlink_to(blob / "key")
            config["blob_root"] = str(blob)
            with self.assertRaisesRegex(ValueError, "Blob"):
                relay.check_credentials(config)

    def test_shared_mount_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            config = self.prepared(Path(directory))
            with patch.object(relay, "local_filesystem", side_effect=ValueError("shared mount")), \
                    self.assertRaisesRegex(ValueError, "shared mount"):
                relay.check_credentials(config)

    def test_dry_run_is_offline_and_runtime_starts_managed_master(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            config_path = Path(directory) / "profile.json"
            config = example()
            config_path.write_text(json.dumps(config))
            with patch.object(relay, "check_credentials") as check, patch.object(relay, "run_master", return_value=0) as execute, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(relay.main(["--config", str(config_path), "--dry-run"]), 0)
            check.assert_not_called()
            execute.assert_not_called()
            with patch.object(relay, "check_credentials"), patch.object(relay.shutil, "which", return_value="/usr/bin/ssh"), \
                    patch.object(relay, "run_master", return_value=0) as execute, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(relay.main(["--config", str(config_path), "--runtime-dir", str(Path(directory) / "runtime")]), 0)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args.args[0]["relay"], profile.normalize(config)["relay"])

    def test_missing_credentials_fail_relay_only_and_session_registers_gpu_service(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            config = example()
            config["local_root"] = str(root / "local")
            config_path = root / "profile.json"
            config_path.write_text(json.dumps(config))
            with patch.object(relay, "check_credentials", side_effect=FileNotFoundError("key pending")), \
                    patch.object(relay, "run_master", return_value=0) as execute, contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(relay.main(["--config", str(config_path), "--runtime-dir", str(Path(directory) / "runtime")]), 1)
            execute.assert_not_called()
            self.assertIn("GPU supervision continues", stderr.getvalue())
            with patch.object(session, "run_services") as run, patch.object(session, "log"), \
                    patch.object(relay, "check_credentials", side_effect=AssertionError("must be checked in child")), \
                    patch.dict(os.environ, {"VSCODE_CLI_ACCESS_TOKEN": "test-only"}):
                self.assertEqual(session.main(["--config", str(config_path)]), 0)
            services = run.call_args.args[0]
            self.assertEqual([service.name for service in services], ["SSH relay", "GPU supervisor"])
            self.assertIn("singularity_placeholder.relay", services[0].command)
            self.assertIn("supervise", services[1].command)
            self.assertNotIn("VSCODE_CLI_ACCESS_TOKEN", services[0].env)
            self.assertTrue((root / "local/bin/placeholder").exists())


if __name__ == "__main__":
    unittest.main()
