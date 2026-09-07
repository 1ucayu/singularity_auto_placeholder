import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from singularity_placeholder.session import prepare_paths, write_environment


class SessionPathsTest(unittest.TestCase):
    def test_blob_and_local_are_separate_and_shell_paths_are_quoted(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mount = base / "blob with ' quote"
            mount.mkdir()
            local = base / "local with ' quote"
            args = argparse.Namespace(blob_root=mount, blob_prefix="lucayu/sglang", local_root=local)
            paths = prepare_paths(args)
            values = write_environment(paths, base / "repository")
            self.assertTrue((mount / "lucayu/sglang/models").is_dir())
            self.assertNotIn("TOKEN", (local / "env.sh").read_text())
            output = subprocess.check_output(["bash", "-c", 'source "$1"; printf "%s" "$BLOB_ROOT"', "_", str(local / "env.sh")], text=True)
            self.assertEqual(output, values["BLOB_ROOT"])
            subprocess.run(["bash", "-n", str(local / "bin/placeholder")], check=True)
            self.assertEqual(os.stat(local / "env.sh").st_mode & 0o777, 0o600)

    def test_rejects_local_inside_blob_and_prefix_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory) / "blob"
            mount.mkdir()
            for prefix, local in [("../escape", Path(directory) / "local"), ("lucayu/sglang", mount / "local")]:
                with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                    prepare_paths(argparse.Namespace(blob_root=mount, blob_prefix=prefix, local_root=local))


if __name__ == "__main__":
    unittest.main()
