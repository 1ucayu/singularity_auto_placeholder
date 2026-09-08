import contextlib
import io
import unittest
from unittest.mock import patch

from singularity_placeholder import __main__ as cli


class SupervisorCLITests(unittest.TestCase):
    def test_auto_and_configured_worker_options_reach_supervisor(self):
        for count_args, expected in (([], None), (["--gpu-count", "auto"], None), (["--gpu-count", "3"], 3)):
            with self.subTest(count_args=count_args), \
                    patch.object(cli, "configure_logging"), \
                    patch.object(cli, "Supervisor") as supervisor:
                supervisor.return_value.run.return_value = 7
                result = cli.main(["supervise", *count_args, "--matrix-size", "2048"])
            self.assertEqual(result, 7)
            config = supervisor.call_args.args[0]
            self.assertEqual(config.gpu_count, expected)
            self.assertEqual(config.matrix_size, 2048)

    def test_invalid_expected_count_fails_before_runtime_startup(self):
        for count in ("0", "-1", "2.5", "none"):
            with self.subTest(count=count), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(cli, "Supervisor") as supervisor, \
                    self.assertRaises(SystemExit) as failure:
                cli.main(["supervise", "--gpu-count", count])
            self.assertEqual(failure.exception.code, 2)
            supervisor.assert_not_called()

    def test_invalid_matrix_size_fails_before_logging_or_runtime_startup(self):
        for size in ("255", "8193"):
            with self.subTest(size=size), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(cli, "configure_logging") as logging, \
                    patch.object(cli, "Supervisor") as supervisor:
                result = cli.main(["supervise", "--matrix-size", size])
            self.assertEqual(result, 1)
            logging.assert_not_called()
            supervisor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
