"""Validate portable JSON profiles and render an Azure ML command offline."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import shlex
import sys
from urllib.parse import urlsplit


DEFAULTS = {
    "schema_version": 1,
    "gpu_model": None,
    "gpu_count": None,
    "gpus": None,
    "blob_root": None,
    "blob_prefix": "",
    "local_root": None,
    "tunnel": False,
    "tunnel_name": None,
    "idle_seconds": 30,
    "poll_seconds": 1,
    "max_utilization": 5,
    "max_memory_mib": 256,
    "matrix_size": 4096,
    "worker_stop_seconds": 10,
    "worker_retry_seconds": 60,
    "retry_seconds": 10,
    "relay": {
        "enabled": False,
        "host": None,
        "user": None,
        "port": 22,
        "identity_file": None,
        "known_hosts_file": None,
        "forwards": [],
        "connect_timeout": 15,
        "server_alive_interval": 30,
        "server_alive_count_max": 3,
    },
    "aml": {
        "input_name": "storage",
        "datastore_uri": None,
        "repository": "https://github.com/1ucayu/singularity_auto_placeholder.git",
        "ref": "main",
        "keep_alive_seconds": 8553600,
        "clone_retry_seconds": 30,
    },
}


def _object(value: object, name: str, allowed: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = value.keys() - allowed.keys()
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(sorted(unknown))}")
    return value


def _number(value: object, name: str, *, minimum: float = 0,
            integer: bool = False, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'}")
    if not math.isfinite(value) or value < minimum or (positive and value <= 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else f'>= {minimum}'}")


def _string(value: object, name: str, *, nullable: bool = False, empty: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be {'null or ' if nullable else ''}a {'nonempty ' if not empty else ''}string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value) or "${{" in value:
        raise ValueError(f"{name} must not contain control characters or AML expressions")


def normalize(data: object) -> dict:
    """Merge defaults and validate without probing the host or creating files."""
    data = _object(data, "profile", DEFAULTS)
    result = copy.deepcopy(DEFAULTS)
    result.update({key: value for key, value in data.items() if key not in ("aml", "relay")})
    for name in ("aml", "relay"):
        result[name].update(_object(data.get(name, {}), name, DEFAULTS[name]))
    _number(result["schema_version"], "schema_version", integer=True, positive=True)
    if result["schema_version"] != 1:
        raise ValueError("Unsupported schema_version; expected 1")
    _string(result["gpu_model"], "gpu_model", nullable=True)
    if result["gpu_count"] is not None:
        _number(result["gpu_count"], "gpu_count", integer=True, positive=True)
    _string(result["gpus"], "gpus", nullable=True)
    if result["gpus"] is not None:
        tokens = [token.strip() for token in result["gpus"].split(",")]
        if any(not re.fullmatch(r"(?:GPU-[A-Za-z0-9-]+|[0-9]+)", token) for token in tokens):
            raise ValueError("gpus must contain full-GPU UUIDs or physical nvidia-smi indices")
        identities = [str(int(token)) if token.isdecimal() else token for token in tokens]
        if len(identities) != len(set(identities)):
            raise ValueError("gpus must not contain duplicate selectors")
        result["gpus"] = ",".join(tokens)
    for name in ("blob_root", "local_root"):
        _string(result[name], name, nullable=True)
        if result[name] is not None:
            path = Path(result[name])
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be an absolute target-node path without '..'")
    _string(result["blob_prefix"], "blob_prefix", empty=True)
    prefix = Path(result["blob_prefix"])
    if prefix.is_absolute() or ".." in prefix.parts:
        raise ValueError("blob_prefix must be relative without '..'; use an empty string for an exact root")
    if result["local_root"] and result["blob_root"]:
        local, mount = Path(result["local_root"]), Path(result["blob_root"])
        if local.is_relative_to(mount) or mount.is_relative_to(local):
            raise ValueError("local_root must be separate from Blob")
    if type(result["tunnel"]) is not bool:
        raise ValueError("tunnel must be a boolean")
    _string(result["tunnel_name"], "tunnel_name", nullable=True)
    if result["tunnel_name"] is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,19}", result["tunnel_name"]):
        raise ValueError("tunnel_name must be 1-20 letters/digits/hyphens, starting with a letter or digit")
    for name in ("poll_seconds", "worker_stop_seconds", "worker_retry_seconds", "retry_seconds"):
        _number(result[name], name, positive=True)
    _number(result["idle_seconds"], "idle_seconds")
    _number(result["max_utilization"], "max_utilization", integer=True)
    if result["max_utilization"] > 100:
        raise ValueError("max_utilization must be between 0 and 100")
    _number(result["max_memory_mib"], "max_memory_mib", integer=True)
    _number(result["matrix_size"], "matrix_size", integer=True, minimum=256)
    if result["matrix_size"] > 8192:
        raise ValueError("matrix_size must be between 256 and 8192")
    relay = result["relay"]
    if type(relay["enabled"]) is not bool:
        raise ValueError("relay.enabled must be a boolean")
    for name in ("host", "user", "identity_file", "known_hosts_file"):
        _string(relay[name], f"relay.{name}", nullable=not relay["enabled"])
    if relay["host"] is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", relay["host"]):
        raise ValueError("relay.host must be an IPv4 address or DNS hostname")
    if relay["user"] is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", relay["user"]):
        raise ValueError("relay.user must be an SSH username, without host or options")
    for name in ("identity_file", "known_hosts_file"):
        if relay[name] is not None:
            path = Path(relay[name])
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"relay.{name} must be an absolute node-local file path without '..'")
            if any(char.isspace() or char in "\"'\\%$~" for char in relay[name]):
                raise ValueError(f"relay.{name} must not contain whitespace, quoting or SSH expansion characters")
            if result["blob_root"] and path.is_relative_to(Path(result["blob_root"])):
                raise ValueError(f"relay.{name} must not be stored on Blob")
    for name in ("port", "connect_timeout", "server_alive_interval", "server_alive_count_max"):
        _number(relay[name], f"relay.{name}", positive=True, integer=True)
    if relay["port"] > 65535:
        raise ValueError("relay.port must be between 1 and 65535")
    if not isinstance(relay["forwards"], list) or (relay["enabled"] and not relay["forwards"]):
        raise ValueError("relay.forwards must be a list, nonempty when relay is enabled")
    seen_sockets = set()
    for forward in relay["forwards"]:
        _object(forward, "relay.forwards entry", {"remote_socket": None, "local_port": None})
        _number(forward.get("local_port"), "relay.forwards.local_port", positive=True, integer=True)
        if forward["local_port"] > 65535:
            raise ValueError("relay.forwards.local_port must be between 1 and 65535")
        socket = forward.get("remote_socket")
        _string(socket, "relay.forwards.remote_socket")
        if (not socket.startswith("/") or ".." in Path(socket).parts
                or not re.fullmatch(r"/[A-Za-z0-9_./-]+", socket)
                or len(socket.encode()) > 100):
            raise ValueError("relay.forwards.remote_socket must be a short absolute path using letters, digits, '_', '.', '-' and '/'")
        if socket in seen_sockets:
            raise ValueError("relay.forwards must have unique remote sockets")
        seen_sockets.add(socket)
    aml = result["aml"]
    _string(aml["input_name"], "aml.input_name")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", aml["input_name"]):
        raise ValueError("aml.input_name must contain only letters, digits and underscores, and not start with a digit")
    _string(aml["datastore_uri"], "aml.datastore_uri", nullable=True)
    if aml["datastore_uri"] is not None and not aml["datastore_uri"].startswith("azureml://"):
        raise ValueError("aml.datastore_uri must be an azureml:// URI")
    _string(aml["repository"], "aml.repository")
    url = urlsplit(aml["repository"])
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("aml.repository must be an HTTPS Git URL without embedded credentials, query or fragment")
    _string(aml["ref"], "aml.ref")
    if aml["ref"].startswith("-") or not re.fullmatch(r"[A-Za-z0-9_./-]+", aml["ref"]):
        raise ValueError("aml.ref must be a branch, tag or commit, not a Git option or expression")
    for name in ("keep_alive_seconds", "clone_retry_seconds"):
        _number(aml[name], f"aml.{name}", positive=True, integer=True)
    return result


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path | None = None) -> dict:
    if path is None:
        return normalize({})
    return normalize(json.loads(path.read_text(), object_pairs_hook=_unique_keys))


def render_aml(profile: dict) -> tuple[str, dict]:
    """Return a pasteable Command and AML input definitions; execute nothing."""
    config = normalize(profile)
    aml = config["aml"]
    if aml["datastore_uri"] is not None and config["blob_root"] is not None:
        raise ValueError("Choose aml.datastore_uri or blob_root, not both, when rendering AML")
    mounted_input = aml["datastore_uri"] is not None
    if config["blob_prefix"] and not (mounted_input or config["blob_root"]):
        raise ValueError("blob_prefix needs blob_root or aml.datastore_uri")
    inputs = {}
    if mounted_input:
        inputs[aml["input_name"]] = {"type": "uri_folder", "path": aml["datastore_uri"], "mode": "rw_mount"}
    # JSON is embedded as literal heredoc data inside an entirely quoted Bash
    # script. User paths never become shell source or AML expressions.
    launch = 'exec bash "$repo/repo/scripts/aml_start.sh" --config "$repo/profile.json"'
    if mounted_input:
        launch += ' --blob-root "$1"'
    inner = '''set -m
cleanup() {
  trap - EXIT TERM INT HUP
  for pid in $(jobs -pr); do kill -TERM -- "-$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap "exit 143" TERM
trap "exit 130" INT
trap "exit 129" HUP
(
  repo="$(mktemp -d /tmp/singularity-placeholder.XXXXXX)" || exit 1
'''
    inner += '  cat > "$repo/profile.json" <<\'PLACEHOLDER_PROFILE_JSON\'\n'
    inner += json.dumps(config, indent=2, allow_nan=False) + "\nPLACEHOLDER_PROFILE_JSON\n"
    # A complete clone permits pinned commits as well as branches/tags. Failed
    # setup retries in the background while the job lifetime remains bounded.
    inner += f'''  until git clone --no-checkout -- {shlex.quote(aml['repository'])} "$repo/repo" &&
    git -C "$repo/repo" fetch origin {shlex.quote(aml['ref'])} &&
    git -C "$repo/repo" checkout --detach FETCH_HEAD; do
    rm -rf -- "$repo/repo"
    sleep {aml['clone_retry_seconds']}
  done
  {launch}
) &
sleep {aml['keep_alive_seconds']} &
wait "$!"
'''
    command = "env -u BASH_ENV bash --noprofile --norc -c " + shlex.quote(inner) + " _"
    if mounted_input:
        # AML substitutes this path before Bash parses Command. A quoted
        # heredoc preserves quotes, dollars and backticks as path data too.
        command += ' "$(cat <<\'PLACEHOLDER_AML_MOUNT\'\n${{inputs.' + aml["input_name"] + '}}\nPLACEHOLDER_AML_MOUNT\n)"'
    return command + "\n", inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("validate", "render-aml"):
        command = commands.add_parser(action)
        command.add_argument("--config", type=Path, required=True)
        if action == "render-aml":
            command.add_argument("--output-dir", type=Path, required=True,
                                 help="New or empty directory for command.sh, aml-inputs.json and resolved-profile.json")
    args = parser.parse_args(argv)
    try:
        config = load(args.config)
        # Validate the complete launch plan, including storage source ambiguity.
        command, inputs = render_aml(config)
        if args.action == "validate":
            print(json.dumps(config, indent=2, allow_nan=False))
            return 0
        output = args.output_dir
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "command.sh").write_text(command)
        (output / "aml-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        (output / "resolved-profile.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"Generated {output / 'command.sh'} and AML input definitions; no job was submitted.")
        return 0
    except (OSError, ValueError, OverflowError) as exc:
        print(f"placeholder profile: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
