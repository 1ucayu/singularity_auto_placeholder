# Profile reference

Use this guide to map interview answers into `schema_version: 1` JSON. Run the selected repository's validator before launch; this reference is packaged with the skill so installation does not require a particular checkout layout.

## Core fields

| Field | Type / default | Decision |
| --- | --- | --- |
| `schema_version` | integer, `1` | Current format. |
| `gpu_model` | string or null, `null` | Planning label, such as A100 or H100. Does not enforce hardware. |
| `gpu_count` | positive integer or null, `null` | Expected discovered count; mismatch is logged. Null uses discovery without an expectation. |
| `gpus` | comma-separated string or null, `null` | Restrict management to allocated devices; prefer physical UUIDs. |
| `blob_root` | absolute path or null, `null` | Existing mount on the target. Mutually exclusive with `aml.datastore_uri`. |
| `blob_prefix` | relative string, `""` | Suffix appended once to the mount. Empty means already at the final root. No `..`. |
| `local_root` | absolute path or null, `null` | Node-local workspace, separate from Blob. Null uses `/tmp/singularity-auto-<UID>`. |
| `tunnel` | boolean, `false` | Enable optional VS Code tunnel. |
| `tunnel_name` | string or null, `null` | Automatic name when null. Fixed names: 1–20 letters/digits/hyphens, first character alphanumeric. |

Without `blob_root` or `aml.datastore_uri`, the session is local-only. A datastore URI is an AML resource reference, not a Linux path. `OUTPUT_ROOT` remains local even when Blob is configured; the tool creates storage directories and session metadata but does not upload experimental outputs or restore models/checkpoints.

The desired GPU model and allocation size belong in AML's compute settings as well as any profile metadata. Do not claim the configuration enforces hardware compatibility. Full NVIDIA GPUs are supported; MIG and MPS workloads are outside the current scope.

## Optional SSH relay object

`relay.enabled` defaults to false. When enabled, provide `host`, `user`, `identity_file`, `known_hosts_file`, and a nonempty `forwards` list. Each forward has a private absolute `remote_socket` path on the jump host and a `local_port` on the GPU node's `127.0.0.1`. Only paths belong in the profile; provision credentials separately on node-local disk. The key must belong to the session user with mode 600. Strict host-key checking is always enabled.

The jump SSH port defaults to 22. `connect_timeout`, `server_alive_interval`, and `server_alive_count_max` default to 15 seconds, 30 seconds, and 3. The jump host needs Python 3 and stream-local SSH forwarding; the GPU image needs the OpenSSH client. The independent relay retries through the session manager without stopping GPU supervision. The jump socket's parent directory must be owned by the jump user with mode 700. Its lease refuses active listeners, cleans refused stale sockets, and unlinks only its own socket inodes on normal shutdown. Use different socket paths for concurrent jobs.

Remote TCP forwarding is deliberately unsupported because a server's `GatewayPorts` setting can override requested loopback binding. Clients can map a local loopback TCP port to the remote Unix socket with `ssh -L 127.0.0.1:30000:/private/remote/api.sock jump-host`. Enable optional GPU SSH forwarding only after verifying the node's existing SSH service and authentication. Relay readiness does not prove SGLang health or successful inference.

## AML object

| Field | Default | Decision |
| --- | --- | --- |
| `input_name` | `"storage"` | AML input name used in `${{inputs.<name>}}`; letters/digits/underscores, not starting with a digit. |
| `datastore_uri` | `null` | User's `azureml://datastores/<name>/paths/<path>` URI; configure a read-write mount from generated input descriptor. |
| `repository` | `"https://github.com/1ucayu/singularity_auto_placeholder.git"` | Requested HTTPS implementation repository. No embedded credentials. |
| `ref` | `"main"` | Requested Git revision; prefer an immutable commit for repeatability. |
| `keep_alive_seconds` | `8553600` | Positive outer sleep duration, 99 days by default. Platform time limits and cancellation still apply. |
| `clone_retry_seconds` | `30` | Positive clone retry delay while the outer command lives. |

For an already mounted path, set `blob_root` and leave `aml.datastore_uri` null. For a new AML input, leave `blob_root` null and set the URI and input name. The renderer embeds the profile and substitutes the actual mounted path when AML starts the command.

## Optional runtime tuning

These top-level fields can normally use their defaults:

| Field | Default |
| --- | --- |
| `idle_seconds` | `30` |
| `poll_seconds` | `1` |
| `max_utilization` | `5` |
| `max_memory_mib` | `256` |
| `matrix_size` | `4096` (integer from 256 to 8192) |
| `worker_stop_seconds` | `10` |
| `worker_retry_seconds` | `60` |
| `retry_seconds` | `10` |

Command-line flags override profile values; profile values override defaults. Use `--no-tunnel` to override an enabled tunnel. See `bash scripts/aml_start.sh --help` for the current flags.

## Example interview result

For a user who chose a new AML job, four A100 GPUs, datastore root plus `experiments/demo`, one day of runtime, and a tunnel with an automatic name:

```json
{
  "schema_version": 1,
  "gpu_model": "A100",
  "gpu_count": 4,
  "blob_prefix": "experiments/demo",
  "tunnel": true,
  "aml": {
    "input_name": "storage",
    "datastore_uri": "azureml://datastores/my_datastore/paths/",
    "keep_alive_seconds": 86400
  }
}
```

`my_datastore`, paths, hardware, and duration in this example are illustrative values; populate them from the current user's answers. If the datastore URI already ends at `experiments/demo/`, use `blob_prefix: ""`. If the user supplies an existing final mount `/mnt/project`, use that `blob_root`, an empty prefix, and no datastore URI.

The AML image must supply Python 3.10+, CUDA-enabled PyTorch, `nvidia-smi`, Git, and Bash. These are target requirements, not dependencies needed merely to generate the profile. Validation/rendering are offline and do not verify hardware, mounted storage, authentication, or access to a private repository.
