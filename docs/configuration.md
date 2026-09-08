# Configuration profiles

A profile is a UTF-8 JSON object with `schema_version: 1`. Partial profiles are supported: omitted fields use defaults. Runtime command-line options override profile values without editing the file.

From the repository root:

```bash
python3 -m singularity_placeholder.profile validate --config configs/my-job.json
bash scripts/aml_start.sh --config configs/my-job.json --dry-run
python3 -m singularity_placeholder.profile render-aml \
  --config configs/my-job.json --output-dir generated/my-job
```

The validator checks configuration structure and values. Dry-run reports the planned runtime configuration. The renderer produces an AML command, an input descriptor, and a resolved profile. These operations do not contact Azure, reserve GPUs, start a tunnel, or prove that a path exists on the target node.

## GPU and runtime settings

| Field | Default | Meaning |
| --- | --- | --- |
| `schema_version` | `1` | Profile format version. |
| `gpu_model` | `null` | Optional label such as `"A100"` or `"H100"` for planning and manifests. Does not select or enforce hardware. |
| `gpu_count` | `null` | Positive expected GPU count. Null uses discovery without a count expectation. A mismatch is logged; it does not increase the allocation or prevent usable GPUs from running. |
| `gpus` | `null` | Comma-separated allocated GPU identifiers to manage. Prefer physical GPU UUIDs for stable selection. Null discovers available devices. |
| `idle_seconds` | `30` | Required idle interval before starting a worker; nonnegative. |
| `poll_seconds` | `1` | Positive monitoring interval. |
| `max_utilization` | `5` | Maximum idle utilization percentage, integer from 0 to 100. |
| `max_memory_mib` | `256` | Maximum idle memory use in MiB, nonnegative integer. |
| `matrix_size` | `4096` | Worker matrix dimension, integer from 256 to 8192. Tune for the actual device if needed. |
| `worker_stop_seconds` | `10` | Positive worker shutdown timeout. |
| `worker_retry_seconds` | `60` | Positive delay before retrying a failed worker. |
| `retry_seconds` | `10` | Positive delay before retrying a failed service or Blob setup. |

Set the GPU model and allocation size in the AML job's compute settings. Changing the profile cannot turn an existing allocation into a different model or a larger allocation. The tool supports full NVIDIA GPUs; MIG and MPS workloads are outside its current scope.

The session's `gpus` field selects physical allocated devices. The indices accepted by `placeholder run --gpus 0,1` instead refer to the managed list shown in `placeholder status`.

## Storage and tunnel settings

| Field | Default | Meaning |
| --- | --- | --- |
| `blob_root` | `null` | Absolute directory already mounted on the target node. Null means no fixed mount path. |
| `blob_prefix` | `""` | Relative suffix appended to the supplied mount. Empty means the root is already final. No absolute path or `..` component. |
| `local_root` | `null` | Absolute node-local workspace path. Null resolves to `/tmp/singularity-auto-<UID>` at runtime. Keep it separate from Blob. |
| `tunnel` | `false` | Enable the VS Code tunnel. |
| `tunnel_name` | `null` | Null generates a name when starting a tunnel. A fixed name must be 1–20 letters, digits, or hyphens, starting with a letter or digit. Use distinct names for concurrent jobs. |

When both `blob_root` and `aml.datastore_uri` are null, storage is local-only. `OUTPUT_ROOT` always points at active local outputs; Blob setup does not upload them automatically. See [storage and recovery](storage-workflow.md).

## AML rendering settings

These fields live inside the `aml` object:

| Field | Default | Meaning |
| --- | --- | --- |
| `input_name` | `"storage"` | Name of the AML input and generated `${{inputs.<name>}}` reference. Letters, digits, or underscores; cannot start with a digit. |
| `datastore_uri` | `null` | Optional `azureml://datastores/<name>/paths/<path>` URI mounted read-write by AML. |
| `repository` | `"https://github.com/1ucayu/singularity_auto_placeholder.git"` | HTTPS repository URL cloned inside the job, without embedded credentials. Change this for a fork containing the desired implementation. |
| `ref` | `"main"` | Git revision to run. Pin an immutable commit for repeatability. |
| `keep_alive_seconds` | `8553600` | Positive lifetime of the outer sleep; defaults to 99 days. Platform limits still apply. |
| `clone_retry_seconds` | `30` | Positive delay between repository clone retries while the command remains alive. |

`blob_root` and `aml.datastore_uri` cannot both be set in a launch profile because they describe competing mount sources. A datastore URI is not a Linux path. Use the renderer for a new AML input; if starting directly inside an existing job, pass its actual mounted path. The generated AML command supplies that mount as a runtime override.

The renderer requires a new or empty output directory to avoid overwriting earlier launch files. For revisions of an existing plan, choose another output directory and keep the profile as the reusable configuration.

The renderer does not generate a complete compute/environment job specification or submit jobs. Choose the GPU resource and an image with Python 3.10+, CUDA-enabled PyTorch, `nvidia-smi`, Git, and Bash in AML. Repository authentication, when required, must come from the execution environment rather than a credential embedded in the profile.

The selected repository revision must contain `singularity_placeholder/profile.py` and the session's `--config` support. While testing a feature before it is merged, set `aml.ref` to that feature branch or commit. The local renderer succeeding does not prove that an older remote `main` can consume the resulting profile.

## Examples

An existing mount at the final project root:

```json
{
  "schema_version": 1,
  "gpu_model": "A100",
  "gpu_count": 4,
  "blob_root": "/mnt/project-data",
  "blob_prefix": "",
  "local_root": "/scratch/placeholder",
  "tunnel": true
}
```

A new AML input with a project suffix:

```json
{
  "schema_version": 1,
  "gpu_model": "H100",
  "gpu_count": 2,
  "blob_prefix": "projects/demo",
  "tunnel": true,
  "aml": {
    "input_name": "storage",
    "datastore_uri": "azureml://datastores/my_datastore/paths/",
    "keep_alive_seconds": 86400
  }
}
```

A local-only session using discovered GPUs, with no tunnel:

```json
{
  "schema_version": 1
}
```

Review the result before launching:

```bash
bash scripts/aml_start.sh --config configs/my-job.json --dry-run
# Example override for an already mounted path:
bash scripts/aml_start.sh --config configs/my-job.json \
  --blob-root /mnt/actual-project --blob-prefix '' --no-tunnel --dry-run
```

Profiles should contain configuration only. Keep access tokens and other credentials out of them. A valid profile proves that its values satisfy the tool's schema, not that its paths, GPU allocation, credentials, or remote services are available.

## Migrating an existing deployment

The previous personal defaults have changed:

| Setting | Previous default | New default |
| --- | --- | --- |
| Blob prefix | `lucayu/sglang` | `""`, using the supplied mount directly. |
| Expected GPU count | `8` | `null`, automatic discovery without a count expectation. |
| Tunnel name | `aml-lucayu` | `null`, generating a name when a tunnel starts. |

To preserve an existing deployment's layout and identity, explicitly set its existing `blob_root`, `blob_prefix`, `gpu_count`, `tunnel`, and `tunnel_name` in the profile. Reuse a fixed tunnel name only when its previous session has ended. Existing explicit CLI values continue to take precedence over the profile.

There is no data migration. Omitting a prefix that was previously implicit changes the directory used by the new session; it does not move the files stored under the old prefix. Check the effective persistent root before restarting an existing deployment.
