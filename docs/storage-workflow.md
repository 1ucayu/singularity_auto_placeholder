# Storage and recovery

Treat an AML job as a replaceable compute node. Git stores the definitions needed to rebuild it, Blob stores durable data, and node-local storage provides the active workspace. Recreating the environment does not recover an experiment unless it saved usable checkpoints and results.

## Choose the persistent root explicitly

The profile supports two ways to supply persistent storage:

- **Existing mount:** set `blob_root` to the absolute mounted directory and leave `aml.datastore_uri` null.
- **New AML input:** set `aml.datastore_uri` and `aml.input_name`, leave `blob_root` null, and generate the AML command and input descriptor. AML substitutes the input's mounted path at runtime.

`blob_prefix` is a relative suffix appended once to the supplied mount. Use `""` when that mount already points at the intended project directory. Do not repeat a username or project component already present in the mounted path or datastore URI. Absolute prefixes and `..` traversal are rejected.

For example, an existing mount `/mnt/team-data` and prefix `projects/my-experiment` produce `/mnt/team-data/projects/my-experiment`. A mount already at `/mnt/team-data/projects/my-experiment` uses an empty prefix. No personal prefix is built into the tool.

With `blob_root: null` and `aml.datastore_uri: null`, the session runs without Blob. There is no durable output destination until the user configures and uses one.

## Directory layout

```text
<configured persistent root>/             # Optional Blob directory
  models/<model>/<immutable-revision>/
  datasets/
  traces/input/
  runs/<experiment>/<run-id>/
    manifest.json
    results/
    traces/
    logs/
    checkpoints/
  sessions/<session-id>/                   # Tool-written startup manifest
  code-snapshots/

<local_root>/                             # Default: /tmp/singularity-auto-<UID>
  workspace/                              # Checkouts, virtualenvs, staged models
  cache/                                  # HF, pip, uv, compilation caches
  outputs/<session-id>/                    # Active experiment output
  control/                                # Node-local locks and reservations
  logs/
  tunnel/                                 # Private authentication cache
  env.sh
```

The tool creates the top-level storage directories and session manifest. The model-version and experiment-run subdirectories above describe a suggested workflow; the tool does not automatically populate or synchronize them.

Choose `local_root` from the target node's actual disks. `/tmp` is a portable default, not a guarantee of the largest SSD. Use a separate node-local directory, outside the Blob mount. Inspect the target's mount and disk information if access is available; local laptop storage does not describe the remote job. A Blob FUSE mount's reported `df` capacity does not establish the backing object store's capacity.

Git checkouts, virtual environments, package installations, build caches, and active traces work best on node-local storage. Large model files and closed result archives can be stored in Blob. Where capacity permits, stage models to local SSD and verify completion before use. Storage throughput and model compatibility depend on the chosen environment.

## Environment variables

Source the exact `env.sh` path printed by the session in each terminal:

| Variable | Meaning |
| --- | --- |
| `BLOB_MOUNT` | Actual mount supplied through the profile or AML input; empty without Blob. |
| `BLOB_ROOT` | Persistent directory after applying `blob_prefix`; empty without Blob. |
| `LOCAL_WORK_ROOT` | Local workspace directory. |
| `OUTPUT_ROOT` | Current session's local output directory. |
| `HF_HOME` / `HF_HUB_CACHE` | Node-local Hugging Face caches. |
| `PLACEHOLDER_CONTROL_DIR` | Control state for this session. |

`OUTPUT_ROOT` is local even when Blob is configured. A successful storage-directory setup does not mean your experiment's outputs are being uploaded. Preserve existing model paths explicitly; changing the profile does not migrate files or deduplicate downloads.

## Rebuild and preserve work

1. **Pin the environment.** Record the code commit, image, dependency lock files, model ID and immutable revision, and experiment configuration. Save unfinished source changes through normal version control or a source snapshot that excludes credentials and caches.
2. **Prepare complete model versions.** Download or copy into a temporary directory, verify all shards, and publish a manifest and completion marker. A later job should reuse only complete versions.
3. **Give each experiment a unique run ID.** Record the actual GPU model/count, model revision, parallelism settings, seeds, inputs, and full command. Separate concurrent jobs' output directories.
4. **Persist during the run.** Close log/trace segments before uploading them, verify transfers, and save recoverable checkpoints periodically. Recovery is bounded by the last confirmed durable data.
5. **Mark completion after upload.** Record checksums and an explicit completion state after all required results are durable.

Do not rely only on an EXIT trap for archival: node loss, `SIGKILL`, and job deletion can prevent it from running. Applications requiring strong durability should verify uploads with their storage service and manifests.

## Recovery boundaries

A new job can clone a pinned revision, rebuild dependencies, reuse complete model files, and resume from an uploaded checkpoint. Old processes, GPU/KV cache, local virtual environments, and unuploaded results do not survive node destruction.

The placeholder tool manages services inside one job. An external controller is needed to detect a missing job and submit a replacement with an Azure identity. Such automation needs explicit scope, bounded retries, and persistent run state; it is not part of this repository's generated command or skill's default workflow.

Tunnel authentication is separate from model and experiment recovery. Its node-local cache survives process restarts, but not node replacement. Use platform-supported credential injection if unattended login is required, and keep credentials out of profiles, generated commands, and shared storage.
