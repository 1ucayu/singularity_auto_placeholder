# Singularity Auto Placeholder

Automatic per-GPU placeholders with optional SSH relay and VS Code tunnel services for a single-node Azure ML / Singularity job. Each GPU yields to real work independently. A JSON profile controls GPU expectations, selected devices, storage paths, local workspace, access services, and job lifetime; no personal datastore or GPU allocation is assumed.

The runtime targets full NVIDIA GPUs with ordinary CUDA processes. It discovers the GPUs available to the job; it does not request or expand a GPU allocation. GPU model names such as A100 and H100 are planning metadata. Choose the actual hardware, allocation size, image, and platform time limit when creating the AML job.

For the **CPU sandbox + 8-H100 GPU backend** topology, use [the prepared profile and launch guide](docs/sandbox-gpu-backend.md). Docker benchmarks run on the sandbox; the GPU node serves SGLang through a private Unix socket on the SSH jump host. The relay is independent of GPU supervision, so pending credentials or a disconnected relay do not stop placeholders.

## Have an AI prepare the configuration

The repository includes the [`singularity-placeholder` skill](skills/singularity-placeholder/SKILL.md). It interviews the user in small rounds, reuses answers already supplied, and produces a validated profile and launch command. It distinguishes a new AML job from an existing remote job and from a request to generate files only.

Install it for Codex from the repository root:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/singularity-placeholder "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Reload skills or start a new task if the client does not discover it immediately, then ask:

```text
Use $singularity-placeholder to interview me and prepare an AML GPU placeholder job.
I need four A100 GPUs and have an existing Blob mount. Generate the files first.
```

```text
用 $singularity-placeholder 帮我配置占卡工具。先问清楚 GPU、Blob 和 tunnel，
我已经有一个运行中的 job，不要重复启动已有服务。
```

The installed skill includes its own configuration reference. It uses an existing repository checkout or clones the requested repository and revision into a user workspace; installation does not require preserving this repository's directory layout. It does not automatically submit jobs, replace existing jobs, or start a service merely because it can generate the command. Existing explicit launch authorization carries forward.

## Configure a profile

From the repository root, copy the example and edit the copy:

```bash
cp configs/example.json configs/my-job.json
python3 -m singularity_placeholder.profile validate --config configs/my-job.json
bash scripts/aml_start.sh --config configs/my-job.json --dry-run
```

Validation and dry-run need only Python 3.10+ and Bash. They do not start GPU workers, create a tunnel, or submit an AML job. See the [configuration guide](docs/configuration.md) for every field and the difference between a mounted path and an AML datastore URI.

Existing deployments should read the [migration notes](docs/configuration.md#migrating-an-existing-deployment): the old personal Blob prefix, eight-GPU expectation, and fixed tunnel name are no longer implicit defaults.

For example, an existing mount whose path is already the intended persistent directory needs no additional prefix:

```json
{
  "schema_version": 1,
  "gpu_model": "A100",
  "gpu_count": 4,
  "blob_root": "/mnt/my-existing-project",
  "blob_prefix": "",
  "local_root": "/tmp/my-placeholder",
  "tunnel": true,
  "tunnel_name": null
}
```

Replace paths and GPU expectations with the target job's actual values. `gpu_count: null` means discover the allocation without an expected-count check. `blob_root: null` with no AML datastore URI enables local-only operation. `tunnel_name: null` generates a name at startup when the tunnel is enabled. Defaults and explicit command-line overrides work together: **CLI > JSON profile > defaults**.

## Start a new Azure ML job

Set `aml.datastore_uri` and `aml.input_name` in the profile when AML should supply the Blob mount. Leave `blob_root` null in this mode. `blob_prefix` is appended to the supplied mount; use an empty string if that mount is already the desired persistent root.

```bash
python3 -m singularity_placeholder.profile render-aml \
  --config configs/my-job.json --output-dir generated/my-job
```

Choose a new or empty output directory. The renderer creates:

| File | Purpose |
| --- | --- |
| `command.sh` | Self-contained command to paste into AML's **Command** field. |
| `aml-inputs.json` | Input names, datastore paths, and mount settings to configure in AML. |
| `resolved-profile.json` | Profile with defaults filled in, for review and reuse. |

Configure the corresponding inputs under **Run a custom training script**, select the required compute allocation and image, and paste the generated command. If using an existing mounted path or local-only operation, the input descriptor contains no datastore input. The renderer generates files only; it does not create the compute allocation or submit the job.

The command clones `aml.repository` at `aml.ref`, starts the services in the background, and keeps the main command alive for `aml.keep_alive_seconds` (default 99 days). Set the revision to an immutable commit for repeatable deployment. That revision must include profile support; when testing an unmerged change, select the feature branch or commit containing it instead of an older default branch. The target Linux image needs Python 3.10+, CUDA-enabled PyTorch, `nvidia-smi`, Git, and Bash. Set `PLACEHOLDER_PYTHON` to use another CUDA-enabled Python executable.

The clean inner Bash avoids reloading platform profile scripts. Outer platform initialization can still log errors before the command starts. Platform reclamation, cancellation, job time limits, and node failure can end the job before its configured lifetime.

## Start inside an existing job

Use the actual remote job's mounted path and GPU allocation in the profile. After reviewing the dry-run output, start the session on that node:

```bash
bash scripts/aml_start.sh --config configs/my-job.json
```

A one-off override does not modify the saved profile:

```bash
bash scripts/aml_start.sh --config configs/my-job.json \
  --gpu-count 2 --blob-root /mnt/another-project --blob-prefix '' --no-tunnel
```

Check the existing job's services before starting another session. Sessions sharing a local root are protected by a lock, but separate roots can still attempt to manage the same GPUs. Keep a single placeholder manager for each GPU.

## Connect with VS Code

When `tunnel` is enabled:

1. Check the job log for the tunnel name and GitHub device-login URL/code. Complete login if prompted.
2. In local VS Code, install **Remote - Tunnels** and use **Remote Tunnels: Connect to Tunnel** with the same GitHub account.
3. Select the logged tunnel name. Use a distinct name for concurrent jobs or leave it null for an automatically generated name.
4. Source the `env.sh` path printed at startup in each new terminal. With the default local root:

```bash
source "/tmp/singularity-auto-$(id -u)/env.sh"
placeholder status
nvidia-smi
```

The tunnel's authentication cache stays on the node and survives process restarts within the same job. A new node needs device login unless the platform injects a compatible `VSCODE_CLI_ACCESS_TOKEN` (and refresh token when required). Keep credentials out of profiles, Git, commands, and shared Blob storage. An arbitrary repository PAT is not necessarily a compatible tunnel credential. Enabling the tunnel downloads the official VS Code CLI and accepts its server license terms; network access to GitHub, VS Code downloads, and Microsoft Dev Tunnels is needed.

## Run experiments

Use the wrapper to release selected GPUs before your program initializes CUDA:

```bash
# Use all managed GPUs.
placeholder run -- python your_script.py

# Use managed GPUs 0 and 1; other placeholders continue running.
placeholder run --gpus 0,1 -- python train.py

# Manual controls.
placeholder pause
placeholder resume
```

The wrapper waits for the selected placeholders to exit, sets the workload's `CUDA_VISIBLE_DEVICES`, forwards Ctrl+C/SIGTERM, and preserves its exit code. GPU numbers here refer to the managed list in `placeholder status`. Keep the wrapped command in the foreground; its reservation lasts while it runs, including CPU initialization. Concurrent wrappers can use disjoint GPU sets.

Programs started directly are detected once they create a CUDA context. Detection is reactive; use `placeholder run` to avoid a startup race with a large model allocation. Download models and install dependencies before releasing GPUs.

## Runtime behavior

- By default, each GPU is checked every second. With no external CUDA process, at most 5% utilization, and at most 256 MiB used memory for 30 seconds, its placeholder starts. These thresholds and worker parameters are configurable.
- An external CUDA process stops only that GPU's placeholder. It resumes after the process exits and the GPU passes the idle interval again.
- GPU supervisor and tunnel run independently and restart after failure. Worker failures retry separately and leave other GPUs running.
- Blob setup runs in the background and retries on failure. Local launchers and the tunnel remain usable while storage is unavailable. Blob setup is skipped in local-only mode.
- CUDA-visible UUIDs identify usable GPUs. A `gpu_count` mismatch is logged; it does not block startup or increase the allocation. Use the profile's `gpus` field to restrict management to an allocated subset, preferably by physical GPU UUID.
- Isolated container PID namespaces are supported. Workers announce readiness after CUDA initialization, and the supervisor maps their NVIDIA PIDs. Workers exit if their supervisor disappears.
- Monitoring errors stop placeholders until monitoring recovers. The tool signals only its own workers.

MIG and MPS workloads are outside the current scope; placeholders use a private MPS pipe directory. A real workload retaining an idle CUDA context keeps control of that GPU. The outer AML sleep remains independent of background worker failures.

## Storage and recovery

Persistent storage is exactly `blob_root / blob_prefix`, or the AML input mount plus `blob_prefix`. No username or project suffix is added implicitly. With no Blob configuration, outputs remain node-local. Code, virtual environments, caches, active outputs, and control state live under `local_root`, which defaults to `/tmp/singularity-auto-<UID>`.

See the [storage and recovery workflow](docs/storage-workflow.md). The tool creates directories, environment variables, and a session manifest when storage is available. It does not download models, upload experiment results, restore checkpoints, or submit replacement jobs.

## Logs and validation

Use `placeholder status`, `nvidia-smi`, and the job log. Local logs are under `<local_root>/logs`; control files are in `<local_root>/control`. Startup prints the resolved environment location and service information.

```bash
bash scripts/aml_start.sh --help
python3 -m singularity_placeholder.profile --help
python3 -m unittest discover -s tests -v
bash -n scripts/aml_start.sh
bash -n docs/aml-command.sh
```

Tests use simulated NVIDIA output and local subprocesses. They do not establish GPU performance, hardware compatibility, a working tunnel login, Blob persistence, or successful job submission on a real cluster. Validate those on the chosen target before relying on the deployment.
