# Singularity Auto Placeholder

Automatic per-GPU placeholders and a VS Code tunnel for a single-node Azure ML / Singularity job. Each GPU yields to real work independently, so an experiment using four GPUs leaves placeholders running on the other four.

The job command starts services in the background and immediately runs **`sleep 8553600` (99 days)**. GPU discovery, worker failures, and Blob setup no longer decide whether the tunnel or the job's main command stays alive.

## Start a new Azure ML job

In **Run a custom training script**, configure this input:

| Setting | Value |
| --- | --- |
| Name | `lucayu` |
| Type | Folder |
| URI | `azureml://datastores/zhiyuhe/paths/` |
| Mode | Read-write mount |

Paste this into **Command**:

```bash
env -u BASH_ENV bash --noprofile --norc -c '
set -m
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
  until git clone --depth 1 https://github.com/1ucayu/singularity_auto_placeholder.git "$repo/repo"; do
    rm -rf -- "$repo/repo"
    sleep 30
  done
  exec bash "$repo/repo/scripts/aml_start.sh" --blob-root "$1" --gpu-count 8 --tunnel --tunnel-name aml-lucayu
) &
sleep 8553600 &
wait "$!"
' _ "${{inputs.lucayu}}"
```

The command follows `main` and clones this public repository without authentication. `${{inputs.lucayu}}` is replaced by AML with the mounted directory. The Linux image needs `python3` (3.10+), CUDA-enabled PyTorch, `nvidia-smi`, Git, and Bash. To use a different image Python, set `PLACEHOLDER_PYTHON` before launching.

The clean inner Bash avoids reloading the platform's broken `/etc/profile.d` scripts. Errors from outer platform initialization can still appear before this command runs.

### Connect with VS Code

1. Check the AML log for the GitHub device-login URL and code, and complete login if prompted.
2. In local VS Code, install **Remote - Tunnels** and use **Remote Tunnels: Connect to Tunnel** with the same GitHub account.
3. Select **`aml-lucayu`**. This is a tunnel name; no local SSH config entry is needed.
4. Open a terminal and run:

```bash
source "/tmp/singularity-auto-$(id -u)/env.sh"
placeholder status
nvidia-smi
```

Login cache survives process restarts within the same job. A fresh job needs device login unless the platform injects a valid `VSCODE_CLI_ACCESS_TOKEN` (and refresh token when required). Keep credentials out of Git, the Command, and shared Blob storage. An arbitrary repository PAT is not necessarily a compatible tunnel credential. `--tunnel` downloads the official VS Code CLI and accepts its server license terms.

Use a distinct tunnel name for concurrent jobs; reuse `aml-lucayu` after the old job ends. The container needs network access to GitHub, VS Code downloads, and Microsoft Dev Tunnels.

## Run experiments

Use the wrapper to release GPUs before your program initializes CUDA:

```bash
# Use all managed GPUs.
placeholder run -- python your_script.py

# Use GPUs 0-3; placeholders continue on the remaining GPUs.
placeholder run --gpus 0,1,2,3 -- python train.py

# SGLang example; adjust model and arguments for your deployment.
placeholder run -- .venv/bin/python -m sglang.launch_server \
  --model-path /your/local/model --tp 8

# Manual controls.
placeholder pause
placeholder resume
```

The wrapper waits for the selected placeholders to exit, sets the workload's `CUDA_VISIBLE_DEVICES`, forwards Ctrl+C/SIGTERM, and preserves its exit code. GPU numbers refer to the managed list in `placeholder status`. Keep the wrapped command in the foreground; its reservation lasts while it runs, including CPU initialization. Concurrent wrappers can use disjoint GPU sets.

Scripts started directly are detected once they create a CUDA context. This detection is reactive; use `placeholder run` to avoid a startup race with a large model allocation. Download models and install dependencies before releasing GPUs.

Do not run the old `gpu_occupy/resnet.py` or `sglang_handson/scripts/gpu_keeper.sh` alongside this tool.

## Runtime behavior

- Every GPU is checked once per second. With no external CUDA process, at most 5% utilization, and at most 256 MiB used memory for 30 seconds, its placeholder starts.
- An external CUDA process causes only that GPU's placeholder to stop. It resumes after the process exits and the GPU passes the idle interval again.
- GPU supervisor and tunnel are independent services. Either service restarts after failure, with a default 10-second retry delay. Worker failures retry separately and leave other GPUs running.
- Blob directory setup happens in the background and retries on failure. Local launchers and the tunnel remain usable while storage is unavailable.
- CUDA-visible UUIDs identify usable GPUs. `--gpu-count 8` is an expected count: a mismatch is logged, and usable GPUs still run. The tool never expands the allocation to satisfy this count.
- Isolated container PID namespaces are supported. A worker announces readiness after initializing CUDA; its GPU's process list establishes the corresponding NVIDIA PID. Additional processes cause that GPU to yield. Workers exit if their supervisor disappears.
- Monitoring errors stop placeholders until monitoring recovers. The tool signals only its own worker processes, never arbitrary Python or CUDA workloads.

The shell waits on the 99-day sleep independently of the background services. In particular, a killed worker cannot skip the sleep as it did with `python resnet.py && sleep ...`. Sleep keeps the command alive; platform reclamation, cancellation, job time limits, and node failure can still end the job earlier.

The target is full NVIDIA GPUs with ordinary CUDA processes. MIG and MPS workloads are outside the current scope; placeholders use a private MPS pipe directory. A real workload can retain a CUDA context while idle, in which case its GPU is left to that workload.

## Storage and recovery

The launcher creates `<Blob mount>/lucayu/sglang/` for persistent models, datasets, completed traces, results, and session manifests. Node-local code, virtual environments, caches, active outputs, and control state live under `/tmp/singularity-auto-<UID>/`. Source `env.sh` to get `BLOB_ROOT`, `LOCAL_WORK_ROOT`, and `OUTPUT_ROOT`.

See the [storage and recovery workflow](docs/storage-workflow.md). This repository sets up the directories and services. It does not yet install `sglang_handson`, download models, upload experiment results, restore checkpoints, or submit replacement AML jobs.

## Logs and troubleshooting

Use `placeholder status`, `nvidia-smi`, and the AML job log. Local rotating logs are under `/tmp/singularity-auto-<UID>/logs/`; control files are in the adjacent `control/` directory. Blob setup logs `Blob personal directory ready` when the persistent layout is available. Until then, files written to `OUTPUT_ROOT` remain local.

For options:

```bash
bash scripts/aml_start.sh --help
python3 -m singularity_placeholder supervise --help
python3 -m singularity_placeholder run --help
```

## Validation

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/aml_start.sh
bash -n docs/aml-command.sh
```

Tests use simulated NVIDIA output and real local subprocesses. They cover per-GPU release, container PID mapping, service restart isolation, unavailable Blob storage, and the command's independent sleep. They do not allocate CUDA memory. Actual MSRA 8×H100 behavior still requires verification on the cluster.
