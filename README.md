# Singularity Auto Placeholder

A foreground supervisor for **single-node Azure ML / Singularity jobs with 8 full NVIDIA GPUs**. Placeholder workers run separately from the job's main process, so killing a worker, a worker failure, or releasing a GPU does not directly terminate the job. The target environment is Linux with a working CUDA-enabled PyTorch installation.

## 1. Start directly from Azure ML

Configure the input as follows: name `lucayu`, type Folder, URI `azureml://datastores/zhiyuhe/paths/`, and Read-write mount. Paste this into **Run a custom training script → Command**:

```bash
bash -lc 'set -euo pipefail; repo="$(mktemp -d /tmp/singularity-placeholder.XXXXXX)"; git clone --depth 1 https://github.com/1ucayu/singularity_auto_placeholder.git "$repo"; exec bash "$repo/scripts/aml_start.sh" --blob-root "$1" --gpu-count 8' _ "${{inputs.lucayu}}"
```

This public repository can be cloned without authentication. Each new job installs the local launcher, checks the CUDA environment, creates a personal Blob directory, and starts the supervisor. There is no need to SSH into the job first or append `sleep`. The image must already provide `python3` (3.10+), CUDA-enabled `torch`, `nvidia-smi`, `git`, and Bash; startup does not replace your PyTorch/CUDA installation. If the correct Python executable is named `python`, set `PLACEHOLDER_PYTHON=python` before `exec bash`.

The command above follows `main` to pick up fixes. For reproducible experiments, pin a verified commit of this repository and record both the image version and the experiment code commit.

AML replaces `${{inputs.lucayu}}` with the mounted directory inside the container. `azureml://...` is a resource URI, not a filesystem path you can pass to `cd`. Passing the input explicitly also resolves the `Missing inputs from command: lucayu` warning.

### Start a VS Code tunnel as well (optional)

Append the following after `--gpu-count 8` in the command above:

```text
--tunnel --tunnel-name aml-lucayu
```

The new job downloads the official VS Code CLI and starts the tunnel automatically. Check the AML logs for device login instructions. GitHub provides authentication; Microsoft Dev Tunnels relays the connection. The container still needs network access to GitHub, VS Code updates, and the tunnel service.

Login state is stored in a private directory on the node's local disk and can be reused when a process restarts within the same job. **Unattended login in a new job requires the platform to securely inject a valid `VSCODE_CLI_ACCESS_TOKEN`** (and a refresh token if required by the provider). An arbitrary repository PAT is not necessarily a compatible replacement. Without a usable credential, GitHub device-code login is still required. Token expiry, revocation, SSO, or network policy changes may require authentication again. Do not store credentials in Git, the Command string, or shared Blob storage. Using `--tunnel` accepts the VS Code Server license terms.

Use different tunnel names for concurrent jobs. You can reuse `aml-lucayu` when replacing a job that has already ended. The GPU supervisor operates independently of tunnel startup success.

## 2. Use from VS Code

When this tool starts the tunnel, new VS Code terminals will usually inherit the environment variables. For an existing independent tunnel or SSH terminal, load the generated environment; no manual installation is needed in each new job:

```bash
source "/tmp/singularity-auto-$(id -u)/env.sh"
placeholder status

# Recommended: release the selected GPUs and wait for confirmation before launching the foreground task.
placeholder run --gpus 0,1,2,3 -- python train.py
# Omitting --gpus releases all managed GPUs.
placeholder run -- .venv/bin/python -m sglang.launch_server --model-path /your/local/model --tp 8

# Manually release all GPUs / resume automatic management.
placeholder pause
placeholder resume
```

The SGLang example only demonstrates the wrapper. Configure the model, memory requirements, quantization, TP/EP, and version-specific arguments for your actual deployment. The wrapper preserves the command's exit code and forwards Ctrl+C / SIGTERM. Run `source` in each terminal as needed, or invoke `/tmp/singularity-auto-$(id -u)/bin/placeholder` directly.

`--gpus 0,1,2,3` selects entries 0–3 in the supervisor's managed GPU list, which is shown by `placeholder status`. The wrapper sets `CUDA_VISIBLE_DEVICES` to the corresponding GPU UUIDs; the experiment sees those GPUs renumbered as 0–3. Full GPU UUIDs are also accepted.

Keep the command after `run` in the foreground: do not append `&`, use `nohup`, or daemonize it. Its reservation remains active while the service runs. Concurrent `run` commands hold separate reservations; a GPU's placeholder can resume only after all reservations for that GPU have ended. The wrapper coordinates placeholders, not scheduling or memory allocation between real workloads. Choose disjoint GPU sets for concurrent services. Complete model downloads and dependency setup that do not require GPUs before using `run` to launch the service, reducing idle time after GPU release.

### Automatic detection versus explicit GPU release

- **Automatic per-GPU mode:** If a managed GPU has no external CUDA process, utilization is at most 5%, and used memory is at most 256 MiB for 30 consecutive seconds, its worker starts. When all 8 GPUs are idle, one worker starts on each GPU.
- **Experiment detected:** When an external CUDA process appears on a GPU, that GPU's placeholder stops. Placeholders on other idle GPUs continue running. A model with temporarily zero utilization still prevents its GPU's placeholder from restarting as long as the process retains a CUDA context.
- **Explicit release:** `placeholder run --gpus 0,1,2,3 -- ...` reserves the selected GPUs **before** the experiment creates a CUDA context, waits for their workers to exit, and then launches the experiment. The reservation remains active during model downloads and CPU initialization, avoiding the startup race inherent in automatic detection. Other GPUs remain under normal management.
- **Errors:** Monitoring failures stop placeholders. Worker failures leave the supervisor running and trigger retries. The supervisor terminates only workers it started and can identify as its own; it does not use `pkill python` or kill other CUDA workloads.

The default polling interval is one second. Automatic detection cannot predict a program's intent before it uses a GPU, nor guarantee that a model launched without the wrapper will avoid running out of memory before the placeholder yields. Prefer `run`.

Per-GPU management matches the reported reclamation policy: when an experiment uses 4 GPUs, placeholders continue on the other 4. Released GPUs may still have low utilization while the experiment is paused, retains only a CUDA context, or performs CPU initialization. The tool prioritizes avoiding interference with the experiment. The platform controls GPU utilization requirements, reclamation thresholds, administrator cancellations, runtime limits, and node failures; the tool cannot guarantee that a job will not be reclaimed.

Do not run `sglang_handson/scripts/gpu_keeper.sh` or the old `gpu_occupy/resnet.py` alongside this supervisor. They will be detected as external workloads and conflict with this tool.

## 3. Troubleshooting and limitations

The original command was:

```bash
python resnet.py && sleep 10540800
```

`&&` continues only if the command on its left exits successfully with status 0. A killed Python process normally exits with a nonzero status, so `sleep` is skipped, the AML main command ends, and the job and tunnel disappear. This repository keeps the session/supervisor in the foreground and runs workers as independently stoppable child processes. **Killing the supervisor or session still ends the job.**

Local control state is stored in `/tmp/singularity-auto-<UID>/control/`, with logs in the adjacent `logs/` directory. The session writes a startup manifest to `lucayu/sglang/sessions/<session-id>/` on Blob. Live status and rotating logs stay on local disk, and log output also goes to the AML job logs. The Blob directory is not used to restore locks, PIDs, or GPU processes.

The tool supports selecting allocated devices by full GPU UUID and does not guess the mapping from CUDA ordinals to physical GPUs. **Automatic process identification currently requires the job to use the host PID namespace with an aligned `/proc` mount.** In an isolated container PID namespace, host PIDs returned by NVML cannot be reliably matched to container PIDs. The tool reports an error before launching workers instead of guessing ownership. MIG, MPS, and isolated environments that hide other CUDA processes also require additional support. The namespace configuration of the current MSRA jobs has not been verified.

When it encounters unknown processes or ambiguous monitoring data, the tool yields GPUs and reports the reason in its status. If `CUDA_VISIBLE_DEVICES` uses a partial list of numeric GPU ordinals, configure that mask with the full UUIDs allocated to the job. `--gpus` does not override or expand the existing visibility mask.

For configurable options, see:

```bash
bash scripts/aml_start.sh --help
python3 -m singularity_placeholder supervise --help
python3 -m singularity_placeholder run --help
```

## 4. Where to store models, code, traces, and results

See the [storage and recovery workflow](docs/storage-workflow.md). The startup script creates the directory layout and generates an `env.sh` for each job. It currently **does not** modify `sglang_handson`, install its dependencies, download models, upload experiment traces, restore checkpoints, or automatically resubmit AML jobs. Those are the next steps for the experiment bootstrap; the placeholder supervisor does not implement them.

## 5. Validation

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/aml_start.sh
```

Tests use simulated NVIDIA output and real local subprocesses to verify monitoring, GPU release handshakes, process termination, and path handling. They do not require GPUs or allocate CUDA memory. Development was performed on macOS; the tool has not yet been tested on an actual MSRA 8×H100 job.

On the first deployment, use `placeholder status` and `nvidia-smi` to verify the actual GPU count and state. Launch an experiment through the wrapper and check that the relevant workers exit first, then resume after the idle grace period once the experiment finishes. Local simulation tests do not prove that the platform will keep the job running.
