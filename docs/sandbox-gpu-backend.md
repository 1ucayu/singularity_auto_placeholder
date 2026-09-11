# CPU sandbox and Singularity GPU backend

The CPU sandbox runs Docker containers, the benchmark harness and the agent. The Singularity allocation runs the existing GPU placeholders and, when requested, a foreground SGLang server on `127.0.0.1:30000`. A reverse SSH connection from the GPU node creates `/home/jumper/.sglang-relay/api.sock` on the jump host. The sandbox and Mac can open independent local forwards to that socket.

```text
Sandbox: mini-swe-agent + Docker
  -> sandbox 127.0.0.1:30000
  -> SSH to jumper@62.146.171.45
  -> private Unix socket /home/jumper/.sglang-relay/api.sock
  -> reverse SSH connection from Singularity
  -> Singularity 127.0.0.1:30000 -> SGLang on 8 H100s
```

The allocation is still an AML/Singularity job and can be reclaimed or cancelled. This changes its role to a persistent serving backend between benchmark runs; it does not remove platform job lifetime limits or automatically request a replacement allocation.

## Create the GPU allocation

Select one node with eight H100 GPUs and an image providing Python 3.10+, CUDA-enabled PyTorch, `nvidia-smi`, Git, Bash, and the OpenSSH client. Enable the platform SSH service for initial credential provisioning and troubleshooting.

Configure the Blob input exactly as supplied:

| AML input setting | Value |
| --- | --- |
| Name | `lucayu` |
| Type | Folder / `uri_folder` |
| URI | `azureml://datastores/zhiyuhe/paths/` |
| Mode | Read-write mount / `rw_mount` |

The [profile](../configs/sandbox-gpu-backend.json) appends `lucayu/sglang` once, so the persistent root is `${{inputs.lucayu}}/lucayu/sglang`. Local code, environments, caches, active outputs and control state remain under `/tmp/singularity-auto-<runtime UID>/`. The expected count of eight GPUs is diagnostic; select the real eight-H100 allocation in the platform UI.

When the uploaded **code directory is this repository's root**, this is the short **Command**:

```bash
env -u BASH_ENV bash --noprofile --norc scripts/aml_start.sh --config configs/sandbox-gpu-backend.json --blob-root "${{inputs.lucayu}}"
```

This runs the session in the foreground until terminated. GPU supervision, relay retries and Blob setup are independent within it. Upload only the Git-tracked source, for example after checking out the published branch:

```bash
git archive --format=tar HEAD | gzip > /tmp/singularity-placeholder-upload.tar.gz
```

Extract that archive before selecting its directory in AML. The archive contains only tracked files and has no Git metadata or local credentials.

For a command that downloads the pinned implementation from GitHub, paste the entire [generated command](sandbox-gpu-backend/command.sh). Its [input descriptor](sandbox-gpu-backend/aml-inputs.json) and [resolved profile](sandbox-gpu-backend/resolved-profile.json) are checked in beside it. This version starts the session in the background and independently waits on a 99-day sleep. Platform limits can end it sooner. Neither command launches SGLang or a benchmark.

## Provision access after the new job starts

The relay initially reports missing credentials and retries every ten seconds. GPU supervision remains active. Through the new job's platform SSH connection, provision these files on node-local disk:

| GPU node file | Source / requirement |
| --- | --- |
| `/tmp/singularity-secrets/id_jumper` | Authorized copy of the user's jump SSH key; source on Mac is `~/.ssh/id_jumper_sandbox_singularity`. |
| `/tmp/singularity-secrets/known_hosts` | Verified host-key entry for `62.146.171.45`; do not disable host-key checking. |

The directory must have mode 700. Both files must be owned by the actual session user; use mode 600. Derive that owner from the live `singularity_placeholder.session` process rather than assuming the SSH shell UID matches it. Keep key contents out of Git, the AML Command, logs and Blob. A destroyed job needs this node-local provisioning again. The [companion `sglang_handson` topology guide](https://github.com/1ucayu/sglang_handson/blob/codex/sandbox-singularity-topology/docs/SANDBOX_SINGULARITY_TOPOLOGY.md) provides the Mac provisioning helper and the complete sandbox/GPU setup.

The jump account must allow stream-local forwarding and Python 3 execution. The helper creates a private socket directory, acquires a lock, and refuses to replace another live session. Remote TCP forwards are not used because the current jump server was observed to force public binds despite a loopback request. No jump `sshd` configuration change is necessary for this socket route.

## Connect the clients and start serving

With the jump key and host key configured in a local `singularity-jumper` SSH alias, run this on the sandbox (or independently on the Mac):

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:30000:/home/jumper/.sglang-relay/api.sock singularity-jumper
```

The host-side agent can then use `http://127.0.0.1:30000/v1`. If the agent itself runs in a Docker container, its loopback is separate; the companion harness uses the CPU sandbox's host-side agent with Docker as its tool environment.

On the GPU node, use a terminal belonging to the placeholder session's actual runtime user. Follow the companion guide to restore the `codex/sandbox-singularity-topology` branch of `sglang_handson`, then run these commands from that checkout in Bash:

```bash
source scripts/env.sh
placeholder status
./scripts/bootstrap.sh --with-model --no-restore-code --no-watch
"$SGLANG_PYTHON" -m por_modeling.gpu_service --preflight-only
"$SGLANG_PYTHON" -u -m por_modeling.gpu_service
```

`--no-restore-code` prevents an old Blob source mirror from overwriting the new branch. `--no-watch` prevents the generic archive watcher from conflicting with this service's own periodic archival. The instrumented `por_modeling.gpu_service` preserves engine hooks, reserves all eight GPUs through `placeholder run` internally, and serves on `127.0.0.1:30000`. Do not wrap it in another `placeholder run` or start the ordinary `serve_h100_fp8.sh` alongside it.

Keep the entire `gpu_service` command in the foreground of its persistent terminal. When that service exits and the GPUs satisfy the idle checks, placeholders resume. On the sandbox, verify the forwarded server before a benchmark:

```bash
curl --fail --max-time 10 http://127.0.0.1:30000/health
curl --fail --max-time 10 http://127.0.0.1:30000/v1/models
```

The relay can be established while no model server exists, so forwarding readiness and model readiness are separate checks. Benchmark results belong on the CPU sandbox and must be archived by its harness; this repository does not upload them automatically.

## Optional SSH through the same jump host

After verifying that the new GPU node has an existing SSH server listening on `127.0.0.1:22`, a separate profile can add:

```json
{"remote_socket": "/home/jumper/.sglang-relay/ssh.sock", "local_port": 22}
```

to `relay.forwards`. A client can then map `127.0.0.1:22022` to that Unix socket and authenticate to the GPU SSH server with its own configured user and host key. This is disabled in the prepared profile: the jump key authenticates the relay, and does not automatically authenticate a login to the GPU node. Initial platform SSH access remains available independently.

## Validation boundary

The unit suite checks profile validation, missing-credential isolation, SSH command construction, private socket ownership, competing-session exclusion, stale-socket recovery and cleanup that preserves replacement sockets. On 2026-09-11, the managed relay also passed two start/HTTP/stop cycles through the actual jump host using a CPU-only HTTP service on the Mac: both exits were zero, owned sockets were removed, and the same socket could reconnect. This verifies the SSH socket route and cleanup with the live jump host.

No unit test starts CUDA, submits a job, downloads a model, or runs a SWE-bench instance. The new GPU allocation's CUDA, Blob mount, credentials and end-to-end inference still need live verification after provisioning.
