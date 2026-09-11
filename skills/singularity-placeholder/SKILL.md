---
name: singularity-placeholder
description: Interview the user and configure the singularity_auto_placeholder GPU placeholder and optional VS Code tunnel for new or existing Azure ML / Singularity jobs. Use for requests to configure GPU placeholders, GPU allocation expectations, Blob paths, or 占卡工具; generate a reusable profile and launch files before any authorized deployment.
---

# Singularity Placeholder

Turn the user's job requirements into a reusable JSON profile and a concrete launch command. Begin with a short interview that fills missing decisions; reuse the current conversation, existing profiles, and confirmed target state. Use the user's language.

This tool manages allocated NVIDIA GPUs within one job. It cannot request a GPU model, enlarge an allocation, or replace a destroyed job. A GPU model label is metadata, and GPU count is an expectation checked against discovery.

## Establish the target and interview

First determine whether the user wants files only, a new AML job, or changes inside an existing job. Identify the intended remote target and any available access. A local checkout or laptop GPU does not describe an AML node. Inspect existing profiles and, where authorized, the target's running sessions, tunnels, GPU allocation, mounts, and local disks without changing them.

Ask a few related questions at a time, only for missing information that changes the result. Cover these decisions in whatever order fits the user's answers:

- **Compute:** GPU model, allocated count or automatic discovery, and all allocated GPUs versus a subset. Use confirmed physical UUIDs for a subset when available. For a new job, record the compute resource the user needs to choose in AML; the profile does not provision it.
- **Persistence:** no Blob, an existing mounted path, or a new AML datastore input. For a mounted path, establish whether it already ends at the desired project directory. For an AML input, collect the datastore URI and input name. Ask about a relative suffix only when needed; never invent a username/project prefix or append an existing suffix twice.
- **Node-local work:** default `/tmp/singularity-auto-<UID>` or a known local SSD path. Use actual target disk information when accessible; do not put virtual environments, control state, or tunnel credentials on Blob.
- **Access and lifetime:** whether a VS Code tunnel is wanted, automatic unique name or a chosen name, intended job duration, and repository/revision. Preserve a working tunnel unless the user asks to replace it. Mention the CLI download/license acceptance when newly enabling the tunnel.
- **Split CPU/GPU access:** when a sandbox hosts Docker and the GPU node serves inference, collect the jump endpoint, node-local key and verified-host-key paths, private remote socket and local API port. Use the independent SSH relay; do not assume a reverse TCP loopback request stays private under the server's `GatewayPorts` setting. Read `docs/sandbox-gpu-backend.md` in the chosen implementation repository for this workflow. Ask only for values not already supplied.

For example, if the user already supplied “four A100s and an existing Blob mount,” ask for the target job and mount semantics rather than asking the GPU questions again. Technical tuning such as idle thresholds and worker matrix size can keep defaults unless the user has a requirement or the target needs adjustment.

Do not treat this interview as a required approval round for every use. A complete existing profile and explicit deployment request may already answer it. Follow-up edits should ask only about unresolved choices.

## Locate the implementation

The installed skill is self-contained documentation; its directory is not assumed to be inside a repository. Use a user-specified checkout first. Otherwise locate the relevant checkout in the current workspace or clone the chosen repository into a writable work directory. The upstream default is `https://github.com/1ucayu/singularity_auto_placeholder.git`; use the user's fork/revision when supplied. Do not rely on `../../scripts` relative to this installed skill.

Verify that the checkout contains `singularity_placeholder/profile.py` and `scripts/aml_start.sh`. If the chosen revision predates profile support, identify that incompatibility and prepare the configuration with a compatible checkout; do not claim the old revision can consume it. Keep unrelated working changes intact.

Read [references/profile.md](references/profile.md) when creating or editing a profile. The repository CLI is the authority for its supported schema; check its help and validation errors if the installed skill and checkout differ.

## Produce reviewable launch files

Write the agreed values to a named JSON profile in the chosen workspace. Preserve meaningful user overrides. Use a partial profile where defaults are sufficient, without adding invented machine paths. Save recurring choices in the profile, not only in a transient command.

Run from the repository root:

```bash
python3 -m singularity_placeholder.profile validate --config configs/my-job.json
bash scripts/aml_start.sh --config configs/my-job.json --dry-run
```

For a new AML job:

```bash
python3 -m singularity_placeholder.profile render-aml \
  --config configs/my-job.json --output-dir generated/my-job
bash -n generated/my-job/command.sh
```

The renderer writes `command.sh`, `aml-inputs.json`, and `resolved-profile.json`; it does not submit a job or select compute. Choose a new or empty output directory so earlier launch files are preserved. A profile using `aml.datastore_uri` is for AML rendering: the generated command supplies the input's actual mount at runtime. For direct startup in an existing job, use its actual `blob_root` path.

Show a concise configuration summary with GPU expectation/subset, exact persistent-root construction or local-only mode, local workspace, tunnel choice, revision, and lifetime. Link the profile and generated files, explain any unresolved target checks, and provide the exact launch command. Validation and dry-run establish configuration correctness only; they do not verify CUDA operation, mount durability, tunnel login, or AML submission.

## Deploy within the user's authorized scope

Generating and validating files is separate from starting services or submitting a job. Complete the configuration and command before seeking launch authorization when it is missing. If the user has already authorized that deployment, continue without asking again. A files-only request ends with the artifacts.

For an existing job, confirm which manager and tunnel are already active before starting anything. Preserve unrelated workloads and services. The session lock protects only a shared `local_root`; choosing another root does not prevent two managers from claiming the same GPUs. Do not solve conflicts by killing arbitrary CUDA processes.

After an authorized start, check the actual GPU discovery, service status, and generated `env.sh` path. Report observed state separately from planned settings. Use `placeholder run --gpus ... -- <command>` when launching authorized real work so its reservation begins before CUDA initialization. Do not submit replacement jobs or add an external recovery controller unless the user requested that work; stop deployment retries on access/configuration failures or the platform's own rejection rather than creating repeated jobs.
