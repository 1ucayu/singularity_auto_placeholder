# Singularity Auto Placeholder

为 **单节点、8 张完整 NVIDIA GPU 的 Azure ML / Singularity job** 提供前台 supervisor。占卡 worker 与 job 主进程分离；worker 被 kill、失败或让卡，不会直接结束整个 job。目标环境是 Linux + 已可使用 CUDA 的 PyTorch。

## 1. Azure ML 中直接启动

保持截图中的 input：名称 `lucayu`、Folder、URI `azureml://datastores/zhiyuhe/paths/`、Read-write mount。在 **Run a custom training script → Command** 粘贴：

```bash
bash -lc 'set -euo pipefail; repo="$(mktemp -d /tmp/singularity-placeholder.XXXXXX)"; git clone --depth 1 https://github.com/1ucayu/singularity_auto_placeholder.git "$repo"; exec bash "$repo/scripts/aml_start.sh" --blob-root "$1" --gpu-count 8' _ "${{inputs.lucayu}}"
```

这个公开仓库可以匿名 clone。每个新 job 都会安装本地启动入口、检查 CUDA 环境、建立个人 Blob 目录并启动 supervisor；不需要先 SSH 进去启动占卡，也不需要最后追加 `sleep`。镜像需已有 `python3`（3.10+）、CUDA 版 `torch`、`nvidia-smi`、`git` 和 Bash；脚本不会在 job 启动时替换你的 PyTorch/CUDA。若正确的 Python 名为 `python`，在 `exec bash` 前设置 `PLACEHOLDER_PYTHON=python`。

上述命令跟随 `main`，适合获取修复。正式实验应固定本仓库的已验证 commit，并同时记录镜像版本和实验代码 commit。

`${{inputs.lucayu}}` 由 AML 替换为容器里的挂载目录。`azureml://...` 是资源 URI，不能直接当作 `cd` 路径。显式传 input 也解决截图里的 `Missing inputs from command: lucayu`。

### 同时启动 VS Code tunnel（可选）

在上一条命令的 `--gpu-count 8` 后追加：

```text
--tunnel --tunnel-name aml-lucayu
```

新 job 会自动获取官方 VS Code CLI 并启动 tunnel；在 AML 日志中查看设备登录提示。GitHub 是身份提供方，连接由 Microsoft Dev Tunnels 中继。容器仍需要访问 GitHub、VS Code 更新和 tunnel 服务。

同一 job 的登录状态保存在节点本地私有目录，供进程重启使用。**新 job 无交互登录需要平台安全注入有效的 `VSCODE_CLI_ACCESS_TOKEN`**（以及提供方需要时的 refresh token）；这不是随便一个仓库 PAT 就一定可替代的凭证。没有可用凭证时，仍需完成 GitHub device-code 登录。token 失效、撤销、SSO 或网络策略变化可能需要重新认证。不要把凭证写进 Git、Command 字符串或共享 Blob。`--tunnel` 表示同意 VS Code Server 许可条款。

同时运行多个 job 时使用不同 tunnel 名字。只替换一个已结束的 job 时可以复用 `aml-lucayu`。占卡部分不依赖 tunnel 是否成功。

## 2. 在 VS Code 里使用

若使用本工具启动 tunnel，VS Code 的新终端通常会继承这些环境变量。已有的独立 tunnel/SSH 终端可载入生成的环境（不需要在每个新 job 中人工安装）：

```bash
source "/tmp/singularity-auto-$(id -u)/env.sh"
placeholder status

# 推荐：先让卡，收到 supervisor 确认后才启动你的前台程序。
placeholder run --gpus 0,1,2,3 -- python train.py
# 不写 --gpus 则让出全部受管 GPU。
placeholder run -- .venv/bin/python -m sglang.launch_server --model-path /your/local/model --tp 8

# 手工让卡 / 恢复自动管理。
placeholder pause
placeholder resume
```

SGLang 示例只展示 wrapper 的用法；模型名、显存、量化、TP/EP 与版本参数仍须按实际模型配置。wrapper 保留命令本来的退出码，并转发 Ctrl+C / SIGTERM。需要多个终端时分别 `source` 即可；也可直接执行 `/tmp/singularity-auto-$(id -u)/bin/placeholder`。

`--gpus 0,1,2,3` 指 supervisor 管理列表中的第 0–3 张卡；列表可在 `placeholder status` 中查看。wrapper 为实验设置对应 GPU UUID 的 `CUDA_VISIBLE_DEVICES`，实验内部会将这些卡重新编号为 0–3。也可以传完整 GPU UUID。

`run` 后的程序要保持前台，不要在命令末尾加 `&`、`nohup` 或自行 daemonize。服务正常运行期间 reservation 会保留；并行的多个 `run` 各自持有 reservation，某张卡上所有 reservation 结束后才允许恢复该卡的占卡程序。这个 wrapper 让的是 placeholder，不负责给两个实验互相排队或分配显存；并发服务应自行选择互不重叠的 GPU 集合。先完成不需要 GPU 的模型下载与依赖准备，再通过 `run` 启动服务，可以缩短让卡后的空闲时间。

### 自动检测与明确让卡的区别

- **逐卡自动模式**：某张受管 GPU 无外部 CUDA 进程，且利用率不超过 5%、已用显存不超过 256 MiB，连续 30 秒后，启动该卡的 worker。8 张卡均空闲时，每卡各启动一个。
- **发现实验**：某卡出现外部 CUDA 进程，停止该卡的占卡 worker，其余空闲卡继续占卡。即使模型暂时利用率为 0，只要进程仍持有 CUDA context，就不会恢复该卡的占卡。
- **明确让卡**：`placeholder run --gpus 0,1,2,3 -- ...` 在实验创建 CUDA context **之前**为指定卡建立 reservation，等待这些卡的 worker 真正退出，再启动实验。模型下载、CPU 初始化期间也保持让卡，避免自动检测的启动竞态；其他卡照常管理。
- **发生错误**：监控失败时停止占卡；worker 失败后 supervisor 继续运行并重试。只终止自己启动和确认归属的 worker，不使用 `pkill python` 或杀其他 CUDA 任务。

默认每秒检测一次。自动检测无法在程序使用 GPU 之前预知其意图，也不能保证未使用 wrapper 的模型不会在让卡前遇到显存不足。优先使用 `run`。

根据实际回收规则，默认使用逐卡管理：实验使用 4 张卡时，另外 4 张继续占卡。实验本身暂停、只保留 CUDA context、处于 CPU 初始化阶段时，已让出的卡仍可能出现低利用率；工具优先保证实验不被占卡程序干扰。GPU 利用率、回收阈值、管理员取消、运行时限和节点故障都由平台决定；工具无法保证 job 不被回收。

不要同时运行 `sglang_handson/scripts/gpu_keeper.sh` 或旧的 `gpu_occupy/resnet.py`，否则它们会被识别为外部任务，并与本工具冲突。

## 3. 故障和边界

原命令：

```bash
python resnet.py && sleep 10540800
```

`&&` 只在左边正常返回 0 时继续。被 kill 的 Python 通常非零退出，`sleep` 因此被跳过，AML 的主命令结束，job/tunnel 也随之消失。本仓库把 session/supervisor 作为前台主进程，worker 是可独立停止的子进程。**杀 supervisor 或 session 仍会结束 job。**

本地控制目录是 `/tmp/singularity-auto-<UID>/control/`，日志位于相邻的 `logs/`。session 将启动 manifest 写入 Blob 的 `lucayu/sglang/sessions/<session-id>/`；实时状态和旋转日志留在本地，并输出到 AML job 日志。Blob 目录不用于恢复锁、PID 或 GPU 进程。

支持按完整 GPU UUID 选择分配给当前 job 的设备；不会凭猜测把 CUDA ordinal 当物理 GPU。**当前自动进程识别要求 job 使用 host PID namespace，并且 `/proc` 与其一致。** 如果容器使用独立 PID namespace，NVML 返回的 host PID 不能可靠对应容器 PID，工具会在启动 worker 前报告错误，不会猜测归属。MIG、MPS、看不到其他 CUDA 进程的隔离环境也需要额外适配；尚未确认 MSRA 当前 job 的 namespace 配置。

出现未知进程或监控歧义时，优先让卡并在状态中报告原因。若 `CUDA_VISIBLE_DEVICES` 使用局部数字映射，须使用分配给 job 的完整 UUID 配置该 mask；`--gpus` 本身不会覆盖或扩大原来的 visibility mask。

可调参数见：

```bash
bash scripts/aml_start.sh --help
python3 -m singularity_placeholder supervise --help
python3 -m singularity_placeholder run --help
```

## 4. 模型、代码、trace 和结果放在哪里

参见 [存储与重建工作流](docs/storage-workflow.md)。启动脚本已经建立目录并生成每个 job 的 `env.sh`；目前**不会**修改 `sglang_handson`、安装它的依赖、下载模型、上传实验 trace、恢复 checkpoint 或自动重提 AML job。这些属于下一步实验 bootstrap 的工作，不应误以为占卡 supervisor 已经完成了它们。

## 5. 验证

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/aml_start.sh
```

测试以模拟 NVIDIA 输出和真实本地子进程验证监控、让卡握手、进程退出和路径处理；不需要 GPU，不会分配 CUDA 显存。本仓库开发环境为 macOS，尚未在 MSRA 的 8×H100 job 上实测。

第一次运行后，用 `placeholder status` 与 `nvidia-smi` 确认实际 GPU 数量和状态；用 wrapper 启动一个实验，检查 worker 先退出、实验结束后经过空闲等待恢复。不要把本地模拟测试当作平台不会回收 job 的证明。
