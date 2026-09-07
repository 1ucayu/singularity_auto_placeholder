# 从临时 AML job 恢复工作环境

把 job 看成随时可以丢弃的计算节点：GitHub 保存重建所需的定义，Blob 保存需要长期保留的数据，本地 SSD 提供运行速度。重新启动 job 是重建环境；恢复一个实验还需要它保存了可用的 checkpoint。

## 推荐目录

截图中的 input `lucayu` 挂载整个 datastore 根目录，并不表示它已经是个人子目录。新工具在该挂载下使用以下个人前缀：

```text
${{inputs.lucayu}}/lucayu/sglang/          # Blob，长期保存
  models/<model>/<immutable-revision>/   # 完整权重、配置、校验清单
  datasets/
  traces/input/
  runs/<experiment>/<run-id>/
    manifest.json
    results/
    traces/                             # 已关闭的 trace 分段
    logs/                               # 已关闭的日志分段/归档
    checkpoints/
  sessions/<session-id>/                 # 占卡工具的启动 manifest
  code-snapshots/                        # 可选，尚未 push 的改动备份

/tmp/singularity-auto-<UID>/              # 节点本地；有更大 SSD 时用 --local-root
  workspace/sglang_handson/              # Git checkout、.venv、SGLang 源码
  workspace/models/                     # 容量允许时，从 Blob stage 的模型副本
  cache/                                # HF、pip、uv、编译缓存
  outputs/<session-id>/                  # 实验正在写的输出
  control/                              # 只在此 job 有意义的锁、reservation、状态
  tunnel/                               # 私有认证缓存，仅保留在当前节点
  env.sh
```

Blob 是对象存储，经 FUSE 映射后也不能当作完整的 POSIX SSD。Git 的锁和小文件、`.venv`、包安装、编译缓存、运行中的高频 trace 放本地。模型大文件和完成的实验结果放 Blob；模型可从 Blob 直接读取，但大量分片/随机读取的吞吐需要实测。优先在容量足够时 stage 到本地 SSD，下载/复制完成并校验后才声明模型可用。

`/tmp` 只是通用默认值，不代表它一定是机器上容量最大的 NVMe。通过节点实际磁盘信息选择本地 SSD，再传 `--local-root /actual/local/ssd/lucayu`。不能根据 Blob 挂载的 `df` 数字来判断后端对象存储容量。

## 对现有 sglang_handson 的对应关系

当前仓库是私有仓库，读取代码需要已有 GitHub 登录或平台安全提供的读取凭证。公开占卡仓库可以匿名 clone，不意味着私有实验代码也可以。

每个 VS Code 终端载入生成的 `env.sh` 后：

|变量|含义|
|---|---|
|`BLOB_MOUNT`|AML 给出的真实挂载根路径|
|`BLOB_ROOT`|个人持久目录：`<mount>/lucayu/sglang`|
|`LOCAL_WORK_ROOT`|本地代码与模型运行副本目录|
|`OUTPUT_ROOT`|当前 session 的本地实验输出目录|
|`HF_HOME` / `HF_HUB_CACHE`|当前节点的 Hugging Face 工作缓存|

现有 `scripts/common.sh` 支持 `BLOB_ROOT`、`MODEL_DIR`、`OUTPUT_ROOT` 覆盖。例如，stage 完模型后，把 `MODEL_DIR` 指向本地完整模型目录，再启动原来的实验脚本。新工具不会搬移原先 datastore 根目录下已有的 `models/`；若已有模型在那里，应显式引用旧路径或另行迁移，避免无意重复下载。

现有 `serve_h200_fp8.sh` 针对 4 张 H200，固定 TP=4 / EP=4，并拒绝非 H200 硬件。你的新节点是 8 张 H100，需要单独检查模型总权重、每卡显存、KV cache、量化/内核兼容性和 TP/EP。绕过硬件检查并不代表这套参数已经验证。

现有依赖安装和模型下载没有完整固定版本；现有输出归档是手工打包上传。因此它们还不足以保证一次 bootstrap 重建全部实验环境。

## 日常工作流

1. **在 GitHub 固定可重建的内容。** 提交代码、实验配置、依赖锁文件、容器镜像版本、SGLang commit、模型 ID 和 immutable revision。模型与完整 trace 不进 Git。每天一次 push 只能保护上次 push 之前的内容；若要保护当天未提交的代码，定期把经过排除凭证/缓存的源代码快照或 patch 存入 Blob。
2. **按版本准备模型。** 首次把模型下载到本地临时目录，检查权重分片和校验清单，上传至 Blob 的版本目录，最后写完成标记。再次部署时验证完成标记及 manifest，复用已有模型。半成品不能被下一次 job 当成完整模型。
3. **为每个实验创建唯一 run ID。** 记录代码 commit、模型 revision、镜像/依赖版本、GPU 型号与数量、TP/EP、输入 trace、随机种子和完整命令。并行 job 不写同一个输出目录。
4. **运行中持续保存。** trace 和日志按大小或时间分段；关闭一个分段后再上传，并记录校验值。可恢复的训练/实验定期写 checkpoint。上传周期为 1 分钟不等于最多只丢 1 分钟，还取决于分段关闭和上传是否成功；以最后已确认持久化的内容为准。
5. **结束时生成 manifest。** 汇总结果和文件校验值，完成上传后才标记 run 完成。输出先在本地形成、再传完整对象，可以降低 Blob 小文件和锁开销。

不能只在 shell 的 EXIT trap 里归档：`SIGKILL`、节点故障、job 被删除时，trap 没机会执行。也不能把“FUSE 写操作已返回”等同于已经完成独立的远端持久性核验；关键结果应使用实际存储服务上传确认、清单校验与明确的完成标记。

## 新 job 的一键重建应做什么

下一阶段的实验 bootstrap 应固定完成以下顺序：

```text
AML 创建新 job 并挂载 Blob
  → 本地 clone 固定 commit（私有库使用安全身份）
  → 固定镜像 / 锁文件重建运行依赖
  → 验证 Blob 模型并 stage 到本地
  → 选择最后完整 checkpoint 或未完成 trace 分片
  → 独立启动 GPU supervisor 与 tunnel；外层 sleep 保持主命令
  → 用 placeholder run 启动服务或实验
  → 周期性持久化新结果
```

准备模型和环境可能花较长时间；实际实现时可先启动 supervisor，在 placeholder 仍运行时完成不需要 GPU 的模型下载和依赖准备，再通过 `placeholder run --gpus ...` 启动实际实验，缩短让卡后的空闲时间。顺序要保证同一时刻只有一套 GPU 管理器。

这次仓库已实现 **job 内的占卡管理、可选 tunnel 启动、目录与环境变量初始化**；其余步骤是明确的下一阶段工作。新 job 可以重用 Blob 的权重和最后保存的 checkpoint，但旧进程、未保存的 GPU/KV cache 和未上传的 trace 不会凭空回来。

完全自动恢复还分两层：相同 Command 可自动部署“已提交成功的新 job”；自动发现 job 消失并提交替代 job，需要运行在 job 外部、拥有 Azure 身份的控制器。它应使用退避、预算/重试上限和持久化 run 状态。已被销毁的 job 无法自己重提自己。

Tunnel 身份也是独立的一层：本地 CLI 缓存能跨进程重启，不能自然跨被销毁的节点。若希望新 job 不再扫码，需使用企业支持的凭证注入方式，并处理过期和撤销；不要把共享 Blob 当作凭证保险箱。
