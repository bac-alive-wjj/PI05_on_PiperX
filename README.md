# PI05 on PiperX

本仓库记录在 AgileX Piper X 上完成 OpenPI π0.5 数据采集、模型微调和 RTC 真机推理的工程实现。项目不依赖 ROS，而是直接使用相机 SDK、pyAgxArm 和 CAN 接口连接硬件。

```text
双相机与机械臂状态采集 → LeRobot 数据集 → π0.5 LoRA 微调 → Linux RTC 真机推理
```

项目的重点是明确模块之间的数据契约，包括图像格式、关节单位、夹爪语义、时间戳、动作表示、归一化和预期执行时刻。单个模块能够运行并不足以保证真机链路正确，各阶段必须对同一份数据保持一致解释。

## 仓库结构

当前仓库由两个独立交付子包组成，不是可以直接从仓库根目录运行的单体工程。

```text
PI05_on_PiperX/
├── capture-to-finetune/
│   └── piper_capture_finetune_bundle_20261007/
│       ├── capture/       双相机、CAN、时间对齐和 LeRobot 写入
│       ├── openpi/        Piper 数据变换与 π0.5 微调配置
│       ├── docs/          数采、环境和微调 runbook
│       ├── tools/         只读硬件检查工具与 udev 规则
│       └── tests/         Linux 设备适配测试
├── inference/
│   └── runtime_root/
│       ├── inference/     RTC policy、调度、传输和 Linux 真机执行器
│       ├── capture/       Linux D435i、DC1 和 Piper CAN 输入适配器
│       ├── openpi/        H40 推理所需的修改版 OpenPI 运行源码
│       ├── docs/          Linux 环境迁移与 RTC 运行手册
│       ├── tools/         GPU、checkpoint 和设备检查工具
│       └── tests/         RTC、传输与设备适配测试
└── LICENSE
```

数采与微调从 [`capture-to-finetune/piper_capture_finetune_bundle_20261007`](capture-to-finetune/piper_capture_finetune_bundle_20261007/) 开始。Linux 推理部署时，应把 [`inference/runtime_root`](inference/runtime_root/) 的内容作为 `PIPER_PROJECT_ROOT`，不要直接把本仓库根目录当作运行根目录。

## 数据契约

| 项目 | 当前约定 |
|---|---|
| 采样频率 | 30 Hz |
| 外部相机 | Intel RealSense D435i，`observation.images.base` |
| 腕部相机 | Orbbec DaBai DC1，`observation.images.wrist` |
| 图像 | RGB `uint8[H,W,3]`；采集端不裁剪、不缩放 |
| 本体状态 | `float32[7] = [q1..q6(rad), gripper_closedness]` |
| 夹爪 | `0` 全开，`1` 全闭 |
| 原始动作 | 绝对 `float32[7]`；当前数采使用下一逻辑时刻的从臂状态作为代理标签 |
| 训练关节动作 | 前六维转换为 `q_target - q_current`，夹爪仍为绝对闭合度 |
| 模型图像处理 | 保持比例并 padding 到 `224×224`，不做中心裁剪 |
| π0.5 动作接口 | Piper 有效动作 7 维；模型内部补到 32 维并投影到 action expert 隐藏空间 |

三路传感器数据在到达电脑后使用 `time.monotonic_ns()` 记录软件时间戳。采集器以 30 Hz 逻辑时刻查询缓存中的最近样本，任一路偏差超过 `±30 ms` 时跳过该采样点。

## 当前状态

已经完成：

- D435i、DaBai DC1 与 Piper CAN 的读取和三路时间对齐。
- LeRobot v2.1 数据写入、逐轨迹 task prompt、假数据生成和数据格式验证。
- 修改版 OpenPI 数据配置、统计量计算、π0.5 LoRA 微调及 checkpoint 下载流程。
- Linux 下的相机、SocketCAN、Piper 状态读取、JAX GPU 和 checkpoint 加载。
- π0.5 RTC 模型服务、WebSocket 契约校验、动作缓存、延迟回退和 Linux 真机执行器。
- H40 部署配置：`action_horizon=40`、`execution_steps=20`、`num_steps=8`、30 Hz。
- 双相机、CAN、真实模型和空执行器的完整只读联调，以及受限真机行为测试。

仍需继续验证或改进：

- H40 连续真机运行时的稳定 RTC 占比、P95 延迟和任务成功率。
- 更完整的关节坐标、限位、跟踪误差、夹爪量程和异常停止验证。
- 当前 `action[t] = state[t+1]` 是运动结果代理，不等价于主臂真实控制目标；快速动作可能产生标签偏差。
- Linux 设备适配器已经通过真机验证，但正式 LeRobot 数采入口仍需完成 Linux 适配器接线。
- 当前客户端退出只会停止发送新目标并断开连接，不会发送立即急停；机械臂可能继续完成最后一个已接收目标。

当前实现已经覆盖从数采到真机动作执行的完整主干，但仍属于研究和受监督实验系统，不适合无人值守运行。

## 使用入口

### 数采与微调

先阅读：

- [数采与微调包说明](capture-to-finetune/piper_capture_finetune_bundle_20261007/README.md)
- [完整数采与云端微调 Runbook](capture-to-finetune/piper_capture_finetune_bundle_20261007/docs/training/CAPTURE_FINETUNE_RUNBOOK.md)
- [数采实现与数据格式](capture-to-finetune/piper_capture_finetune_bundle_20261007/docs/collection/DATA_COLLECTION_DESIGN.md)

正式数采入口位于：

```text
capture-to-finetune/piper_capture_finetune_bundle_20261007/capture/run_aligned_capture.py
```

假数据生成入口位于：

```text
capture-to-finetune/piper_capture_finetune_bundle_20261007/capture/generate_fake_lerobot_dataset.py
```

### Linux H40 RTC 推理

先阅读：

- [推理运行子包说明](inference/runtime_root/README_MIGRATION.md)
- [Linux 环境迁移手册](inference/runtime_root/docs/environment/LINUX_INFERENCE_MIGRATION_GUIDE.md)
- [Linux H40 RTC 标准运行流程](inference/runtime_root/docs/inference/LINUX_RTC_INFERENCE_RUNBOOK.md)

模型服务入口：

```text
inference/runtime_root/inference/serve_piper_rtc.py
```

Linux 设备与 RTC 主入口：

```text
inference/runtime_root/inference/run_piper_rtc_linux.py
```

## 仓库未包含的内容

为控制仓库大小并避免平台混用，本仓库不包含：

- π0.5 基础权重与 H10/H40 微调 checkpoint；
- 真实或假 LeRobot 数据集；
- Python 虚拟环境、pip/uv/Hugging Face 缓存；
- NVIDIA 驱动和系统 CAN 内核模块；
- 完整 Orbbec SDK 二进制文件；
- 真机运行日志。

推理部署时必须另外放置与配置名匹配的 checkpoint 和 `norm_stats.json`。H10 与 H40 的配置、horizon、权重和 stats 不能混用。

## 安全说明

数采子包只读取相机和 CAN，不使能机械臂，也不发送运动指令。推理子包只有在显式选择真机执行器后才会控制机械臂。

真机运行前必须依次完成设备只读检查、checkpoint 检查、单次 action 检查、完整 RTC 只读测试和工作空间确认。不要通过试错选择关节符号，不要在无人监护的情况下运行连续动作。
