# Multi-view LingBot

完整多视角项目的源码整理仓库：数据与相机对齐、地图与场景记忆、条件渲染、模型训练、长时序生成、动力学、多视角评估和 baseline。**Interaction 是其中一条条件建模路线。**

基于原仓库 `runtime-complete-20260912` 的 `b61bd38` 建立独立历史。金融项目、Yitao/Hongfeng 项目、历史整树、视频、权重、日志与缓存不进入新仓库。原始备份保持不变。

## 项目组成

| 环节 | 内容 | 主要入口 |
|---|---|---|
| 数据 | 视频切片、相机/帧对齐、match split、latent cache、多视角配对 | `data.*` |
| 地图与场景 | 静态地图、episode 玩家状态、BSP/mesh 投影、dense 渲染 | `geometry.*` |
| 条件建模 | Dense、state v2、action 与 interaction sidecar | `conditions.*` |
| 训练 | Dense、base expert、state、interaction 四套训练入口 | `train.*` |
| 视频生成 | Dense/state/interaction 采样、跨窗口 AR、history guidance | `infer.*` |
| 动力学 | v0–v3 玩家动力学、运动参数拟合、长时域评估 | `dynamics.*` |
| 场景 rollout | 多场景 10 ego、原生动力学 10 视角、SRCDS 引擎 | `rollout.*` |
| 评估 | 共视一致性、结构遵循、零可见幻觉、AR 漂移、显著性与消融 | `eval.*` |
| Baseline | LingBot camera-only LoRA、SCOPE action-only bridge | `baseline.*` |

实际代码关系与尚未接通的部分见 [项目路线图](docs/PROJECT_MAP.md)。入口可导入、组件测试通过、GPU 端到端成功是不同的验证层级，分别记录在 [验证结果](docs/VALIDATION.md)。

## 目录

```text
multiview.py                  全项目入口 + interaction 配置入口
project_routes.py             按功能组织的路线目录与独立进程启动
 tools/                       Xiaoqi 当前数据/几何/训练/推理/评估代码
 pipelines/scene_rollout/      Zhengqi 场景、动力学与配套版本
 baselines/lingbot_cam/        Camera-only baseline
 baselines/scope/              SCOPE bridge 与 action 转换
 vendor/lingbot/               上传中实际使用的 LingBot/Wan 源码及许可证
 configs/interaction/         Interaction 的便携 JSON 配方
 tests/                       跨模块接口、梯度和多视角数据测试
 requirements/                基础、数据评估、CUDA 与测试环境
 docs/                        方法关系、使用方式、验证边界
 provenance/                  逐文件来源、选择依据、原始/整理后哈希
 scripts/                     源码装配、路径迁移和完整性检查
```

同名模块在两套来源中有实质差异，因此保留配套目录和文件名。路线入口在对应目录启动；源码明确引用另一套代码时保留该关系。不要任意交换两套 `state`、renderer 或 checkpoint。

## 环境

Python 3.11+。在独立环境中安装匹配平台的 PyTorch/torchvision，再安装项目依赖：

```bash
python -m venv .venv
source .venv/bin/activate
# Linux / NVIDIA 的原项目环境：
python -m pip install -r requirements/cuda124.txt
python -m pip install -r requirements/base.txt -r requirements/data-eval.txt -r requirements/dev.txt
# GPU kernel 按当前机器构建；CPU 组件测试不需要：
# python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

本次验证使用 macOS、Python 3.11、PyTorch 2.6 CPU；Linux GPU 环境尚未重新安装验证。SCOPE 和 LingBot Fast v2 实验各自需要其匹配的外部源码与模型环境，详见 [外部资产](docs/ASSETS.md)。

## 使用

```bash
python multiview.py list
python multiview.py run data.multiview -- --help
python multiview.py run train.state -- --help
python multiview.py run rollout.multiscene -- prepare --help
python multiview.py run baseline.lingbot-train -- --help
```

`run 路线 -- 参数` 保留原入口的参数和子命令。包装层的 `--dry-run` 放在分隔符前，仅打印命令。该模式不自动启动多卡；需要分布式训练时使用对应 trainer 的 torchrun 接口。原参数中的相对路径以所选路线目录为基准，建议数据与输出使用绝对路径。

```bash
python multiview.py run train.state --dry-run -- train --out-dir /absolute/path/run
python multiview.py run rollout.dynamics -- self-test
```

Interaction 另外提供 JSON 配方，路径相对于配置文件目录解析：

```bash
cp configs/interaction/smoke.example.json configs/interaction/smoke.local.json
# 编辑真实资产路径后，先查看启动命令
python multiview.py interaction-smoke --config configs/interaction/smoke.local.json --dry-run
```

完整步骤见 [Interaction 配方](docs/INTERACTION.md)、[各路线输入输出](docs/PROJECT_MAP.md) 和 baseline 目录的 README。模型、训练数据、地图、manifest 及引擎安装由外部提供；可用 `MULTIVIEW_ASSETS` 设置资产根目录。已有 manifest 内部的服务器路径需要在目标机器上可解析。

## 检查与来源

```bash
python scripts/check_codebase.py --help-smoke
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python multiview.py run rollout.dynamics -- self-test
```

源码来源、选择范围与改动边界见 [来源说明](docs/PROVENANCE.md)。历史实验运行记录没有被当成本仓库的运行证明。
