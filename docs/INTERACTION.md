# Interaction 路线运行说明

此文档只描述完整多视角项目中的 dense + state + interaction 路线。全项目见 [README](../README.md)。

## 环境

服务器建议按既有项目使用 Python 3.12；本地接口测试支持 Python 3.11。创建独立环境后：

```bash
python -m venv .venv
source .venv/bin/activate

# Linux / NVIDIA CUDA 12.4 profile
python -m pip install -r requirements/cuda124.txt
python -m pip install -r requirements/base.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -r requirements/dev.txt
```

FlashAttention 需要与 Python、PyTorch、CUDA 对应的构建环境或已有匹配 wheel。CPU 开发环境先安装匹配的 torch/torchvision，再安装 base/dev，跳过 CUDA profile 和 FlashAttention；CPU 测试不等于可以在 CPU 上执行实际模型生成。

这些依赖文件是整理后的环境配置，完整 GPU 安装及真实资产复跑尚未在本次机器上验证，详见 [验证记录](VALIDATION.md)。

## 配置路径

复制示例为本机配置；`*.local.json` 已忽略，不提交机器路径。

```bash
cp configs/interaction/smoke.example.json configs/interaction/smoke.local.json
cp configs/interaction/train.example.json configs/interaction/train.local.json
cp configs/interaction/infer.example.json configs/interaction/infer.local.json
```

修改 `args` 内的 manifest、checkpoint 和输出路径。相对路径始终相对于 **配置文件目录**，不受启动目录影响，也支持 `${MY_DATA_ROOT}` 等环境变量。
示例使用 `assets/` 作为外部资产位置；配置中可以直接指向服务器真实挂载，不需要复制大文件。manifest 内部引用的文件路径仍需能在运行机器上解析，本入口不批量改写数据内容。

LingBot 代码由统一入口固定到 `vendor/lingbot`；不用重新安装另一份 `wan`。底层模块的资产默认目录可用 `MULTIVIEW_ASSETS` 设置，显式配置优先。

## 训练与推理

先查看解析后的真实命令；`--dry-run` 不启动模型、不创建输出。

```bash
python multiview.py interaction-smoke --config configs/interaction/smoke.local.json --dry-run
python multiview.py interaction-infer --config configs/interaction/infer.local.json --dry-run
```

准备好 [外部资产](DATA_CONTRACT.md) 后：

```bash
python multiview.py interaction-preflight --config configs/interaction/smoke.local.json
python multiview.py interaction-smoke --config configs/interaction/smoke.local.json
python multiview.py interaction-train --config configs/interaction/train.local.json
python multiview.py interaction-infer --config configs/interaction/infer.local.json
```

- `smoke` 固定为单卡、1 optimizer step，并重置 optimizer/本轮步数。不会沿用配置中的长训练步数。
- `train` 按 `distributed` 选择单机或多机 torchrun；多机需要显式设置 `nnodes/node_rank/master_addr`。
- 训练示例默认 LOW expert，输出步数和学习率是可编辑配置，不是本次验证过的实验配方。训练 HIGH expert 时要使用对应的 warm-start checkpoint。
- `infer` 要求显式 LOW adapter checkpoint，防止误将 clean base 输出当成 interaction 结果。HIGH checkpoint 可选；缺省时 HIGH expert 为 clean base，此模式会进入原 sampler 报告。
- `interaction-variant` 支持 `true/blank/shuffled`。固定样本、seed、checkpoint 和相机再比较条件贡献；视频生成成功并不直接证明多视角一致性提升。

每次有 `out-dir` 的真实运行写入独立 `launch_*.json`，记录完整参数和 tools/vendor 源码哈希。训练与推理产物沿用底层脚本格式。

## 数据准备与校验

```bash
python multiview.py prepare-state --config configs/interaction/prepare-state.example.json --dry-run
python multiview.py prepare-interaction --config configs/interaction/prepare-interaction.example.json --dry-run
python multiview.py prepare-eval --config configs/interaction/prepare-eval.example.json --dry-run

# 查看任何底层入口所需参数，然后编写相应 args 配置
python multiview.py prepare-aligned --tool-help
python multiview.py render-dense --tool-help
python multiview.py evaluate-loss --tool-help
```

工具关系和原始数据要求见 [数据契约](DATA_CONTRACT.md)。本仓库从既有模型、游戏轨迹、地图和 latent/cache 资产继续工作，不包含原始视频采集与基模预训练。

## 验证与版本

```bash
python -m pytest -q
python scripts/check_codebase.py
```

检查范围包括便携路径、重复参数、单步限制、交互事件/缓存/梯度、state 投影契约、adapter 重计算一致性、vendored Wan 导入。无 GPU、数据与 checkpoint 时，真实训练／视频推理仍需在服务器验收。

[来源及改动边界](PROVENANCE.md) 记录当前源码与历史训练版本的关系。缺失原始运行日志不妨碍整理源码，但不据此认证历史 8500 步或视频质量。
