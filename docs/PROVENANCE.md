# 源码来源与改动边界

来源仓库：`ZhengqiSun/multi-views-lingbot`；分支 `runtime-complete-20260912`；提交 `b61bd38007eb1bc87214989c83760a9e02d75bc5`。2026-09-12 已再次核对该远端分支，仍为该提交。

## 选取范围

- `tools/`：`xiaoqi/multiview-map-v0-qxq/tools` 中的数据、几何、模型、训练、生成、评估实现及依赖。
- `pipelines/scene_rollout/tools/`：`zhengqi/multiview-map-v0/code_interaction_v1_20260731_v0/tools` 中的场景、多视角、动力学、AR、评估与 Fast v2 实验及配套依赖。
- `baselines/`：Xiaoqi 工作树内的 LingBot Cam / SCOPE 源码、接口规范与测试。
- `vendor/lingbot/`：上传中的 `xiaoqi/lingbot-world` Python runtime、generator 和许可证。

共 268 个导入文件，约 4.8 MB，包含小型代码 fixture 与许可证。静态 import 和源码中引用的动态模块/脚本用于追踪依赖；保留原文件名及配套版本。来源明确跨工作树引用的 sampler 路径继续指向相应目录。

金融、Yitao/Hongfeng 项目按用户要求排除。未使用的 legacy、固定任务 watchdog/launcher、历史整树、演示拼图脚本、输出文档与缓存留在原备份。部分实验程序仍被测试或方法模块直接引用，作为内部依赖保留，不被注册成全项目默认运行入口。

`provenance/selection_inventory.json` 对两套项目目录逐文件记录纳入位置或排除理由；`source_manifest.json` 记录导入源路径、原始 SHA256 与依赖边。筛选以代码关系为依据，不以目录所有者或文件时间推断算法有效性。

## 实际改动

64 个导入文件发生修改：63 个项目文件调整源码/资产路径及 `ffmpeg` 默认位置；一个 Wan T5 文件将导入时的 CUDA device 求值延后到构造时，以支持 CPU 导入检查。CUDA 可用时仍默认使用当前 CUDA device。

没有改写网络结构、loss 公式、checkpoint 字段、采样算法或 split 协议。新增的是项目分类入口、interaction JSON 配方、路径辅助模块、测试、文档和来源工具。原文件及整理后哈希在 `clean_manifest.json`；全部导入源码改动在 `source_patches.patch`，可逐行复核。

`scripts/assemble_from_backup.py SOURCE EMPTY_OUTPUT` 可复现原始源码选取；生成的是路径调整前版本。随后在输出目录应用本仓库的 `provenance/source_patches.patch`，可复现导入源码的整理版本；新增的入口、路径辅助文件、测试与文档直接以本仓库为准。`port_source_paths.py` 记录批量路径迁移规则，最初的八个入口路径精简也完整包含在 patch 中。

## 历史执行证据

历史整树仍在原仓库，不复制到新历史。原 runtime snapshot 的 `b44b1bc...` trainer、后续 PPU launcher 的 `108d136...` 和当前 trainer 不是相同源码。当前 trainer 比 `108d136` 增加可配置的 per-release 最小 record 下限；默认值不变。

原上传的 Zhengqi 1-step 日志是历史版本的执行证据。Xiaoqi 8500 步/77 输出的原始运行日志没有上传。本仓库的 CPU 检查不能用来追认这些历史运行或视频质量；新仓库的真实 GPU 复跑尚未进行。

## 第三方代码

`vendor/lingbot` 保留原 `LICENSE.txt` 和模块版权信息，来源是项目已修改的工作树，不等价于未修改上游。SCOPE 和官方 causal Fast v2 runtime 未上传，只有项目集成代码。2026-09-12 经项目所有者授权，仓库更名为 OmniScene Forge 并公开；项目自有代码以 Apache 2.0 发布。第三方文件保留各自条款，特别是 LIA 衍生动画模块的非商业许可，见 [第三方说明](../THIRD_PARTY_NOTICES.md)。
