# 外部资产与不可替代的依赖

源码仓库不包含研究数据、模型权重和服务器安装。`MULTIVIEW_ASSETS` 默认指仓库的 `assets/`；已迁移的默认路径可由环境变量或显式参数配置。manifest 内部的路径不会自动重写。

| 路线 | 需要提供的资产 | 本次状态 |
|---|---|---|
| 数据与几何 | 原始 match/episode JSON、世界事件、视频、camera pose/intrinsics、BSP faces/navmesh | 未上传本地验证资产 |
| 条件训练 | map/cache manifest、latent/text cache、state cache；interaction 另需事件/sidecar | 未做真实数据 preflight |
| LingBot 训练/推理 | 匹配的 base camera low/high experts、T5/tokenizer、Wan2.1 VAE、对应 adapter/LoRA checkpoint | 未下载模型、未进行 GPU 训练或生成 |
| 场景 10 ego | 同步 10 人片段、场景选择、dense/state cache、脚本约束的 W5 LOW/HIGH checkpoint | 仅 CPU 场景协议测试 |
| 玩家动力学 | 轨迹/动作训练 manifest、地图约束、运动参数和动力学 checkpoint | 仅规则 self-test；未训练新模型 |
| SRCDS | `engine_smoke_v0/csgo_ds/srcds_linux`、游戏资源、`engine_inference_bridge_v1.smx` 等安装 | 引擎与 bridge 源码/二进制未上传 |
| SCOPE | 官方源码 checkout、DiT/VAE/T5/tokenizer 权重、SCOPE action parquet、训练集 mouse calibration | bridge 已保留；官方实现和模型未上传 |
| LingBot Fast v2 实验 | 官方 `lingbot-world-v2-code` 和 `lingbot-world-v2-14b-causal-fast` 模型 | 上传里只有集成实验代码；此官方 runtime 未上传 |
| 评估 | 相同样本/seed/checkpoint 的视频，匹配协议的 pair index、几何/相机/mask 等 | 未重做真实实验评分 |

CPU 测试可以验证本仓库的接口和组件逻辑，无法证明这些外部资产齐备，也无法证明研究指标达到历史记录。

部分导入工具仍保留旧 mount 前缀，用于识别旧 manifest 的路径并做兼容转换；它们不应被当作新机器的挂载要求。`provenance/remaining_server_paths.json` 列出剩余字符串所在位置，用于迁移资产时逐项核对。

## Demo 06/07 data preview

The [06/07 sample guide](demo_reproduction/06_07/README.md) describes a separate Release archive containing six historical views, GT, context/camera inputs and 126 legacy dense NPZ files. Extract it under `assets/demos/06_07/`. Model-specific setup information is not provided. Source files and existing inference routes are unchanged; this does not supersede the GPU-validation limits above.
