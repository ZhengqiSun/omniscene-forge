# 多视角项目的代码关系

## 数据与场景链路

同一 match / episode 的视频、动作和世界事件首先构成 source manifest。切片与相机生成确定 raw frame、video frame、latent frame 的对应关系；地图几何和 episode memory 为每个相机渲染 dense 条件；aligned cache 将这些条件与 latent/text cache 配对。state 和 interaction 是可选的附加条件。

`tools/` 保留当前工作树的版本。`pipelines/scene_rollout/tools/` 保留场景生成和动力学所依赖的版本，包含 state v0/v1 等历史协议。它们是项目的不同方法链路，不能仅凭文件名相近相互替换。

| 阶段 | 主要文件 | 输入 → 输出 |
|---|---|---|
| 视频切片 | `materialize_clips_v1`、`materialize_event50h_clips_v1` | 原视频/对齐行 → 81 帧视频 |
| 地图记忆 | `build_static_memory_v0`、`build_episode_memory_v0` | match 数据/地图 → static/episode memory |
| 几何渲染 | `build_mesh_projection_v0`、`gpu_mesh_renderer_v0`、`run_dense_v2_gpu_v1` | BSP/mesh、玩家状态、相机 → dense cache |
| latent 缓存 | 场景目录中的 `cache_tier69h_latents_dlc_v0` | 视频、文本、VAE/T5 → latent/text cache |
| 对齐与索引 | `build_memory_dense_aligned_cache_v0`、`build_v2_map_manifest_and_cache_v1` | source、map、cache → 训练 manifest |
| 附加条件 | `build_state_channels_v2`、`build_interaction_sidecar_v0` | cache、动作/事件 → state/interaction sidecar |

这些文件名省略 `.py`。所有函数和 CLI 保留原命名；分类入口见本页末尾的全量目录。

## 训练与生成

Dense → state → interaction 是逐步增加条件的模型路线。四个 trainer 均保留，并非只提供 interaction。它们共享部分数据/adapter 工具，但不同版本具有不同的 checkpoint 和条件协议。

训练读取对齐数据，构造 dense/state/interaction token，通过上传中已修改的 LingBot 注入模型；各入口分别支持其原有训练模式。生成阶段读取对应 checkpoint，按目标相机采样视频。AR wrapper 将前一窗口的生成尾帧用于后一窗口，history guidance 和 anchor 变体也保留在场景目录。

**当前工作树的 trainer 没有导入 `multiview_consistency_data_v1.py` 或配套 pair manifest builder。** 配对数据工具已存在，但本次未发现它接入联合多视角训练 loss 的代码证据。此次整理保留这个现状，没有新增一个未经验证的训练算法。

## 多视角与评估

多视角组从同一场景的不同玩家相机建立，正样本要求 21 个采样位置的 raw indices、frame counts 和 world ticks 完全一致。配对工具还生成 time-shift、wrong-window、cross-episode、cross-match 对照。RGB/camera loader 与预检负责视频解码、相机矩阵和分组关系。

这一配对数据集使用自己的 match 分割 salt；`canonical_match_split_v0/v1` 是另一套历史数据分割协议。两者不能未经核对就被视为相同 train/val/test 集合。v1 扩充的 16 个 heldout match 曾被某些旧 checkpoint 训练使用，旧模型对比应遵守源码中的 v0 heldout 约束。

多视角评分包括 homography-aware SymmVC、结构遵循、零可见区域幻觉、AR 漂移和配对显著性。各评分工具需要自己的几何、视频、mask 或 pair index 输入；能够输出分数不等于研究结论成立。

## 场景与动力学

- `run_multiscene_10ego_demo_v1.py`：discover → prepare → render-dense → build-state → generate-one → verify → package；保留 10 视角、时间窗口和 W5 checkpoint 哈希约束。
- `train_player_dynamics_v0/v1/v2` 与 `train_player_dynamics_v3_rollout_v0`：独立的玩家状态/动作预测路线，v3 增加多步 rollout 训练。
- `run_dynamics_inference_demo_v0.py`：规则/地图约束动力学或引擎输出 → episode memory → 各相机 dense/state → 生成。内置 self-test 可在 CPU 上运行。
- `run_native_dynamics_10view_10s_v1.py`：10 视角、10 秒场景流程，使用对应的场景目录版本。
- `run_srcds_inference_rollout_v1.py`：需要真实 SRCDS 安装和 bridge plugin。`--project-root` 在此用于定位外部 `engine_smoke_v0/` 安装；其他场景工具的 `--project-root` 用于定位配套 `tools/`，一般指本仓库的 `pipelines/scene_rollout`。不要混淆。
- LingBot Fast v2 的 injection、conditioning、direct-tune、checkpoint 迁移和测试代码保留为实验模块。其官方 causal v2 源码与模型没有上传，不能用 `vendor/lingbot` 自动替代。

场景脚本中显式切换到 Xiaoqi sampler 的调用仍会进入顶层 `tools/`。启动器首先设置当前源码族的 import 路径；源文件自己的显式路径操作按原逻辑保留。

## Baseline

LingBot Cam 使用初始 RGB、prompt 和 camera；SCOPE 使用初始 RGB、prompt 和 10-DoF action。两者保留数据校验与模型输入边界测试，SCOPE 的评估专用字段不会透传给模型。使用对应 baseline 的 manifest schema，不能直接传 Map Memory manifest。

## 全量公开路线

通过 `python multiview.py list` 查看；下面的表由 `project_routes.py` 生成。`run` 使用独立进程，baseline 使用 Python package 模式，保留其相对导入。

| 路线 | 源码入口 | 功能 |
|---|---|---|
| `data.materialize` | [materialize_clips_v1.py](../tools/materialize_clips_v1.py) | Materialize synchronized video clips |
| `data.events` | [materialize_event50h_clips_v1.py](../tools/materialize_event50h_clips_v1.py) | Materialize aligned event clips |
| `data.align` | [build_memory_dense_aligned_cache_v0.py](../tools/build_memory_dense_aligned_cache_v0.py) | Join source, dense and latent caches |
| `data.manifest` | [build_v2_map_manifest_and_cache_v1.py](../tools/build_v2_map_manifest_and_cache_v1.py) | Build training map/cache manifests |
| `data.multiview` | [build_multiview_consistency_manifest_v1.py](../tools/build_multiview_consistency_manifest_v1.py) | Build same-scene multiview groups |
| `data.inspect-multiview` | [multiview_consistency_data_v1.py](../tools/multiview_consistency_data_v1.py) | Validate multiview group data |
| `data.visible-pairs` | [find_mutual_visible_clip_pairs_v0.py](../tools/find_mutual_visible_clip_pairs_v0.py) | Find mutually visible clip pairs |
| `geometry.static` | [build_static_memory_v0.py](../tools/build_static_memory_v0.py) | Build static map memory |
| `geometry.episode` | [build_episode_memory_v0.py](../tools/build_episode_memory_v0.py) | Build dynamic episode memory |
| `geometry.render` | [run_dense_v2_gpu_v1.py](../tools/run_dense_v2_gpu_v1.py) | Render dense v2 on GPU |
| `geometry.mesh` | [build_mesh_dense_condition_v0.py](../tools/build_mesh_dense_condition_v0.py) | Render mesh-based conditions |
| `geometry.compare` | [compare_geometry_backends_v0.py](../tools/compare_geometry_backends_v0.py) | Compare geometry backends |
| `conditions.state` | [build_state_channels_v2.py](../tools/build_state_channels_v2.py) | Build state v2 channels |
| `conditions.interaction` | [build_interaction_sidecar_v0.py](../tools/build_interaction_sidecar_v0.py) | Build interaction sidecars |
| `conditions.actions` | [build_interaction_action_cache_v0.py](../tools/build_interaction_action_cache_v0.py) | Build action cache |
| `train.dense` | [train_memory_dense_adapter_v0.py](../tools/train_memory_dense_adapter_v0.py) | Dense adapter training |
| `train.base` | [train_memory_dense_adapter_base_v0.py](../tools/train_memory_dense_adapter_base_v0.py) | Base-expert dense training |
| `train.state` | [train_memory_dense_adapter_state_v0.py](../tools/train_memory_dense_adapter_state_v0.py) | Dense + state training |
| `train.interaction` | [train_memory_dense_adapter_interaction_v1.py](../tools/train_memory_dense_adapter_interaction_v1.py) | Dense + state + interaction training |
| `infer.dense` | [sample_memory_dense_adapter_phase2a_v0.py](../tools/sample_memory_dense_adapter_phase2a_v0.py) | Dense-conditioned sampling |
| `infer.state` | [sample_memory_dense_adapter_state_v0.py](../tools/sample_memory_dense_adapter_state_v0.py) | Dense + state sampling |
| `infer.interaction` | [sample_memory_dense_adapter_interaction_eval_v1.py](../tools/sample_memory_dense_adapter_interaction_eval_v1.py) | Interaction/control sampling |
| `infer.ar` | [qxq_sample_candidate1_ar_v0.py](../tools/qxq_sample_candidate1_ar_v0.py) | Sequential autoregressive video windows |
| `eval.loss` | [evaluate_memory_dense_adapter_base_v1.py](../tools/evaluate_memory_dense_adapter_base_v1.py) | Validation diffusion loss |
| `eval.subsets` | [build_interaction_eval_subsets_v1.py](../tools/build_interaction_eval_subsets_v1.py) | Construct held-out evaluation subsets |
| `eval.supervision` | [audit_interaction_supervision_v0.py](../tools/audit_interaction_supervision_v0.py) | Audit supervision alignment |
| `eval.symmvc` | [symmvc_v2_score_haware_v1.py](../tools/symmvc_v2_score_haware_v1.py) | Homography-aware multiview consistency |
| `eval.ablation-rows` | [analyze_ablation_eval_rows_v0.py](../tools/analyze_ablation_eval_rows_v0.py) | Summarize ablation rows |
| `baseline.lingbot-train` | [train.py](../baselines/lingbot_cam/train.py) | Camera-only LingBot LoRA |
| `baseline.lingbot-infer` | [infer.py](../baselines/lingbot_cam/infer.py) | Camera-only LingBot inference |
| `baseline.lingbot-validate` | [validate_manifest.py](../baselines/lingbot_cam/validate_manifest.py) | Validate LingBot baseline data |
| `baseline.scope-train` | [train.py](../baselines/scope/train.py) | SCOPE ActionModule fine-tuning |
| `baseline.scope-infer` | [infer.py](../baselines/scope/infer.py) | SCOPE zero-shot / fine-tuned inference |
| `baseline.scope-validate` | [validate_manifest.py](../baselines/scope/validate_manifest.py) | Validate SCOPE baseline data |
| `baseline.scope-actions` | [convert_csgo_to_scope.py](../baselines/scope/actions/convert_csgo_to_scope.py) | Convert CSGO actions to SCOPE format |
| `data.latents` | [cache_tier69h_latents_dlc_v0.py](../pipelines/scene_rollout/tools/cache_tier69h_latents_dlc_v0.py) | Cache LingBot video/text latents |
| `data.covis` | [mine_scene_covis_pairs_v3.py](../pipelines/scene_rollout/tools/mine_scene_covis_pairs_v3.py) | Mine scene co-visibility pairs |
| `data.heldout-pairs` | [mine_strong_covis_pairs_heldout_v1.py](../pipelines/scene_rollout/tools/mine_strong_covis_pairs_heldout_v1.py) | Mine held-out strong co-visibility |
| `dynamics.train-v0` | [train_player_dynamics_v0.py](../pipelines/scene_rollout/tools/train_player_dynamics_v0.py) | Player dynamics v0 |
| `dynamics.train-v1` | [train_player_dynamics_v1.py](../pipelines/scene_rollout/tools/train_player_dynamics_v1.py) | Player dynamics v1 |
| `dynamics.train-v2` | [train_player_dynamics_v2.py](../pipelines/scene_rollout/tools/train_player_dynamics_v2.py) | Player dynamics v2 |
| `dynamics.train-v3` | [train_player_dynamics_v3_rollout_v0.py](../pipelines/scene_rollout/tools/train_player_dynamics_v3_rollout_v0.py) | Scheduled multistep dynamics training |
| `dynamics.fit-motion` | [fit_motion_params_v1.py](../pipelines/scene_rollout/tools/fit_motion_params_v1.py) | Fit map-aware motion parameters |
| `dynamics.evaluate` | [evaluate_player_dynamics_v2_long_horizon_v0.py](../pipelines/scene_rollout/tools/evaluate_player_dynamics_v2_long_horizon_v0.py) | Long-horizon dynamics evaluation |
| `rollout.dynamics` | [run_dynamics_inference_demo_v0.py](../pipelines/scene_rollout/tools/run_dynamics_inference_demo_v0.py) | Map-aware dynamics to multiview generation |
| `rollout.multiscene` | [run_multiscene_10ego_demo_v1.py](../pipelines/scene_rollout/tools/run_multiscene_10ego_demo_v1.py) | Ten-ego two-window scene pipeline |
| `rollout.native` | [run_native_dynamics_10view_10s_v1.py](../pipelines/scene_rollout/tools/run_native_dynamics_10view_10s_v1.py) | Native dynamics ten-view 10-second pipeline |
| `rollout.srcds` | [run_srcds_inference_rollout_v1.py](../pipelines/scene_rollout/tools/run_srcds_inference_rollout_v1.py) | SRCDS engine-backed inference rollout |
| `eval.structure` | [structure_adherence_score_v0.py](../pipelines/scene_rollout/tools/structure_adherence_score_v0.py) | Condition structure adherence |
| `eval.zero-visibility` | [eval_zero_visibility_hallucination_v0.py](../pipelines/scene_rollout/tools/eval_zero_visibility_hallucination_v0.py) | Zero-visibility hallucination evaluation |
| `eval.ar-drift` | [ar_drift_curve_v0.py](../pipelines/scene_rollout/tools/ar_drift_curve_v0.py) | Autoregressive temporal drift |
| `eval.significance` | [paired_significance_v0.py](../pipelines/scene_rollout/tools/paired_significance_v0.py) | Paired significance analysis |
| `infer.history-guidance` | [qxq_sample_candidate1_ar_hg_v0.py](../pipelines/scene_rollout/tools/qxq_sample_candidate1_ar_hg_v0.py) | AR sampling with history guidance |
