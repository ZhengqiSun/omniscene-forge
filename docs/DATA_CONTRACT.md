# 数据契约与保留工具

## 核心关系

```text
地图几何 + 轨迹 ── renderer ── dense release [7,240,416]
既有 latent/cache + 原始 sample IDs ── 对齐 ── aligned cache manifest
aligned cache + 人物轨迹 ── build_state_channels_v2 ── state cache
state cache + action JSON ── build_interaction_sidecar ── state+interaction cache
上述三个 manifest + 基模 + warm-start ── train ── checkpoint
同一数据协议 + checkpoint ── interaction sampler ── 视频与 JSON 报告
```

### 必需外部资产

| 输入 | 内容 |
|---|---|
| map manifest | dense sample IDs、7 通道文件路径、shape、backend、match/split 等元数据 |
| aligned cache manifest | video latent、first-frame condition、text cache、camera、dense sample ID 的帧对齐 |
| state cache manifest | 按 clip_id 引用 NPZ；包含 alive/health/dead-marker 和 interaction sidecar 字段 |
| base checkpoint root | HIGH/LOW noise 模型、T5/tokenizer、Wan2.1 VAE；使用既有 camera 基模资产 |
| adapter checkpoint | adapter、LoRA、dense encoder、state projector，以及 interaction projector；结构与所选版本相容 |
| 推理 clip | image、prompt、poses、intrinsics，或原 sampler 支持的对应缓存路径 |
| 可选 renderer 输入 | BSP faces、地图/轨迹/action JSON；不是从 RGB 自动重建地图的流程 |

interaction 固定 134 维：4 类事件 + 5×26 武器 one-hot；投影到 128 维，再与 dense/state token 相加。事件在相邻 raw-frame 采样点之间聚合，避免降采样漏事件。武器名称遵循 `interaction_channels_v0.py` 的词表映射。

当前 dense 规范为 240×416、视频 480×832、latent 60×104、Wan token 网格 30×52。旧 176×320 数据不能仅通过改文件名混入，需要显式选择对应 dense-hw 与 resize 方式。

state v2 投影为 `far=3000`、`pitch_sign=+1`。`verify_state_projection_contract(strict=True)` 会拒绝明确的旧参数，但没有找到 build report 时返回 `unknown`；验收时不能把 unknown 算作通过。sidecar 及其源 state 的 build report 应保留可追溯关系。

## 工具分工

| 统一命令 | 底层文件 | 输入/作用 |
|---|---|---|
| `train/smoke/preflight/prepare-index` | `train_memory_dense_adapter_interaction_v1.py` | 主训练、真实输入预检与索引 |
| `infer` | `sample_memory_dense_adapter_interaction_eval_v1.py` | 加载三路条件并生成 |
| `prepare-state` | `build_state_channels_v2.py` | 从人物轨迹构建 v2 state |
| `prepare-interaction` | `build_interaction_sidecar_v0.py` | 在 state 上添加交互字段 |
| `prepare-aligned` | `build_memory_dense_aligned_cache_v0.py` | 对齐已有 latent 与 dense；不重新编码所有原始视频 |
| `prepare-manifest` | `build_v2_map_manifest_and_cache_v1.py` | 合成当前 release/cache 视图；`zhengqi-root` 是源数据相对路径的根，必须显式配置 |
| `render-dense` | `run_dense_v2_gpu_v1.py` | 几何渲染；复用 droplog renderer 的实际校准参数 |
| `prepare-eval` | `build_interaction_eval_subsets_v1.py` | 按事件类别固定 val/test 子集 |
| `audit-supervision` | `audit_interaction_supervision_v0.py` | 交互监督数据审计 |
| `evaluate-loss` | `evaluate_memory_dense_adapter_base_v1.py` | loss/条件对照；不等价于视觉一致性评分 |
| `prepare-multiview/inspect-multiview` | 两个 multiview consistency 数据模块 | 多视角样本关系的构建与检查 |

其余 tools 是上述入口的共享组件和动态导入依赖。保留现有 Python 名字便于加载旧 checkpoint 和核对来源；独立 W3–W7 launcher、旧 dynamics/demo/FastV2 实验不在公开入口内。

配置结构为 `{"args": {...}, "distributed": {...}}`。键使用原 CLI 参数去掉 `--` 后的名称，布尔值用 JSON true/false；`append` 参数用列表；`prepare-eval` 的 `pair` 用 `[["name","map.json","cache.jsonl"], ...]`。

配置示例没有隐含服务器路径。使用前填入真实数据，保留 match split、state 投影和 clip/frame 对齐。外部 manifest 中的绝对路径需要原挂载可访问，或先生成独立重映射文件。
