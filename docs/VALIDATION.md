# 本次验证结果

日期：2026-09-12。环境：macOS、Python 3.11、PyTorch 2.6 CPU。未使用原始训练日志充当本次执行结果。

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 全部 Python 语法 | 278 个文件通过 | 源码可以解析 |
| 公开 CLI | 53/53 真实 `--help` 通过 | 当前环境能导入入口、创建参数解析器；不证明后续模型/数据可加载 |
| 组件与接口测试 | 58 passed | 条件生产/消费、adapter 梯度、checkpoint、两套 baseline 数据/参数边界、多视角对齐与场景契约 |
| 动力学内置 self-test | 4/4 通过 | 规则/准备阶段的合成输入检查 |
| 导入文件完整性 | 268/268 在位 | 与 source manifest 的选取一致 |
| 源码整理复现 | 268/268 哈希一致 | 原上传选取 + source patch 可以重建导入源码 |
| GPU 模型加载、单步训练、长训练、视频推理 | 未执行 | 本地缺少匹配模型、研究数据与 CUDA 环境 |
| 真实多视角指标与 baseline 比较 | 未执行 | 没有相同样本、checkpoint、seed 的新生成结果 |
| SCOPE / SRCDS / causal Fast v2 外部 runtime | 未实跑 | 外部实现、引擎或权重未上传 |

## 测试覆盖

- Interaction sidecar 的生成、序列化、加载与投影；事件错误处理、split 内 shuffle。
- 非零 adapter 的前向和反向；开启/关闭 activation checkpoint 的输出、梯度一致性。
- 新配对数据的 21 个 world ticks / frame counts 匹配约束，相机采样位置与非有限数值检查；canonical v1 对原 heldout 的保留。
- Camera-only LingBot 的 LoRA、checkpoint、media 和 schema。
- SCOPE 的 action 转换、模型输入边界、ActionModule 梯度边界与 checkpoint。
- 十视角窗口在首次死亡前的选择、帧索引、原子写入与 worker task 解析。
- 路线启动的代码目录隔离、baseline package 模式、dry-run；interaction 配置解析、显式 checkpoint、单步与多机参数约束。
- Vendored Wan 可在 CPU 上导入，实际 `WanModel` 保留项目所需条件接口。

原项目中依赖官方 Fast v2 runtime、固定实验资产或 GPU 的测试仍随对应代码保存，不在默认 CPU 测试集中。它们没有被标记成已通过。

## 已修复问题与保留边界

T5 原默认参数在导入时执行 `torch.cuda.current_device()`，CPU 环境会失败；现改为构造时求值。服务器代码与常见资产路径已迁移到本地源码/可配置资产根目录，剩余 mount 字符串是旧数据前缀兼容逻辑，已列明供资产迁移核对。

配对 loader 尚未接入当前 trainer；checkpoint/schema、旧 state 协议和不同 split 规则也没有强行合并。缺少日志不影响代码整理，但当前证据不足以认证历史 8500 步完成、视频数量或研究质量结论。

复查命令：

```bash
python scripts/check_codebase.py --help-smoke
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python multiview.py run rollout.dynamics -- self-test
```

入口检查的机器可读结果为 `provenance/validation.json`。依赖配置是本次可用的开发配置；Linux CUDA 配置沿用上传环境信息，尚未重新安装验证。
