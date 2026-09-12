# SCOPE action-only baseline bridge

S0 为官方权重零样本推理；S1 仅训练 `blocks.*.action_attn.*`。保留原项目的官方模型加载边界、flow-matching、action 转换/校准、checkpoint 和数据校验。官方 SCOPE 源码与模型需要外部提供。

```bash
python multiview.py run baseline.scope-actions -- --help
python multiview.py run baseline.scope-validate -- --help
python multiview.py run baseline.scope-train -- --help
python multiview.py run baseline.scope-infer -- --help
```

输入是初始 RGB、prompt、逐帧 action；camera/dense/state/其他视角保留在 evaluator 区域而不传给模型。CSGO 转换轴向符号必须显式指定，mouse gain 只能在训练集上拟合。详见 [接口协议](SCOPE_INTERFACE_SPEC.md)。

`--scope-repo` 指官方源码、`--model-dir` 指权重，使用官方版本匹配的独立环境。不要把本项目 `vendor/lingbot` 作为 SCOPE runtime。CPU 测试覆盖 schema、action、模型可训练参数边界和 checkpoint；官方模型未在本次机器上实跑。
