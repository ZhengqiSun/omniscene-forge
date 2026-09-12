# LingBot camera-only baseline

保留上传中的 camera-only LoRA 训练、推理、checkpoint 配对、manifest 校验以及测试。底层模型使用本仓库的 `vendor/lingbot`；base 模型和 LoRA 权重外置。

```bash
python multiview.py run baseline.lingbot-validate -- --help
python multiview.py run baseline.lingbot-train -- --help
python multiview.py run baseline.lingbot-infer -- --help
```

以上命令从仓库根目录执行。也可使用 `python -m baselines.lingbot_cam.train ...`。源码 `schema.py` 定义 `lingbot_sample_v1`：初始图或原视频二选一，显式 pose/intrinsics、时间/尺寸、prompt 或 text cache，训练另需 target。Dense、state、world events 等额外条件被拒绝。

测试覆盖 LoRA 梯度边界、checkpoint 恢复与基座身份、media 编码接口和 schema。未在本次机器上完成真实 GPU 训练/生成。
