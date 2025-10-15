# ResNet-1D for EEG Motor Imagery Classification

深度残差网络用于脑电运动想象分类的实现。

---

## 📁 新增文件

### 1. 模型架构
**`src/models/resnet1d.py`**
- `ResNet1D`: 标准ResNet-1D模型（~4.5M参数）
- `ResNet1DLarge`: 更大的ResNet-1D模型（~10M参数）
- 包含测试代码，可独立运行验证模型

### 2. 训练脚本
**`src/train_resnet_cross_subject.py`**
- Leave-One-Subject-Out (LOSO) 交叉验证
- 混合精度训练（Mixed Precision）
- GPU优化（大batch size，pin_memory等）
- 自动保存结果和报告

### 3. 便捷运行脚本
**`run_resnet_experiment.py`**
- 预设快速测试和完整实验配置
- 实时输出训练进度
- 自动生成输出目录

---

## 🚀 快速开始

### 测试模型架构

```bash
# 激活虚拟环境
source .venv_eegnet/bin/activate

# 测试ResNet-1D模型
python src/models/resnet1d.py
```

**输出示例：**
```
Testing ResNet1D Model
======================================================================

Configuration: 8-ch, 321 timepoints
----------------------------------------------------------------------
Parameters: 4,514,818
✓ Gradient flow check passed
```

---

## 🔬 运行实验

### 方法1：使用便捷脚本（推荐）

```bash
# 快速测试（10个受试者）
python run_resnet_experiment.py --quick

# 完整实验（100个受试者）- 在Windows RTX 5080上运行
python run_resnet_experiment.py --full

# 使用更大的模型
python run_resnet_experiment.py --quick --large_model

# 自定义受试者
python run_resnet_experiment.py --subjects 1 2 3 4 5 --epochs 100 --batch_size 128
```

### 方法2：直接调用训练脚本

```bash
# 基本用法
python src/train_resnet_cross_subject.py \
    --subjects 1 2 3 4 5 6 7 8 9 10 \
    --epochs 150 \
    --batch_size 128 \
    --output outputs_resnet

# 完整参数示例
python src/train_resnet_cross_subject.py \
    --subjects 1 2 3 4 5 6 7 8 9 10 \
    --epochs 150 \
    --batch_size 128 \
    --lr 0.001 \
    --config 8-channel-motor \
    --output outputs_resnet_10subjects \
    --seed 42
```

---

## ⚙️ 参数说明

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--subjects` | 1-10 | 受试者ID列表 |
| `--epochs` | 150 | 训练轮数 |
| `--batch_size` | 128 | 批大小（RTX 5080推荐128） |
| `--lr` | 0.001 | 学习率 |
| `--config` | 8-channel-motor | 通道配置 |
| `--large_model` | False | 是否使用更大的模型 |
| `--output` | outputs_resnet_cross_subject | 输出目录 |

### GPU优化参数（自动启用）

- **混合精度训练**: `torch.cuda.amp.autocast()`
- **DataLoader优化**: `num_workers=4`, `pin_memory=True`
- **大batch size**: 默认128（vs EEGNet的32）

---

## 📊 模型架构详情

### ResNet1D（标准版）

```
输入: (batch, 8 channels, 321 timepoints)

Stem:
  Conv1D(8 → 64, kernel=25, stride=2) + BN + ReLU
  输出: (batch, 64, 160)

Layer 1 (ResBlock):
  Conv1D(64 → 64, kernel=15) + BN + ReLU + Dropout
  Conv1D(64 → 64, kernel=15) + BN
  + Skip Connection
  输出: (batch, 64, 160)

Layer 2 (ResBlock with downsample):
  Conv1D(64 → 128, kernel=15, stride=2) + BN + ReLU + Dropout
  Conv1D(128 → 128, kernel=15) + BN
  + Skip Connection (1x1 conv)
  输出: (batch, 128, 80)

Layer 3 (ResBlock):
  Conv1D(128 → 128, kernel=15) + BN + ReLU + Dropout
  Conv1D(128 → 128, kernel=15) + BN
  + Skip Connection
  输出: (batch, 128, 80)

Layer 4 (ResBlock with downsample):
  Conv1D(128 → 256, kernel=15, stride=2) + BN + ReLU + Dropout
  Conv1D(256 → 256, kernel=15) + BN
  + Skip Connection (1x1 conv)
  输出: (batch, 256, 40)

Layer 5 (ResBlock):
  Conv1D(256 → 256, kernel=15) + BN + ReLU + Dropout
  Conv1D(256 → 256, kernel=15) + BN
  + Skip Connection
  输出: (batch, 256, 40)

Head:
  Global Average Pooling → (batch, 256)
  Dropout(0.5)
  FC(256 → 128) + ReLU
  Dropout(0.3)
  FC(128 → 2)

总参数: ~4.5M
```

---

## 📈 预期结果

### 性能目标

| 指标 | EEGNet (baseline) | ResNet-1D (目标) |
|------|-------------------|------------------|
| 交叉受试者准确率 | 60.62% ± 11.28% | **65-70%** |
| 模型参数量 | 5K | 1.2-4.5M |
| 训练速度 | 慢（GPU利用率低） | **3-5x 更快** |
| GPU利用率 | 低 | **高** |

### 输出文件

训练完成后会生成：

```
outputs_resnet_cross_subject/
├── loso_results.json          # 详细结果（JSON格式）
└── LOSO_REPORT.md             # 结果报告（Markdown）
```

**`LOSO_REPORT.md` 示例内容：**
```markdown
# Cross-Subject ResNet-1D Results (LOSO)

## Summary
- **Model**: ResNet1D
- **Mean Accuracy**: 67.50% ± 12.34%
- **Subjects**: 10
- **Batch size**: 128

## Per-Subject Results
| Subject | Accuracy | Status |
|---------|----------|--------|
| 01 | 72.50% | ✅ |
| 02 | 65.30% | ⚠️ |
...

## Comparison with Baselines
| Model | Mean Accuracy | Std |
|-------|---------------|-----|
| CSP+LDA (cross-subject) | 51.85% | ± 10.48% |
| EEGNet (cross-subject) | 60.62% | ± 11.28% |
| **ResNet-1D (this)** | **67.50%** | **± 12.34%** |

**Improvement over EEGNet**: +6.88%
```

---

## 🔍 技术细节

### GPU优化策略

1. **混合精度训练（Mixed Precision）**
   - 使用 `torch.cuda.amp.autocast()`
   - 降低显存占用，提升训练速度
   - 在RTX 5080上效果显著

2. **大Batch Size**
   - 默认128（vs EEGNet的32）
   - 充分利用GPU并行计算能力
   - 如遇显存不足可降至64

3. **DataLoader优化**
   - `num_workers=4`: 多进程数据加载
   - `pin_memory=True`: 加速CPU→GPU数据传输

4. **学习率调度**
   - `ReduceLROnPlateau`: 当验证准确率停止提升时降低学习率
   - `patience=10`: 10个epoch后降低学习率

### 正则化策略

1. **Batch Normalization**
   - 每个卷积层后添加BN
   - 加速收敛，提升泛化能力

2. **Dropout**
   - 残差块中: 0.3
   - 分类头中: 0.5 和 0.3
   - 防止过拟合

3. **Early Stopping**
   - `patience=30`: 30个epoch内验证准确率无提升则停止
   - 节省训练时间

---

## 🐛 故障排除

### 问题1: CUDA Out of Memory

**解决方案：**
```bash
# 降低batch size
python run_resnet_experiment.py --quick --batch_size 64

# 或使用标准模型而非大模型
python run_resnet_experiment.py --quick
```

### 问题2: 训练速度慢

**检查项：**
1. 确认使用GPU而非CPU
   ```python
   import torch
   print(torch.cuda.is_available())  # 应为True
   ```

2. 确认batch_size足够大
   - RTX 5080: 建议128
   - RTX 3080: 建议64-128

3. 确认混合精度已启用
   - 脚本中 `use_amp=True` 默认已开启

### 问题3: 准确率低于预期

**可能原因：**
1. 训练epoch不足 → 增加 `--epochs 200`
2. 学习率不合适 → 尝试 `--lr 0.0005` 或 `--lr 0.002`
3. 数据量不足 → 使用更多受试者训练

---

## 📝 与EEGNet对比

| 特性 | EEGNet | ResNet-1D |
|------|--------|-----------|
| **参数量** | 5K | 4.5M |
| **深度** | 3层 | 5个残差块 |
| **Skip connections** | ❌ | ✅ |
| **准确率（10受试者LOSO）** | ~60% | **预期65-70%** |
| **GPU利用率** | 低 | **高** |
| **训练速度（batch=128）** | 慢 | **快** |
| **过拟合风险** | 低（模型小） | 中（需正则化） |

---

## 🎯 下一步建议

### 1. 快速验证（本地Mac）
```bash
# 10个受试者快速测试
python run_resnet_experiment.py --quick
```

### 2. 完整实验（Windows RTX 5080）
```bash
# 100个受试者完整评估
python run_resnet_experiment.py --full --batch_size 128
```

### 3. 超参数调优（可选）
```bash
# 尝试不同学习率
python run_resnet_experiment.py --subjects 1 2 3 4 5 --lr 0.0005
python run_resnet_experiment.py --subjects 1 2 3 4 5 --lr 0.002

# 尝试更大模型
python run_resnet_experiment.py --subjects 1 2 3 4 5 --large_model
```

### 4. 数据增强（进阶）
可在 `src/train_resnet_cross_subject.py` 中添加：
- 时间翻转
- 高斯噪声
- 时间扭曲

---

## 📚 参考文献

**ResNet原论文：**
> He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.

**EEG深度学习综述：**
> Craik, A., He, Y., & Contreras-Vidal, J. L. (2019). Deep learning for electroencephalogram (EEG) classification tasks: a review. Journal of Neural Engineering.

**PhysioNet EEGBCI数据集：**
> Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. IEEE TBME.

---

## ✅ 成功标准

1. ✅ 模型成功训练（无错误）
2. ✅ GPU利用率高（RTX 5080应感觉发热）
3. ✅ 训练时间 < 3小时（100个受试者）
4. ✅ 平均准确率 > 65%（超过EEGNet的60.62%）
5. ✅ 生成完整报告文件

---

## 🙋 常见问题

**Q: 为什么参数量是4.5M而不是文档中的1.2M？**

A: 当前实现优先保证性能，使用了更多的卷积核。如需减少参数，可以：
- 减少每层的通道数（64→32，128→64，256→128）
- 减少残差块数量（5→3）

**Q: 能否在CPU上训练？**

A: 可以，但会很慢。建议：
- 减少受试者数量（--subjects 1 2 3）
- 减少epoch（--epochs 50）
- 减少batch_size（--batch_size 16）

**Q: 如何保存和加载训练好的模型？**

A: 可以在 `train_resnet_cross_subject.py` 中添加：
```python
# 保存模型
torch.save(model.state_dict(), 'resnet1d_best.pth')

# 加载模型
model.load_state_dict(torch.load('resnet1d_best.pth'))
```

---

**Good luck with your experiments! 🚀**