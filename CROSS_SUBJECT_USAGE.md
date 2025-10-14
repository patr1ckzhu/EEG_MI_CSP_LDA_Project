# Cross-Subject EEGNet 使用指南

## 🎯 什么是Cross-Subject模型？

**Individual模型**: 为每个人单独训练，需要该人的45个trials
**Cross-Subject模型**: 用9个人的数据(405 trials)训练，测试第10个人

## 📊 算法: Leave-One-Subject-Out (LOSO)

```
10轮训练:

轮1: 用S2-S10训练 (9×45=405 trials) → 测试S1
轮2: 用S1,S3-S10训练 (405 trials) → 测试S2
...
轮10: 用S1-S9训练 (405 trials) → 测试S10

最终准确率 = 10次测试的平均值
```

---

## 🚀 快速开始 (Windows PC)

### 1. 克隆最新代码

```bash
git clone https://github.com/patr1ckzhu/EEG_MI_CSP_LDA_Project.git
cd EEG_MI_CSP_LDA_Project
```

### 2. 安装依赖 (如果还没装)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn matplotlib mne
```

### 3. 运行Cross-Subject训练 (一条命令!)

```bash
python src/train_eegnet_cross_subject.py
```

**就这么简单！**

---

## ⏱️ 时间估算

**RTX 5080:**
- 每轮训练: ~10-15分钟
- 总共10轮: **~2-3小时**

**M1 Max:**
- 每轮训练: ~15-20分钟
- 总共10轮: **~3-4小时**

---

## 📁 输出结果

训练完成后会生成:

### 1. JSON结果文件

```bash
outputs_eegnet_cross_subject/loso_results.json
```

包含:
- 平均准确率
- 每个受试者的测试准确率
- 训练配置

### 2. Markdown报告

```bash
outputs_eegnet_cross_subject/LOSO_REPORT.md
```

包含:
- 总结统计
- 每个受试者的结果表格
- 与CSP+LDA的对比

### 3. 查看结果

**Windows:**
```bash
type outputs_eegnet_cross_subject\LOSO_REPORT.md
```

**Mac/Linux:**
```bash
cat outputs_eegnet_cross_subject/LOSO_REPORT.md
```

---

## 🎛️ 高级选项

### 调整训练参数

```bash
# 增加训练轮数
python src/train_eegnet_cross_subject.py --epochs 200

# 降低学习率
python src/train_eegnet_cross_subject.py --lr 0.0001

# 使用大模型
python src/train_eegnet_cross_subject.py --large_model

# 调整batch size
python src/train_eegnet_cross_subject.py --batch_size 64
```

### 只测试部分受试者 (快速验证)

```bash
# 只用前5个人
python src/train_eegnet_cross_subject.py --subjects 1 2 3 4 5
```

### 使用更多受试者 (下载后)

```bash
# 先下载更多数据
python download_all_subjects.py --subjects 1-50

# 然后训练50人LOSO (需要很长时间!)
python src/train_eegnet_cross_subject.py --subjects $(seq 1 50)  # Mac/Linux
```

---

## 📊 预期结果

### Baseline (你已有的结果)

| 方法 | 模式 | 准确率 |
|------|------|--------|
| CSP+LDA | Within-Subject | 62.00% ± 18.09% |
| CSP+LDA | Cross-Subject | 51.85% ± 10.48% |

### 新的Cross-Subject EEGNet

```
预期: 60-70% (希望能超过CSP+LDA的52%)
```

---

## ✅ 成功标准

**如果 EEGNet > 52%:**
- ✅ 证明深度学习在跨受试者泛化上更好
- ✅ 可以写论文说明深度学习的优势

**如果 EEGNet ≈ 52%:**
- ✅ 两种方法相当，但EEGNet需要GPU
- ✅ 仍然是有价值的对比研究

**如果 EEGNet < 52%:**
- ✅ 证明CSP+LDA更稳定
- ✅ 说明传统方法在这个场景下更优

**无论结果如何，都是好的研究发现！**

---

## 🔍 监控训练进度

训练时会实时显示:

```
==================================================
FOLD: Testing on Subject 01
==================================================
Train subjects: [2, 3, 4, 5, 6, 7, 8, 9, 10]
Test subject: 1

Loading training data...
  Subject 02: 45 trials
  Subject 03: 45 trials
  ...
Training data: (405, 8, 321)

Training...
Epoch  20: Train Loss=0.6234 Acc=0.6543
Epoch  40: Train Loss=0.5123 Acc=0.7234
...

Testing...
Subject 01 Test Accuracy: 65.43%
```

---

## ⚠️ 常见问题

### Q: 训练太慢怎么办？

```bash
# 减少epochs
python src/train_eegnet_cross_subject.py --epochs 100

# 增大batch size (如果显存够)
python src/train_eegnet_cross_subject.py --batch_size 64
```

### Q: 显存不足？

```bash
# 减小batch size
python src/train_eegnet_cross_subject.py --batch_size 16
```

### Q: 想先快速测试？

```bash
# 只用3个人测试
python src/train_eegnet_cross_subject.py --subjects 1 2 3 --epochs 50
```

---

## 📝 完整命令参考

```bash
python src/train_eegnet_cross_subject.py \
    --subjects 1 2 3 4 5 6 7 8 9 10 \
    --datadir data/raw \
    --config 8-channel-motor \
    --output outputs_eegnet_cross_subject \
    --epochs 150 \
    --batch_size 32 \
    --lr 0.001 \
    --band 8 30 \
    --tmin 1.0 \
    --tmax 3.0
```

---

## 🎯 推荐工作流

### Step 1: 快速测试 (5分钟)

```bash
# 只用2个人测试，确保代码能跑
python src/train_eegnet_cross_subject.py --subjects 1 2 --epochs 10
```

### Step 2: 小规模验证 (30分钟)

```bash
# 用5个人验证效果
python src/train_eegnet_cross_subject.py --subjects 1 2 3 4 5 --epochs 100
```

### Step 3: 完整训练 (2-3小时)

```bash
# 全部10个人，完整训练
python src/train_eegnet_cross_subject.py
```

### Step 4: 查看和分析结果

```bash
cat outputs_eegnet_cross_subject/LOSO_REPORT.md
```

---

## 🚀 现在就开始！

在你的Windows PC上:

```bash
cd EEG_MI_CSP_LDA_Project
python src/train_eegnet_cross_subject.py
```

然后去喝杯咖啡，2-3小时后回来看结果！☕

Good luck! 🍀
