# GPR热导率预测模型

## 模型概述

本目录包含使用**高斯过程回归（GPR）**训练的热导率预测模型，用于根据材料的元素组分预测其热导率。

## 训练信息

- **训练日期**: 2025-12-10 18:32:57
- **训练样本数**: 220
- **特征数**: 14个元素 (Ag, As, Bi, Cu, Ge, In, Pb, S, Sb, Se, Sn, Te, Ti, V)
- **目标变量**: 热导率 k (W/Km)
- **数据变换**: log变换（训练时对热导率取对数）

## 模型性能

### 测试集指标
- **R² (log空间)**: 0.8369
- **R² (原始空间)**: 0.7425
- **RMSE**: 0.9134 W/Km
- **MAE**: 0.5748 W/Km

### 核函数
```
1.73² × RBF(length_scale=2.24e+04) × RationalQuadratic(alpha=0.0411, length_scale=0.356) + WhiteKernel(noise_level=0.0828)
```

## 文件说明

| 文件名 | 说明 |
|--------|------|
| `gpr_thermal_conductivity.joblib` | 训练好的GPR模型 |
| `gpr_scaler.joblib` | 特征标准化器 |
| `gpr_model_info.json` | 模型元信息 |
| `gpr_prediction_plot.png` | 预测结果可视化 |
| `train_gpr_model.py` | 训练脚本 |

## 使用方法

### 1. 加载模型

```python
import joblib
import numpy as np

# 加载模型和scaler
model = joblib.load('src/models/gpr_thermal_conductivity.joblib')
scaler = joblib.load('src/models/gpr_scaler.joblib')
```

### 2. 准备输入数据

输入数据应为14维特征向量，表示各元素的含量：

```python
# 元素顺序: Ag, As, Bi, Cu, Ge, In, Pb, S, Sb, Se, Sn, Te, Ti, V
# 示例: Ag2Se (2个Ag原子, 1个Se原子)
X_new = np.array([[2.0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0]])

# 标准化
X_scaled = scaler.transform(X_new)
```

### 3. 预测热导率

```python
# 预测（返回log空间的均值和标准差）
y_pred_log, y_std = model.predict(X_scaled, return_std=True)

# 转换回原始空间
k_pred = np.exp(y_pred_log)

print(f"预测热导率: {k_pred[0]:.4f} W/Km")
print(f"不确定性 (log空间): {y_std[0]:.4f}")

# 计算95%置信区间
k_lower = np.exp(y_pred_log - 1.96 * y_std)
k_upper = np.exp(y_pred_log + 1.96 * y_std)
print(f"95%置信区间: [{k_lower[0]:.4f}, {k_upper[0]:.4f}] W/Km")
```

### 4. 批量预测

```python
# 多个样本
X_batch = np.array([
    [2.0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],  # Ag2Se
    [0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 3.0, 0, 0],  # Bi2Te3
    # ... 更多样本
])

X_scaled = scaler.transform(X_batch)
y_pred_log, y_std = model.predict(X_scaled, return_std=True)
k_pred = np.exp(y_pred_log)

for i, k in enumerate(k_pred):
    print(f"样本 {i+1}: {k:.4f} W/Km")
```

## 重新训练模型

如果需要使用新数据重新训练模型：

```bash
# 激活conda环境
conda activate gnome

# 运行训练脚本
python src/models/train_gpr_model.py
```

## 注意事项

1. **输入特征**: 必须是14维向量，元素顺序固定
2. **数据变换**: 模型在log空间训练，预测结果需要用exp转换回原始空间
3. **不确定性**: GPR模型提供预测的不确定性估计（标准差）
4. **适用范围**: 模型基于220个样本训练，对于训练数据范围外的组分，预测可能不准确

## 相关文件

- 训练数据: `data/original_data.csv`
- 工具函数: `data/algorithms/utils.py`

