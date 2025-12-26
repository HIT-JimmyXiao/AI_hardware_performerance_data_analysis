# 99_appendix 目录说明

## 📁 目录用途

`99_appendix/` 目录用于存放**额外的探索性图表**，这些图表不属于主要分析流程，但可能对深入理解数据有帮助。

## 🎯 使用场景

### 1. **探索性分析图**
- 临时生成的图表，用于验证假设
- 尝试不同的可视化方法
- 测试新的分析角度

### 2. **补充分析图**
- 不在主要研究问题（RQ1-RQ5）范围内的图表
- 但可能对理解数据有帮助
- 例如：按制程工艺分组、按代工厂分组等

### 3. **草稿/实验图**
- 在最终确定图表样式前的实验版本
- 用于对比不同可视化方案
- 保留作为参考

### 4. **详细分析图**
- 某些分析的详细版本（如按厂商的详细趋势）
- 可能因为信息量太大而不适合放在主报告中
- 但可以作为补充材料

## 📝 命名建议

建议文件名包含：
- 分析主题（如 `manufacturer_trend_detailed.png`）
- 日期或版本（如 `exploration_20251226.png`）
- 用途说明（如 `draft_xxx.png`）

## ⚠️ 注意事项

- 本目录的图表**不会自动生成**，需要手动添加
- 主分析脚本（`hw_analysis_full.py`）不会向此目录输出
- 如需在此目录生成图表，请手动调用绘图函数并指定 `subdir='99_appendix'`

## 🔧 如何使用

### 方法一：在主脚本中添加

```python
# 在 hw_analysis_full.py 中添加
plot_scatter_trend(
    df,
    x_col='release_year',
    y_col='int8_tops',
    title="探索性分析：按制程分组",
    filename="exploration_by_process.png",
    subdir='99_appendix'  # 指定输出到 appendix
)
```

### 方法二：在 Notebook 中手动生成

```python
# 在 notebook 中
fig, ax = plt.subplots(figsize=(10, 6))
# ... 绘图代码 ...
save_fig(fig, "my_exploration.png", subdir='99_appendix')
```

---

**总结**：`99_appendix/` 是一个"自由发挥"的目录，用于存放不在主要分析流程中，但对理解数据有帮助的额外图表。

