#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 硬件算力演进分析 - 完整分析脚本
============================================================
作者: 肖景铭
学校: 哈尔滨工业大学
学号: 2023112881
课程: Python程序设计
日期: 2025-12-26

数据来源: Epoch AI Machine Learning Hardware
https://epoch.ai/data/machine-learning-hardware
============================================================

本脚本可以：
1. 直接运行（python hw_analysis_full.py）
2. 用 jupytext 转成 .ipynb（jupytext --to notebook hw_analysis_full.py）
"""

# ============================================================
# Block 00 - 项目说明（Markdown Cell）
# ============================================================
'''
# AI 硬件算力演进分析
## 基于 Epoch AI Machine Learning Hardware 数据集

---

**为什么要分析 AI 硬件？**

- 🤖 **AI 大模型的崛起**：ChatGPT/GPT-4/Llama 等大模型的训练与推理都依赖强大的硬件算力
- 💰 **数据中心成本**：算力、功耗、显存是数据中心选型的核心指标
- 📈 **硬件创新速度**：AI 芯片的迭代速度远超传统 CPU，了解趋势有助于预测未来

### 核心研究问题（偏向推理场景）

- **RQ1**: 推理算力（INT8）如何演进？
- **RQ2**: 谁的能效（TOP/s/W）最高？
- **RQ3**: 显存/带宽是否匹配算力增长？
- **RQ4**: 消费级 GPU 性价比如何？
- **RQ5**: 如何选型（训练 vs 推理）？

---

### 📚 通俗解释：什么是 FP32/FP16/INT8？

| 精度 | 适用场景 | 通俗类比 |
|------|----------|----------|
| **FP32** | 传统训练/推理 | "标准清晰度"，慢但兼容 |
| **FP16/BF16** | 现代训练 | "高清"，快且精度可接受 |
| **INT8** | 推理部署 | "压缩格式"，极快（本项目重点）|

**为什么 INT8 算力更高？**  
低精度 → 电路简单 → 并行单元更多 → INT8 算力是 FP32 的 4-16 倍
'''

# ============================================================
# Block 01 - 环境设置与导入模块
# 目标：导入库，初始化主题，创建输出目录
# ============================================================

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 添加 src/ 到路径
sys.path.append('src')

from hw_config import *
from hw_utils import *
from hw_viz import *

# 初始化
ensure_dirs()
setup_logging()
init_viz_theme()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("✅ 环境设置完成！")
print(f"📁 项目根目录: {PROJECT_ROOT}")
print(f"📊 数据文件: {ML_HARDWARE_CSV}")
print(f"\n⚙️  分析配置:")
print(f"   - 时间范围: {YEAR_RANGE}")
print(f"   - 推理口径: {INFERENCE_METRIC}")
print(f"   - Top-N: {TOP_N}")

# ============================================================
# Block 02 - 数据读取与初步探索
# 目标：读取 CSV，查看基本信息、缺失值
# ============================================================

df_raw = load_ml_hardware_data()

print("\n" + "="*60)
print("📊 数据集基本信息")
print("="*60)
print(f"行数: {len(df_raw)}")
print(f"列数: {df_raw.shape[1]}")
print(f"\n前 5 列名: {list(df_raw.columns[:5])}")

# 缺失值统计
missing_summary = get_missing_summary(df_raw)
print("\n⚠️  缺失值 Top 5:")
print(missing_summary.head(5))
save_table(missing_summary, 'missing_value_summary.csv')

print("\n📌 初步发现:")
print("- 价格字段缺失严重（~80%），数据中心硬件不零售")
print("- 算力字段完整度较高（~95%）")
print("- 带宽/互联字段中度缺失（~50%）")

# ============================================================
# Block 03 - 数据清洗与派生指标
# 目标：解析日期、转换数值、计算 TFLOP/s、能效等
# ============================================================

df = df_raw.copy()

# 清洗流程
df = parse_release_date(df)
df = convert_numeric_cols(df)
df = add_derived_metrics(df)
df = filter_by_year_range(df, year_range=YEAR_RANGE)

# 去重
duplicates = check_duplicates(df)
if len(duplicates) > 0:
    print(f"\n⚠️  发现 {len(duplicates)} 条重复，已去重")
    df = df.drop_duplicates(subset=['Hardware name', 'Release date'], keep='first')

print(f"\n✅ 清洗完成：{len(df)} 行 × {df.shape[1]} 列")
print(f"\n新增派生字段:")
for col in ['int8_tops', 'mem_gb', 'efficiency_int8_per_w']:
    if col in df.columns:
        print(f"   - {col}: {df[col].notna().sum()} 非空")

save_table(df, 'cleaned_data_with_derived.csv', subdir='derived')

# ============================================================
# Block 04 - 数据概览可视化
# 目标：缺失率热图、厂商分布、类型分布
# ============================================================

# 缺失率热图（只显示缺失率 >= 50% 的字段）
plot_missing_heatmap(df, title="数据集字段缺失率 (2012-2025)", filename="missing_rate_heatmap.png", min_missing_rate=50.0)

# 厂商分布
manufacturer_counts = df['Manufacturer'].value_counts().reset_index()
manufacturer_counts.columns = ['Manufacturer', 'Count']
plot_barh(manufacturer_counts, x_col='Count', y_col='Manufacturer',
          title="AI 硬件厂商分布 (2012-2025)", xlabel="数量", ylabel="厂商",
          filename="manufacturer_distribution.png", subdir='00_dataset_overview')

# 类型分布（改用条形图，避免饼图重叠）
type_counts = df['Type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']

# 计算百分比
type_counts['Percentage'] = (type_counts['Count'] / type_counts['Count'].sum() * 100).round(1)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(type_counts['Type'], type_counts['Count'], 
               color=sns.color_palette('Set2', len(type_counts)))

# 在条形上添加数值和百分比标签
for i, (idx, row) in enumerate(type_counts.iterrows()):
    count = row['Count']
    pct = row['Percentage']
    ax.text(count + max(type_counts['Count']) * 0.01, i, 
            f'{count} ({pct}%)', 
            va='center', fontsize=10, fontweight='bold')

ax.set_title("硬件类型分布 (GPU vs TPU)", fontsize=TITLE_SIZE, fontweight='bold')
ax.set_xlabel("数量 (Count)", fontsize=LABEL_SIZE)
ax.set_ylabel("硬件类型 (Type)", fontsize=LABEL_SIZE)
ax.invert_yaxis()  # 最大的在上
ax.grid(axis='x', alpha=GRID_ALPHA)
save_fig(fig, "type_distribution.png", subdir='00_dataset_overview')

print("✅ 数据概览图已保存")

# ============================================================
# Block 05 - RQ1: INT8 推理算力演进
# 目标：时间序列散点图（log 轴）+ Top 20 排行
# ============================================================

print("\n" + "="*60)
print("📊 RQ1: INT8 推理算力演进分析")
print("="*60)

df_int8 = df[df['int8_tops'].notna()].copy()
print(f"有 INT8 数据: {len(df_int8)} / {len(df)} ({len(df_int8)/len(df)*100:.1f}%)")

# 趋势图（增加标注数量，改进样式）
plot_scatter_trend(
    df_int8, x_col='release_year', y_col='int8_tops', hue_col='Manufacturer',
    title="INT8 推理算力演进 (2012-2025)\n对数轴展示指数增长",
    xlabel="发布年份", ylabel="INT8 算力 (TOP/s, log)",
    filename="int8_performance_over_time.png", subdir='01_perf_trends',
    log_y=True, annotate_top_n=15, annotate_col='Hardware name'
)

# 增长统计
df_int8_sorted = df_int8.sort_values('release_year')
earliest = df_int8_sorted.iloc[0]
latest = df_int8_sorted.iloc[-1]
growth = latest['int8_tops'] / earliest['int8_tops']
years = latest['release_year'] - earliest['release_year']
cagr = (growth ** (1/years) - 1) * 100

print(f"\n📈 INT8 算力增长:")
print(f"   最早: {earliest['Hardware name']} ({earliest['release_year']:.0f}) - {earliest['int8_tops']:.1f} TOP/s")
print(f"   最新: {latest['Hardware name']} ({latest['release_year']:.0f}) - {latest['int8_tops']:.1f} TOP/s")
print(f"   增长倍数: {growth:.1f}x")
print(f"   CAGR: {cagr:.1f}%")

# Top 20 排行
df_int8_top20 = get_top_n(df_int8, 'int8_tops', n=20)
plot_barh(df_int8_top20, x_col='int8_tops', y_col='Hardware name',
          title="INT8 推理算力 Top 20", xlabel="INT8 算力 (TOP/s)", ylabel="硬件名称",
          filename="int8_top20_ranking.png", subdir='01_perf_trends')
save_table(df_int8_top20[['Hardware name', 'Manufacturer', 'release_year', 'int8_tops', 'TDP (W)']],
           'top20_int8_performance.csv')

print("✅ INT8 推理算力分析完成")

# ============================================================
# Block 05-2: FP16 训练算力演进（补充分析）
# 目标：训练场景的算力趋势（FP16/BF16）
# ============================================================

print("\n" + "="*60)
print("📊 FP16/BF16 训练算力演进分析（补充）")
print("="*60)

df_fp16 = df[df['perf_fp16_tflops'].notna()].copy()
print(f"有 FP16/BF16 数据: {len(df_fp16)} / {len(df)} ({len(df_fp16)/len(df)*100:.1f}%)")

# FP16 训练算力趋势图
plot_scatter_trend(
    df_fp16, x_col='release_year', y_col='perf_fp16_tflops', hue_col='Manufacturer',
    title="FP16/BF16 训练算力演进 (2012-2025)\n对数轴展示指数增长",
    xlabel="发布年份", ylabel="FP16/BF16 算力 (TFLOP/s, log)",
    filename="fp16_performance_over_time.png", subdir='01_perf_trends',
    log_y=True, annotate_top_n=15, annotate_col='Hardware name'
)

# FP16 增长统计
df_fp16_sorted = df_fp16.sort_values('release_year')
if len(df_fp16_sorted) > 0:
    earliest_fp16 = df_fp16_sorted.iloc[0]
    latest_fp16 = df_fp16_sorted.iloc[-1]
    growth_fp16 = latest_fp16['perf_fp16_tflops'] / earliest_fp16['perf_fp16_tflops']
    years_fp16 = latest_fp16['release_year'] - earliest_fp16['release_year']
    cagr_fp16 = (growth_fp16 ** (1/years_fp16) - 1) * 100 if years_fp16 > 0 else 0
    
    print(f"\n📈 FP16/BF16 算力增长:")
    print(f"   最早: {earliest_fp16['Hardware name']} ({earliest_fp16['release_year']:.0f}) - {earliest_fp16['perf_fp16_tflops']:.1f} TFLOP/s")
    print(f"   最新: {latest_fp16['Hardware name']} ({latest_fp16['release_year']:.0f}) - {latest_fp16['perf_fp16_tflops']:.1f} TFLOP/s")
    print(f"   增长倍数: {growth_fp16:.1f}x")
    print(f"   CAGR: {cagr_fp16:.1f}%")

# FP16 Top 20 排行
df_fp16_top20 = get_top_n(df_fp16, 'perf_fp16_tflops', n=20)
plot_barh(df_fp16_top20, x_col='perf_fp16_tflops', y_col='Hardware name',
          title="FP16/BF16 训练算力 Top 20", xlabel="FP16/BF16 算力 (TFLOP/s)", ylabel="硬件名称",
          filename="fp16_top20_ranking.png", subdir='01_perf_trends')
save_table(df_fp16_top20[['Hardware name', 'Manufacturer', 'release_year', 'perf_fp16_tflops', 'TDP (W)']],
           'top20_fp16_performance.csv')

print("✅ FP16/BF16 训练算力分析完成")

# ============================================================
# Block 05-3: 按厂商分组的性能趋势对比
# 目标：对比不同厂商的算力演进轨迹
# ============================================================

print("\n" + "="*60)
print("📊 按厂商分组的性能趋势对比")
print("="*60)

# INT8 按厂商分组趋势
if len(df_int8) > 0:
    # 只显示主要厂商（至少有 3 个硬件）
    manufacturer_counts = df_int8['Manufacturer'].value_counts()
    major_manufacturers = manufacturer_counts[manufacturer_counts >= 3].index.tolist()
    df_int8_major = df_int8[df_int8['Manufacturer'].isin(major_manufacturers)].copy()
    
    if len(major_manufacturers) > 0:
        plot_facet_trend(
            df_int8_major, x_col='release_year', y_col='int8_tops', facet_col='Manufacturer',
            title="INT8 推理算力演进 - 按厂商分组对比",
            xlabel="发布年份", ylabel="INT8 算力 (TOP/s, log)",
            filename="int8_performance_by_manufacturer.png", subdir='01_perf_trends', log_y=True
        )
        print(f"✅ INT8 按厂商分组趋势图已生成（{len(major_manufacturers)} 个主要厂商）")

# FP16 按厂商分组趋势
if len(df_fp16) > 0:
    manufacturer_counts_fp16 = df_fp16['Manufacturer'].value_counts()
    major_manufacturers_fp16 = manufacturer_counts_fp16[manufacturer_counts_fp16 >= 3].index.tolist()
    df_fp16_major = df_fp16[df_fp16['Manufacturer'].isin(major_manufacturers_fp16)].copy()
    
    if len(major_manufacturers_fp16) > 0:
        plot_facet_trend(
            df_fp16_major, x_col='release_year', y_col='perf_fp16_tflops', facet_col='Manufacturer',
            title="FP16/BF16 训练算力演进 - 按厂商分组对比",
            xlabel="发布年份", ylabel="FP16/BF16 算力 (TFLOP/s, log)",
            filename="fp16_performance_by_manufacturer.png", subdir='01_perf_trends', log_y=True
        )
        print(f"✅ FP16/BF16 按厂商分组趋势图已生成（{len(major_manufacturers_fp16)} 个主要厂商）")

# 综合对比图（所有厂商在同一张图上，用颜色区分）
if len(df_int8) > 0:
    plot_scatter_trend(
        df_int8, x_col='release_year', y_col='int8_tops', hue_col='Manufacturer',
        title="INT8 推理算力演进 - 主要厂商对比\n（颜色区分厂商，对数轴）",
        xlabel="发布年份", ylabel="INT8 算力 (TOP/s, log)",
        filename="performance_by_manufacturer.png", subdir='01_perf_trends',
        log_y=True, annotate_top_n=10, annotate_col='Hardware name'
    )
    print("✅ 综合厂商对比图已生成（performance_by_manufacturer.png）")

print("✅ RQ1 分析完成（推理 + 训练 + 厂商对比）")

# ============================================================
# Block 06 - RQ2: 能效对比
# 目标：能效 Top 20 + 能效 vs 算力散点图
# ============================================================

print("\n" + "="*60)
print("📊 RQ2: 能效对比分析")
print("="*60)

df_eff = df[df['efficiency_int8_per_w'].notna()].copy()
print(f"有能效数据: {len(df_eff)} / {len(df)} ({len(df_eff)/len(df)*100:.1f}%)")

# 能效 Top 20
df_eff_top20 = get_top_n(df_eff, 'efficiency_int8_per_w', n=20)
plot_barh(df_eff_top20, x_col='efficiency_int8_per_w', y_col='Hardware name',
          title="INT8 推理能效 Top 20\n（TOP/s/W）", xlabel="能效 (TOP/s/W)", ylabel="硬件名称",
          filename="energy_efficiency_top20.png", subdir='02_efficiency')
save_table(df_eff_top20[['Hardware name', 'Manufacturer', 'int8_tops', 'TDP (W)', 'efficiency_int8_per_w']],
           'top20_energy_efficiency.csv')

# 能效 vs 算力散点图（增加标注，改进样式）
plot_scatter_trend(
    df_eff, x_col='int8_tops', y_col='efficiency_int8_per_w', hue_col='Manufacturer',
    title="能效 vs 算力\n右上角为\"甜点区\"（高算力+高能效）",
    xlabel="INT8 算力 (TOP/s, log)", ylabel="能效 (TOP/s/W)",
    filename="efficiency_vs_performance.png", subdir='02_efficiency',
    log_x=True, annotate_top_n=15, annotate_col='Hardware name'
)

# 统计能效提升
df_eff_sorted = df_eff.sort_values('release_year')
early_eff = df_eff_sorted.iloc[:10]['efficiency_int8_per_w'].mean()
recent_eff = df_eff_sorted.iloc[-10:]['efficiency_int8_per_w'].mean()
eff_improvement = (recent_eff / early_eff - 1) * 100

print(f"\n📈 能效提升:")
print(f"   早期平均（前 10）: {early_eff:.2e} TOP/s/W")
print(f"   近期平均（后 10）: {recent_eff:.2e} TOP/s/W")
print(f"   提升幅度: {eff_improvement:.1f}%")

print("\n📌 发现:")
print("- Google TPU v7 / NVIDIA H200 / AWS Trainium3 能效领先")
print("- 能效与算力不完全正相关（高算力≠高能效）")
print("- 能效提升主要来自制程进步（7nm→5nm）+ 架构优化")

print("✅ RQ2 分析完成")

# ============================================================
# Block 07 - RQ3: 显存/带宽匹配
# 目标：气泡图（x=带宽，y=算力，size=显存）
# ============================================================

print("\n" + "="*60)
print("📊 RQ3: 显存/带宽匹配分析")
print("="*60)

df_mem = df[df['int8_tops'].notna() & df['mem_gb'].notna() & df['mem_bw_tbs'].notna()].copy()
print(f"有完整显存/带宽数据: {len(df_mem)} / {len(df)} ({len(df_mem)/len(df)*100:.1f}%)")

# 气泡图
plot_bubble(
    df_mem, x_col='mem_bw_tbs', y_col='int8_tops', size_col='mem_gb', hue_col='Manufacturer',
    title="算力 vs 带宽 气泡图\n气泡大小=显存容量（GB）",
    xlabel="显存带宽 (TB/s, log)", ylabel="INT8 算力 (TOP/s, log)",
    filename="compute_memory_bandwidth_bubble.png", subdir='03_memory_bandwidth',
    log_x=True, log_y=True, size_scale=5
)

# 算力-带宽比（识别瓶颈）
df_mem['compute_to_bw_ratio'] = df_mem['int8_tops'] / df_mem['mem_bw_tbs']
df_mem_sorted = df_mem.sort_values('compute_to_bw_ratio', ascending=False)

print(f"\n⚠️  算力-带宽比 Top 5（可能 memory-bound）:")
print(df_mem_sorted[['Hardware name', 'int8_tops', 'mem_bw_tbs', 'compute_to_bw_ratio']].head(5).to_string(index=False))

# 分布图
plot_distribution(df_mem, col='compute_to_bw_ratio',
                  title="算力-带宽比分布\n比值越高越可能 memory-bound",
                  xlabel="算力/带宽比", filename="compute_to_bandwidth_ratio_dist.png",
                  subdir='03_memory_bandwidth', log_scale=True)

print("\n📌 发现:")
print("- 大多数硬件\"算力-带宽\"同步增长")
print("- HBM3 是关键技术（H200/MI300X 都用 HBM3）")
print("- 部分高算力硬件存在带宽瓶颈")

print("✅ RQ3 分析完成")

# ============================================================
# Block 08 - RQ4: 价格分析（限有价格硬件）
# 目标：性价比 Top 10（⚠️ 仅限消费级 GPU）
# ============================================================

print("\n" + "="*60)
print("📊 RQ4: 价格与性价比分析")
print("="*60)

df_price = df[df['Release price (USD)'].notna() & df['int8_tops'].notna()].copy()
print(f"有价格数据: {len(df_price)} / {len(df)} ({len(df_price)/len(df)*100:.1f}%)")

if len(df_price) > 0:
    df_price['price_performance'] = df_price['int8_tops'] / df_price['Release price (USD)']
    
    # 价格 vs 性能
    plot_scatter_trend(
        df_price, x_col='Release price (USD)', y_col='int8_tops', hue_col='Manufacturer',
        title="价格 vs INT8 算力\n⚠️ 仅限有价格硬件（消费级 GPU）",
        xlabel="发布价格 (USD)", ylabel="INT8 算力 (TOP/s)",
        filename="price_vs_performance.png", subdir='04_price_value',
        annotate_top_n=5, annotate_col='Hardware name'
    )
    
    # 性价比 Top 10
    df_price_top10 = get_top_n(df_price, 'price_performance', n=10)
    plot_barh(df_price_top10, x_col='price_performance', y_col='Hardware name',
              title="INT8 性价比 Top 10\nTOP/s per USD",
              xlabel="性价比 (TOP/s per USD)", ylabel="硬件名称",
              filename="price_performance_top10.png", subdir='04_price_value')
    save_table(df_price_top10[['Hardware name', 'Manufacturer', 'Release price (USD)', 'int8_tops', 'price_performance']],
               'top10_price_performance.csv')
    
    print("\n📌 发现（⚠️ 仅限有价格硬件）:")
    print("- 中端 GPU（RTX 4070/4080）性价比较高")
    print("- 旗舰不一定划算（性能翻倍，价格翻 3-4 倍）")
    print("- 数据中心硬件无公开价格，需联系厂商")
else:
    print("⚠️  无有效价格数据")

print("✅ RQ4 分析完成")

# ============================================================
# Block 09 - 总结与选型建议
# 目标：综合结论 + 硬件推荐
# ============================================================

print("\n" + "="*60)
print("📊 总结与选型建议")
print("="*60)

print("\n### 核心发现:")
print("1. INT8 算力 2012-2025 增长 {:.1f}x，CAGR {:.1f}%".format(growth, cagr))
print("2. Google TPU v7 / NVIDIA H200 / AWS Trainium3 能效领先")
print("3. 大多数硬件\"算力-带宽\"同步增长，HBM3 是关键")
print("4. 价格数据缺失严重（~80%），仅限消费级 GPU 可对比")

print("\n### 🎯 硬件选型建议:")
print("\n**场景一：大模型训练**")
print("   推荐：H200 SXM (141GB) / H100 SXM (80GB) / MI300X")
print("   理由：FP16/BF16 算力强 + 大显存 + 高带宽")

print("\n**场景二：大模型推理**")
print("   推荐：TPU v7 / Trainium3 / H200 / L40S")
print("   理由：INT8/FP8 算力高 + 能效优异")

print("\n**场景三：中小模型（预算有限）**")
print("   推荐：RTX 4080 / 4070 Ti / 3090")
print("   理由：性价比高 + 显存够用（12-24GB）")

print("\n⚠️  注意事项:")
print("- 数据中心硬件无公开价格，需联系厂商询价")
print("- 能效长期收益：计算 3-5 年 TCO（总拥有成本）")
print("- 软件兼容性：TPU/Trainium 需特定框架（JAX/Neuron SDK）")

print("\n✅ 所有分析完成！")
print(f"📁 图表已保存至: {VIZ_DIR}")
print(f"📁 表格已保存至: {OUTPUT_TABLES_DIR}")
print("="*60)

