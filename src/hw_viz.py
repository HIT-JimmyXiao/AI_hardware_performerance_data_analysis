"""
AI 硬件分析项目 - 可视化函数库
作者：HIT_Jimmy
用途：统一风格的可视化函数，减少重复代码
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 导入配置
from hw_config import *

# ============================================================
# 1. 初始化可视化主题（全局调用一次）
# ============================================================

def init_viz_theme():
    """
    初始化全局可视化主题（调用一次即可）
    包含中文显示修正（跨平台支持）
    """
    # seaborn 主题（先设置，避免覆盖字体）
    sns.set_style(SNS_STYLE)
    sns.set_context(SNS_CONTEXT)
    sns.set_palette(SNS_PALETTE)
    
    # matplotlib 全局配置（包含字体设置）
    plt.rcParams.update({
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'figure.figsize': FIGSIZE,
        'axes.grid': True,
        'grid.alpha': GRID_ALPHA,
        'figure.autolayout': True,  # 自动调整布局
    })
    
    # 中文显示修正（必须在 update 之后设置，确保不被覆盖）
    # 检测并设置可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_candidates = FONT_SANS_SERIF + ['Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']  # 添加更多候选字体
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + [f for f in FONT_SANS_SERIF if f != selected_font]
        logging.info(f"✅ 找到中文字体: {selected_font}")
    else:
        # 如果找不到，使用配置的字体（让 matplotlib 尝试）
        plt.rcParams['font.sans-serif'] = FONT_SANS_SERIF
        logging.warning(f"⚠️ 未找到中文字体，使用配置: {FONT_SANS_SERIF}")
        logging.warning(f"   可用字体示例: {available_fonts[:5]}...")
    
    plt.rcParams['font.family'] = ['sans-serif']  # 使用 sans-serif 字体族
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题
    
    # 验证字体设置
    current_font = plt.rcParams['font.sans-serif']
    logging.info(f"✅ 可视化主题初始化完成（当前字体: {current_font[0] if current_font else 'None'}）")

# ============================================================
# 2. 保存图表（统一接口）
# ============================================================

def save_fig(fig, filename, subdir='00_dataset_overview', tight=True, transparent=False):
    """
    保存图表到 visualization/ 指定子目录
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        图表对象
    filename : str
        文件名（含扩展名，如 'missing_rate.png'）
    subdir : str
        visualization/ 下的子目录名
    tight : bool
        是否使用 tight_layout
    transparent : bool
        是否透明背景
    """
    ensure_dirs()
    
    # 映射子目录
    subdir_map = {
        '00_dataset_overview': VIZ_OVERVIEW_DIR,
        '01_perf_trends': VIZ_PERF_TRENDS_DIR,
        '02_efficiency': VIZ_EFFICIENCY_DIR,
        '03_memory_bandwidth': VIZ_MEMORY_DIR,
        '04_price_value': VIZ_PRICE_DIR,
        '99_appendix': VIZ_APPENDIX_DIR,
    }
    
    out_dir = subdir_map.get(subdir, VIZ_DIR / subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = out_dir / filename
    
    if tight:
        fig.tight_layout()
    
    fig.savefig(filepath, dpi=DPI, transparent=transparent, bbox_inches='tight')
    logging.info(f"✅ 图表已保存: {filepath}")
    
    # 清理内存
    plt.close(fig)

# ============================================================
# 3. 缺失率热力图
# ============================================================

def plot_missing_heatmap(df, title="Missing Rate Heatmap", filename="missing_rate_heatmap.png", min_missing_rate=50.0):
    """
    绘制缺失率热力图
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    title : str
        图表标题
    filename : str
        保存文件名
    min_missing_rate : float
        最小缺失率阈值（只显示缺失率 >= 此值的字段）
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        图表对象
    """
    # 计算缺失率
    missing_rate = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_rate = missing_rate[missing_rate >= min_missing_rate]  # 只显示缺失率 >= 阈值的字段
    
    if len(missing_rate) == 0:
        logging.warning(f"⚠️ 无缺失率 >= {min_missing_rate}% 的字段，跳过热力图")
        return None
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, max(6, len(missing_rate) * 0.3)))
    
    # 热力图（竖向排列）
    sns.heatmap(
        missing_rate.values.reshape(-1, 1),
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Missing Rate (%)'},
        yticklabels=missing_rate.index,
        xticklabels=[''],
        ax=ax
    )
    
    ax.set_title(f"{title}\n(仅显示缺失率 ≥ {min_missing_rate}% 的字段)", fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Field', fontsize=LABEL_SIZE)
    
    save_fig(fig, filename, subdir='00_dataset_overview')
    return fig

# ============================================================
# 4. 条形图（横向/纵向）
# ============================================================

def plot_barh(data, x_col, y_col, title, xlabel, ylabel, filename, subdir='00_dataset_overview', color=None, top_n=None):
    """
    绘制横向条形图（适合排名/Top-N）
    
    Parameters:
    -----------
    data : pd.DataFrame
        数据框
    x_col : str
        X 轴列名（数值）
    y_col : str
        Y 轴列名（类别）
    title : str
        图表标题
    xlabel/ylabel : str
        轴标签
    filename : str
        保存文件名
    subdir : str
        子目录
    color : str or list
        颜色（单色或列表）
    top_n : int
        只显示前 N 个
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    if top_n:
        data = data.nlargest(top_n, x_col)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.4)))
    
    # 横向条形图
    ax.barh(data[y_col], data[x_col], color=color or sns.color_palette()[0])
    
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.invert_yaxis()  # Top 在上
    ax.grid(axis='x', alpha=GRID_ALPHA)
    
    save_fig(fig, filename, subdir=subdir)
    return fig

# ============================================================
# 5. 散点图（趋势分析）
# ============================================================

def plot_scatter_trend(df, x_col, y_col, hue_col=None, title="", xlabel="", ylabel="", 
                       filename="scatter_trend.png", subdir='01_perf_trends', 
                       log_y=False, log_x=False, annotate_top_n=10, annotate_col='Hardware name',
                       annotate_threshold=None):
    """
    绘制散点图 + 趋势（适合时间序列/相关分析）
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    x_col, y_col : str
        X/Y 轴列名
    hue_col : str
        分组列（颜色区分）
    title, xlabel, ylabel : str
        标题与轴标签
    filename : str
        文件名
    subdir : str
        子目录
    log_y, log_x : bool
        是否对数轴
    annotate_top_n : int
        标注前 N 个点
    annotate_col : str
        标注文本列名
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # 过滤有效数据
    df_plot = df[[x_col, y_col, hue_col, annotate_col]].dropna(subset=[x_col, y_col]) if hue_col else df[[x_col, y_col, annotate_col]].dropna(subset=[x_col, y_col])
    
    if len(df_plot) == 0:
        logging.warning(f"⚠️ {title}: 无有效数据，跳过绘图")
        return None
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # 散点图
    if hue_col:
        for label, group in df_plot.groupby(hue_col):
            color = get_manufacturer_color(label) if hue_col == 'Manufacturer' else None
            ax.scatter(group[x_col], group[y_col], label=label, alpha=0.7, s=80, color=color)
    else:
        ax.scatter(df_plot[x_col], df_plot[y_col], alpha=0.7, s=80, color=sns.color_palette()[0])
    
    # 对数轴
    if log_y:
        ax.set_yscale('log')
    if log_x:
        ax.set_xscale('log')
    
    # 标注 Top-N 点（改进版：避免重叠，美化样式）
    if annotate_top_n and annotate_col:
        # 选择要标注的点（Top-N 或超过阈值的）
        if annotate_threshold is not None:
            top_data = df_plot[df_plot[y_col] >= annotate_threshold].nlargest(annotate_top_n, y_col)
        else:
            top_data = df_plot.nlargest(annotate_top_n, y_col)
        
        # 使用 adjustText 避免重叠（如果可用）
        try:
            from adjustText import adjust_text
            texts = []
            for _, row in top_data.iterrows():
                text = ax.annotate(
                    row[annotate_col],
                    (row[x_col], row[y_col]),
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor='white',
                        edgecolor='gray',
                        alpha=0.8,
                        linewidth=1
                    ),
                    arrowprops=dict(
                        arrowstyle='->',
                        connectionstyle='arc3,rad=0.2',
                        color='gray',
                        alpha=0.6,
                        lw=1
                    )
                )
                texts.append(text)
            # 自动调整文本位置避免重叠
            adjust_text(texts, ax=ax, 
                       expand_points=(1.2, 1.2),
                       expand_text=(1.2, 1.2),
                       force_points=(0.3, 0.3),
                       force_text=(0.3, 0.3),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        except ImportError:
            # 如果没有 adjustText，使用改进的手动标注
            logging.warning("⚠️ adjustText 未安装，使用基础标注（可能重叠）")
            for i, (_, row) in enumerate(top_data.iterrows()):
                # 交替标注位置，减少重叠
                offset_x = 10 if i % 2 == 0 else -10
                offset_y = 10 if i % 3 == 0 else -10
                ax.annotate(
                    row[annotate_col],
                    (row[x_col], row[y_col]),
                    xytext=(offset_x, offset_y),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor='white',
                        edgecolor='gray',
                        alpha=0.8,
                        linewidth=1
                    ),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='gray',
                        alpha=0.6,
                        lw=1
                    )
                )
    
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    if hue_col:
        ax.legend(loc='best', fontsize=LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)
    
    save_fig(fig, filename, subdir=subdir)
    return fig

# ============================================================
# 6. 气泡图（多维度）
# ============================================================

def plot_bubble(df, x_col, y_col, size_col, hue_col, title, xlabel, ylabel, 
                filename, subdir='03_memory_bandwidth', log_x=False, log_y=False, size_scale=200):
    """
    绘制气泡图（x/y/size/color 四维展示）
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    x_col, y_col : str
        X/Y 轴
    size_col : str
        气泡大小列
    hue_col : str
        颜色分组列
    title, xlabel, ylabel : str
        标题与标签
    filename : str
        文件名
    subdir : str
        子目录
    log_x, log_y : bool
        对数轴
    size_scale : int
        气泡大小缩放系数
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # 过滤有效数据
    df_plot = df[[x_col, y_col, size_col, hue_col]].dropna()
    
    if len(df_plot) == 0:
        logging.warning(f"⚠️ {title}: 无有效数据，跳过绘图")
        return None
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # 按 hue 分组绘制
    for label, group in df_plot.groupby(hue_col):
        color = get_manufacturer_color(label) if hue_col == 'Manufacturer' else None
        ax.scatter(
            group[x_col],
            group[y_col],
            s=group[size_col] * size_scale,
            label=label,
            alpha=0.6,
            color=color,
            edgecolors='black',
            linewidth=0.5
        )
    
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.legend(loc='best', fontsize=LEGEND_SIZE, framealpha=0.9, scatterpoints=1)
    ax.grid(True, alpha=GRID_ALPHA)
    
    save_fig(fig, filename, subdir=subdir)
    return fig

# ============================================================
# 7. 分布图（直方图 + KDE）
# ============================================================

def plot_distribution(df, col, title, xlabel, filename, subdir='00_dataset_overview', bins=30, log_scale=False):
    """
    绘制分布图（直方图 + KDE）
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    col : str
        列名
    title, xlabel : str
        标题与标签
    filename : str
        文件名
    subdir : str
        子目录
    bins : int
        直方图分箱数
    log_scale : bool
        是否对数轴
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    data = df[col].dropna()
    
    if len(data) == 0:
        logging.warning(f"⚠️ {title}: 无有效数据，跳过绘图")
        return None
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # 直方图 + KDE
    sns.histplot(data, bins=bins, kde=True, ax=ax, color=sns.color_palette()[0], alpha=0.6)
    
    if log_scale:
        ax.set_xscale('log')
    
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel('Frequency', fontsize=LABEL_SIZE)
    ax.grid(True, alpha=GRID_ALPHA)
    
    save_fig(fig, filename, subdir=subdir)
    return fig

# ============================================================
# 8. 分面图（按类别分组展示趋势）
# ============================================================

def plot_facet_trend(df, x_col, y_col, facet_col, title, xlabel, ylabel, filename, subdir='01_perf_trends', log_y=False):
    """
    绘制分面图（按 facet_col 分组，每组一个子图）
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    x_col, y_col : str
        X/Y 轴
    facet_col : str
        分面列
    title, xlabel, ylabel : str
        标题与标签
    filename : str
        文件名
    subdir : str
        子目录
    log_y : bool
        Y 轴对数
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    df_plot = df[[x_col, y_col, facet_col]].dropna()
    
    if len(df_plot) == 0:
        logging.warning(f"⚠️ {title}: 无有效数据，跳过绘图")
        return None
    
    # 使用 seaborn FacetGrid
    g = sns.FacetGrid(df_plot, col=facet_col, col_wrap=3, height=4, aspect=1.2, sharey=False)
    g.map(sns.scatterplot, x_col, y_col, alpha=0.7, s=50)
    
    if log_y:
        for ax in g.axes.flat:
            ax.set_yscale('log')
    
    g.set_titles(col_template="{col_name}", size=TITLE_SIZE)
    g.set_axis_labels(xlabel, ylabel, fontsize=LABEL_SIZE)
    g.fig.suptitle(title, fontsize=TITLE_SIZE + 2, fontweight='bold', y=1.02)
    
    save_fig(g.fig, filename, subdir=subdir)
    return g.fig

# ============================================================
# 9. 相关性热力图
# ============================================================

def plot_corr_heatmap(df, cols, title, filename, subdir='00_dataset_overview', method='pearson'):
    """
    绘制相关性热力图
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    cols : list
        要计算相关性的列
    title : str
        标题
    filename : str
        文件名
    subdir : str
        子目录
    method : str
        相关系数方法（'pearson' / 'spearman'）
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    df_corr = df[cols].dropna()
    
    if len(df_corr) == 0:
        logging.warning(f"⚠️ {title}: 无有效数据，跳过绘图")
        return None
    
    corr = df_corr.corr(method=method)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    
    save_fig(fig, filename, subdir=subdir)
    return fig

# ============================================================
# 测试（运行本文件时自动测试）
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI 硬件分析项目 - 可视化函数测试")
    print("=" * 60)
    
    # 初始化主题
    init_viz_theme()
    
    # 生成测试数据
    np.random.seed(42)
    test_df = pd.DataFrame({
        'Year': np.repeat([2015, 2018, 2021, 2024], 10),
        'Performance': np.random.lognormal(12, 1, 40),
        'TDP': np.random.uniform(150, 400, 40),
        'Memory': np.random.uniform(8, 80, 40),
        'Manufacturer': np.random.choice(['NVIDIA', 'AMD', 'Google'], 40)
    })
    
    # 测试散点图
    print("\n测试散点图...")
    plot_scatter_trend(
        test_df,
        x_col='Year',
        y_col='Performance',
        hue_col='Manufacturer',
        title="Test Performance Trend",
        xlabel="Year",
        ylabel="Performance (TFLOP/s)",
        filename="test_scatter.png",
        log_y=True,
        annotate_top_n=3,
        annotate_col='Manufacturer'
    )
    
    print("\n✅ 可视化测试完成，请检查 visualization/ 目录")
    print("=" * 60)

