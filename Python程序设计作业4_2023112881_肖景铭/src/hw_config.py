"""
AI 硬件分析项目 - 全局配置文件
作者：HIT_Jimmy
用途：集中管理路径、参数、主题、默认设置等
"""

import os
import platform
from pathlib import Path

# ============================================================
# 1. 路径配置
# ============================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "output"
VIZ_DIR = PROJECT_ROOT / "visualization"

# 数据文件
ML_HARDWARE_CSV = DATA_DIR / "ml_hardware.csv"

# 输出子目录
OUTPUT_TABLES_DIR = OUTPUT_DIR / "tables"
OUTPUT_DERIVED_DIR = OUTPUT_DIR / "derived"
OUTPUT_LOGS_DIR = OUTPUT_DIR / "logs"

# 可视化子目录
VIZ_OVERVIEW_DIR = VIZ_DIR / "00_dataset_overview"
VIZ_PERF_TRENDS_DIR = VIZ_DIR / "01_perf_trends"
VIZ_EFFICIENCY_DIR = VIZ_DIR / "02_efficiency"
VIZ_MEMORY_DIR = VIZ_DIR / "03_memory_bandwidth"
VIZ_PRICE_DIR = VIZ_DIR / "04_price_value"
VIZ_APPENDIX_DIR = VIZ_DIR / "99_appendix"

# 自动创建目录
ALL_DIRS = [
    OUTPUT_DIR, OUTPUT_TABLES_DIR, OUTPUT_DERIVED_DIR, OUTPUT_LOGS_DIR,
    VIZ_DIR, VIZ_OVERVIEW_DIR, VIZ_PERF_TRENDS_DIR, VIZ_EFFICIENCY_DIR,
    VIZ_MEMORY_DIR, VIZ_PRICE_DIR, VIZ_APPENDIX_DIR
]

def ensure_dirs():
    """确保所有输出目录存在"""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. 分析参数配置
# ============================================================

# 时间范围（主要分析）
YEAR_RANGE = (2012, 2025)  # 2012 年前数据稀疏

# 算力口径（字段名）
TRAIN_METRIC = "Tensor-FP16/BF16 performance (FLOP/s)"  # 训练主口径
INFERENCE_METRIC = "INT8 performance (OP/s)"            # 推理主口径（本项目重点）
FP32_METRIC = "FP32 (single precision) performance (FLOP/s)"  # 传统基准
FP8_METRIC = "FP8 performance (FLOP/s)"                 # 新一代推理/训练

# 显示/排名数量
TOP_N = 20  # Top-N 排名

# 能效口径（可选：用哪个算力 / TDP）
EFFICIENCY_METRIC = INFERENCE_METRIC  # 默认用 INT8 算力计算能效

# 缺失值处理策略
MISSING_THRESHOLD = 0.5  # 若某字段缺失率 > 50%，标注为"高缺失"

# ============================================================
# 3. 可视化配置
# ============================================================

# 图表尺寸与分辨率
FIGSIZE = (12, 8)       # 默认尺寸（英寸）
FIGSIZE_SMALL = (10, 6) # 小图尺寸
FIGSIZE_WIDE = (14, 6)  # 宽图尺寸
DPI = 300               # 分辨率（适合论文打印）

# 字体与主题
# 跨平台中文字体配置
SYSTEM = platform.system()
if SYSTEM == 'Windows':
    FONT_SANS_SERIF = ['SimHei']  # Windows 黑体
elif SYSTEM == 'Darwin':  # macOS
    FONT_SANS_SERIF = ['PingFang SC', 'Arial Unicode MS']
elif SYSTEM == 'Linux':
    FONT_SANS_SERIF = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
else:
    FONT_SANS_SERIF = ['DejaVu Sans']  # 默认

FONT_FAMILY = "sans-serif"  # 使用 sans-serif 字体族
FONT_SIZE = 11
TITLE_SIZE = 14
LABEL_SIZE = 12
LEGEND_SIZE = 10

# seaborn 主题
SNS_STYLE = "whitegrid"      # 背景风格
SNS_PALETTE = "tab10"        # 配色方案（颜色盲友好）
SNS_CONTEXT = "notebook"     # 上下文（'paper', 'notebook', 'talk', 'poster'）

# 颜色配置
COLOR_NVIDIA = "#76B900"     # NVIDIA 绿
COLOR_AMD = "#ED1C24"        # AMD 红
COLOR_GOOGLE = "#4285F4"     # Google 蓝
COLOR_INTEL = "#0071C5"      # Intel 蓝
COLOR_AWS = "#FF9900"        # AWS 橙
COLOR_META = "#0668E1"       # Meta 蓝

# 厂商颜色映射（可扩展）
MANUFACTURER_COLORS = {
    "NVIDIA": COLOR_NVIDIA,
    "AMD": COLOR_AMD,
    "Google": COLOR_GOOGLE,
    "Intel": COLOR_INTEL,
    "Amazon AWS": COLOR_AWS,
    "Meta": COLOR_META,
}

# 网格透明度
GRID_ALPHA = 0.3

# 图例位置（默认）
LEGEND_LOC = "best"

# ============================================================
# 4. 数据清洗配置
# ============================================================

# 日期格式
DATE_FORMAT = "%Y-%m-%d"

# 数值字段（需转换为 float）
NUMERIC_COLS = [
    "TDP (W)",
    "FP64 (double precision) performance (FLOP/s)",
    "FP32 (single precision) performance (FLOP/s)",
    "TF32 (TensorFloat-32) performance (FLOP/s)",
    "FP16 (half precision) performance (FLOP/s)",
    "Tensor-FP16/BF16 performance (FLOP/s)",
    "FP8 performance (FLOP/s)",
    "FP4 performance (FLOP/s)",
    "INT16 performance (OP/s)",
    "INT8 performance (OP/s)",
    "INT4 performance (OP/s)",
    "Memory (bytes)",
    "Memory bandwidth (byte/s)",
    "Intranode bandwidth (byte/s)",
    "Internode bandwidth (bit/s)",
    "Release price (USD)",
    "Energy efficiency",
    "Max performance",
    "Total processing performance (bit-OP/s)",
    "Price-performance",
    "ML OP/s",
]

# 类别字段
CATEGORICAL_COLS = [
    "Manufacturer",
    "Type",
    "Foundry",
]

# ============================================================
# 5. 单位换算常数
# ============================================================

# 算力换算
TERA = 1e12   # 1 TFLOP/s = 10^12 FLOP/s
PETA = 1e15   # 1 PFLOP/s = 10^15 FLOP/s

# 存储换算
KB = 1024
MB = KB ** 2
GB = KB ** 3
TB = KB ** 4

# 带宽换算（二进制 vs 十进制）
GB_DECIMAL = 1e9     # 1 GB = 10^9 bytes（十进制，常用于带宽）
TB_DECIMAL = 1e12    # 1 TB = 10^12 bytes

# ============================================================
# 6. 随机种子（可复现）
# ============================================================

RANDOM_SEED = 42

# ============================================================
# 7. 日志配置（可选）
# ============================================================

LOG_LEVEL = "INFO"  # DEBUG / INFO / WARNING / ERROR
LOG_FILE = OUTPUT_LOGS_DIR / "analysis.log"

# ============================================================
# 8. Notebook 设置（可选）
# ============================================================

# Jupyter 中是否内联显示图表
INLINE_PLOTS = True

# 是否自动保存图表（即使在 notebook 中也落盘）
AUTO_SAVE_FIGS = True

# ============================================================
# 9. 辅助函数
# ============================================================

def get_manufacturer_color(manufacturer):
    """获取厂商对应的颜色（如果没有则返回默认色）"""
    return MANUFACTURER_COLORS.get(manufacturer, "#808080")  # 默认灰色

def format_tflops(value):
    """格式化 TFLOP/s（保留 1 位小数）"""
    if value >= PETA:
        return f"{value / PETA:.1f} PFLOP/s"
    elif value >= TERA:
        return f"{value / TERA:.1f} TFLOP/s"
    else:
        return f"{value:.0f} GFLOP/s"

def format_tops(value):
    """格式化 TOP/s（保留 1 位小数）"""
    if value >= PETA:
        return f"{value / PETA:.1f} PETA-OP/s"
    elif value >= TERA:
        return f"{value / TERA:.1f} TOP/s"
    else:
        return f"{value:.0f} GOP/s"

def format_memory(value_bytes):
    """格式化显存（bytes → GB）"""
    if value_bytes >= TB:
        return f"{value_bytes / TB:.1f} TB"
    elif value_bytes >= GB:
        return f"{value_bytes / GB:.1f} GB"
    else:
        return f"{value_bytes / MB:.0f} MB"

def format_bandwidth(value_bps):
    """格式化带宽（byte/s → TB/s）"""
    if value_bps >= TB_DECIMAL:
        return f"{value_bps / TB_DECIMAL:.1f} TB/s"
    elif value_bps >= GB_DECIMAL:
        return f"{value_bps / GB_DECIMAL:.1f} GB/s"
    else:
        return f"{value_bps / 1e6:.0f} MB/s"

# ============================================================
# 测试（运行本文件时自动测试）
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI 硬件分析项目 - 配置文件测试")
    print("=" * 60)
    
    # 测试路径
    print(f"\n📁 项目根目录: {PROJECT_ROOT}")
    print(f"📊 数据文件: {ML_HARDWARE_CSV}")
    print(f"📈 可视化目录: {VIZ_DIR}")
    
    # 测试参数
    print(f"\n⚙️  分析时间范围: {YEAR_RANGE}")
    print(f"⚙️  推理口径: {INFERENCE_METRIC}")
    print(f"⚙️  Top-N: {TOP_N}")
    
    # 测试格式化
    print(f"\n🎨 格式化测试:")
    print(f"   6.71e14 FLOP/s → {format_tflops(6.71e14)}")
    print(f"   2.517e15 FLOP/s → {format_tflops(2.517e15)}")
    print(f"   1.92e11 bytes → {format_memory(1.92e11)}")
    print(f"   7.37e12 byte/s → {format_bandwidth(7.37e12)}")
    
    # 测试创建目录
    print(f"\n📂 创建输出目录...")
    ensure_dirs()
    print(f"   ✅ 完成！")
    
    print("\n" + "=" * 60)

