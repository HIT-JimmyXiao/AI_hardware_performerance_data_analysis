"""
AI 硬件分析项目 - 通用工具函数
作者：HIT_Jimmy
用途：数据读取、单位换算、格式化、保存等通用操作
"""

import pandas as pd
import numpy as np
from pathlib import Path
import csv
import logging
from datetime import datetime

# 导入配置
from hw_config import *

# ============================================================
# 1. 日志配置
# ============================================================

def setup_logging(log_file=LOG_FILE, level=LOG_LEVEL):
    """
    配置日志系统
    
    Parameters:
    -----------
    log_file : Path
        日志文件路径
    level : str
        日志级别（DEBUG/INFO/WARNING/ERROR）
    """
    ensure_dirs()  # 确保日志目录存在
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logging.info("日志系统初始化完成")

# ============================================================
# 2. 数据读取（鲁棒解析 CSV）
# ============================================================

def load_ml_hardware_data(filepath=ML_HARDWARE_CSV, encoding='utf-8'):
    """
    读取 ML Hardware 数据集（处理 Notes 多行/引号问题）
    
    Parameters:
    -----------
    filepath : Path or str
        CSV 文件路径
    encoding : str
        文件编码
    
    Returns:
    --------
    df : pd.DataFrame
        读取的数据（含原始列名）
    
    Notes:
    ------
    - Notes 字段可能包含多行引号，使用 QUOTE_ALL 模式
    - 失败时会尝试 python engine 回退
    """
    logging.info(f"正在读取数据: {filepath}")
    
    try:
        # 方式1：默认 C engine（快速但可能失败）
        df = pd.read_csv(
            filepath,
            encoding=encoding,
            low_memory=False,  # 避免 DtypeWarning
            na_values=['', 'NA', 'N/A', 'nan', 'NaN', 'null']  # 统一缺失值
        )
        logging.info(f"✅ 数据读取成功（C engine）: {df.shape[0]} 行 × {df.shape[1]} 列")
        return df
    
    except Exception as e:
        logging.warning(f"⚠️ C engine 解析失败: {e}")
        logging.info("尝试使用 Python engine 回退...")
        
        try:
            # 方式2：Python engine（慢但鲁棒）
            df = pd.read_csv(
                filepath,
                encoding=encoding,
                engine='python',
                quoting=csv.QUOTE_MINIMAL,
                escapechar='\\',
                na_values=['', 'NA', 'N/A', 'nan', 'NaN', 'null']
            )
            logging.info(f"✅ 数据读取成功（Python engine）: {df.shape[0]} 行 × {df.shape[1]} 列")
            return df
        
        except Exception as e2:
            logging.error(f"❌ 数据读取失败: {e2}")
            raise

# ============================================================
# 3. 数据清洗辅助函数
# ============================================================

def parse_release_date(df, date_col='Release date'):
    """
    解析发布日期（字符串 → datetime）
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    date_col : str
        日期列名
    
    Returns:
    --------
    df : pd.DataFrame
        添加了 'release_year', 'release_month' 列
    """
    logging.info(f"解析日期字段: {date_col}")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['release_year'] = df[date_col].dt.year
    df['release_month'] = df[date_col].dt.month
    
    # 统计缺失
    missing_count = df[date_col].isna().sum()
    if missing_count > 0:
        logging.warning(f"⚠️ {date_col} 缺失 {missing_count} 条记录")
    
    return df

def convert_numeric_cols(df, cols=None):
    """
    将指定列转为数值类型（处理科学计数法）
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    cols : list
        要转换的列名列表（默认使用 hw_config.NUMERIC_COLS）
    
    Returns:
    --------
    df : pd.DataFrame
        转换后的数据框
    """
    if cols is None:
        cols = NUMERIC_COLS
    
    logging.info(f"转换数值列: {len(cols)} 个字段")
    
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def add_derived_metrics(df):
    """
    添加派生指标（单位换算 + 能效 + 比率）
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始数据框
    
    Returns:
    --------
    df : pd.DataFrame
        添加了派生列的数据框
    
    派生列:
    -------
    - perf_fp32_tflops: FP32 算力（TFLOP/s）
    - perf_fp16_tflops: FP16 算力（TFLOP/s）
    - perf_fp8_tflops: FP8 算力（TFLOP/s）
    - int8_tops: INT8 算力（TOP/s）
    - mem_gb: 显存（GB）
    - mem_bw_tbs: 显存带宽（TB/s）
    - efficiency_int8_per_w: INT8 能效（TOP/s/W）
    - efficiency_fp16_per_w: FP16 能效（TFLOP/s/W）
    - compute_to_mem_ratio: 算力-带宽比
    - price_per_int8_top: 价格-INT8 性价比（USD/TOP/s）
    """
    logging.info("计算派生指标...")
    
    # 算力换算（FLOP/s → TFLOP/s, OP/s → TOP/s）
    if FP32_METRIC in df.columns:
        df['perf_fp32_tflops'] = df[FP32_METRIC] / TERA
    
    if TRAIN_METRIC in df.columns:
        df['perf_fp16_tflops'] = df[TRAIN_METRIC] / TERA
    
    if FP8_METRIC in df.columns:
        df['perf_fp8_tflops'] = df[FP8_METRIC] / TERA
    
    if INFERENCE_METRIC in df.columns:
        df['int8_tops'] = df[INFERENCE_METRIC] / TERA
    
    # 显存换算（bytes → GB）
    if 'Memory (bytes)' in df.columns:
        df['mem_gb'] = df['Memory (bytes)'] / GB
    
    # 带宽换算（byte/s → TB/s）
    if 'Memory bandwidth (byte/s)' in df.columns:
        df['mem_bw_tbs'] = df['Memory bandwidth (byte/s)'] / TB_DECIMAL
    
    # 能效（算力 / TDP）
    if INFERENCE_METRIC in df.columns and 'TDP (W)' in df.columns:
        df['efficiency_int8_per_w'] = df[INFERENCE_METRIC] / df['TDP (W)']
    
    if TRAIN_METRIC in df.columns and 'TDP (W)' in df.columns:
        df['efficiency_fp16_per_w'] = df[TRAIN_METRIC] / df['TDP (W)']
    
    # 算力-带宽比（用于瓶颈分析）
    if 'Max performance' in df.columns and 'Memory bandwidth (byte/s)' in df.columns:
        df['compute_to_mem_ratio'] = df['Max performance'] / df['Memory bandwidth (byte/s)']
    
    # 价格-性能比（仅限有价格数据）
    if INFERENCE_METRIC in df.columns and 'Release price (USD)' in df.columns:
        df['price_per_int8_top'] = df['Release price (USD)'] / (df[INFERENCE_METRIC] / TERA)
    
    logging.info("✅ 派生指标计算完成")
    return df

# ============================================================
# 4. 数据质量检查
# ============================================================

def get_missing_summary(df):
    """
    生成缺失值汇总表
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    
    Returns:
    --------
    summary : pd.DataFrame
        缺失值汇总（字段名、缺失数、缺失率、非空数）
    """
    missing_count = df.isnull().sum()
    missing_rate = (missing_count / len(df) * 100).round(2)
    non_missing = len(df) - missing_count
    
    summary = pd.DataFrame({
        'Field': df.columns,
        'Missing Count': missing_count.values,
        'Missing Rate (%)': missing_rate.values,
        'Non-Missing': non_missing.values
    })
    
    # 按缺失率降序排列
    summary = summary.sort_values('Missing Rate (%)', ascending=False)
    summary = summary[summary['Missing Count'] > 0]  # 只显示有缺失的字段
    
    return summary

def check_duplicates(df, subset=['Hardware name', 'Release date']):
    """
    检查重复记录
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    subset : list
        用于判断重复的列
    
    Returns:
    --------
    duplicates : pd.DataFrame
        重复记录
    """
    duplicates = df[df.duplicated(subset=subset, keep=False)]
    
    if len(duplicates) > 0:
        logging.warning(f"⚠️ 发现 {len(duplicates)} 条重复记录")
    else:
        logging.info("✅ 无重复记录")
    
    return duplicates

# ============================================================
# 5. 数据筛选与分组
# ============================================================

def filter_by_year_range(df, year_col='release_year', year_range=YEAR_RANGE):
    """
    按年份范围筛选数据
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    year_col : str
        年份列名
    year_range : tuple
        (起始年, 结束年)
    
    Returns:
    --------
    df_filtered : pd.DataFrame
        筛选后的数据
    """
    df_filtered = df[
        (df[year_col] >= year_range[0]) &
        (df[year_col] <= year_range[1])
    ].copy()
    
    logging.info(f"按年份筛选 ({year_range[0]}-{year_range[1]}): {len(df)} → {len(df_filtered)} 条记录")
    
    return df_filtered

def get_top_n(df, metric, n=TOP_N, ascending=False):
    """
    获取 Top-N 记录
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    metric : str
        排序指标列名
    n : int
        Top N
    ascending : bool
        是否升序（False=降序，即最大的 N 个）
    
    Returns:
    --------
    df_top : pd.DataFrame
        Top-N 数据
    """
    df_top = df.nlargest(n, metric) if not ascending else df.nsmallest(n, metric)
    return df_top

# ============================================================
# 6. 保存结果
# ============================================================

def save_table(df, filename, subdir='tables', index=False):
    """
    保存表格到 output/tables/ 或指定子目录
    
    Parameters:
    -----------
    df : pd.DataFrame
        要保存的数据框
    filename : str
        文件名（含扩展名，如 'top_int8.csv'）
    subdir : str
        output/ 下的子目录名（默认 'tables'）
    index : bool
        是否保存索引
    """
    ensure_dirs()
    
    if subdir == 'tables':
        out_dir = OUTPUT_TABLES_DIR
    elif subdir == 'derived':
        out_dir = OUTPUT_DERIVED_DIR
    else:
        out_dir = OUTPUT_DIR / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = out_dir / filename
    
    # 根据扩展名选择保存格式
    if filepath.suffix == '.csv':
        df.to_csv(filepath, index=index, encoding='utf-8-sig')  # utf-8-sig 兼容 Excel
    elif filepath.suffix in ['.xlsx', '.xls']:
        df.to_excel(filepath, index=index)
    else:
        logging.warning(f"⚠️ 不支持的文件格式: {filepath.suffix}，默认保存为 CSV")
        filepath = filepath.with_suffix('.csv')
        df.to_csv(filepath, index=index, encoding='utf-8-sig')
    
    logging.info(f"✅ 表格已保存: {filepath}")

# ============================================================
# 7. 格式化工具（包装 hw_config 的函数）
# ============================================================

def format_large_number(value):
    """格式化大数字（自动添加千分位）"""
    if pd.isna(value):
        return "N/A"
    if value >= 1e12:
        return f"{value/1e12:.2f}T"
    elif value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    else:
        return f"{value:,.0f}"

def format_percentage(value):
    """格式化百分比"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"

# ============================================================
# 8. 快速统计
# ============================================================

def quick_stats(df, col):
    """
    快速统计（均值、中位数、标准差、最大最小）
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    col : str
        列名
    
    Returns:
    --------
    stats : dict
        统计字典
    """
    data = df[col].dropna()
    
    if len(data) == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    
    return {
        "count": len(data),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "min": data.min(),
        "max": data.max()
    }

# ============================================================
# 测试（运行本文件时自动测试）
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI 硬件分析项目 - 工具函数测试")
    print("=" * 60)
    
    # 测试日志
    setup_logging()
    logging.info("日志测试")
    
    # 测试数据读取
    try:
        df = load_ml_hardware_data()
        print(f"\n✅ 数据读取成功: {df.shape}")
        print(f"\n前 3 行:")
        print(df.head(3))
        
        # 测试清洗
        df = parse_release_date(df)
        df = convert_numeric_cols(df)
        df = add_derived_metrics(df)
        
        print(f"\n✅ 派生指标添加完成，新增列:")
        derived_cols = ['perf_fp32_tflops', 'int8_tops', 'mem_gb', 'efficiency_int8_per_w']
        for col in derived_cols:
            if col in df.columns:
                print(f"   - {col}: {df[col].notna().sum()} 非空")
        
        # 测试缺失值汇总
        missing_summary = get_missing_summary(df)
        print(f"\n缺失值 Top 5:")
        print(missing_summary.head(5))
        
    except Exception as e:
        logging.error(f"❌ 测试失败: {e}")
    
    print("\n" + "=" * 60)

