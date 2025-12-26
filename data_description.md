# 数据集说明文档（Data Description）

**作者**：HIT_ Jimmy（哈尔滨工业大学）  
**数据来源**：[Epoch AI - Machine Learning Hardware](https://epoch.ai/data/machine-learning-hardware)  
**数据文件**：`ml_hardware.csv`（286 条记录，36 个字段）  
**数据时间跨度**：2007-2025（按硬件发布日期）

---

## 📋 1. 数据集来源与授权

### 1.1 数据集简介

本数据集由 **Epoch AI** 整理并公开发布，收录了 **170+ AI 加速器**（GPU、TPU 等）的详细规格参数，涵盖算力、显存、功耗、价格、制程等核心指标。Epoch AI 是一个专注于 AI 发展趋势研究的独立机构，其数据被 OpenAI、DeepMind 以及多国政府机构引用。  ​​

### 1.2 授权与引用

- **授权协议**：[Creative Commons Attribution (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **引用格式**：
  
  ```
  Epoch AI, 'Data on Machine Learning Hardware'. Published online at epoch.ai. 
  Retrieved from 'https://epoch.ai/data/machine-learning-hardware' [online resource].
  ```
  
- **BibTeX 引用**：
  
  ```bibtex
  @misc{EpochMachineLearningHardware2024,
    title = {Data on Machine Learning Hardware},
    author = {{Epoch AI}},
    year = {2024},
    month = {10},
    url = {https://epoch.ai/data/machine-learning-hardware}
  }
  ```

### 1.3 数据版本说明

- 数据集包含 `Last modified` 字段（最后修改时间），本项目使用的版本截至 **2025-12-05**
- Epoch AI 会持续更新数据，建议定期从官网下载最新版本

---

## 📊 2. 字段详细说明（按功能分组）

本数据集共 **36 个字段**，下面按照"基本信息 → 算力 → 内存/互联 → 功耗/能效 → 制程/结构 → 价格/衍生指标"分组说明。

### 2.1 基本信息字段

| 字段名 | 中文含义 | 数据类型 | 示例值 | 说明与用途 |
|--------|----------|----------|--------|------------|
| `Hardware name` | 硬件名称 | 字符串 | "NVIDIA H200 SXM" | 唯一标识硬件型号，用于所有图表标注 |
| `Manufacturer` | 制造厂商 | 字符串 | "NVIDIA" | 用于按厂商分组对比（NVIDIA/AMD/Google/Intel/AWS/Meta 等） |
| `Type` | 硬件类型 | 字符串 | "GPU" / "TPU" | GPU（图形处理器）或 TPU（张量处理单元），用于分类分析 |
| `Release date` | 发布日期 | 日期字符串 | "2024-11-18" | 格式：YYYY-MM-DD，用于时间序列趋势分析 |
| `Notes` | 备注说明 | 文本（多行） | （见数据） | 可能包含 TDP 来源、TODO 事项、外部链接等；需处理多行引号 |

---

### 2.2 算力（Performance）字段 ⭐ 核心指标

AI 硬件的算力通常用 **FLOP/s**（浮点运算/秒）或 **OP/s**（整数运算/秒）衡量，不同数值格式的算力差异可达数十倍甚至百倍。

| 字段名 | 中文含义 | 单位 | 示例值 | 适用场景 | 量级换算 |
|--------|----------|------|--------|----------|----------|
| `FP64 (double precision) performance (FLOP/s)` | 双精度算力 | FLOP/s | 极少提供 | 科学计算（非 AI 主流） | 1 TFLOP/s = 10¹² FLOP/s |
| `FP32 (single precision) performance (FLOP/s)` | 单精度算力 | FLOP/s | 1.83×10¹⁴ | 传统训练/推理基准 | 183 TFLOP/s |
| `TF32 (TensorFloat-32) performance (FLOP/s)` | TF32 算力 | FLOP/s | 6.71×10¹⁴ | NVIDIA 特有，训练加速 | 671 TFLOP/s |
| `FP16 (half precision) performance (FLOP/s)` | 半精度算力 | FLOP/s | - | 通用半精度 | - |
| `Tensor-FP16/BF16 performance (FLOP/s)` | Tensor Core 加速的 FP16/BF16 | FLOP/s | 6.71×10¹⁴ | **训练主口径** | 671 TFLOP/s |
| `FP8 performance (FLOP/s)` | 8 位浮点算力 | FLOP/s | 2.517×10¹⁵ | **新一代推理/训练** | 2517 TFLOP/s |
| `FP4 performance (FLOP/s)` | 4 位浮点算力 | FLOP/s | 2.517×10¹⁵ | 极低精度推理（实验性） | 2517 TFLOP/s |
| `INT16 performance (OP/s)` | 16 位整数算力 | OP/s | - | 少见 | - |
| `INT8 performance (OP/s)` | 8 位整数算力 | OP/s | - | **推理主口径** | 1 TOP/s = 10¹² OP/s |
| `INT4 performance (OP/s)` | 4 位整数算力 | OP/s | - | 极低精度推理 | - |

#### 📐 单位换算示例

- **FLOP/s → TFLOP/s（TeraFLOP/s）**：除以 10¹²  
  例：`6.71×10¹⁴ FLOP/s = 671 TFLOP/s`

- **FLOP/s → PFLOP/s（PetaFLOP/s）**：除以 10¹⁵  
  例：`2.517×10¹⁵ FLOP/s = 2.517 PFLOP/s`

- **OP/s → TOP/s（TeraOP/s）**：除以 10¹²

#### 🎯 不同精度的适用场景（通俗解释）

| 精度 | 适用场景 | 通俗类比 |
|------|----------|----------|
| **FP32** | 传统深度学习训练/推理 | 相当于"标准清晰度"，兼容性最好但速度较慢 |
| **FP16/BF16** | 现代训练主流（混合精度） | "高清"画质，速度快且精度损失可接受 |
| **FP8** | 新一代推理/训练 | "超高清 HDR"，速度更快，新硬件才支持 |
| **INT8** | 推理部署（量化后模型） | "高效压缩格式"，速度极快但需要精心调优 |
| **INT4/FP4** | 极致推理优化 | "极限压缩"，用于边缘设备或超大模型 |

**为什么低精度算力更高？**  

- 低精度计算需要的晶体管/电路更少 → 同一个芯片可以并行更多计算单元  
- 例如：1 个 FP32 单元 ≈ 2 个 FP16 单元 ≈ 4 个 INT8 单元  
- 因此 INT8 算力常常是 FP32 的 4-16 倍

---

### 2.3 内存与互联字段

| 字段名 | 中文含义 | 单位 | 示例值 | 换算与说明 |
|--------|----------|------|--------|------------|
| `Memory (bytes)` | 显存容量 | 字节 | 1.92×10¹¹ | 192 GB（除以 10⁹）；决定能跑多大的模型 |
| `Memory bandwidth (byte/s)` | 显存带宽 | 字节/秒 | 7.37×10¹² | 7.37 TB/s（除以 10¹²）；决定数据传输速度 |
| `Intranode bandwidth (byte/s)` | 节点内互联带宽 | 字节/秒 | - | NVLink/自研互联；多卡通信速度 |
| `Internode bandwidth (bit/s)` | 节点间互联带宽 | **比特/秒** | - | ⚠️ 注意单位是 bit/s，需除以 8 转 byte/s |

**⚠️ 重要区别**：

- `Memory bandwidth` / `Intranode bandwidth` 单位是 **byte/s**
- `Internode bandwidth` 单位是 **bit/s**（1 byte = 8 bits）

**瓶颈分析指标**（派生）：

- `compute_to_mem_ratio` = 算力 / 显存带宽  
  若比值过大 → "算力强但显存跟不上"（可能出现 memory-bound 瓶颈）

---

### 2.4 功耗与能效字段

| 字段名 | 中文含义 | 单位 | 示例值 | 说明 |
|--------|----------|------|--------|------|
| `TDP (W)` | 热设计功耗 | 瓦特 | 700 | 硬件运行时的最大功耗（决定数据中心电力/散热成本） |
| `Energy efficiency` | 能效 | FLOP/s/W | 9.59×10¹¹ | **已计算好的能效**（通常是 Max performance / TDP） |

**能效口径说明**：
- 数据集的 `Energy efficiency` = `Max performance` / `TDP`
- 单位：FLOP/s/W 或 OP/s/W（看 Max performance 用的哪个口径）
- 能效越高 → 同样功耗下算力越强 → 数据中心成本越低

---

### 2.5 制程与硬件结构字段

| 字段名 | 中文含义 | 单位 | 示例值 | 说明 |
|--------|----------|------|--------|------|
| `Foundry` | 代工厂 | 字符串 | "TSMC" | 台积电（TSMC）/ 三星（Samsung）/ Intel 等 |
| `Process size (nm)` | 制程工艺 | 纳米 | 5 | 数字越小 → 晶体管越密集 → 性能/能效越高 |
| `Transistors (millions)` | 晶体管数量 | 百万 | - | 芯片复杂度指标 |
| `Die Size (mm²)` | 芯片面积 | 平方毫米 | - | 面积越大 → 成本越高（良率下降） |
| `Tensor cores` | Tensor Core 数量 | 整数 | - | NVIDIA 特有，专门加速矩阵运算的单元 |
| `Base clock (MHz)` | 基础频率 | 兆赫兹 | - | 芯片的默认运行频率 |
| `Boost clock (MHz)` | 加速频率 | 兆赫兹 | - | 短时冲刺的最高频率 |
| `Memory clock (MHz)` | 显存频率 | 兆赫兹 | - | 显存运行频率 |
| `Memory bus (bit)` | 显存位宽 | 比特 | - | 位宽越大 → 带宽越高 |

---

### 2.6 价格与引用字段

| 字段名 | 中文含义 | 单位 | 示例值 | 说明 |
|--------|----------|------|--------|------|
| `Release price (USD)` | 发布价格 | 美元 | - | ⚠️ **缺失严重**，仅部分消费级 GPU 有价格 |
| `Source for the price` | 价格来源 | URL | - | 价格的参考链接 |
| `Link to datasheet` | 数据手册链接 | URL | （AWS 文档） | 官方规格说明，可追溯验证 |
| `ML models` | 关联的 ML 模型 | 文本 | - | 用该硬件训练过的知名模型（可能为空） |

**⚠️ 价格数据局限性**：
- 数据中心级硬件（H200/TPU v7 等）不对外零售，无公开价格
- 仅消费级 GPU（RTX 系列等）有参考价格
- 价格分析需声明"仅限有价格子集"

---

### 2.7 衍生/汇总字段（Epoch AI 已计算）

| 字段名 | 中文含义 | 计算口径 | 说明 |
|--------|----------|----------|------|
| `Max performance` | 最大算力 | FLOP/s 或 OP/s | **取所有精度中的最大值**（通常是 FP8/INT8） |
| `ML OP/s` | ML 相关算力 | FLOP/s 或 OP/s | 可能优先选择 Tensor-FP16/BF16 或 INT8 |
| `Total processing performance (bit-OP/s)` | 总处理性能 | bit-OP/s | 加权综合指标（考虑不同位宽） |
| `Price-performance` | 性价比 | FLOP/s/USD | `Max performance` / `Release price`（若价格存在） |
| `Last modified` | 最后修改时间 | 日期时间 | 数据维护时间戳 |

**⚠️ 使用注意**：
- `Max performance` 可能混合了不同精度口径（FP8 vs INT8 vs FP16）
- 对比时需**统一口径**（如只用 FP32，或只用 INT8）
- 本项目会显式声明"对比口径"并在图表标题注明

---

## 🧹 3. 数据质量与清洗策略

### 3.1 缺失值情况（初步分析）

| 字段 | 预期缺失情况 | 处理策略 |
|------|--------------|----------|
| `Release price (USD)` | **严重缺失**（~80%） | 仅对非空子集做价格分析，并声明局限 |
| `Intranode/Internode bandwidth` | 中度缺失 | 跳过或标注"未披露" |
| `FP64/INT16/INT4` | 新硬件不支持 | 正常，按设计缺失 |
| `Foundry/Process size` | 少量缺失 | 保留缺失值，不强行填充 |
| `Notes` | 全部非空（但内容差异大） | 需鲁棒解析多行文本 |

### 3.2 异常值检测

- **TDP = 0 或过大（>2000W）**：可能是数据错误或特殊设计，需标注
- **Release date 缺失**：无法做时间序列分析，考虑剔除
- **算力为 0 但其他字段正常**：可能是"规格未披露"，保留但不参与算力对比

### 3.3 数据类型转换

| 字段 | 原始类型 | 转换后类型 | 方法 |
|------|----------|------------|------|
| `Release date` | 字符串 | `datetime64[ns]` | `pd.to_datetime()` |
| 算力/带宽字段 | 科学计数法字符串 | `float64` | pandas 自动识别（但需检查溢出） |
| `Memory (bytes)` | `float64` | `float64` | 读取后除以 10⁹ 转 GB |

### 3.4 重复值检查

- 按 `Hardware name` + `Release date` 去重
- 若同一硬件有多个条目（不同配置/版本），保留或标注版本差异

---

## 🧮 4. 本项目的派生指标定义

为便于分析与可视化，本项目会派生以下字段（在脚本中计算）：

| 派生字段名 | 公式 | 单位 | 用途 |
|-----------|------|------|------|
| `perf_fp32_tflops` | `FP32 performance / 1e12` | TFLOP/s | 标准化算力（训练基准） |
| `perf_fp16_tflops` | `Tensor-FP16/BF16 / 1e12` | TFLOP/s | 现代训练主口径 |
| `perf_fp8_tflops` | `FP8 performance / 1e12` | TFLOP/s | 新一代推理/训练 |
| `int8_tops` | `INT8 performance / 1e12` | TOP/s | 推理主口径 ⭐ |
| `mem_gb` | `Memory / 1e9` | GB | 显存容量（便于阅读） |
| `mem_bw_tbs` | `Memory bandwidth / 1e12` | TB/s | 显存带宽（标准化） |
| `efficiency_int8_per_w` | `INT8 performance / TDP` | TOP/s/W | **推理能效**（本项目重点） |
| `compute_to_mem_ratio` | `Max performance / Memory bandwidth` | 无量纲 | 算力-带宽匹配度（>某阈值可能 memory-bound） |
| `release_year` | 从 `Release date` 提取年份 | 整数 | 用于按年分组统计 |

---

## ❓ 5. 研究问题（RQ）与字段/图表映射

本项目围绕"**AI 推理硬件的算力演进与选型建议**"展开，重点关注 **INT8/FP8 推理能力**。

| 研究问题 | 涉及字段 | 计划图表 | 预期发现 |
|----------|----------|----------|----------|
| **RQ0: 数据理解**<br>不同精度口径的含义与量级差异 | FP32/FP16/FP8/INT8 | 口径对比表<br>算力量级分布图 | INT8 通常是 FP32 的 4-16 倍 |
| **RQ1: 算力演进**<br>2012-2025 年推理算力（INT8）增长趋势 | `INT8 performance`<br>`Release date`<br>`Manufacturer` | 时间序列散点图（log 轴）<br>按厂商分组趋势 | NVIDIA 领先；2020 后增速加快 |
| **RQ2: 能效对比**<br>同等功耗下谁的推理能力更强？ | `INT8 performance`<br>`TDP`<br>`Energy efficiency` | 能效排名条形图<br>能效-算力散点图 | 新架构（H200/TPU v7）能效显著提升 |
| **RQ3: 显存/带宽匹配**<br>算力增长是否伴随显存/带宽同步增长？ | `Memory`<br>`Memory bandwidth`<br>`INT8 performance` | 气泡图（x=带宽，y=算力，size=显存）<br>算力-带宽比分布 | 部分高算力硬件存在带宽瓶颈 |
| **RQ4: 价格与性价比**<br>消费级 GPU 推理性价比对比 | `Release price`<br>`INT8 performance` | 价格-性能散点图<br>性价比排名（仅非空子集） | 仅限有价格硬件；数据中心硬件不适用 |
| **RQ5: 选型建议**<br>训练 vs 推理场景的硬件推荐 | 综合上述字段 | 决策矩阵表格<br>推荐清单 | 训练优先 FP16 能效；推理优先 INT8 性价比 |

---

## 📚 6. 参考资料与延伸阅读

### 6.1 数值格式详解

- **FP32/FP16/BF16**：[NVIDIA Mixed-Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- **INT8 量化**：[TensorFlow Lite Post-Training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- **FP8 标准**：[NVIDIA Hopper Architecture FP8](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

### 6.2 硬件选型指南

- [Epoch AI Trends in ML Hardware](https://epoch.ai/blog/trends-in-machine-learning-hardware)
- [Stanford MLSys Hardware Lecture](https://cs229.stanford.edu/proj2022spr/report/116.pdf)

---

## ✅ 7. 数据完整性声明

- 本数据集由 **Epoch AI** 维护，数据来源于各硬件厂商的官方规格手册、学术论文、行业报告等
- 部分字段（如 TDP、价格）可能基于合理推测或第三方披露，已在 `Notes` 中注明来源
- 本项目使用的数据版本：**2025-12-05**，后续如有更新请访问 [Epoch AI 官网](https://epoch.ai/data/machine-learning-hardware)

---

**文档更新日期**：2025-12-26  
**联系方式**：xiao.jm44@qq.com

