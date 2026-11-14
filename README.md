# 统一 SNN 加速器：FC / SRNN / SCNN

面向 FPGA 的多拓扑脉冲神经网络加速器研究项目。通过一套 **12 × H40 / P4** 物理计算底座，以互斥调度支持 FC、SRNN、SCNN，并对选定参数实施在线适应。

本仓库是大创项目的**核心源码精简版**，用于项目备份、阅读和继续开发。采用 **统一 V2 / RUN120 R2** 基线，推荐顶层为 [`unified_h40_final_composition_top_v1.sv`](rtl/top/unified_h40_final_composition_top_v1.sv)。文件名 `v1` 是接口命名，与项目整体 V2 基线不冲突；RUN122/123 是消融对照。

## 核心工作

- **共享推理底座**：profile、descriptor、owner 和排空控制管理一套阵列。
- **选择性在线适应**：结合算法消融和状态成本，只让选定参数集合保持可写。
- **原子参数发布**：采用 `forward → optional learn → commit/detach → next forward` 顺序事务，新参数在发布后对下一次前向可见。
- **有边界的学习复用**：FC/SCNN 共享兼容的输出学习服务，SRNN 保留独立数值协调器。

| 拓扑 | 数据集与主模型 | 主更新范围 |
|---|---|---|
| FC | N-MNIST，2312 → 160 → 10 | W2 |
| SRNN | SHD，冻结 700 → 450 循环骨干，连接 450 → 256 → 20 的 M2 | M2 的 Wout |
| SCNN | DVS-Gesture，VGG-9 SNN，2048 → 11 分类器 | classifier |

三拓扑按时间复用阵列。`TOPOLOGY_MASK=1/2/4/7` 属于编译期裁剪；运行时切换由 profile/owner 协议控制。

## 阅读入口

| 目的 | 入口 |
|---|---|
| 理解共享阵列、数值格式与学习事务 | [架构说明](docs/architecture.md) |
| 定位顶层和模块分层 | [RTL 导航](rtl/README.md) |
| 查看运行入口与外部依赖 | [使用指南](docs/reproduction.md) |
| 核对成果范围和局限 | [验证边界](docs/validation.md) |
| 核对随库保留的原实验摘要 | [证据说明](evidence/README.md) |

## 已有物理实现记录

原项目采用 **Vivado 2022.2**，目标器件 **XC7VX690T-FFG1761-3**。RUN120 R2 的 OOC 核心实现记录：

| 指标 | 原报告结果 |
|---|---:|
| 时钟约束 | 15 ns，约 66.667 MHz |
| LUT / FF | 144,637 / 72,731 |
| BRAM36 等效 / DSP48E1 | 513.5 / 1,193 |
| WNS / WHS | +0.075 ns / +0.055 ns |
| TNS / THS | 0 / 0 |
| 已路由网络 | 279,676 / 279,676 |

来源：[终态摘要](evidence/run120_r2/terminal_summary.json)、[资源报告](evidence/run120_r2/utilization_post_route.rpt)、[路由状态](evidence/run120_r2/route_status_post_route.rpt)。本次整理没有重新综合或实现，完整 timing/DRC/methodology 报告与 DCP 没有上传。

同源、同器件、15 ns 条件下的**综合阶段**固定外壳对照，相较三套独立外壳资源之和，统一设计减少约 56% LUT、57% FF、62% BRAM 和 65% DSP。详见[资源对照 CSV](evidence/comparison/resource_comparison.csv)。综合对照与上表后布线结果属于不同阶段。

## 仓库结构

```text
rtl/           基线完整 RTL 编译输入和小型配套文件
scripts/       Python 算法/汇总工具、生成器、3 个 RTL 验证入口
tb/            3 个定向测试文件（含生成器源模板）
profiles/      SCNN 配置与小型控制表
constraints/   原 OOC 核心时钟约束
docs/          架构、使用与验证范围说明
evidence/      精选算法摘要、资源对照、RUN120 R2 小型报告
```

## 快速使用

阅读源码和随库摘要不需要下载数据集。Python 工具建议使用 **Python 3.10+**。RTL runner 还需要 Windows PowerShell、MSYS2、Verilator、C++ 编译器和 make；默认工具路径为 `C:\msys64`。

从仓库根目录执行：

```powershell
# 最终顶层展开检查
powershell -ExecutionPolicy Bypass -File scripts/run_verilator_unified_h40_final_composition_v1_lint.ps1

# 最终顶层 FC/SRNN 训推事务
powershell -ExecutionPolicy Bypass -File scripts/run_verilator_unified_h40_final_composition_v1.ps1

# 三拓扑无全局复位切换
powershell -ExecutionPolicy Bypass -File scripts/run_verilator_h40_array16_three_topology_graph_top_v1.ps1
```

lint 可加 `-LearningOff` 或 `-TopologyMask 1`、`2`、`4`、`7`。以上是保留的入口，**本次上传整理未执行 RTL 仿真**。完整训练和真实数据回归需要额外数据、权重及原训练工程，详见[使用指南](docs/reproduction.md)。

## 精简范围

保留核心代码、少量测试、配置和结果摘要。未收录数据集、权重、训练 checkpoint、大型 fixture、完整实验树、Vivado DCP/bitstream、综合工程、仿真缓存、重复版本、PPT 和论文 PDF。完整 RTL 约 6 MiB，保留它是为了维持编译依赖；本仓库不等于完整复现包。

## 当前验证边界

- 已有证据分别覆盖算法消融、定向 RTL 事务和 OOC 物理实现。
- 真实 SCNN 学习验证从 classifier 特征边界开始，不能扩大为全卷积网络端到端在线适应验证。
- 完整数据集端到端 RTL 在线适应精度、完整 SRNN episode 硬件延迟、上板运行和实测功耗尚未建立。
- OOC 外部 I/O 延迟未约束，WNS 裕量仅 +0.075 ns，不能视为板级时序签核。
- 三个数据集采用不同算法协议，不能直接横向比较，也不能据此主张全局最优更新范围。

本仓库用于研究资料归档，未授予额外开源许可。第三方数据、代码和模型的权利归各自权利人。
