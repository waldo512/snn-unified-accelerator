# 精简仓库使用指南

## 定向 RTL 入口

- `scripts/run_verilator_unified_h40_final_composition_v1_lint.ps1`：最终顶层 lint。
- `scripts/run_verilator_unified_h40_final_composition_v1.ps1`：FC/SRNN 集成事务。
- `scripts/run_verilator_h40_array16_three_topology_graph_top_v1.ps1`：三拓扑往返切换。

前两个入口先调用 Python 生成器；生成器仅依赖标准库，源模板和对应 TB 已保留。runner 默认 MSYS2 位于 `C:\msys64`，应按本机环境调整。输出目录 `sim_work/` 已忽略。完整编译需要时间与内存。

本次整理只做文件完整性、Python 语法、文档链接、凭据模式和入口依赖静态检查，没有安装仿真工具或重跑 RTL/物理流程。原实验记录通过不等于本次重新验证通过。

## 算法源码及包外依赖

| 拓扑 | 主要入口（scripts/ 下） | 额外需要 |
|---|---|---|
| FC | run_fc_tp_h160_mask_ablation.py、run_fc_tp_h160_report.py | N-MNIST、原 traces_propagation 工程和权重 |
| SRNN | compare_srnn_shd_shifted_optimizers.py、run_srnn_shd_report.py | SHD、原 SRNN 工程、formal_m2 checkpoint 与配置 |
| SCNN | run_scnn_tp_subject_pair_ablation.py、run_scnn_tp_subject_pair_report.py | DVS-Gesture 预处理输入、标签、受试者信息、原 TP 工程和 checkpoint |

按入口参数定义配置自己的路径。原机器绝对路径仅定位来源，不是通用安装要求。部分工具导入时即需要 NumPy、PyTorch 或包外工程，不能假设只有标准库时 `--help` 也可执行。本次未锁定或安装完整训练环境。

小型 Python 源码保留在同一 scripts 目录，以维持本地 import 关系。部分汇总器、fixture 构建器和 SCNN 测试生成器仍需要原包完整结果树、测试文件或权重；只保留脚本不代表这些数据依赖也已包含。随库证据是摘要，不能直接重建所有原始统计表。

## 物理流程和原包

`constraints/core_ooc.xdc` 仅保存原核心时钟约束。本仓库没有 Vivado 工程、完整服务器打包流程或 DCP，不能一条命令复现后布线结果。需要重跑时应从原 V2 交接包取用 Vivado 2022.2 流程，单独记录新条件。

[架构说明](architecture.md)和[验证边界](validation.md)沿用原交接包说明，其中“本包已包含”等语境指原包证据范围。仓库实际保留内容见[证据说明](../evidence/README.md)。完整 timing、DRC、methodology 报告和真实数据测试输入留在原项目目录。
