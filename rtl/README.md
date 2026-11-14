# RTL 源码导航

推荐顶层：[unified_h40_final_composition_top_v1.sv](top/unified_h40_final_composition_top_v1.sv)。保留原 V2 基线全部 443 个 SystemVerilog 编译输入，避免精简时破坏编译与生成器依赖。

| 目录 | 用途 |
|---|---|
| top | 统一顶层、拓扑适配与集成 |
| tiles | H40/P4 计算与神经元逻辑 |
| memory | 参数、状态、特征和工作集存储 |
| learning | 观察、贡献计算、RMW 与提交 |
| control | owner、调度和几何控制 |
| common | 公共算术及弹性传输 |

实际物理阵列是 12×H40/P4；部分 array16、candidate、旧尺寸文件名是历史标识，不能据此推导实际实例数。详见[架构说明](../docs/architecture.md)。TOPOLOGY_MASK 是编译期裁剪，运行时拓扑由 profile/descriptor/owner 控制。
