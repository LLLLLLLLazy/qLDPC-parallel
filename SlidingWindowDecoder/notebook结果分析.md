# Notebook 结果分析

说明：这份分析基于 notebook 中已经保存的输出，没有重新大规模跑实验。由于 Monte Carlo 采样有随机波动，尤其是错误数很少时，数值应按量级和趋势理解。

## 总体结论

1. 滑动窗口思想有效：在 `Sliding Window OSD.ipynb` 中，窗口宽度 `W` 增大时逻辑错误率明显下降，但运行时间也明显增加。
2. BP+OSD 很稳：多数 circuit-level sliding-window OSD 实验中 flagged error 都是 0，说明 OSD 基本都能给出满足 syndrome 的解。
3. GDG 的定位是低延迟/迭代式窗口解码：它通常有少量 flagged/convergence failure；最后一个窗口再用 OSD 有时会显著改善，有时反而略差，说明 last-window OSD 不是无条件收益。
4. 对 noisy syndrome code，GDG 在 syndrome noise 较高时明显优于普通 OSD，这和项目里强调的 Appendix E/shortening/noisy-syndrome 问题相呼应。
5. 数据比特噪声下，GDG 的逻辑错误率可以优于 OSD10，但 notebook 也明确说当前实现不针对 data noise 的吞吐优化，运行时间未必占优。

## `Sliding Window OSD.ipynb`

这是最核心的 BP+OSD 滑动窗口实验。

| 参数 | 结果 | 运行时间 | 解读 |
| --- | ---: | ---: | --- |
| `N=144, p=0.004, W=3, F=1, shots=10000` | `254/10000`, per round `2.14e-3` | 407s | 基准 BP+OSD |
| 同上，`shorten=True` | `183/10000`, per round `1.54e-3` | 275s | shortening 在这里又快又更准 |
| 同上，CUDA-Q/NV batch，`shots=1000` | `24/1000`, per round `2.02e-3` | 7.3s | GPU batch 吞吐很强，但样本更少 |
| `N=144, p=0.004, W=4` | `131/10000`, per round `1.10e-3` | 671s | 比 W=3 好很多 |
| `N=144, p=0.004, W=5` | `108/10000`, per round `9.04e-4` | 987s | 继续变好，但耗时更高 |
| `N=144, p=0.003, W=3, shots=100000` | `351/100000`, per round `2.93e-4` | 2345s | 低物理错误率下明显更低 |
| `N=144, p=0.003, W=4` | `159/100000`, per round `1.33e-4` | 3724s | 增大窗口收益明显 |
| `N=144, p=0.003, W=5` | `119/100000`, per round `9.92e-5` | 5433s | 最好但最慢 |

关键观察：

- `W=3 -> 4 -> 5` 时，logical error per round 大致递减，说明更宽窗口包含更多时域相关信息。
- 代价是运行时间显著增加，`W=5` 通常比 `W=3` 慢 2 倍以上。
- 所有这些 OSD 结果 flagged error 都是 0，说明 OSD 的失败主要表现为逻辑错误，而不是 syndrome 不满足。

## `Sliding Window GDG.ipynb`

这个 notebook 在每个窗口使用 GDG，并可选择最后一个窗口改用 OSD。

| 参数 | last-window OSD | logical per round | flagged | 解读 |
| --- | --- | ---: | ---: | --- |
| `N=144, p=0.005, W=3, shots=5000` | 否 | `6.92e-3` | `180/5000` | 基准 GDG |
| 同上 | 是 | `6.89e-3` | `78/5000` | flagged 降很多，LER 只微降 |
| `N=288, p=0.004, W=4, shots=50000` | 否 | `1.42e-4` | `120/50000` | 很强的结果 |
| 同上 | 是 | `1.37e-4` | `53/50000` | flagged 和 LER 都小幅改善 |
| `N=144, p=0.004, W=5, F=2, shots=100000` | 否 | `6.01e-4` | `175/100000` | F=2 提交更激进 |
| 同上 | 是 | `6.83e-4` | `144/100000` | flagged 降了，但 LER 变差 |
| `N=90, p=0.003, W=5, F=2, shots=100000` | 否 | `2.43e-4` | `28/100000` | 小码低噪声 |
| 同上 | 是 | `3.18e-4` | `6/100000` | OSD 降 flagged 但 LER 变差 |
| `N=288, p=0.005, W=4, repeat=6, shots=20000` | 否 | `1.14e-3` | `129/20000` | 最后窗口较难 |
| 同上 | 是 | `7.10e-4` | `9/20000` | 这里 last-window OSD 帮助很大 |

关键观察：

- 最后窗口往往 flagged 最多，因为末尾边界条件和窗口截断更难。
- last-window OSD 几乎总能降低 flagged error，但不总能降低 logical error。说明“满足 syndrome”不等于“逻辑上正确”，路径度量/先验排序仍然关键。
- 对 `N=288, p=0.005, repeat=6` 这种最后窗口困难的设置，last-window OSD 的收益很明显：per-round LER 从 `1.14e-3` 降到 `7.10e-4`。
- 对 `F=2` 的设置，last-window OSD 反而可能变差，可能是前面窗口提交区域较大，早期决策错误已经固定，最后窗口无法纠正。

## `Data noise.ipynb`

这个 notebook 主要展示 GDG 在 data qubit noise 下的用法，并和 BP+OSD 比较。

保存的 3 组 `p=0.02, shots=10000000` 结果：

| decoder | LER 范围 | 运行时间 |
| --- | ---: | ---: |
| OSD0 | `1.20e-5` 到 `1.21e-5` | 包含在 OSD 总时间中 |
| OSD10 | `5e-7` 到 `1.3e-6` | 560s 到 597s |
| GDG | `1e-7` 到 `2e-7` | 763s 到 766s |

关键观察：

- GDG 的 LER 在这些样本里优于 OSD10，大约低一个小倍数。
- 但 GDG 更慢；notebook 自己也说明当前 GDG 实现为 circuit-level noise 写得更多，data noise 下内存复制/预处理不是最优。
- flagged 基本为 0 或 1/10000000，说明 GDG 在这些 data-noise 设置下非常稳定。

## `Syndrome code.ipynb`

这个 notebook 研究 noisy syndrome code，也就是把 syndrome measurement error 作为额外变量节点并入解码。

`[[288,12,18]]`，`p_data=0.03`：

| `p_syndrome` | OSD LER | GDG LER | 解读 |
| ---: | ---: | ---: | --- |
| `1e-5` | `2.1e-5` | `2.0e-5` | 两者接近 |
| `1e-4` | `3.02e-4` | `4.4e-5` | GDG 约好 6.9 倍 |
| `1e-3` | `2.0164e-2` | `1.356e-3` | GDG 约好 14.9 倍 |

但 notebook 同时指出一个结构性问题：

- matrix A 的 column span 中有 216 个 weight-two syndrome codewords。
- GDG 只成功解出其中 2 个：`(0,72)` 和 `(1,73)`。

这说明 GDG 在随机 noisy-syndrome 采样上表现很好，但对某些低重量 syndrome-code codeword 存在明确失败模式；这也是 Appendix E 里提到的问题来源。

`[[254,28,14<=d<=20]]`：

- `p_data=0.01, p_syndrome=1e-5/1e-4`，GDG 在 `10^7` shots 下只有 `0` 到 `2` 个 logical errors，LER 为 `0` 到 `2e-7`。
- 这组结果很强，但错误数太少，置信区间相对宽；它更适合说明“量级很低”，不宜过度比较 `0`、`1e-7`、`2e-7` 的细微差别。

## `IBM.ipynb`

目标是复现 IBM 论文中的 `[[72,12,6]]` circuit-level 结果。

| 参数 | 结果 | 运行时间 |
| --- | ---: | ---: |
| `p=0.004, shots=10000` | `76/10000`, per round `6.36e-4` | 1658s |
| `p=0.003, shots=100000` | `77/100000`, per round `6.42e-5` | 4199s |
| `p=0.004, shorten=True, shots=10000` | `90/10000`, per round `7.53e-4` | 429s |

关键观察：

- 从 `p=0.004` 到 `p=0.003`，per-round LER 大约降低 10 倍，说明对物理错误率很敏感。
- `shorten=True` 速度快约 3.9 倍，但这里 logical error 略差：`76 -> 90`，即 per-round 从 `6.36e-4` 到 `7.53e-4`。
- notebook markdown 也提醒：它对 IBM arXiv v1/v2 和 circuit 实现细节敏感，因此这些数值更适合作为复现实验参考，而不是最终 benchmark。

## `SHYPS.ipynb`

`r=3, p=0.001, num_repeat=4, shots=20000, osd_order=0`。

| 模式 | Logical Errors | per round | 运行时间 |
| --- | ---: | ---: | ---: |
| sliding window, `W=3,F=1` | `170/20000` | `2.13e-3` | 54.7s |
| global decoding | `187/20000` | `2.35e-3` | 30.7s |

关键观察：

- 两者都 flagged 0。
- sliding window 的 LER 略好，但运行更慢。
- 差距只有 17 个 errors/20000，统计上不能说特别稳；更像是“滑窗没有明显损失，甚至略好”的初步迹象。
- check matrix 是 `(105, 833)`，row weight 较大，说明这个电路/码的 Tanner graph 和 BB code 的窗口结构不完全一样。

## `Misc.ipynb`

这是多个 decoder/code 的杂项实验。

### BP4+OSD

`GHP_n882_k24, Depolarize(0.1), shots=100000`：

- BP4 flagged：`18372/100000`
- BP4+OSD0 LER：`7.7e-4`
- BP4+OSD LER：`2.2e-4`
- OSD 明显修正了 BP4 的不收敛/残余错误。

### 2BGA codes

`N=96`：

| p | OSD0 | OSD10 | GDG |
| ---: | ---: | ---: | ---: |
| `0.03`, `shots=100000` | `1.356e-2` | `4.45e-3` | `3.14e-3` |
| `0.02`, `shots=1000000` | `1.844e-3` | `4.85e-4` | `2.69e-4` |

GDG 优于 OSD10，但运行时间略高。

### CAMEL

CAMEL 示例中：

- BP4 flagged：`26/100000`
- CAMEL LER：`2.6e-4`
- 运行时间：2447s

notebook 说明没有完全复现原论文，因为这里用了 normalized min-sum，而原工作用 sum-product。

### BPGD / GDG / OSD 比较

`N=882, p=0.04, shots=1000000`：

- Extra decoder 1：`3.4e-5`
- Extra decoder 2：`5.51e-4`
- OSD0：`2.6e-5`
- OSD10：`1e-6`
- GDG：`2e-5`

这里 OSD10 最强，但最耗时；GDG 不是最优 LER，但保持了较低错误率。

## `Round Analysis.ipynb`

这个 notebook 不是结果 benchmark，主要解释滑动窗口矩阵结构。

核心信息：

- 示例码：`BB_n144_k12`
- check matrix：`(360, 3024)`
- row weight：最大 35，最小 16
- column weight：最大 6，最小 2

重要解释：

- 每一列代表 syndrome measurement circuit 中一种 fault realization。
- 每一行代表一个 detector，越靠上的行对应越早的 syndrome round。
- `(W,F)` sliding window 表示窗口覆盖连续 `W` 轮 detector，每次向前移动 `F` 轮。
- 窗口右侧某些列可以合并成 identity matrix 的 virtual nodes，其 prior 是相关列 prior 的和；这就是 noisy syndrome decoding 中常见的技巧。

## 推荐下一步

1. 若目标是复现论文主线，优先看 `Round Analysis.ipynb` -> `Sliding Window OSD.ipynb` -> `Sliding Window GDG.ipynb`。
2. 若目标是比较 decoder，重点重跑同一组参数下的 OSD、GDG、GDG+last-window-OSD，确保 shots、`W/F`、`num_repeat` 完全一致。
3. 若目标是调参，优先扫 `W`、`F`、`max_iter`、`max_step`、`max_tree_depth/max_side_depth`，并同时记录 flagged 和 logical error，因为 flagged 降低不一定代表 logical error 降低。
4. 若目标是延迟评估，应单独测 per-sample decode latency；当前很多 notebook 输出的是整批总时间，不能直接代表 worst-case latency。
