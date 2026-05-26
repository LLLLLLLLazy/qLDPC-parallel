# TSAE + gated interface + joint-B-DP 策略说明

本文档总结当前 `ParallelWindowDecoder` 中最强的小窗口组合策略：

```text
TSAE + gated interface branch + joint-B-DP
```

它的目标是在不直接扩大单个 A/B 解码窗口的前提下，降低 parallel window decoding 的 LER。

## 1. 背景问题

普通 parallel A/B window decoder 的基本结构是：

```text
A windows 先并行解码并提交 boundary / seam 变量；
B windows 再在固定 A commit 的基础上并行解码。
```

这种方式很快，但问题是：

```text
A 的 hard decision 一旦提交错误，
相邻 B window 往往只能用 noisy boundary 虚拟列吸收 residual，
最终可能留下 physical residual 或 logical error。
```

实验中常见现象：

```text
plain parallel flagged 多；
TSAE 能显著降低 flagged；
但 TSAE 在 high-p 下仍不如 sliding / 大窗口 a9/b9。
```

直接扩大窗口，例如 `a_solve=9,b_width=9`，可以改善物理闭合和 LER，但单个局部矩阵变大，不符合“小核并行”的资源目标。

因此当前策略的核心思想是：

```text
不扩大单个小窗口；
而是让多个小窗口候选通过联合选择，获得接近大窗口的信息。
```

## 2. 策略概览

最终策略由三层组成：

```text
1. TSAE:
   用多个 shifted A 小窗口产生同一 commit 区域的候选。

2. joint-B-DP:
   对相邻 A candidate pair 实际运行 B 小窗口解码，
   用 B 是否能物理闭合作为 DP 边权，选择整条 A/B path。

3. gated interface branch:
   只有当轻量 joint-B-DP 仍留下 physical residual 时，
   才对对应 shot 触发 interface bit branching 并重跑 joint-B-DP。
```

简化流程：

```text
shifted A candidates
        |
        v
light joint-B-DP over A/B chain
        |
        +-- physical residual = 0 --> accept
        |
        +-- physical residual > 0 --> interface branch for that shot only
                                      rerun joint-B-DP
                                      accept if residual/cost improves
```

### 2.1 数学形式化与推导

本节把当前策略写成一个近似 MAP / ML 解码问题。对每个 shot，探测事件 syndrome 记为：

```math
s \in \mathbb{F}_2^m
```

物理错误向量记为：

```math
e \in \mathbb{F}_2^n
```

校验矩阵为：

```math
H \in \mathbb{F}_2^{m \times n}
```

理想解码要求：

```math
H e = s \pmod 2
```

如果第 `j` 个物理列的先验错误率为 `p_j`，独立 Bernoulli 近似下，某个 error pattern 的负 log-likelihood 可以写成：

```math
\mathcal{C}(e)
= -\log P(e)
= \text{const}
  + \sum_{j=1}^{n} e_j \log \frac{1-p_j}{p_j}
```

去掉与 `e` 无关的常数，定义列代价：

```math
w_j = \log \frac{1-p_j}{p_j}
```

则全局 ML / MAP 解码可写为：

```math
e^\star
= \arg\min_{e \in \mathbb{F}_2^n}
  \sum_j w_j e_j
\quad
\text{s.t.}
\quad
H e = s \pmod 2
```

这个问题直接在全局矩阵上求解太大；sliding window 和 parallel window 都是在做局部近似。

#### 2.1.1 A/B 分块后的变量分解

在 buffer-aligned A/B 排布中，物理变量按 ownership 大致分为：

```text
A commit variables: x_1, x_2, ..., x_M
B interior variables: y_1, y_2, ..., y_{M-1}
```

其中：

```text
x_i: 第 i 个 A window 最终提交的 seam / boundary 变量；
y_i: 第 i 个 B window 内部变量。
```

第 `i` 个 B window 连接相邻两个 A commit：

```text
x_i -- B_i -- x_{i+1}
```

把第 `i` 个 B window 的校验方程单独写出来：

```math
H^{(i)}_{B} y_i
+ H^{(i)}_{L} x_i
+ H^{(i)}_{R} x_{i+1}
= s^{(i)}_B
\pmod 2
```

等价地，如果相邻 A candidate 已经固定为 `(x_i, x_{i+1})`，则 B window 看到的有效 syndrome 是：

```math
\tilde{s}^{(i)}_B(x_i, x_{i+1})
= s^{(i)}_B
  + H^{(i)}_{L} x_i
  + H^{(i)}_{R} x_{i+1}
\pmod 2
```

B window 的局部解码问题变成：

```math
y_i^\star(x_i, x_{i+1})
= \arg\min_{y_i}
  \mathcal{C}^{(i)}_B(y_i)
\quad
\text{s.t.}
\quad
H^{(i)}_{B} y_i
= \tilde{s}^{(i)}_B(x_i, x_{i+1})
\pmod 2
```

这就是 joint-B-DP 的关键：它不是只看 `x_i` 或 `x_{i+1}` 的局部好坏，而是看这对 A commit 是否能让中间 B window 实际闭合。

#### 2.1.2 TSAE 候选集

普通 parallel 对每个 A window 只产生一个 commit：

```math
x_i = \pi_i(\operatorname{Dec}_A(W_i, s_{W_i}))
```

其中：

```text
W_i: 第 i 个 A solve window
\pi_i: 从局部解中投影出 commit columns
```

TSAE 对 interior A window 使用多个 shifted solve windows。设 offset 集合为：

```math
D_i = \{-2, 0, +2\}
```

则第 `i` 个 A window 的 candidate set 是：

```math
\mathcal{K}_i
= \left\{
    x_{i,\delta}
    =
    \pi_i\left(
      \operatorname{Dec}_A(W_i+\delta, s_{W_i+\delta})
    \right)
    :
    \delta \in D_i
  \right\}
```

边界 A window 的合法 offset 可能只有 `0`：

```math
\mathcal{K}_1 = \{x_{1,0}\},
\quad
\mathcal{K}_M = \{x_{M,0}\}
```

TSAE 的本质是把全局搜索空间从一个 hard commit 扩展为一个小的候选笛卡尔积：

```math
\mathcal{K}
= \mathcal{K}_1
  \times \mathcal{K}_2
  \times \cdots
  \times \mathcal{K}_M
```

如果只做 greedy selection，每个 `x_i` 独立选择，无法保证相邻 A commit 与 B window 同时一致。

#### 2.1.3 B edge potential

joint-B-DP 把每个 B window 看成连接两个 A candidate 的边势函数。

对任意：

```math
x_i \in \mathcal{K}_i,
\quad
x_{i+1} \in \mathcal{K}_{i+1}
```

先运行 B decoder：

```math
\hat{y}_i(x_i, x_{i+1})
=
\operatorname{Dec}_B
\left(
  H^{(i)}_B,
  \tilde{s}^{(i)}_B(x_i, x_{i+1})
\right)
```

局部 B 代价是：

```math
\mathcal{C}^{(i)}_B(\hat{y}_i)
=
\sum_{j \in B_i} w_j \hat{y}_{i,j}
```

如果启用了 `b_noisy_boundary`，B decoder 实际上在 augmented matrix 上解：

```math
\left[
  H^{(i)}_{B,\text{phys}}
  \;\middle|\;
  H^{(i)}_{B,\text{virtual}}
\right]
\begin{bmatrix}
  y_i \\
  z_i
\end{bmatrix}
=
\tilde{s}^{(i)}_B
\pmod 2
```

其中 `z_i` 是 noisy boundary 虚拟列，只用于局部吸收 residual，不能成为最终 physical correction。

因此 joint-B-DP 使用 physical residual：

```math
r_i(x_i, x_{i+1})
=
\left\|
  H^{(i)}_{B,\text{phys}}
  \hat{y}_i
  +
  \tilde{s}^{(i)}_B(x_i, x_{i+1})
\right\|_0
```

如果 `r_i = 0`，说明这对 A commit 允许 B window 用真实 physical variables 闭合。

如果 `r_i > 0`，说明 B decoder 即使能借助 noisy boundary 虚拟列局部满足 augmented equation，physical correction 本身仍没有闭合。

最终定义 B edge potential：

```math
\psi_i(x_i, x_{i+1})
=
\mathcal{C}^{(i)}_B(\hat{y}_i)
+
\lambda r_i(x_i, x_{i+1})
```

其中：

```math
\lambda = \texttt{a\_shifted\_joint\_flag\_penalty}
```

默认 `lambda=1000`，远大于普通 bit likelihood cost，所以优化顺序近似为：

```text
先最小化 physical residual；
再在 residual 相同的候选之间最小化 likelihood cost。
```

#### 2.1.4 A node potential

每个 A candidate 也有局部 cost：

```math
\phi_i(x_i)
```

它来自 A shifted local decoder 给出的局部 error cost。直观上：

```math
\phi_i(x_i)
\approx
\min_{u_i:\,\pi_i(u_i)=x_i}
\mathcal{C}^{(i)}_A(u_i)
```

其中 `u_i` 是 A solve window 的完整局部解，`x_i` 是其 commit projection。

#### 2.1.5 联合目标函数

在 TSAE candidate set 上，joint-B-DP 近似求解：

```math
(x_1^\star,\ldots,x_M^\star)
=
\arg\min_{x_i \in \mathcal{K}_i}
\left[
  \sum_{i=1}^{M} \phi_i(x_i)
  +
  \sum_{i=1}^{M-1} \psi_i(x_i,x_{i+1})
\right]
```

这是一个链式 pairwise Markov random field：

```text
node potential: A local cost
edge potential: B actual decode cost + physical residual penalty
```

因为图是链，最优解可以用动态规划精确求出。

#### 2.1.6 DP 递推

令第 `i` 个 A window 的候选编号为：

```math
k_i \in \{1,\ldots,K_i\}
```

对应候选：

```math
x_i^{(k_i)} \in \mathcal{K}_i
```

定义：

```math
F_i(k)
=
\text{从 } A_1 \text{ 到 } A_i
\text{ 且 } A_i \text{ 选第 } k \text{ 个候选时的最小总代价}
```

初始化：

```math
F_1(k)
=
\phi_1(x_1^{(k)})
```

递推：

```math
F_i(k)
=
\phi_i(x_i^{(k)})
+
\min_{h}
\left[
  F_{i-1}(h)
  +
  \psi_{i-1}
  \left(
    x_{i-1}^{(h)},
    x_i^{(k)}
  \right)
\right]
```

最终：

```math
k_M^\star
=
\arg\min_k F_M(k)
```

再通过 backpointer 恢复：

```math
(k_1^\star,\ldots,k_M^\star)
```

这个递推正是当前 `joint-B-DP` 的数学形式。

#### 2.1.7 与大窗口解码的关系

如果某个大窗口解码器能在局部大矩阵中看到更完整的上下文，它实际是在更大的变量空间中求近似 ML：

```math
e^\star_{\text{large}}
=
\arg\min_e \mathcal{C}(e)
\quad
\text{s.t.}
\quad
H_{\text{large}}e=s_{\text{large}}
```

TSAE + joint-B-DP 不扩大单个矩阵，而是构造一个小候选空间：

```math
\mathcal{K}_1 \times \cdots \times \mathcal{K}_M
```

然后在这个候选空间内做精确链式优化。

因此它可以理解为：

```text
用多个 shifted 小窗口采样大窗口可能给出的 commit；
用 B edge decode 检查相邻 commit 是否可共同闭合；
用 DP 在这些候选中选最一致的一条 path。
```

如果全局最优大窗口解的 A commit projection 恰好包含在候选集中：

```math
\left(
  x_1^\star,
  \ldots,
  x_M^\star
\right)
\in
\mathcal{K}_1
\times \cdots \times
\mathcal{K}_M
```

并且 B edge decoder 能给出对应的低 residual / 低 cost 解，那么 joint-B-DP 就有机会恢复接近大窗口的选择。

如果候选集中根本没有正确 logical class，则再好的 DP selector 也无法完全修复。这就是 high-p / high-repeat 下仍然可能失败的原因。

#### 2.1.8 interface branch 的数学解释

interface branch 是对 TSAE candidate set 的条件扩展。

设第 `i` 个 interior A commit 的 interface columns 集合为：

```math
I_i \subseteq \operatorname{cols}(x_i)
```

每个 branch assignment 是：

```math
b \in \mathbb{F}_2^{|I_i|}
```

在 A 局部解码时强制：

```math
x_i|_{I_i} = b
```

得到 branch candidate：

```math
x_{i,\delta,b}
=
\pi_i
\left(
  \operatorname{Dec}_A
  (
    W_i+\delta,
    s_{W_i+\delta}
    \mid
    x_i|_{I_i}=b
  )
\right)
```

于是 candidate set 从：

```math
\mathcal{K}_i
=
\{x_{i,\delta}:\delta\in D_i\}
```

扩展为：

```math
\mathcal{K}^{\text{branch}}_i
=
\{x_{i,\delta,b}:
  \delta\in D_i,\;
  b\in\mathbb{F}_2^{|I_i|}
\}
```

当每侧选 1 个 interface column，总共 `|I_i|=2` 时：

```math
|\mathcal{K}^{\text{branch}}_i|
=
|D_i| \cdot 2^{|I_i|}
=
3 \cdot 4
=
12
```

这相当于主动枚举少量高影响 seam bits 的不同取值，逼迫 A decoder 产生更丰富的 commit class。

#### 2.1.9 gated branch 的数学形式

轻量 joint-B-DP 先在未 branch 的候选集上得到：

```math
x^{(0)}
=
(x_1^{(0)},\ldots,x_M^{(0)})
```

对应 B 解为：

```math
y^{(0)}
=
(y_1^{(0)},\ldots,y_{M-1}^{(0)})
```

定义该 shot 的 physical residual 总量：

```math
R^{(0)}
=
\sum_{i=1}^{M-1}
r_i(x_i^{(0)},x_{i+1}^{(0)})
```

gated 触发规则是：

```math
R^{(0)} > 0
```

如果：

```math
R^{(0)} = 0
```

则直接接受轻量 joint-B-DP 结果。

如果：

```math
R^{(0)} > 0
```

则为该 shot 构造 branch candidate set，并再次运行 joint-B-DP，得到：

```math
x^{(1)}, y^{(1)}, R^{(1)}, C^{(1)}
```

旧结果 cost 记为：

```math
C^{(0)}
```

当前实现采用字典序接受规则：

```math
(R^{(1)}, C^{(1)})
<
_{\text{lex}}
(R^{(0)}, C^{(0)})
```

也就是：

```math
R^{(1)} < R^{(0)}
```

或者：

```math
R^{(1)} = R^{(0)}
\quad\text{and}\quad
C^{(1)} < C^{(0)}
```

这个规则保证：

```text
branch 不会让 physical residual 变差；
只有 physical closure 改善或 likelihood 改善时才替换。
```

#### 2.1.10 复杂度

设：

```text
M: A window 数量
K_i: 第 i 个 A window candidate 数量
T_B: 一次 B window BP-OSD 解码时间
```

light joint-B-DP 每个 shot 需要的 B pair decode 数约为：

```math
N_{\text{B-decode}}^{\text{light}}
=
\sum_{i=1}^{M-1} K_i K_{i+1}
```

repeat=12 的典型 TSAE candidate 数是：

```text
K = [1, 3, 1]
```

所以：

```math
N_{\text{B-decode}}^{\text{light}}
=
1\cdot3 + 3\cdot1
=
6
```

full interface branch 时：

```text
K = [1, 12, 1]
```

所以：

```math
N_{\text{B-decode}}^{\text{full}}
=
1\cdot12 + 12\cdot1
=
24
```

gated 版本对所有 shots 先做 light pass，只对 `G` 个 gated shots 做 branch pass。若总 shots 为 `S`：

```math
N_{\text{B-decode}}^{\text{gated}}
\approx
6S + 24G
```

因此当：

```math
G \ll S
```

时，gated 可以接近 light joint-B-DP 的 runtime；当 high-p 下 `G` 增大时，runtime 会逐渐接近 full interface branch。

#### 2.1.11 streaming frontier-DP

上面的 full joint-B-DP 是离线形式：

```text
A_0 -- B_0 -- A_1 -- B_1 -- ... -- A_{M-1}
```

它先获得整条链的所有边势函数，然后从最右端回溯，选出整条最优路径。
这会带来一个问题：`A_0` 的提交依赖最右端的 syndrome，因此没有 bounded-latency realtime 解码能力。

新增的 streaming frontier-DP 用固定 lag `L` 代替整链回溯。递推仍然是：

```math
F_i(k)
=
\phi_i(x_i^k)
+
\min_h \left[
F_{i-1}(h)
+
\psi_{i-1}(x_{i-1}^h,x_i^k)
\right].
```

区别在于：当 frontier 到达 `A_i` 后，只回溯 `L` 个 A 节点并提交
`A_{i-L}`。提交之后，把所有与这个已提交 state 不兼容的 frontier path
置为无穷大，相当于把 Viterbi trellis 剪到一个固定前缀：

```math
F_i(k)=\infty
\quad
\text{if path}(k \rightarrow A_{i-L}) \ne \hat{x}_{i-L}.
```

因此：

- `lag=0`：保持旧的 full-chain offline joint-B-DP。
- `lag=1`：`A_i-B_i-A_{i+1}` 到齐后提交 `A_i`。
- `lag=2`：看到两个未来 A window 后提交旧 A，更接近 full-chain，但延迟更大。

B window 的提交要滞后一拍：`B_i` 依赖 `A_i` 和 `A_{i+1}`，所以只有两侧 A 都已经固定后，才把对应 B 解写入最终 correction。

在命令行中使用：

```bash
--a-shifted-joint-b-dp \
--a-shifted-joint-b-dp-lag 1
```

实验脚本仍然以 batch 形式接收 `det_data`，这和 sliding baseline 的测试方式一致；
但 frontier-DP 的决策规则只允许每次提交使用固定 lag 内的未来 syndrome，
不再用整条链末端的信息来决定最左侧 A。

这也解释了实验中的诊断：

```text
repeat=12, p=0.007, shots=1000:
S = 1000
G = 163
light part = 6000 B pair decodes
gated part = 3912 B pair decodes
total = 9912 B pair decodes
```

## 3. TSAE: Tri-Shift A Ensemble

TSAE 的全称是：

```text
Tri-Shift A Ensemble
```

它只改变 A candidate 的生成方式，不扩大单个 A 窗口。

以 `repeat=12, a_solve=5, b_width=5` 的典型布局为例，interior A2 的 center window 是：

```text
A2-center solve: s5-s9
A2 commit:       e11-e13
```

TSAE 对同一个 commit 区域 `e11-e13` 同时使用三个同规模小窗口：

```text
A2-left   solve: s3-s7   -> commit e11-e13
A2-center solve: s5-s9   -> commit e11-e13
A2-right  solve: s7-s11  -> commit e11-e13
```

也就是：

```bash
--a-shifted-ensemble
--a-shift-offsets=-2,0,2
```

边界 A window 由于 commit 必须落在 solve cols 内，通常只有 center offset。

因此 repeat=12 的典型 candidate 数为：

```text
A1: 1
A2: 3
A3: 1
total = 5 A candidates per shot
```

TSAE 的价值：

```text
每个 A 子问题仍是 5-row 小窗口；
但 interior A commit 可以从 left / center / right 三个上下文视角中选择。
```

## 4. joint-B-DP

### 4.1 为什么需要 joint-B-DP

原始 TSAE 的选择分数类似：

```text
score = A local cost + beta * neighbor B residual popcount
```

这个 residual popcount 只是候选对 B syndrome 的局部风险估计，并没有真正问：

```text
在这对 A candidate 固定后，B window 自己能不能解出 physical closure？
```

实验表明，单纯 residual popcount、chain-DP、bit-level stitching、z_B repair 都有收益上限。

joint-B-DP 的改动是：

```text
对每个相邻 A candidate pair，真实运行对应 B window decoder；
用 B 解码后的 cost 和 physical residual 作为边权；
最后用链式 DP 选全局 A/B candidate path。
```

### 4.2 图模型

把 A windows 看成链上的节点：

```text
A1 -- B1 -- A2 -- B2 -- A3
```

每个 A 节点有若干 candidate 状态：

```text
A1: center
A2: left / center / right
A3: center
```

每条 B 边连接相邻 A candidate pair：

```text
edge_cost(Bi, A_left_candidate, A_right_candidate)
```

边权由实际 B 解码给出：

```text
B local syndrome = det_data[B_rows]
                 xor H_B,A_left  * A_left_candidate
                 xor H_B,A_right * A_right_candidate

B decode -> local_e

physical_residual = H_B,physical * physical_e xor local_syndrome

edge_cost = error_cost(local_e, B_prior)
          + flag_penalty * physical_residual_weight
```

默认：

```bash
--a-shifted-joint-flag-penalty 1000.0
```

这个 penalty 很大，意思是：

```text
优先选择能让 B physical residual 清零的 A/B path；
只有 residual 相同或都为 0 时，再比较 likelihood cost。
```

### 4.3 为什么叫 joint-B-DP

它不是把 A/B 合成一个大矩阵一次性解码，而是：

```text
每个 B edge 仍然是小 B window decode；
A candidate pair 之间通过 DP 联合选择；
整体形成一个小状态链式动态规划。
```

所以它保持了小窗口结构，又比 greedy TSAE 更接近大窗口联合信息。

## 5. interface branch

### 5.1 全量 interface branch

interface branch 的目标是增加 A commit 在 seam 附近的 candidate diversity。

对 interior A 的左右 interface region，各选若干最相关的物理列：

```bash
--tsae-interface-branch
--tsae-interface-cols-per-side 1
```

每侧 1 个 interface bit 时，左右两侧共 2 个 bit，因此有：

```text
2^2 = 4 branch assignments
```

结合 A2 的 3 个 shift：

```text
A2 candidates: 3 shifts * 4 branches = 12
```

全量 interface branch 的准确率最好，但很慢，因为每个 shot 都要对更多 A candidate pair 跑 B decode。

repeat=12, shots=100 的对比中：

```text
p=0.007:
joint-B-DP:                 LER 0.470, flagged 14/100, 19.824s
full interface + joint-B-DP: LER 0.400, flagged 8/100, 77.605s
```

## 6. gated interface branch

### 6.1 动机

全量 interface branch 有两个问题：

```text
1. 许多 shot 在轻量 joint-B-DP 后已经 physical residual = 0；
   对这些 shot 再 branch 很贵。

2. 我们主要想修的是 physical non-closure；
   所以可以只对 residual shot 开启更大的 candidate pool。
```

因此加入：

```bash
--tsae-interface-gated
```

它必须和下面两个参数一起使用：

```bash
--tsae-interface-branch
--a-shifted-joint-b-dp
```

### 6.2 gated 流程

算法流程：

```text
for all shots:
    run light joint-B-DP using unbranched TSAE candidates
    record selected physical residual per shot

gated_shots = shots whose selected physical residual > 0

for gated shots only:
    build interface-branch A candidates
    rerun joint-B-DP
    accept branch result if:
        new_residual < old_residual
        or new_residual == old_residual and new_cost < old_cost

for non-gated shots:
    keep light joint-B-DP result
```

接受规则是保守的：

```text
不允许 branch 结果让 physical residual 变差；
只有 residual 改善，或者 residual 持平且 cost 更低，才覆盖原结果。
```

### 6.3 gated 的优缺点

优点：

```text
1. 显著降低 runtime；
2. 基本保留 physical closure 改善；
3. 对 low/mid p 很省，因为 gated shots 很少。
```

缺点：

```text
1. 如果某个 shot 已经 physical residual = 0，
   但 full interface branch 可以改善 logical class，
   gated 不会触发，因此可能错过 logical-only improvement。

2. high-p 下 gated shots 增多，runtime 仍会升高。
```

## 7. 关键参数

典型命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 1000 \
  --p-list 0.003,0.004,0.005,0.007 \
  --decoders parallel \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --top-k-boundary 1 \
  --a-noisy-boundary \
  --b-noisy-boundary \
  --window-shorten \
  --a-shifted-ensemble \
  --a-shift-offsets=-2,0,2 \
  --a-shifted-beta 1.0 \
  --tsae-interface-branch \
  --tsae-interface-gated \
  --tsae-interface-cols-per-side 1 \
  --a-shifted-joint-b-dp \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots1000_TSAE_gated_interface_joint_b_dp_ab_noisy_shorten_parallel_only.csv
```

参数解释：

| 参数 | 含义 |
|---|---|
| `--a-shifted-ensemble` | 启用 TSAE shifted A candidates |
| `--a-shift-offsets=-2,0,2` | interior A 使用 left / center / right 三个视角 |
| `--a-shifted-joint-b-dp` | 启用基于真实 B decode 边权的 chain DP |
| `--a-shifted-joint-flag-penalty` | B physical residual 的惩罚权重，默认 1000 |
| `--tsae-interface-branch` | 启用 interface bit branch 能力 |
| `--tsae-interface-gated` | 只在 light joint-B-DP residual shot 上触发 branch |
| `--tsae-interface-cols-per-side` | 每侧选择多少个 interface columns，目前常用 1 |
| `--a-noisy-boundary` | A window 增加 noisy boundary columns |
| `--b-noisy-boundary` | B window 增加 noisy boundary columns |
| `--window-shorten` | 使用 shortening 版本局部 OSD |

## 8. 结果摘要

### 8.1 repeat=12, shots=1000

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots1000_TSAE_gated_interface_joint_b_dp_ab_noisy_shorten_parallel_only.csv
```

| p | parallel | TSAE | sliding | gated interface+joint |
|---:|---:|---:|---:|---:|
| 0.003 | 10/1000, LER 0.011, 29.4s | 4/1000, LER 0.007, 29.2s | 0/1000, LER 0.001, 71.4s | 0/1000, LER 0.001, 30.7s |
| 0.004 | 32/1000, LER 0.038, 31.3s | 11/1000, LER 0.022, 40.1s | 0/1000, LER 0.021, 75.3s | 0/1000, LER 0.010, 50.5s |
| 0.005 | 119/1000, LER 0.148, 34.0s | 38/1000, LER 0.111, 42.9s | 0/1000, LER 0.093, 70.2s | 1/1000, LER 0.050, 90.9s |
| 0.007 | 423/1000, LER 0.585, 56.7s | 275/1000, LER 0.556, 67.5s | 0/1000, LER 0.513, 92.8s | 77/1000, LER 0.449, 339.8s |

诊断：

| p | B pair decodes | gated shots | accepted | improved to zero | gated B decodes | final physical residual |
|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 6000 | 0 | 0 | 0 | 0 | 0 |
| 0.004 | 6024 | 1 | 1 | 1 | 24 | 0 |
| 0.005 | 6144 | 6 | 5 | 5 | 144 | 6 |
| 0.007 | 9912 | 163 | 146 | 86 | 3912 | 200 |

结论：

```text
gated interface + joint-B-DP 在 1000 shots 下持续降低 LER：

p=0.004: sliding 0.021, TSAE 0.022, gated 0.010
p=0.005: sliding 0.093, TSAE 0.111, gated 0.050
p=0.007: sliding 0.513, TSAE 0.556, gated 0.449
```

### 8.2 repeat=18, shots=100

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat18_shots100_TSAE_gated_interface_joint_b_dp_ab_noisy_shorten_parallel_only.csv
```

| p | parallel | TSAE | sliding | gated interface+joint |
|---:|---:|---:|---:|---:|
| 0.003 | 3/100, LER 0.030, 8.1s | 0/100, LER 0.000, 11.4s | 0/100, LER 0.000, 34.7s | 0/100, LER 0.000, 10.1s |
| 0.004 | 6/100, LER 0.060, 7.4s | 4/100, LER 0.050, 10.6s | 0/100, LER 0.020, 30.5s | 0/100, LER 0.000, 15.6s |
| 0.005 | 32/100, LER 0.350, 7.7s | 14/100, LER 0.270, 12.7s | 0/100, LER 0.150, 25.0s | 0/100, LER 0.140, 54.3s |
| 0.007 | 68/100, LER 0.830, 9.6s | 43/100, LER 0.800, 19.9s | 0/100, LER 0.620, 28.3s | 13/100, LER 0.680, 228.2s |

repeat=18 结论：

```text
low/mid p 下 gated 仍然强；
p=0.007 下 gated 显著优于 plain parallel / TSAE，
但本轮 100-shot 没有超过 sliding。
```

## 9. flagged 与 LER 的关系

项目里的 scoring 定义为：

```text
flagged = final syndrome residual nonzero
logical = observable mismatch
failed  = flagged OR logical
LER     = failed / shots
```

因此：

```text
LER >= flagged_rate
```

例如 repeat=12, p=0.007, shots=1000：

```text
flagged = 77/1000 = 0.077
LER     = 0.449
```

这并不矛盾：

```text
77 个 flagged shots 一定算失败；
另外还有大量 syndrome 已闭合但 logical class 错误的 shots。
```

也就是说，high-p 下 LER 的主要来源仍然是 logical error，而不是 flagged。

这也解释了为什么 full interface branch 有时比 gated LER 更低：

```text
full branch 也会处理 physical residual = 0 的 shots，
可能改善 logical-only 错误；
gated 只处理 physical residual > 0 的 shots，
主要改善 flagged / physical closure。
```

## 10. 并行性分析

这个策略不是单纯 runtime 优化，而是 parallel 框架里的 accuracy 优化。

保留的并行结构：

```text
1. A shifted candidates 可并行；
2. 不同 shots 可并行；
3. 不同 B windows 可并行；
4. 不同 A-candidate pair 的 B decode 可并行。
```

当前 Python 实现中的限制：

```text
joint-B-DP 内部的 B pair decode 目前主要在 Python 循环中执行；
还没有把所有 pair decode 展开成 worker payload。
```

因此：

```text
算法形态具有并行性；
当前实现还没有完全吃满这种并行性。
```

从硬件角度看，它适合小核并行：

```text
单个 decoder core 仍然只处理小 A/B window；
额外代价来自 candidate pair 数量增加；
不需要一个更大的 monolithic BP-OSD window。
```

## 11. 当前局限

主要局限：

```text
1. high-p runtime 仍然高。
   p=0.007 时 gated shots 增多，额外 B pair decode 明显增加。

2. gated 不处理 zero-residual logical-only shots。
   因此 LER 可能不如 full interface branch。

3. high repeat / high p 下，candidate pool 可能仍不包含正确 logical class。
   这时再优化 selector 收益有限，需要增加 candidate diversity。

4. 目前 joint-B-DP 的 pair decode 没有充分并行化。
```

## 12. 后续优化方向

推荐优先级：

```text
1. 并行化 joint-B-DP 的 B pair decode。
   这是最直接的 runtime 优化。

2. 改进 gated 触发条件。
   除 physical residual > 0 外，加入低 margin / high uncertainty 触发，
   以捕获部分 logical-only improvement。

3. 对 p=0.007 的失败 shot 做 oracle diagnostics。
   判断失败来自：
       candidate pool 不含正确 A commit；
       还是正确 candidate 存在但 score 选错。

4. 做 adaptive interface branching。
   low-p 保持 gated；
   high-p 或 low-margin shot 才扩大 branch cols。

5. 将 full interface branch 改为 two-stage branch。
   先尝试单侧 branch；
   仍 residual 时再尝试双侧 branch。
```

## 13. 一句话总结

```text
TSAE + gated interface + joint-B-DP
是在不扩大单个小窗口的前提下，
用 shifted A candidate diversity 和真实 B-window closure score
构造接近大窗口信息的 parallel decoder。

它显著降低 LER，
尤其在 repeat=12 的 1000-shot 测试中优于 sliding；
代价是 high-p 下额外 candidate-pair B decode 带来的 runtime 增加。
```
