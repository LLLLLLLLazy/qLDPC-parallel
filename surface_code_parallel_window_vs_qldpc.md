# Surface Code Parallel Window Decoding 与 qLDPC 的结构差异

可以。我们不用“错误链很局域”这种口头说法，而用一个更严谨的数学框架说明：

> **surface code 适合 parallel window decoding 的关键，是它的时空解码矩阵 \(D_{\mathrm{SC}}\) 具有几何局域的边界算子结构；而一般 qLDPC code 的解码矩阵 \(D_{\mathrm{qLDPC}}\) 虽然稀疏，但不一定具有这种几何局域性。因此，surface code 的窗口边界只产生局部 residual / artificial defects，而 qLDPC 的窗口切分后可能留下非局域耦合，导致 A/B window 不能简单并行缝合。**

下面一步步推导。

## 1. 统一写法：所有解码都可以写成

无论 surface code 还是 qLDPC，多轮 syndrome 解码都可以抽象为：

$$
D e = s \pmod 2.
$$

其中：

- \(e \in \mathbb{F}_2^N\)：所有可能错误事件的指示向量；
- \(s \in \mathbb{F}_2^M\)：观测到的 syndrome / defect 向量；
- \(D \in \mathbb{F}_2^{M \times N}\)：错误事件到 syndrome 的线性映射。

第 \(j\) 列 \(D_{:,j}\) 表示第 \(j\) 个错误事件会触发哪些 syndrome bits。

解码器要找一个低权重解：

$$
\hat e
=
\arg\min_e w(e)
\quad
\text{s.t.}
\quad
D e = s.
$$

所以从最抽象层面看，surface code 和 qLDPC 都是同一个问题。

区别在于：

$$
D_{\mathrm{SC}}
\quad \text{和} \quad
D_{\mathrm{qLDPC}}
$$

的结构不同。

## 2. surface code 的 \(D_{\mathrm{SC}}\)：几何局域边界算子

surface code 的多轮 syndrome history 可以嵌入到一个时空晶格：

$$
\Lambda
\subset
\mathbb{Z}^2 \times \mathbb{Z}.
$$

其中：

- \(\mathbb{Z}^2\)：二维空间坐标；
- \(\mathbb{Z}\)：时间轮数。

一个 syndrome defect 是某个时空点：

$$
v = (x,y,t).
$$

一个可能错误事件对应一个局部边或局部超边：

$$
a \subset \Lambda.
$$

在 phenomenological noise model 下，可以近似看成普通边：

$$
a = (v_1,v_2).
$$

于是 surface code 的 \(D_{\mathrm{SC}}\) 可以看成时空图 \(G=(V,E)\) 的边界算子：

$$
D_{\mathrm{SC}}
=
\partial_1.
$$

如果某条错误边 \(e_j\) 连接两个 defects \(v_a,v_b\)，那么：

$$
D_{\mathrm{SC}}[:,j]
=
\mathbf{1}_{v_a}
+
\mathbf{1}_{v_b}.
$$

也就是说：

$$
\partial e = s.
$$

这就是 matching graph 语言中的：

> 一组错误边的边界等于观测 defects。

论文中也正是这样描述 matching decoder：vertices 是 potential defects，edges 是 possible errors，decoder 输入 triggered defects，输出一组 correction edges。

## 3. surface code 的关键性质：有限传播半径

surface code 的局域性可以形式化为：

存在常数 \(r=O(1)\)，使得对任意错误变量 \(e_j\)，其 syndrome 支持集满足：

$$
\operatorname{diam}
\left(
\operatorname{supp}(D_{\mathrm{SC}}[:,j])
\right)
\le r.
$$

这里 \(\operatorname{diam}\) 是在时空几何度量中的直径。

直观上：

- data qubit error 只影响附近 stabilizers；
- measurement error 只影响相邻时间轮；
- circuit-level hook error 也只影响局部区域。

所以每个错误事件只在局部时空区域产生 syndrome。

这意味着 \(D_{\mathrm{SC}}\) 的每一列不仅稀疏，而且是 **几何局域稀疏**：

$$
\text{sparse}
+
\text{geometrically local}.
$$

这点非常关键。

## 4. 定义时间窗口和 commit region

现在在时间方向上定义一个 A window：

$$
W_A
=
[t_0,t_0+3w].
$$

其中：

$$
W_A
=
B_L
\cup
C
\cup
B_R,
$$

分别是：

$$
B_L = [t_0,t_0+w],
$$

$$
C = [t_0+w,t_0+2w],
$$

$$
B_R = [t_0+2w,t_0+3w].
$$

即：

$$
W_A
=
\underbrace{B_L}_{\text{left buffer}}
\cup
\underbrace{C}_{\text{commit}}
\cup
\underbrace{B_R}_{\text{right buffer}}.
$$

论文中取：

$$
n_{\mathrm{buf}}
=
n_{\mathrm{com}}
=
w,
\qquad
n_W = 3w,
$$

并用和 sliding window 相同的 reasoning 取：

$$
w = d.
$$

也就是：

$$
W_A
=
d+d+d.
$$

论文 Fig. 3 的 A layer 就是这个结构：A window 两侧都有 buffer，中间 commit region 被提交。

## 5. 为什么 surface code 的 A window 可以只提交 commit region？

定义一个投影算子：

$$
P_C
$$

把 correction 限制到 commit region 内。

A window 内部解码得到一个局部解：

$$
\hat e_{W_A}
\quad
\text{s.t.}
\quad
D_{W_A}\hat e_{W_A}
=
s_{W_A}
+
b_{\partial W_A}.
$$

其中 \(b_{\partial W_A}\) 表示边界条件，例如 rough boundary 上允许把 defect 接到 boundary。

A layer 最终不提交整个 \(\hat e_{W_A}\)，而只提交：

$$
\hat e_C
=
P_C \hat e_{W_A}.
$$

提交后，剩余 syndrome 变成：

$$
s'
=
s
+
D_{\mathrm{SC}}\hat e_C.
$$

因为

$$
D_{\mathrm{SC}}\hat e_C
$$

只会改变 \(\hat e_C\) 的边界附近 syndrome，所以 \(s'\) 中新产生的 residual syndrome 只出现在 commit region 的边界附近。

这就是 artificial defects。

形式化地说，如果 \(\hat e_C\) 的支持在 \(C\) 内，那么由于 \(D_{\mathrm{SC}}\) 是局域边界算子：

$$
\operatorname{supp}
\left(
D_{\mathrm{SC}}\hat e_C
\right)
\subset
N_r(\partial C),
$$

其中 \(N_r(\partial C)\) 是 commit boundary 的 \(r\)-邻域。

所以 surface code 中：

$$
\text{artificial defects}
\subset
\text{commit boundary 的局部邻域}.
$$

这就是 A layer 可以向 B layer 传递少量边界信息的数学原因。

## 6. surface code 中 A windows 之间为什么可以并行？

取一组 A windows：

$$
W_A^{(1)},W_A^{(2)},\dots,W_A^{(K)}.
$$

它们的 commit regions 为：

$$
C_1,C_2,\dots,C_K.
$$

设计它们满足：

$$
\operatorname{dist}(C_i,C_j) > 2w
\quad
(i \neq j).
$$

由于 \(D_{\mathrm{SC}}\) 的局域半径是 \(r=O(1)\)，并且 \(w=d \gg r\)，则一个局部错误事件不可能同时影响两个不同的 commit regions。

数学上：

$$
\operatorname{supp}(D_{\mathrm{SC}}[:,j])
\cap C_i \neq \varnothing
$$

和

$$
\operatorname{supp}(D_{\mathrm{SC}}[:,j])
\cap C_k \neq \varnothing
$$

不能同时发生，除非错误链很长。

更准确地说，单个局部 error event 只能影响一个 \(O(1)\) 邻域；一条长度 \(\ell\) 的 error chain 的影响范围最多随 \(\ell\) 增长。

如果 error chain 长度满足：

$$
\ell \le d,
$$

而 buffer 宽度取：

$$
w = d,
$$

则影响某个 commit region 的低权重链会被该 window 完整包含。

论文中正是利用这个性质：只要 A layer 的 window size 和 commit region 选择合适，每条长度 \(\le d\) 的 error chain 都会被某个 window 完整捕获，因此 logical fidelity 不会显著下降。

因此，不同 A windows 可以并行解码，因为它们只对彼此分离的 \(C_i\) 做最终提交：

$$
\hat e_{C_i}
=
P_{C_i}\hat e_{W_A^{(i)}}.
$$

它们之间的相互作用只会以局部 artificial defects 的形式留在：

$$
\partial C_i.
$$

然后交给 B layer。

## 7. B layer 的数学作用

A layer 提交后，新的 syndrome 是：

$$
s^{(A)}
=
s
+
D_{\mathrm{SC}}
\left(
\sum_i \hat e_{C_i}
\right).
$$

由于每个 \(\hat e_{C_i}\) 只在局部边界产生 artificial defects，所以：

$$
\operatorname{supp}(s^{(A)})
$$

集中在相邻 A commit regions 之间的区域。

于是定义 B windows：

$$
W_B^{(i)}
$$

覆盖：

$$
C_i
\quad \text{和} \quad
C_{i+1}
$$

之间的未解决区域。

B layer 解：

$$
D_{W_B^{(i)}} \hat e_{B_i}
=
s^{(A)}_{W_B^{(i)}}.
$$

因为两侧 A commit regions 已经 resolved，所以 B window 使用 smooth time boundaries，不需要 buffer。论文也明确说，B windows 前后 correction 已在 A layer resolved，因此 B windows 有 smooth time boundaries and do not require buffers。

最终 correction 是：

$$
\hat e
=
\sum_i \hat e_{C_i}
+
\sum_i \hat e_{B_i}.
$$

满足：

$$
D_{\mathrm{SC}}\hat e
=
s
$$

除了可能的逻辑等价差异。

## 8. surface code parallel window 成立的核心数学条件

可以总结为三个条件。

### 条件 1：局域列支持

存在常数 \(r\)，使得：

$$
\operatorname{diam}
\left(
\operatorname{supp}(D[:,j])
\right)
\le r
\quad
\forall j.
$$

surface code 满足。

### 条件 2：窗口 buffer 大于局部相关长度

取：

$$
w = d.
$$

使得长度不超过 \(d\) 的 error chain 不能在不被看到的情况下穿过整个 buffer。

### 条件 3：边界 residual 是局部的

提交 commit correction 后：

$$
D\hat e_C
$$

的新增 syndrome 只出现在：

$$
N_r(\partial C).
$$

因此可以用 B window 局部缝合。

## 9. qLDPC 的问题：稀疏不等于几何局域

现在看一般 qLDPC code。

单轮可以写成：

$$
H e = s.
$$

多轮可以写成：

$$
D_{\mathrm{hist}} e_{\mathrm{hist}}
=
s_{\mathrm{hist}}.
$$

qLDPC 的 LDPC 性质是：

$$
\operatorname{wt}(H_{i,:}) = O(1),
\qquad
\operatorname{wt}(H_{:,j}) = O(1).
$$

即每行每列权重都是常数。

但是这只说明：

$$
D_{\mathrm{qLDPC}}
\text{ sparse}.
$$

不说明：

$$
D_{\mathrm{qLDPC}}
\text{ geometrically local}.
$$

也就是说，对一般 qLDPC，可能不存在一个二维或低维几何嵌入 \(\phi\)，使得：

$$
\operatorname{diam}
\left(
\phi(\operatorname{supp}(D[:,j]))
\right)
\le O(1)
\quad
\forall j.
$$

一个 check 可能连接 Tanner graph 上少量 qubits，但这些 qubits 在任何简单几何排序下都不局部。

例如某个 check：

$$
h_i
=
q_3 + q_{19} + q_{287} + q_{901}.
$$

它行重是 4，满足 LDPC，但几何上可能非常非局部。

## 10. qLDPC 为什么不适合直接套 surface-code parallel window？

如果只做 **时间 sliding window**，qLDPC 是可以的，因为 syndrome 确实按时间产生。

问题在于 surface-code parallel window 的 A/B 层逻辑需要更强条件：

$$
\text{A window 提交后，residual 只留在局部边界。}
$$

对于一般 qLDPC，这个性质不一定成立。

假设我们试图把变量或 syndrome 分成若干局部区域：

$$
V = V_1 \cup V_2 \cup \cdots \cup V_K.
$$

对 surface code，一个局部 correction \(\hat e_{V_i}\) 只会在 \(V_i\) 的几何边界产生 residual：

$$
\operatorname{supp}(D\hat e_{V_i})
\subset
\partial V_i.
$$

但一般 qLDPC 中，可能有：

$$
\operatorname{supp}(D\hat e_{V_i})
\cap V_j
\neq \varnothing
$$

对于很多远处的 \(j\)。

也就是说：

$$
D\hat e_{V_i}
$$

不只影响 \(V_i\) 附近，而可能影响很多非相邻区域。

这会破坏 B layer 的设计。

因为 B layer 原本只负责缝合相邻 A windows：

$$
B_i
\sim
\text{between } A_i \text{ and } A_{i+1}.
$$

但 qLDPC 可能需要一个 B window 同时缝合：

$$
A_i,A_j,A_k,\dots
$$

的非局部耦合。

于是并行结构从：

$$
A_i
\rightarrow
B_{i,i+1}
$$

变成：

$$
A_i
\rightarrow
B_{i,j,k,\dots}.
$$

这就不再是论文 Fig. 3 那种简单的两层局部并行结构。

## 11. 用矩阵块形式严格表达区别

对 surface code，按照几何窗口排序后，\(D_{\mathrm{SC}}\) 近似是块带状矩阵：

$$
D_{\mathrm{SC}}
=
\begin{bmatrix}
D_1 & C_{12} & 0 & 0 & \cdots \\
C_{21} & D_2 & C_{23} & 0 & \cdots \\
0 & C_{32} & D_3 & C_{34} & \cdots \\
0 & 0 & C_{43} & D_4 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}.
$$

其中：

- \(D_i\)：window 内部解码；
- \(C_{i,i+1}\)：相邻 window 的边界耦合；
- 远距离耦合块为 0：

$$
C_{ij}=0
\quad
\text{if}
\quad
|i-j|>1.
$$

所以局部 A/B window 能覆盖所有耦合。

而一般 qLDPC 经过任意简单时间/空间窗口排序后，可能是：

$$
D_{\mathrm{qLDPC}}
=
\begin{bmatrix}
D_1 & C_{12} & C_{13} & 0 & C_{15} \\
C_{21} & D_2 & 0 & C_{24} & C_{25} \\
C_{31} & 0 & D_3 & C_{34} & 0 \\
0 & C_{42} & C_{43} & D_4 & C_{45} \\
C_{51} & C_{52} & 0 & C_{54} & D_5
\end{bmatrix}.
$$

虽然每个 \(C_{ij}\) 可能很稀疏，但非零块可能分布在远距离位置。

这意味着：

$$
C_{ij} \neq 0
\quad
\text{for many non-neighbor } i,j.
$$

那么 A window 的提交会影响多个远处区域，B layer 不能只处理相邻窗口之间的缝合。

## 12. 关键区别：surface code 是“局部边界”，qLDPC 是“图论耦合”

surface code 中：

$$
\partial C
$$

是几何边界。

提交 \(C\) 内 correction 后，人工缺陷位于：

$$
\partial C.
$$

一般 qLDPC 中，如果没有几何嵌入，所谓“边界”只能定义为 Tanner graph cut：

$$
\partial_{\mathrm{Tanner}} V_i
=
\left\{
c \in C:
c \text{ connected to both } V_i \text{ and } V \setminus V_i
\right\}.
$$

如果这个 cut 很大或连接到很多远处区域，则 residual syndrome 不是局部边界上的少量 defects，而是广泛分布的约束。

parallel window 需要的是：

$$
|\partial V_i| \ll |V_i|
$$

并且边界只连接相邻区域。

一般 qLDPC 不能保证这个性质。

## 13. 所以不是 qLDPC 绝对不能用，而是不能“直接套用”

更严谨地说：

> **qLDPC 不适合直接使用 surface-code Fig. 3 的 parallel window 结构。**

不是说 qLDPC 永远不能并行窗口解码。

如果某个 qLDPC family 的 Tanner graph 有良好的局部分解：

$$
D_{\mathrm{qLDPC}}
\approx
\begin{bmatrix}
D_1 & C_{12} & 0 & \cdots \\
C_{21} & D_2 & C_{23} & \cdots \\
0 & C_{32} & D_3 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix},
$$

或者可以通过某个离线变换得到：

$$
D'
=
TDP
=
\left(
\operatorname{diag}(D_1,\dots,D_K),
A
\right),
$$

其中 \(A\) 足够稀疏，那么 qLDPC 也有可能做类似 parallel window / hierarchical decoding。

这正好和你之前研究 Vegapunk 的思想相近：先离线寻找结构化变换，把全局问题变成局部块加少量耦合。

## 14. 最终严谨总结

surface code 可以使用 parallel window decoding，是因为它满足：

$$
D_{\mathrm{SC}} = \partial
$$

是一个低维时空几何复形上的局部边界算子，并且存在常数 \(r\)：

$$
\operatorname{diam}
\left(
\operatorname{supp}(D_{\mathrm{SC}}[:,j])
\right)
\le r.
$$

因此对一个 commit region \(C\)，提交局部 correction 后：

$$
\operatorname{supp}(D_{\mathrm{SC}}\hat e_C)
\subset
N_r(\partial C),
$$

artificial defects 只出现在局部边界附近，可以由相邻 B window 处理。

一般 qLDPC 虽然也满足：

$$
D_{\mathrm{qLDPC}} e = s,
$$

并且 \(D_{\mathrm{qLDPC}}\) 稀疏，但通常只保证：

$$
\operatorname{wt}(D[:,j]) = O(1),
$$

不保证：

$$
\operatorname{diam}
\left(
\operatorname{supp}(D[:,j])
\right)
=
O(1)
$$

在某个低维几何嵌入中成立。因此切成窗口后，局部 correction 的 syndrome 影响可能分布到多个非相邻区域：

$$
\operatorname{supp}(D_{\mathrm{qLDPC}}\hat e_C)
\not\subset
N_r(\partial C).
$$

所以 qLDPC 的 residual syndrome 不一定能被相邻 B window 局部缝合。

一句话：

**surface code 的 \(D\) 是几何局域的边界算子，window 边界上的残差仍然局部；一般 qLDPC 的 \(D\) 只是稀疏，不一定几何局域，window 切分后可能产生非局部跨块耦合，所以不能直接套用 surface code 的 parallel window decoding。**









空间几何局域性和A、B又有什么关系呢，并行求解每个A对应的Di ei = si部分，再求解B对应的Di ei = si部分，为什么这样不行呢？
