# qLDPC 并行窗口解码

---

## 1. 动机

传统 qLDPC 电路级解码可以写成全局线性 syndrome 方程：

$$
De =s \rightarrow H_{\mathrm{circ}} e = s.
$$

其中：

- \(H_{\mathrm{circ}}\)：电路级 fault-to-detector 矩阵；
- \(e\)：可能故障机制的二进制向量；
- \(s\)：detector 取值；
- 每一行是一个 detector；
- 每一列是一个可能的电路故障。

普通滑动窗口解码按时间顺序逐个 window 解码：

$$
W_1 \rightarrow W_2 \rightarrow W_3 \rightarrow \cdots
$$

这种方式虽然降低了单次解码规模，但仍然存在串行依赖。

因此，仿照 surface code 中并行窗口的思想，将 qLDPC 的 detector history 拆成多个可并行处理的 A windows 和 B windows：

$$
\text{A layer 并行解码}
\rightarrow
\text{residual 更新}
\rightarrow
\text{B layer 并行解码}.
$$

本文将该方案称为：

$$
\boxed{\text{qLDPC  Parallel Window decoder}.}
$$

---

## 2. Detector 与 \(H_{\mathrm{circ}}\)

在电路级噪声模型下，我们不直接使用原始 qLDPC parity-check matrix \(H_X,H_Z\)，而是构造电路级 detector matrix：

$$
H_{\mathrm{circ}}e=s.
$$

detector 通常由相邻两轮 syndrome 测量结果的 XOR 构造。

若某个 check \(c\) 的测量结果为：

$$
m_1,m_2,\dots,m_R,
$$

则 detector 取值为：

$$
m_1,\quad m_2\oplus m_1,\quad m_3\oplus m_2,\dots.
$$

因此，detector 表示某个 check 的测量结果是否在相邻轮之间发生变化。

在 qLDPC sliding window 论文中，\(H_{\mathrm{circ}}\) 的行是 detectors，列是 single fault mechanisms；矩阵元素为 1 表示该 fault 会触发该 detector。论文也指出，detector 的 XOR 构造使得单个 fault 不会触发超过两轮 detectors，从而得到时间方向的块带状结构。

---

## 3. \(H_{\mathrm{circ}}\) 的块结构

按照 detector 时间块排列，\(H_{\mathrm{circ}}\) 具有如下结构：

$$
H_{\mathrm{circ}}
=
\begin{bmatrix}
H_0 & H_1 \\
& H_2 & H_0 & H_1 \\
& & & H_2 & H_0 & H_1 \\
& & & & & H_2 & H_0 & H_1 \\
& & & & & & \ddots & \ddots
\end{bmatrix}.
$$

设 detector blocks 为：

$$
s_1,s_2,s_3,\dots
$$

设 fault variables 按时间排列为：

$$
e_0,e_1,e_2,e_3,\dots
$$

其中：

- \(e_{2t-2}\)：只影响第 \(t\) 轮 detector block 的 \(H_0\)-type 变量；
- \(e_{2t-1}\)：同时影响第 \(t\) 和第 \(t+1\) 轮 detector blocks 的跨轮变量。

于是：

$$
s_1 = H_0e_0 + H_1e_1,
$$

$$
s_2 = H_2e_1 + H_0e_2 + H_1e_3,
$$

$$
s_3 = H_2e_3 + H_0e_4 + H_1e_5,
$$

$$
s_4 = H_2e_5 + H_0e_6 + H_1e_7.
$$

一般地：

$$
\boxed{
s_t
=
H_2e_{2t-3}
+
H_0e_{2t-2}
+
H_1e_{2t-1}.
}
$$

其中：

- \(H_0e_{2t-2}\)：当前 detector block 的内部贡献；
- \(H_1e_{2t-1}\)：跨轮变量对较早 detector block \(s_t\) 的贡献；
- \(H_2e_{2t-1}\)：同一个跨轮变量对较晚 detector block \(s_{t+1}\) 的贡献。

因此：

$$
e_{2t-1}
$$

是连接相邻 detector blocks \(s_t\) 和 \(s_{t+1}\) 的桥变量。

---

## 4. 从 Sliding Window 到 Parallel Window

普通 qLDPC sliding window 以 \((3,1)\)-window 为例，每次处理三轮 detector 取值：

$$
(s_t,s_{t+1},s_{t+2}),
$$

解局部方程，然后提交最左边不再出现在下一窗口中的变量。

例如第一个 window：

$$
(s_1,s_2,s_3)
$$

解出局部估计后，提交：

$$
\hat e_0,\hat e_1.
$$

必须满足 partial equation：

$$
\boxed{
\begin{bmatrix}
H_0 & H_1
\end{bmatrix}
\begin{bmatrix}
\hat e_0 \\
\hat e_1
\end{bmatrix}
=
s_1.
}
$$

否则 \(s_1\) 离开窗口后再也无法被后续 window 修复，因此全局方程 \(H_{\mathrm{circ}}\hat e=s\) 不可能满足。该要求与论文中的 Eq. (7) 一致；论文也强调，如果该 partial equation 不成立，则整体 syndrome equation 无论后续 window decoding 如何进行都无法满足。

提交 \(\hat e_1\) 后，需要更新下一轮 detector：

$$
s_2'
=
s_2+H_2\hat e_1.
$$

这是 residual update：

$$
\boxed{
s \leftarrow s+H_{\mathrm{circ}}\hat e_C.
}
$$

其中加法在 \(\mathbb F_2\) 上进行，即 XOR。

---

## 5. A/B Window 思路

qLDPC 并行窗口解码的核心思想是：

1. 先并行解多个相互分离的 A windows；
2. A windows 提供左右边界变量；
3. 用这些边界变量更新中间 residual detectors；
4. 再并行解多个 B windows。

整体流程为：

$$
\boxed{
\text{A layer 并行}
\rightarrow
\text{边界 residual 更新}
\rightarrow
\text{B layer 并行}.
}
$$

与普通 sliding window 的区别是：

$$
W_1 \rightarrow W_2 \rightarrow W_3 \rightarrow \cdots
$$

这种串行链被压缩成两层同步处理：

$$
A_1,A_2,A_3,\dots \quad \text{并行},
$$

然后：

$$
B_1,B_2,B_3,\dots \quad \text{并行}.
$$

---

## 6. A Window 与 B Window 的范围

考虑一个中间 B window：

$$
\boxed{
B=[u,v].
}
$$

它负责处理 detector rows：

$$
s_u,s_{u+1},\dots,s_v.
$$

假设左侧 A window 已经给出了左边界变量：

$$
\hat e_{2u-3},
$$

右侧 A window 已经给出了右边界变量：

$$
\hat e_{2v-1}.
$$

那么 B window 不再解这两个边界外变量，而是把它们作为已知边界条件。



再考虑如何确定A的大小：

不妨定义一个变量：
$$
n_A
$$
令：
$$
n_A = A_i矩阵的行数/H_i矩阵的行数，
n_B = B_i矩阵的行数/H_i矩阵的行数
$$
除开始和结束的A1和Ak外

例如，当\(n_{\mathrm{A}}\)=3，\(n_{\mathrm{B}}\)=3时，

\(A_{\mathrm{1}}\)矩阵如下：
$$
\boxed{
A_1
=
\begin{bmatrix}
H_0 & H_1 & & & \\
				& H_2 & H_0 & H_1 & \\
\end{bmatrix}
}
$$
负责处理：
$$
s_1,s_2.
$$
commit：
$$
e_0,e_1.
$$
\(A_{\mathrm{2}}\)矩阵如下：
$$
\boxed{
A_2
=
\begin{bmatrix}
H_2 & H_0 & H_1 & & & \\
				&  & H_2 & H_0 & H_1 & \\
							&	& & & H_2 & H_0 & H_1 \\
\end{bmatrix}
}
$$
负责处理：
$$
s_4,s_5,s_6.
$$
commit：
$$
e_7,e_8,e_9
$$
\(B_{\mathrm{1}}\)矩阵如下：
$$
\boxed{
B_1
=
\begin{bmatrix}
H_0 & H_1 & & & \\
			& H_2 & H_0 & H_1 & \\
							& & & H_2 & H_0\\
\end{bmatrix}
}
$$
负责处理：
$$
s^{(L)}_2,s_3,s^{(R)}_4.
$$

$$
s^{(L)}_2 = s_2 + H_2e_1\\
s^{(R)}_4 = s_4 + H_1e_7
$$

commit：
$$
e_2,e_3,....,e_6
$$

### 6.1 左边界更新

全局第 \(u\) 行为：

$$
s_u
=
H_2e_{2u-3}
+
H_0e_{2u-2}
+
H_1e_{2u-1}.
$$

左侧 A window 已经给出：

$$
\hat e_{2u-3}.
$$

因此 B window 的左端 residual 为：

$$
\boxed{
s_u^{(L)}
=
s_u+H_2\hat e_{2u-3}.
}
$$

这表示将左侧 A window 已经确定的边界贡献从 \(s_u\) 中抵消掉。

---

### 6.2 右边界更新

全局第 \(v\) 行为：

$$
s_v
=
H_2e_{2v-3}
+
H_0e_{2v-2}
+
H_1e_{2v-1}.
$$

右侧 A window 已经给出：

$$
\hat e_{2v-1}.
$$

因此 B window 的右端 residual 为：

$$
\boxed{
s_v^{(R)}
=
s_v+H_1\hat e_{2v-1}.
}
$$

这不是“未来影响过去”，而是因为 \(e_{2v-1}\) 本来就是连接 \(s_v\) 和 \(s_{v+1}\) 的跨轮变量。右侧 A window 先估计了该边界变量，B window 在求解前将其对 \(s_v\) 的贡献 XOR 掉。

---

## 7. B Window 局部方程

B window 要解的变量为：

$$
\boxed{
e_{2u-2},e_{2u-1},\dots,e_{2v-2}.
}
$$

它不包括：

$$
e_{2u-3}
$$

和：

$$
e_{2v-1},
$$

因为这两个变量已经分别由左侧 A 和右侧 A 给出。

因此 B window 的局部方程为：

$$
\boxed{
H_B\hat e_B=s_B^{\mathrm{res}}.
}
$$

其中：

$$
\hat e_B
=
\begin{bmatrix}
\hat e_{2u-2} \\
\hat e_{2u-1} \\
\vdots \\
\hat e_{2v-2}
\end{bmatrix}.
$$

右端 residual detector vector 为：

$$
\boxed{
s_B^{\mathrm{res}}
=
\begin{bmatrix}
s_u+H_2\hat e_{2u-3} \\
s_{u+1} \\
\vdots \\
s_{v-1} \\
s_v+H_1\hat e_{2v-1}
\end{bmatrix}.
}
$$

局部矩阵为：

$$
\boxed{
H_B
=
\begin{bmatrix}
H_0 & H_1 & & & \\
& H_2 & H_0 & H_1 & \\
& & & H_2 & H_0 & H_1 \\
& & & & \ddots & \ddots & \ddots \\
& & & & & H_2 & H_0
\end{bmatrix}.
}
$$

注意：

- 第一行没有 \(H_2\)，因为左边界变量已经被移到右端；
- 最后一行没有 \(H_1\)，因为右边界变量已经被移到右端；
- 中间行保持完整的 \(H_2,H_0,H_1\) 结构。

---

## 8. 示例：\(B=[2,4]\)

若：

$$
B=[2,4],
$$

则：

$$
u=2,\quad v=4.
$$

左边界变量是：

$$
e_{2u-3}=e_1.
$$

右边界变量是：

$$
e_{2v-1}=e_7.
$$

B window 解的变量是：

$$
e_2,e_3,e_4,e_5,e_6.
$$

局部方程为：

$$
\boxed{
\begin{bmatrix}
H_0 & H_1 & 0 & 0 & 0 \\
0 & H_2 & H_0 & H_1 & 0 \\
0 & 0 & 0 & H_2 & H_0
\end{bmatrix}
\begin{bmatrix}
\hat e_2 \\
\hat e_3 \\
\hat e_4 \\
\hat e_5 \\
\hat e_6
\end{bmatrix}
=
\begin{bmatrix}
s_2+H_2\hat e_1 \\
s_3 \\
s_4+H_1\hat e_7
\end{bmatrix}.
}
$$

展开为：

$$
H_0\hat e_2+H_1\hat e_3
=
s_2+H_2\hat e_1,
$$

$$
H_2\hat e_3+H_0\hat e_4+H_1\hat e_5
=
s_3,
$$

$$
H_2\hat e_5+H_0\hat e_6
=
s_4+H_1\hat e_7.
$$

---

## 9. 正确性检查

若 B window 的局部方程成立，则它可以恢复对应区间的全局方程。

### 第一行 \(t=u\)

B 方程给出：

$$
H_0\hat e_{2u-2}+H_1\hat e_{2u-1}
=
s_u+H_2\hat e_{2u-3}.
$$

两边加上 \(H_2\hat e_{2u-3}\)，得到：

$$
H_2\hat e_{2u-3}
+
H_0\hat e_{2u-2}
+
H_1\hat e_{2u-1}
=
s_u.
$$

这正是全局第 \(u\) 行。

---

### 中间行 \(u<t<v\)

B 方程直接给出：

$$
H_2\hat e_{2t-3}
+
H_0\hat e_{2t-2}
+
H_1\hat e_{2t-1}
=
s_t.
$$

这正是全局第 \(t\) 行。

---

### 最后一行 \(t=v\)

B 方程给出：

$$
H_2\hat e_{2v-3}
+
H_0\hat e_{2v-2}
=
s_v+H_1\hat e_{2v-1}.
$$

两边加上 \(H_1\hat e_{2v-1}\)，得到：

$$
H_2\hat e_{2v-3}
+
H_0\hat e_{2v-2}
+
H_1\hat e_{2v-1}
=
s_v.
$$

这正是全局第 \(v\) 行。

因此，如果左右 A windows 给出的边界变量可信，并且 B window 局部方程满足，则 B 区间内的全局 syndrome equations 都被满足。

---

## 10. 并行性

qLDPC 并行窗口解码的并行性来自两个层面。

### 10.1 A Layer 并行性

多个 A windows 如果时间上不重叠，并且最终提交的变量集合不冲突，则可以同时解：

$$
H_{A_i}\hat e_{A_i}=s_{A_i},
\quad i=1,2,\dots
$$

即：

$$
\boxed{
A_1,A_2,A_3,\dots \text{ 可以并行解码。}
}
$$

每个 A window 输出左右边界变量，并更新相邻 B window 的 residual detector。

---

### 10.2 B Layer 并行性

A layer 完成后，多个 B windows 的左右边界条件都已经确定。

若不同 B windows 的 detector intervals 不重叠，则可以同时解：

$$
H_{B_i}\hat e_{B_i}=s_{B_i}^{\mathrm{res}},
\quad i=1,2,\dots
$$

即：

$$
\boxed{
B_1,B_2,B_3,\dots \text{ 可以并行解码。}
}
$$

---

### 10.3 Layer 依赖关系

A 和 B 不能完全同时独立解，因为 B 的输入依赖 A 输出的边界变量：

$$
s_u^{(L)}
=
s_u+H_2\hat e_{2u-3},
$$

$$
s_v^{(R)}
=
s_v+H_1\hat e_{2v-1}.
$$

因此调度为：

$$
\boxed{
\text{A layer}
\rightarrow
\text{residual 更新}
\rightarrow
\text{B layer}.
}
$$

这是一种逐层并行，而不是所有窗口完全同时独立。

---

## 11. 与 Surface-Code Parallel Window 的区别

Surface code parallel window 通常使用：

$$
w=d
$$

来选择 A/B/buffer 的大小。原因是 surface code 的 decoding graph 有二维空间几何局部性，穿过 \(d\)-round buffer 的 error chain 至少有 \(d\) 个局部错误事件，已经达到 code distance 尺度。

但 qLDPC 的 detector graph 是 hypergraph。一个 fault 可以触发多个 detectors：

$$
H_{\mathrm{circ}}[:,j]
=
\mathbf 1_{v_1}+\cdots+\mathbf 1_{v_r}.
$$

因此 qLDPC 的 code distance \(d\) 不一定对应二维几何错误链长度，也不能直接用 \(w=d\) 来规定 A/B/buffer。

qLDPC 并行窗口解码的依据不是几何错误链，而是：

$$
\boxed{
H_{\mathrm{circ}}\text{ 的时间方向块带状结构。}
}
$$

即每个 fault 只影响有限个连续 detector blocks，因此可以沿时间轴拆分为 A/B windows。

---

## 12. 算法草图

### 输入

- Detector 取值：

$$
s_1,s_2,\dots,s_R.
$$

- Circuit detector matrix：

$$
H_{\mathrm{circ}}.
$$

- Window 划分：

$$
A_1,B_1,A_2,B_2,\dots
$$

- 内部解码器：

$$
\mathcal D
$$

例如 BP+OSD、GDG、Vegapunk-like local decoder 等。

---

### 步骤 1：A Layer 解码

对每个 A window \(A_i\)，并行执行：

$$
\hat e_{A_i}
=
\mathcal D(H_{A_i},s_{A_i}).
$$

提取边界变量，例如：

$$
\hat e_{2u_i-3},\quad \hat e_{2v_i-1}.
$$

只提交分配给 A 的变量。属于后续 B windows 的变量应视为临时估计，而不是最终提交。

---

### 步骤 2：Residual 更新

对每个 B window \(B_i=[u_i,v_i]\)，计算：

$$
s_{u_i}^{(L)}
=
s_{u_i}+H_2\hat e_{2u_i-3},
$$

$$
s_{v_i}^{(R)}
=
s_{v_i}+H_1\hat e_{2v_i-1}.
$$

构造：

$$
s_{B_i}^{\mathrm{res}}
=
\begin{bmatrix}
s_{u_i}^{(L)} \\
s_{u_i+1} \\
\vdots \\
s_{v_i-1} \\
s_{v_i}^{(R)}
\end{bmatrix}.
$$

---

### 步骤 3：B Layer 解码

对每个 B window \(B_i\)，并行执行：

$$
\hat e_{B_i}
=
\mathcal D(H_{B_i},s_{B_i}^{\mathrm{res}}).
$$

提交属于 \(B_i\) 的变量。

---

### 步骤 4：组装全局 correction

组合 A 和 B 已提交的变量：

$$
\hat e
=
\sum_i \hat e_{A_i}^{\mathrm{commit}}
+
\sum_i \hat e_{B_i}^{\mathrm{commit}}.
$$

如果每个 A/B partial equation 都成立，则组装后的 correction 满足：

$$
H_{\mathrm{circ}}\hat e=s.
$$

---

## 13. 关键要求

该算法需要满足：

1. **有限时间带宽**

   $$
   \operatorname{supp}_t(H_{\mathrm{circ}}[:,j])
   \subseteq \{t,t+1\}
   $$

   或者更一般地说，每个 fault 只影响有界数量的连续 detector rounds。

2. **无冲突提交**

   A 和 B windows 不能以不一致的方式提交同一个 fault variable。

3. **边界一致性**

   B windows 必须通过 residual update 使用 A 提供的边界变量。

4. **Partial equation 满足性**

   任何离开 window 的 detector row 都必须由已提交变量完全解释。

5. **可靠的内部解码器**

   内部解码器应输出一个满足 window syndrome equation 的向量，或者至少满足已提交部分对应的 partial equation。Plain BP 可能因为不收敛而无法满足这一点；qLDPC sliding window 论文使用 BP+OSD 和 GDG 来改善这一问题。

---

## 14. 总结

qLDPC 并行窗口解码可以总结为：

$$
\boxed{
\text{使用 }H_{\mathrm{circ}}\text{ 暴露时间局域的 detector 耦合。}
}
$$

$$
\boxed{
\text{并行解码相互分离的 A windows。}
}
$$

$$
\boxed{
\text{将 A 输出作为边界变量，用于更新 residual detectors。}
}
$$

$$
\boxed{
\text{并行解码中间的 B windows。}
}
$$

对于一个中间 B window \(B=[u,v]\)，其核心局部方程为：

$$
\boxed{
\begin{bmatrix}
H_0 & H_1 & & & \\
& H_2 & H_0 & H_1 & \\
& & & \ddots & \ddots \\
& & & & H_2 & H_0
\end{bmatrix}
\begin{bmatrix}
\hat e_{2u-2} \\
\hat e_{2u-1} \\
\vdots \\
\hat e_{2v-2}
\end{bmatrix}
=
\begin{bmatrix}
s_u+H_2\hat e_{2u-3} \\
s_{u+1} \\
\vdots \\
s_{v-1} \\
s_v+H_1\hat e_{2v-1}
\end{bmatrix}.
}
$$

这是 surface-code parallel window stitching 在矩阵层面的对应形式；不过这里是通过 \(H_{\mathrm{circ}}\) 适配 qLDPC 的 detector hypergraph，而不是依赖二维几何错误链的论证。

---

## 15. 严谨对比下的 Buffer 对齐策略

前面的 \(n_A=3,n_B=3\) 和 \(n_A^{\mathrm{solve}}=5,n_B=3\) 排布可以说明 A/B stitching 的矩阵结构，但它们不应作为和 sliding baseline 的最终公平对比设置。

原因是：若 parallel window 使用更大的 A solve context，而 B window 仍保持较小宽度，那么 parallel 与 sliding 看到的边界 buffer 厚度并不一致。这样的实验能说明扩大 A solve window 有帮助，但不能严格说明 parallel window 在同等 buffer 条件下优于或接近 sliding。

因此，正式对比实验应显式对齐 buffer：

$$
\boxed{
\text{parallel 中每个局部窗口的 buffer 厚度应与 sliding baseline 的 buffer 厚度一致。}
}
$$

如果 sliding baseline 使用：

$$
\text{buffer}=2,
$$

则 parallel window 也使用同样的 buffer 厚度。

需要注意的是，parallel 的 interior A window 位于两个 B window 之间，因此它有左右两个 buffer：

$$
\boxed{
\text{A solve} = \text{left buffer} + \text{center} + \text{right buffer}.
}
$$

所以当 buffer 为 2 时：

$$
n_A^{\mathrm{solve}}=2+1+2=5.
$$

这并不是给 parallel A window 使用更厚的 buffer，而是把与 sliding 相同厚度的 buffer 分别放在 A commit 区域两侧。因此 parallel 的完整 interior A solve window 总行数会更大，但单侧 buffer 厚度仍与 sliding 一致。

---

### 15.1 采用的正式参数

后续严谨对比应采用：

```text
buffer = 2
a_solve_size = 5
b_width = 5
step = 6
```

其中：

- `buffer = 2`：与 sliding baseline 的 buffer 厚度对齐；
- `a_solve_size = 5`：interior A window 包含左右各 2 行 buffer；
- `b_width = 5`：B window 也使用相同尺度的局部范围；
- `step = 6`：同类窗口之间的起点间隔，使 A windows 之间不重叠，B windows 之间也不重叠。

这里的 `step` 为：

$$
\boxed{
\mathrm{step}=a\_solve\_size+1=6.
}
$$

---

### 15.2 非重叠 A 与 B 的交错排布

在 `buffer = 2`、`a_solve_size = 5`、`b_width = 5` 时，排布应为：

```text
A1 solve: s1-s3
B1 solve:    s2-s6
A2 solve:          s5-s9
B2 solve:             s8-s12
A3 solve:                   s11-s15
B3 solve:                      s14-s18
A4 solve:                            s17-s21
```

对应的 A solve windows 是：

```text
A1 solve: s1-s3
A2 solve: s5-s9
A3 solve: s11-s15
A4 solve: s17-s21
```

对应的 B solve windows 是：

```text
B1 solve: s2-s6
B2 solve: s8-s12
B3 solve: s14-s18
B4 solve: s20-s24
```

这样有两个性质：

```text
A windows 之间不重叠
B windows 之间不重叠
```

但 A 和 B 之间允许交错覆盖。例如：

```text
A2 solve: s5,s6,s7,s8,s9
B2 solve:          s8,s9,s10,s11,s12
```

它们共享 \(s_8,s_9\) 两行。这正对应 `buffer = 2` 的理解：B window 覆盖 A 的右侧 buffer 和下一段 residual 区域。由于 B layer 在 A layer commit 并完成 residual update 后才执行，A/B 之间的交错不是并行冲突。

---

### 15.3 A Window 的边界与 Interior

左边界 A window 没有完整的左 buffer，因此：

```text
A1 solve: s1-s3
```

它可以看成：

```text
center + right buffer = 1 + 2
```

从 \(A_2\) 开始，interior A window 有完整左右 buffer：

```text
A2 solve: s5-s9
```

也就是：

```text
left buffer = s5,s6
center      = s7
right buffer= s8,s9
```

如果仍然用 \(A_2\) 提供一段局部边界变量，则这些 commit 变量应位于 A solve window 的中间区域，而不是贴在 \(s_5\) 或 \(s_9\) 的边缘。核心原则保持不变：

$$
\boxed{
\text{A window 可以解更大的局部方程，但只 commit 中间可靠区域。}
}
$$

---

### 15.4 生成规则

使用 0-based detector block index 时，上面的排布为：

```text
A1: rows [0, 3)    -> s1-s3
A2: rows [4, 9)    -> s5-s9
A3: rows [10, 15)  -> s11-s15
A4: rows [16, 21)  -> s17-s21

B1: rows [1, 6)    -> s2-s6
B2: rows [7, 12)   -> s8-s12
B3: rows [13, 18)  -> s14-s18
B4: rows [19, 24)  -> s20-s24
```

可写成：

```text
buffer = 2
a_solve_size = 2 * buffer + 1
b_width = a_solve_size
step = a_solve_size + 1

A1.row_start = 0
A1.row_stop = buffer + 1

for i >= 2:
    A_i.row_start = buffer + 2 + (i - 2) * step
    A_i.row_stop = A_i.row_start + a_solve_size

for i >= 1:
    B_i.row_start = 1 + (i - 1) * step
    B_i.row_stop = B_i.row_start + b_width
```

最后一个 window 需要截断到 detector history 末尾：

```text
row_stop = min(row_stop, num_detector_blocks)
```

---

### 15.5 A/B 分别 Commit 哪些变量

在该排布中，每个 \(B_i=[u_i,v_i]\) 处理连续 detector rows：

$$
s_{u_i},s_{u_i+1},\dots,s_{v_i}.
$$

它的左右边界变量分别由相邻 A windows 提供：

$$
\text{left boundary}=e_{2u_i-3},
\qquad
\text{right boundary}=e_{2v_i-1}.
$$

因此 \(B_i\) 自己 commit 的变量为：

$$
\boxed{
B_i\text{ commit}: e_{2u_i-2},e_{2u_i-1},\dots,e_{2v_i-2}.
}
$$

也就是说，B window 不 commit 左右两个边界外变量，只 commit 它 residual 方程内部的变量。

以 `b_width = 5` 的新排布为例：

```text
B1 solve:   s2-s6
B1 commit:  e2,e3,e4,e5,e6,e7,e8,e9,e10

B2 solve:   s8-s12
B2 commit:  e14,e15,e16,e17,e18,e19,e20,e21,e22

B3 solve:   s14-s18
B3 commit:  e26,e27,e28,e29,e30,e31,e32,e33,e34
```

A window 的作用是补上 B windows 之间的边界变量。左边界 \(A_1\) commit 初始变量：

```text
A1 solve:   s1-s3
A1 commit:  e0,e1
```

其中 \(e_1\) 是 \(B_1\) 的左边界变量。

从 \(A_2\) 开始，每个 interior A window commit 位于自身 solve window 中间的 3 个变量：

```text
A2 solve:   s5-s9
A2 commit:  e11,e12,e13

A3 solve:   s11-s15
A3 commit:  e23,e24,e25

A4 solve:   s17-s21
A4 commit:  e35,e36,e37
```

这里：

- \(e_{11}\) 是 \(B_1=[2,6]\) 的右边界变量；
- \(e_{13}\) 是 \(B_2=[8,12]\) 的左边界变量；
- \(e_{12}\) 是夹在两个边界变量中间、与该 A 区域绑定的内部变量。

因此全局提交顺序可以看成：

```text
A1 commit: e0,e1
B1 commit: e2-e10
A2 commit: e11,e12,e13
B2 commit: e14-e22
A3 commit: e23,e24,e25
B3 commit: e26-e34
A4 commit: e35,e36,e37
...
```

这个分配保证：

```text
每个 e 只由一个 window commit
A commit 的变量正好作为相邻 B window 的边界条件
B commit 的变量正好覆盖两个 A 边界之间的 residual 区域
```

---

### 15.6 对旧实验结论的修正

旧的 `a_solve_size = 5, b_width = 3` 或 `a_solve_size = 9, b_width = 3` 实验仍然有诊断价值：它们说明 A boundary 的可靠性会随着 A solve context 增大而提升，也说明 A/B residual stitching 的代数结构可以闭合。

但这些实验不应作为与 sliding baseline 的最终严谨性能对比，因为：

```text
parallel 的 A 局部上下文被增大了，
但 B window 和 sliding baseline 的 buffer 尺度没有同步对齐。
```

正式对比应使用本节的 buffer-aligned 排布。只有在相同 buffer 厚度下比较 LER、flagged count 和 runtime，才能说明 parallel window 的收益来自并行调度和 A/B 分层结构，而不是来自更宽的局部解码上下文。
