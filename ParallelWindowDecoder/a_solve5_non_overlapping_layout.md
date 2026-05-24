# a_solve_size = 5 的非重叠 A Window 排布

本文说明一种新的 A/B parallel window 排布方式：当

```text
a_size = 3
a_solve_size = 5
b_width = 5
```

时，让不同的 A solve window 之间不再共享 syndrome rows。

这个排布的目的不是单纯缩放窗口，而是为了保持对比实验的严谨性：

```text
parallel window 中每个局部窗口的 buffer 大小
需要和 sliding baseline 中使用的 buffer 大小一致
```

这样比较 parallel 和 sliding 时，窗口边界可见的上下文大小是一致的，
不会因为 parallel 使用了更大的局部上下文而获得不公平优势。

需要注意的是，parallel 的 A window 是夹在两个方向之间的局部解码窗口，
所以一个完整的 interior A window 会同时带有：

```text
left buffer = 2
right buffer = 2
```

因此从总窗口宽度看，parallel 的 interior A solve window 会比单侧 buffer
意义下的 sliding window 更大。这里的严谨性指的是 buffer 厚度一致，而不是
所有局部矩阵的总行数完全相同。

核心变化是：

```text
旧排布：A solve window 围绕原来的 A commit 区域向左右扩张
新排布：把 buffer 理解为 2，让 A/B window 交错前进，但同类 A 之间不重叠
```

也就是说，`a_solve_size = 5` 时，每个 A window 解连续 5 个 syndrome
block，但相邻 A window 的 solve 区间不再交叉。左边界的 `A1` 因为没有完整左
buffer，所以只解 `s1-s3`。

---

## 1. 原来的重叠排布

原 staggered A/B 排布中，A commit 区间保持较小：

```text
A1 commit: e1,e2
A2 commit: e7,e8,e9
A3 commit: e15,e16,e17
...
```

当使用

```text
a_size = 3
a_solve_size = 5
```

时，A solve window 会在原 A commit 附近增加左右 buffer。

典型排布类似：

```text
A1 solve: s1,s2,s3
A2 solve: s3,s4,s5,s6,s7
A3 solve: s7,s8,s9,s10,s11
A4 solve: s11,s12,s13,...
```

这里的问题是：

```text
A1 和 A2 共享 s3
A2 和 A3 共享 s7
A3 和 A4 共享 s11
```

因此虽然每个 A window 可以并行解码，但它们在 syndrome rows 上不是完全独立的。

---

## 2. 新的非重叠排布

新的想法是：把 `a_solve_size = 5` 看成：

```text
left buffer = 2
center = 1
right buffer = 2
```

因此 interior A window 占据连续 5 个 syndrome block。为了让不同 A
之间不重叠，相邻 A 的起点间隔设为：

```text
step = a_solve_size + 1 = 6
```

例如：

```text
A1 solve: s1,s2,s3
A2 solve: s5,s6,s7,s8,s9
A3 solve: s11,s12,s13,s14,s15
A4 solve: s17,s18,s19,s20,s21
...
```

其中你提出的例子就是：

```text
A2 solve: s5,s6,s7,s8,s9
```

这样：

```text
A1 solve 和 A2 solve 没有重叠
A2 solve 和 A3 solve 没有重叠
A3 solve 和 A4 solve 没有重叠
```

如果写成区间形式：

```text
A1 solve: s1-s3
A2 solve: s5-s9
A3 solve: s11-s15
A4 solve: s17-s21
```

这里 `A1` 根据边界条件少两行，因为最左侧没有完整的左 buffer。
从 `A2` 开始，每个 interior A 都使用完整的 5 行 solve window。

---

## 3. 为什么这样对齐 Sliding Buffer

在 sliding baseline 中，窗口边界附近会保留一个固定厚度的 buffer。
如果 sliding 的 buffer 设为 2，那么 parallel 的局部窗口也应该使用同样的
buffer 厚度：

```text
buffer = 2
```

这样对比时，两个 decoder 在窗口边界处看到的额外 syndrome 上下文是同一尺度。

不过 parallel A window 和 sliding window 的几何位置不同。parallel 的
interior A window 位于中间，需要同时保护左右两侧边界，因此它自然包含两个
buffer：

```text
A_k solve = left buffer + center + right buffer
          = 2 + 1 + 2
          = 5
```

这就是为什么在 `buffer = 2` 时，parallel 的完整 interior A solve window
会写成：

```text
A2 solve: s5,s6,s7,s8,s9
```

它并不是把 sliding 的 buffer 变大了，而是把相同厚度的 buffer 分别放在
A commit 区域的两侧。

---

## 4. 与 B Window 的关系

B window 也随之改成宽度 5：

```text
b_width = 5
```

一种对应排布可以写成：

```text
A1 solve: s1-s3
B1 solve: s2-s6

A2 solve: s5-s9
B2 solve: s8-s12

A3 solve: s11-s15
B3 solve: s14-s18

A4 solve: s17-s21
B4 solve: s20-s24
```

也就是说，B window 也按长度 5 向右铺开。相对对应的 interior A window，
B 向右错开 3 个 syndrome block：

```text
A_k solve:  s_m, s_{m+1}, s_{m+2}, s_{m+3}, s_{m+4}
B_k solve:                 s_{m+3}, s_{m+4}, s_{m+5}, s_{m+6}, s_{m+7}
```

例如：

```text
A2 solve: s5,s6,s7,s8,s9
B2 solve:          s8,s9,s10,s11,s12
```

你提到的第一个 B window 就是：

```text
B1 solve: s2,s3,s4,s5,s6
```

这样 A 和 B 形成交错覆盖：

```text
A1 solve: s1-s3
B1 solve:    s2-s6
A2 solve:          s5-s9
B2 solve:             s8-s12
A3 solve:                   s11-s15
```

A window 之间仍然不重叠；B window 之间也不重叠。A/B 之间有 2 行交叠，
这正对应这里的 `buffer = 2` 理解：B 会覆盖 A 的右侧 buffer 和下一段区域，
在 A 层 commit 后处理 residual syndrome。

---

## 5. A Commit 区间

非重叠排布只改变 A 的 solve rows，不必让 A commit 也变大。

对于 interior A window，仍然可以保持：

```text
A commit size = 3
```

例如：

```text
A2 solve:  s5,s6,s7,s8,s9
A2 commit: e11,e12,e13
```

这里 commit 变量位于 A2 solve window 的中间，而不是贴在 solve window
边缘。直观上：

```text
s5 | s6 | s7 | s8 | s9
      \  commit  /
```

这保留了扩大 A solve window 的主要目的：

```text
让 A commit 变量远离局部窗口边界
```

同时又避免了不同 A window 之间在 syndrome rows 上重叠。

---

## 6. 与旧 a_solve = 5 排布的对比

旧排布：

```text
A1 solve: s1-s3
A2 solve: s3-s7
A3 solve: s7-s11
A4 solve: s11-s15
```

特点：

```text
每个 interior A 有 5 行
相邻 A 之间共享 1 行
A 起点间隔是 4
```

新排布：

```text
A1 solve: s1-s3
A2 solve: s5-s9
A3 solve: s11-s15
A4 solve: s17-s21
```

特点：

```text
每个 interior A 有 5 行
相邻 A 之间没有共享 syndrome rows
A 起点间隔是 6
```

因此，关键变化可以概括为：

```text
旧：next_A_start = current_A_start + 4
新：next_A_start = current_A_start + 6
```

对于 `a_solve_size = 5`，新的 A solve 起点可以简单写成：

```text
A1: start = 1
A2: start = 5
A3: start = 11
A4: start = 17
...
```

对应的 B solve 起点则是：

```text
B1: start = 2
B2: start = 8
B3: start = 14
B4: start = 20
...
```

也就是 B window 自己也按步长 6 向右移动：

```text
B_{k+1}.start = B_k.start + 6
```

注意：因为 `A1` 是左边界窗口，只有 `s1-s3`，所以 `B1` 相对 `A1`
只右移 1 行；从 interior window 开始，`B2` 相对 `A2` 是右移 3 行：

```text
A2 start = 5
B2 start = 8
```

如果统一使用 0-based detector block index，则是：

```text
A1: rows [0, 3)
A2: rows [4, 9)
A3: rows [10, 15)
A4: rows [16, 21)
...
```

---

## 7. 实现时的生成规则

可以把新的 A solve window 生成规则写成：

```text
buffer = 2
step = a_solve_size + 1

A1 solve = [0, buffer + 1)
A2 solve = [buffer + 2, buffer + 2 + a_solve_size)
A3 solve = [buffer + 2 + step, buffer + 2 + step + a_solve_size)
...
```

代入 `a_solve_size = 5`：

```text
A1 solve = [0, 3)   -> s1-s3
A2 solve = [4, 9)   -> s5-s9
A3 solve = [10, 15) -> s11-s15
A4 solve = [16, 21) -> s17-s21
```

最后一个 A window 如果超过总 syndrome block 数，则截断到末尾：

```text
row_stop = min(row_start + a_solve_size, num_blocks)
```

这样可以保证：

```text
对于任意相邻 A_i, A_{i+1}：
A_i.row_stop <= A_{i+1}.row_start
```

也就是 A solve window 之间没有重叠。

B solve window 可以类似生成：

```text
B1 solve = [1, 6)    -> s2-s6
B2 solve = [7, 12)   -> s8-s12
B3 solve = [13, 18)  -> s14-s18
B4 solve = [19, 24)  -> s20-s24
```

也就是：

```text
b_width = a_solve_size = 5
b_row_start = 1 + b_index * step
b_row_stop = b_row_start + b_width
```

最后一个 B window 同样需要截断到 syndrome block 末尾：

```text
b_row_stop = min(b_row_start + b_width, num_blocks)
```
