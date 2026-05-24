# Buffer-aligned parallel window test

记录时间：

```text
2026-05-23 17:59:26 CST
```

## 1. 实验目的

之前的 parallel 对比实验使用过：

```text
a_solve_size = 5 或 9
b_width = 3
```

这种设置可以说明扩大 A solve window 会提高 A boundary 的可靠性，但和
sliding baseline 的对比不够严谨。原因是 parallel 的 A 局部上下文被增大，
而 B window 和 sliding 的 buffer 尺度没有同步对齐。

本次实验改为 buffer-aligned 策略：

```text
buffer = 2
a_solve_size = 5
b_width = 5
step = 6
```

这样每个局部窗口的 buffer 厚度和 sliding baseline 保持一致。需要注意的是，
parallel 的 interior A window 有左右两个 buffer，因此 A solve window 的总
宽度会更大：

```text
A solve = left buffer + center + right buffer
        = 2 + 1 + 2
        = 5
```

## 2. 代码修改

修改文件：

```text
ParallelWindowDecoder/parallel_window_decoder.py
```

当参数满足：

```text
--a-size 3
--a-solve-size 5
--b-width 5
```

时，`decode_staggered_ab` 自动使用新的 `buffer_aligned` schedule。

新 schedule 生成：

```text
A1 solve: s1-s3
B1 solve:    s2-s6
A2 solve:          s5-s9
B2 solve:             s8-s12
A3 solve:                   s11-s15
```

在 `N=144, num_repeat=12` 的实际 detector history 中，诊断输出为：

```text
schedule: buffer_aligned
A rows: ['s1-s3', 's5-s9', 's11-s13']
B rows: ['s2-s6', 's8-s12']
A commit: ['e0-e1', 'e11-e13', 'e23-e24']
B commit: ['e2-e10', 'e14-e22']
step: 6
```

最后一个 A window 被截断为 `s11-s13`，因为 detector blocks 总数为 13。

## 3. 运行命令

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --p-list 0.003,0.004,0.005 \
  --num-repeat 12 \
  --num-shots 100 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 5 \
  --osd-order 10 \
  --max-iter 200 \
  --parallel-workers 4 \
  --parallel-backend process \
  --decoders sliding,parallel,oracle \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_buffer_aligned_a_solve5_b5_process_workers4.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_buffer_aligned_a_solve5_b5_process_workers4.csv
```

## 4. 实验结果

| p | decoder | LER | flagged | logical_or_flagged | elapsed |
|---:|---|---:|---:|---:|---:|
| 0.003 | sliding | 0.00 | 0/100 | 0/100 | 6.853s |
| 0.003 | parallel | 0.01 | 1/100 | 1/100 | 3.180s |
| 0.003 | oracle | 0.00 | 0/100 | 0/100 | 0.918s |
| 0.004 | sliding | 0.02 | 0/100 | 2/100 | 9.871s |
| 0.004 | parallel | 0.07 | 6/100 | 7/100 | 3.889s |
| 0.004 | oracle | 0.01 | 0/100 | 1/100 | 1.340s |
| 0.005 | sliding | 0.06 | 0/100 | 6/100 | 13.756s |
| 0.005 | parallel | 0.18 | 17/100 | 18/100 | 5.444s |
| 0.005 | oracle | 0.01 | 0/100 | 1/100 | 1.370s |

## 5. 结论

本次结果和学长的判断一致：在 buffer 对齐之后，parallel 的准确率并不高。

具体表现为：

```text
p = 0.003: parallel LER 0.01, flagged 1/100
p = 0.004: parallel LER 0.07, flagged 6/100
p = 0.005: parallel LER 0.18, flagged 17/100
```

相比 sliding：

```text
p = 0.003: sliding LER 0.00
p = 0.004: sliding LER 0.02
p = 0.005: sliding LER 0.06
```

parallel 速度更快，但准确率明显更差。oracle 结果仍然较好，说明 A/B residual
stitching 的代数结构基本成立；主要瓶颈仍然是 decoded A boundary 的可靠性。

---

## 6. 1000-shot 修正版：sliding window = 3

上面的 100-shot 结果已经说明趋势。为了进一步确认，并加入更高错误率
`p = 0.007`，重新运行 1000-shot 实验。

注意：这里 sliding baseline 必须使用：

```text
sliding_width = 3
```

因为 sliding 的 window 大小为 3 时，对应的 buffer 厚度才是 2，才能和
parallel 的：

```text
a_solve_size = 5
b_width = 5
buffer = 2
```

严格对齐。曾短暂启动过 `sliding_width = 5` 的实验，但该设置不符合这一定义，
已中止，不作为正式结果。

### 6.1 运行命令

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --p-list 0.003,0.004,0.005,0.007 \
  --num-repeat 12 \
  --num-shots 1000 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --osd-order 10 \
  --max-iter 200 \
  --parallel-workers 4 \
  --parallel-backend process \
  --decoders sliding,parallel,oracle \
  --out ParallelWindowDecoder/results/N144_repeat12_shots1000_buffer_aligned_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots1000_buffer_aligned_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

parallel diagnostics 确认仍然使用 buffer-aligned 排布：

```text
schedule: buffer_aligned
A rows: ['s1-s3', 's5-s9', 's11-s13']
B rows: ['s2-s6', 's8-s12']
A commit: ['e0-e1', 'e11-e13', 'e23-e24']
B commit: ['e2-e10', 'e14-e22']
step: 6
```

### 6.2 1000-shot 结果

| p | decoder | LER | flagged | logical_or_flagged | elapsed |
|---:|---|---:|---:|---:|---:|
| 0.003 | sliding | 0.003 | 0/1000 | 3/1000 | 58.705s |
| 0.003 | parallel | 0.014 | 11/1000 | 14/1000 | 21.166s |
| 0.003 | oracle | 0.001 | 0/1000 | 1/1000 | 6.607s |
| 0.004 | sliding | 0.023 | 0/1000 | 23/1000 | 69.483s |
| 0.004 | parallel | 0.052 | 42/1000 | 52/1000 | 30.141s |
| 0.004 | oracle | 0.004 | 0/1000 | 4/1000 | 7.534s |
| 0.005 | sliding | 0.107 | 0/1000 | 107/1000 | 82.850s |
| 0.005 | parallel | 0.161 | 123/1000 | 161/1000 | 42.166s |
| 0.005 | oracle | 0.026 | 0/1000 | 26/1000 | 10.499s |
| 0.007 | sliding | 0.561 | 0/1000 | 561/1000 | 120.528s |
| 0.007 | parallel | 0.665 | 500/1000 | 665/1000 | 68.944s |
| 0.007 | oracle | 0.257 | 0/1000 | 257/1000 | 21.331s |

### 6.3 结论

1000-shot 结果进一步确认：

```text
parallel 的 wall-clock 更短，但准确率低于 sliding；
错误率越高，parallel 的 decoded A boundary 问题越明显。
```

尤其在 `p = 0.007` 时：

```text
sliding LER  = 0.561
parallel LER = 0.665
parallel flagged = 500/1000
```

oracle 仍然比 decoded parallel 好很多，说明如果 A boundary 给对，B residual
结构仍有较强闭合能力；但真实 decoded A boundary 在该 buffer-aligned 设置下
不够可靠。

---

## 7. Top-K Boundary Stitching 初步尝试

### 7.1 代码备份

尝试 Top-K 前，已备份当前代码：

```text
ParallelWindowDecoder/backups/parallel_window_decoder.before_topk.py
ParallelWindowDecoder/backups/run_experiments.before_topk.py
```

### 7.2 实现方式

新增参数：

```text
--top-k-boundary
```

默认值为：

```text
--top-k-boundary 1
```

因此不传该参数时，仍保持原来的 hard-boundary 行为。

当使用：

```text
--top-k-boundary 2
```

且 schedule 为 `buffer_aligned` 时，A window 不只输出一个 commit 结果，而是输出
2 个候选。

候选生成方式：

1. A window 先正常 BP+OSD 解码，得到 top-1 commit；
2. 读取 `BpOsdDecoder.log_prob_ratios`；
3. 在 A commit 区域中找到最低可靠度的 bit；
4. 修改该 bit 的 prior 后重新解 A window，得到另一个候选；
5. B window 枚举左右 A 候选组合；
6. 用 B 局部 residual 是否闭合和 channel cost 选择总代价最低的拼接。

这不是完整的 OSD top-K list，而是一个工程上较小的 top-K 近似版本。它的目的
是验证：

```text
A boundary hard decision 是否确实是主要瓶颈
```

### 7.3 运行命令

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --p-list 0.003,0.004,0.005,0.007 \
  --num-repeat 12 \
  --num-shots 100 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --osd-order 10 \
  --max-iter 200 \
  --parallel-workers 4 \
  --parallel-backend process \
  --top-k-boundary 2 \
  --decoders parallel \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_buffer_aligned_topk2_redecode_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_buffer_aligned_topk2_redecode_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

### 7.4 Top-K=2 pilot 结果

| p | hard parallel LER | hard flagged | Top-K=2 LER | Top-K=2 flagged | Top-K=2 elapsed |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 0.01 | 1/100 | 0.00 | 0/100 | 7.385s |
| 0.004 | 0.07 | 6/100 | 0.06 | 5/100 | 10.606s |
| 0.005 | 0.18 | 17/100 | 0.14 | 11/100 | 16.023s |
| 0.007 | - | - | 0.60 | 42/100 | 30.806s |

其中 hard parallel 的 100-shot baseline 来自：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_buffer_aligned_a_solve5_b5_process_workers4.csv
```

该 baseline 当时只跑到：

```text
p = 0.003, 0.004, 0.005
```

所以表中 `p = 0.007` 没有对应的 hard 100-shot baseline。

### 7.5 初步结论

Top-K=2 有改善信号：

```text
p = 0.003: flagged 1/100 -> 0/100
p = 0.004: flagged 6/100 -> 5/100
p = 0.005: flagged 17/100 -> 11/100
```

但代价也明显：

```text
runtime 明显增加
准确率仍然低于 sliding
```

这说明方向是有价值的：A boundary 的 hard decision 的确会损失信息，给 B 一点
重新选择边界的空间可以降低 flagged/LER。

不过当前实现还只是近似 Top-K：它只围绕低可靠 bit 重新解 A，而不是枚举真正的
多个低代价 syndrome-satisfying A 解。后续若继续推进，应该尝试更正式的：

```text
OSD-derived top-K boundary candidates
或 reliability-aware Top-K candidate generation
```

---

## 8. Naive Soft Boundary Message 初步尝试

### 8.1 实现方式

在不采用 Top-K 的情况下，尝试 soft boundary message：

```text
--soft-boundary-message
```

核心变化：

```text
原 hard boundary:
A 先 commit 边界，B 用 residual syndrome 固定接受这些边界。

naive soft boundary:
A 仍先给出边界估计；
B 解码时把左右边界变量放回 B 的局部矩阵；
根据 A 的输出修改这些边界变量的 prior；
B 可以选择和 A 不同的边界值。
```

本次使用较强的 soft prior：

```text
--soft-boundary-one-prior 0.49
--soft-boundary-zero-prior 0.000001
```

也就是说：

```text
A 说 e_j = 1 时，B 中该变量 prior 设为 0.49；
A 说 e_j = 0 时，B 中该变量 prior 设为 1e-6。
```

这样仍不是 hard constraint，因为 B 可以根据 syndrome 反驳 A。

### 8.2 运行命令

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --p-list 0.003,0.004,0.005,0.007 \
  --num-repeat 12 \
  --num-shots 100 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --osd-order 10 \
  --max-iter 200 \
  --parallel-workers 1 \
  --parallel-backend thread \
  --soft-boundary-message \
  --soft-boundary-one-prior 0.49 \
  --soft-boundary-zero-prior 0.000001 \
  --decoders parallel \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_buffer_aligned_soft_boundary_a_solve5_b5_sliding3_thread_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_buffer_aligned_soft_boundary_a_solve5_b5_sliding3_thread_p003_to_007.csv
```

### 8.3 结果

| p | hard parallel LER | hard flagged | naive soft LER | naive soft flagged | naive soft elapsed |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 0.01 | 1/100 | 0.09 | 9/100 | 3.971s |
| 0.004 | 0.07 | 6/100 | 0.14 | 14/100 | 5.474s |
| 0.005 | 0.18 | 17/100 | 0.24 | 22/100 | 7.696s |
| 0.007 | - | - | 0.70 | 60/100 | 13.946s |

### 8.4 结论

这个 naive soft boundary 版本明显变差。

主要原因是：B 虽然可以反驳 A 的边界变量，但 A commit block 中间的变量没有随之
重新优化。例如：

```text
A2 commit: e11,e12,e13
```

如果 B1 soft-decode 后改写了 `e11`，B2 soft-decode 后改写了 `e13`，那么
`e12` 仍然是 A 原先 hard decode 的结果。这样 A2 的局部一致性可能被破坏，
最终全局 residual 更容易不闭合。

因此，soft boundary 不能只让 B 单方面改边界。更合理的版本应该是：

```text
A -> B soft boundary
B 改边界后
再做一次 A-center repair / A-B-A 迭代
```

也就是说，soft boundary message 需要配合 boundary repair 或一轮 A-B-A
消息传递，否则会破坏 A commit block 的内部一致性。

---

## 9. Cython OSD-list Top-K 尝试

### 9.1 目的

之前的 Top-K=4 使用的是 re-decode 近似：

```text
为每个候选修改 prior
重新跑一次 BP+OSD
```

这会导致 A 侧开销约随 K 线性增长，B 侧还要枚举 K^2 个边界组合。

本次尝试把候选提取改成：

```text
one BP run
one OSD basis
extract top-K candidates
```

即只跑一次 BP+OSD，然后根据 BP/OSD 的 reliability order，在 GF(2) 消元后的
OSD basis 上枚举低可靠自由位翻转，生成 top-K 候选。

### 9.2 实现

新增 Cython 文件：

```text
ParallelWindowDecoder/osd_list.pyx
ParallelWindowDecoder/setup_osd_list.py
```

编译命令：

```bash
cd ParallelWindowDecoder
../SlidingWindowDecoder/.conda-gdg/bin/python setup_osd_list.py build_ext --inplace
```

生成本地扩展：

```text
ParallelWindowDecoder/osd_list.cpython-312-darwin.so
```

`parallel_window_decoder.py` 中优先使用：

```text
from osd_list import osd_list_candidates
```

如果扩展不存在，则 fallback 到 Python 版本。

### 9.3 实现细节

`osd_list_candidates` 做：

1. 按 reliability 从高到低排列列；
2. 对排列后的局部矩阵做 GF(2) 消元；
3. 用 BP+OSD 输出作为 seed；
4. 选最低可靠的自由位；
5. 枚举单 bit / 双 bit 翻转；
6. 解出 pivot 位；
7. 取 channel cost 最低的 top-K 个候选。

调试时发现一个关键 bug：

```text
rhs = np.ascontiguousarray(syndrome)
```

在 syndrome 已连续时不会复制，导致 Cython 消元原地修改传入 syndrome，
污染后续 residual。已修复为：

```text
rhs = np.array(syndrome, dtype=np.uint8, copy=True, order="C")
```

一致性检查通过：

```text
seed residual before: 0
seed residual after:  0
candidate residuals:  0,0,0,0
```

### 9.4 20-shot 测试

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --p-list 0.003,0.004,0.005,0.007 \
  --num-repeat 12 \
  --num-shots 20 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --osd-order 10 \
  --max-iter 200 \
  --parallel-workers 4 \
  --parallel-backend process \
  --top-k-boundary 4 \
  --decoders parallel \
  --out ParallelWindowDecoder/results/N144_repeat12_shots20_buffer_aligned_topk4_cython_osdlist_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots20_buffer_aligned_topk4_cython_osdlist_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

结果：

| p | LER | flagged | logical_or_flagged | elapsed |
|---:|---:|---:|---:|---:|
| 0.003 | 0.00 | 0/20 | 0/20 | 24.176s |
| 0.004 | 0.00 | 0/20 | 0/20 | 29.469s |
| 0.005 | 0.05 | 1/20 | 1/20 | 33.686s |
| 0.007 | 0.75 | 11/20 | 15/20 | 49.312s |

### 9.5 速度结论

Cython 版比 Python OSD-list 有明显加速：

```text
p=0.003, 20 shots:
Python OSD-list: 44.247s
Cython OSD-list: 24.176s
```

大约快了：

```text
45%
```

但它仍然比 re-decode Top-K=4 的 100-shot 版本按比例更慢，不适合作为当前主线。

原因是：

```text
每个 shot / A window 仍然要做一次 dense GF(2) 消元
局部矩阵列数很大
当前 Cython 版本没有缓存同一个 window 的消元结构
```

如果继续这个方向，下一步需要：

```text
1. 缓存每个 A window 的 reliability-independent 结构；
2. 或直接 patch ldpc 的 C++ OsdDecoder，让它在内部 OSD 搜索时返回 candidate list；
3. 或只在低 reliability 的 A window 上触发 OSD-list。
```

当前结论：

```text
Cython OSD-list 可行，但仍太慢；
更适合后续作为底层优化方向，不适合作为当前实验主线。
```

## 10. Patch ldpc C++ OsdDecoder 返回 OSD candidate list

为避免外部 Python/Cython OSD-list 重复做 dense GF(2) 消元，改为直接 patch `ldpc` 的 C++ `OsdDecoder`：

```text
source: /private/tmp/ldpc_repo
installed env: SlidingWindowDecoder/.conda-gdg
ldpc version: 2.4.1
saved patch: ParallelWindowDecoder/patches/ldpc_osd_candidate_list.patch
```

### 10.1 修改内容

在 `src_cpp/osd.hpp` 中新增：

```text
osd_candidate_decodings
osd_candidate_weights
osd_candidate_list_size
set_osd_candidate_list_size(...)
```

在 OSD 内部搜索循环中，每生成一个完整 `candidate_solution`，立即计算 weight 并维护 top-K candidate list。

默认 `osd_candidate_list_size=0`，普通 `decode()` 不收集 candidate，因此默认路径没有额外 candidate-list 开销。

在 `src_python/ldpc/bposd_decoder/_bposd_decoder.pyx` 中新增：

```text
BpOsdDecoder.decode_list(syndrome, top_k=4, force_osd=False)
BpOsdDecoder.osd_candidate_weights
```

其中：

```text
force_osd=False:
  BP 收敛时只返回 BP 解

force_osd=True:
  强制运行一次 OSD，并返回 C++ 内部 OSD 搜索得到的 top-K candidate
```

### 10.2 parallel Top-K 接入

`ParallelWindowDecoder/parallel_window_decoder.py` 的 Top-K A-window candidate 路径已改为：

```text
优先使用 patched ldpc BpOsdDecoder.decode_list(..., force_osd=True)
若当前 ldpc 没有 decode_list，则回退到外部 Cython/Python OSD-list
```

这样 Top-K 不再需要在 parallel decoder 外部重新实现一套 OSD-list 搜索。

### 10.3 API 验证

小矩阵测试确认：

```text
has_decode_list True
candidates_shape (4, 6)
syndromes_ok [True, True, True, True]
```

说明返回的 4 个 candidate 都满足：

```text
H @ candidate = syndrome mod 2
```

### 10.4 冒烟实验

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 5 \
  --p-list 0.003 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --top-k-boundary 4 \
  --parallel-workers 1 \
  --parallel-backend thread \
  --out ParallelWindowDecoder/results/native_ldpc_decode_list_smoke.csv
```

结果：

| decoder | LER | flagged | elapsed |
|---|---:|---:|---:|
| global | 0 | 0/5 | 0.105s |
| sliding | 0 | 0/5 | 0.681s |
| parallel | 0 | 0/5 | 0.903s |
| oracle | 0 | 0/5 | 0.093s |

输出文件：

```text
ParallelWindowDecoder/results/native_ldpc_decode_list_smoke.csv
```

### 10.5 100-shot 延迟与准确率对比

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
  --p-list 0.003,0.004,0.005,0.007 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --top-k-boundary 4 \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_native_ldpc_topk4_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_native_ldpc_topk4_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

结果：

| p | global LER/time | sliding LER/time | parallel Top-K=4 LER/time | oracle LER/time | parallel/sliding time | parallel flagged |
|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 0.000/4.752s | 0.000/11.574s | 0.010/23.016s | 0.000/1.793s | 1.99x | 1/100 |
| 0.004 | 0.000/7.951s | 0.050/13.902s | 0.060/39.437s | 0.010/2.401s | 2.84x | 5/100 |
| 0.005 | 0.050/19.991s | 0.150/16.083s | 0.180/72.209s | 0.010/2.692s | 4.49x | 16/100 |
| 0.007 | 0.490/40.590s | 0.600/23.421s | 0.630/167.108s | 0.190/5.109s | 7.13x | 39/100 |

结论：

```text
patched ldpc native Top-K=4 可以正常运行；
但在 100-shot 对比中，parallel Top-K=4 没有优于 sliding 的准确率；
同时延迟随 p 增大显著上升，约为 sliding 的 2x 到 7x。
```

因此当前 Top-K=4 不适合作为主线对比方案。它更适合作为后续研究边界候选机制的工具，而不是当前严谨对比实验中的高效 decoder。

### 10.6 Top-K=2 100-shot 对比

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
  --p-list 0.003,0.004,0.005,0.007 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --top-k-boundary 2 \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_native_ldpc_topk2_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_native_ldpc_topk2_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

与 K=4 对比：

| p | sliding LER/time | K=2 parallel LER/time | K=2 ratio | K=4 parallel LER/time | K=4 ratio |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 0.000/11.626s | 0.010/11.251s | 0.97x | 0.010/23.016s | 1.98x |
| 0.004 | 0.050/14.717s | 0.070/17.150s | 1.17x | 0.060/39.437s | 2.68x |
| 0.005 | 0.150/16.083s | 0.180/22.845s | 1.42x | 0.180/72.209s | 4.49x |
| 0.007 | 0.600/23.060s | 0.640/46.786s | 2.03x | 0.630/167.108s | 7.25x |

结论：

```text
K=2 将 B 层组合从 K^2=16 降到 K^2=4；
延迟明显下降，p=0.005 从 72.209s 降到 22.845s，p=0.007 从 167.108s 降到 46.786s；
但准确率没有明显改善，整体仍略差于 sliding。
```

这说明当前 Top-K 的主要收益不足以抵消 boundary stitching 的复杂度；K=2 是更合理的延迟点，但不是更好的准确率点。

## 11. 引入 SlidingWindowDecoder 的 noisy syndrome boundary 和 shortening trick

本轮将 `SlidingWindowDecoder` 中两个窗口优化机制接入当前对比实验：

```text
1. noisy syndrome boundary
2. shortening trick
```

### 11.1 代码修改

新增 CLI 参数：

```text
--sliding-method
--window-shorten
--shorten-pre-max-iter
--b-noisy-boundary
```

已有参数：

```text
--a-noisy-boundary
```

当前含义：

```text
sliding:
  --sliding-method 1 使用 noisy syndrome boundary
  --window-shorten 使用 src.osd_window 的 shortening trick

parallel:
  --a-noisy-boundary 给 A window 加 noisy boundary columns
  --b-noisy-boundary 给 B window 加 noisy boundary columns
  --window-shorten 对 A/B window 使用 src.osd_window
```

注意：

```text
top_k_boundary > 1 暂不和 window_shorten / b_noisy_boundary 同时使用；
因为 shortening decoder 不返回 candidate list，B 的 top-K stitching 也需要额外处理 noisy columns。
```

### 11.2 100-shot 对比实验

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
  --p-list 0.003,0.004,0.005,0.007 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --top-k-boundary 1 \
  --a-noisy-boundary \
  --b-noisy-boundary \
  --window-shorten \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_noisy_boundary_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_noisy_boundary_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

与 baseline K=2 对比：

| p | baseline sliding LER/time | noisy+shorten sliding LER/time | baseline parallel LER/time | noisy+shorten parallel LER/time |
|---:|---:|---:|---:|---:|
| 0.003 | 0.000/11.626s | 0.000/11.323s | 0.010/11.251s | 0.010/4.224s |
| 0.004 | 0.050/14.717s | 0.020/12.561s | 0.070/17.150s | 0.060/5.122s |
| 0.005 | 0.150/16.083s | 0.120/13.926s | 0.180/22.845s | 0.200/7.324s |
| 0.007 | 0.600/23.060s | 0.450/18.113s | 0.640/46.786s | 0.590/10.609s |

### 11.3 结论

对 sliding：

```text
noisy syndrome boundary + shortening 明显有帮助。
p=0.004: LER 0.050 -> 0.020
p=0.005: LER 0.150 -> 0.120
p=0.007: LER 0.600 -> 0.450
```

对 parallel：

```text
速度明显改善。
p=0.007: 46.786s -> 10.609s
```

但准确率改善不稳定：

```text
p=0.004: 0.070 -> 0.060
p=0.005: 0.180 -> 0.200
p=0.007: 0.640 -> 0.590
```

说明 shortening 可以降低窗口内部解码成本，noisy boundary 可以缓解局部边界截断；但 parallel 的 A/B hard commit 错误传播仍然存在，尤其 A commit 错后 B 仍会在错误残差下解码。

当前更严谨的 baseline 应该使用：

```text
sliding method=1 + shortening
parallel A/B noisy boundary + shortening
```

而不是之前的 plain BP+OSD window。

## 12. 让 shortening decoder 支持 Top-K candidate list

为支持：

```text
parallel top-k=2 + shortening
```

修改了 `SlidingWindowDecoder/src/osd_window.pyx` 和 `SlidingWindowDecoder/src/osd_window.pxd`：

```text
新增 osd_window.decode_list(input_vector, top_k=2, force_osd=False)
新增 osd_window.candidate_weights
```

内部修改：

```text
1. 在 osd_window 内部维护 candidate_decodings / candidate_weights；
2. OSD_0 解和 OSD_CS/OSD_E 搜索中的每个候选都会 push 到 top-K list；
3. force_osd=True 时，即使 BP 已经收敛，也继续执行 OSD candidate extraction。
```

重新编译：

```bash
cd SlidingWindowDecoder
.conda-gdg/bin/python setup.py build_ext --inplace
```

API 验证：

```text
shape (2, 6)
weights [2.9444389791664403, 5.8888779583328805]
ok [True, True]
```

### 12.1 100-shot parallel top-k=2 + shortening 实验

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
  --p-list 0.003,0.004,0.005,0.007 \
  --a-size 3 \
  --a-solve-size 5 \
  --b-width 5 \
  --sliding-width 3 \
  --top-k-boundary 2 \
  --a-noisy-boundary \
  --window-shorten \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_parallel_topk2_a_noisy_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_parallel_topk2_a_noisy_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

对比：

| p | sliding noisy+shorten LER/time | parallel noisy+shorten LER/time | parallel top-k=2 + shorten LER/time |
|---:|---:|---:|---:|
| 0.003 | 0.000/11.289s | 0.010/4.224s | 0.000/7.458s |
| 0.004 | 0.020/12.547s | 0.060/5.122s | 0.050/11.173s |
| 0.005 | 0.120/16.677s | 0.200/7.324s | 0.190/20.100s |
| 0.007 | 0.450/18.628s | 0.590/10.609s | 0.580/44.355s |

结论：

```text
top-k=2 + shortening 可以正常运行；
低 p=0.003 时准确率改善到 0；
但 p>=0.004 时仍明显差于 sliding noisy+shorten；
高 p 下延迟增加明显，p=0.007 达到 44.355s。
```

说明：

```text
shortening 解决的是单个窗口内部复杂度；
top-k 增加的是 A 边界候选；
但 B 仍然要做 K^2 stitching，且 A/B hard commit 的结构性错误传播仍未根除。
```
