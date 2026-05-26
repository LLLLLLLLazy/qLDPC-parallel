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

## 13. Flag-triggered single boundary flip repair

为避免全量 Top-K 的 `K^2` stitching 延迟，新增低延迟 repair 策略：

```text
1. A layer 正常 top-1 解码，并记录 A commit 区域 reliability；
2. B layer 正常 top-1 解码；
3. 若 B 不 flagged，直接提交；
4. 若 B flagged，只选择相邻 A boundary 中 reliability 最低且影响该 B syndrome 的一个变量；
5. 翻转该 boundary 变量，更新 B residual，并只重跑该 B window 一次；
6. 只有 residual weight 下降时才接受该 repair。
```

新增参数：

```text
--flag-triggered-single-flip
```

当前限制：

```text
只用于 top_k_boundary=1；
不和 soft_boundary_message / b_noisy_boundary 同时使用；
oracle 不启用该 repair。
```

### 13.1 100-shot 实验

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
  --window-shorten \
  --flag-triggered-single-flip \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_flag_single_flip_a_noisy_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_flag_single_flip_a_noisy_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

对比：

| p | sliding noisy+shorten | parallel noisy+shorten | parallel top-k=2+shorten | flag-triggered single-flip | repair attempts/accepted/success |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 0.000/5.874s | 0.010/4.224s | 0.000/7.458s | 0.010/3.138s | 2/1/1 |
| 0.004 | 0.020/6.398s | 0.060/5.122s | 0.050/11.173s | 0.040/3.771s | 10/5/4 |
| 0.005 | 0.120/7.136s | 0.200/7.324s | 0.190/20.100s | 0.190/5.869s | 34/18/5 |
| 0.007 | 0.450/10.323s | 0.590/10.609s | 0.580/44.355s | 0.590/11.097s | 86/31/2 |

### 13.2 结论

延迟方面：

```text
flag-triggered single-flip 明显优于全量 Top-K=2。
p=0.007: Top-K=2+shorten 44.355s, single-flip 11.097s
```

准确率方面：

```text
p=0.004 有收益: 0.060 -> 0.040
p=0.005 有小收益: 0.200 -> 0.190
p=0.007 基本无收益: 0.590 -> 0.590
```

repair 统计显示高 p 下虽然 triggered 多，但真正修成 unflagged 的次数很少：

```text
p=0.007: attempts=86, accepted=31, success=2
```

这说明高 p 下 B flagged 往往不是单个 A boundary bit 错误造成的，而是多个 boundary / B 内部局部解共同出错。single-flip 是很好的低延迟 repair baseline，但它不能替代更强的边界联合优化。

### 13.3 控制变量说明

当前最严谨的 sliding-vs-parallel 对比应固定：

```text
N=144
num_repeat=12
shots=100
p-list=0.003,0.004,0.005,0.007
same random seeds
max_iter=200
osd_order=10
sliding_width=3
a_solve_size=5
b_width=5
buffer=2
sliding method=1
window shortening enabled
```

然后只改变 parallel 的 boundary policy：

```text
plain top-1
flag-triggered single-flip
top-k=2
```

这样能保证比较的是 boundary 策略本身，而不是 window size、buffer size、inner decoder 或 noise sample 的差异。

## 14. Experiment 1: residual-aware single-flip

在 `flag-triggered single-flip` 框架下，新增 flip 选择规则：

```text
--single-flip-rule reliability
--single-flip-rule residual
```

其中 `residual` 规则对每个候选 A boundary bit 计算：

```text
|r_B + H_boundary,j|
```

选择翻转后 B residual weight 最小的 bit；若有并列，再用 A reliability 做 tie-break。

### 14.1 100-shot 实验

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
  --window-shorten \
  --flag-triggered-single-flip \
  --single-flip-rule residual \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_residual_single_flip_a_noisy_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_residual_single_flip_a_noisy_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

对比：

| p | sliding noisy+shorten | reliability single-flip | residual-aware single-flip | residual repair attempts/accepted/success |
|---:|---:|---:|---:|---:|
| 0.003 | 0.000/5.859s | 0.010/3.138s | 0.010/3.232s | 2/2/2 |
| 0.004 | 0.020/6.304s | 0.040/3.771s | 0.060/3.917s | 10/10/0 |
| 0.005 | 0.120/7.234s | 0.190/5.869s | 0.200/5.885s | 34/34/4 |
| 0.007 | 0.450/9.078s | 0.590/11.097s | 0.590/10.752s | 86/86/26 |

### 14.2 结论

residual-aware rule 没有优于 reliability rule：

```text
p=0.004: reliability 0.040, residual-aware 0.060
p=0.005: reliability 0.190, residual-aware 0.200
p=0.007: 两者都是 0.590
```

原因可能是：

```text
residual-aware 只优化当前 B 的 syndrome residual；
但翻转 A boundary 变量会影响相邻 B window 或全局 logical class；
所以局部 residual weight 降低不等于最终 logical error 降低。
```

从 repair 统计看，residual-aware 会更激进地接受 repair：

```text
p=0.007: attempts=86, accepted=86, success=26
```

但最终 LER 没变，说明它主要修了 syndrome flagged，而没有稳定改善 logical class。

当前结论：

```text
保留 reliability single-flip 作为低延迟 repair baseline；
residual-aware single-flip 不作为主路线。
```

## 15. Flag-triggered bounded-2flip

按后续判断，删除运行代码中的 residual-aware single-flip 分支，保留 reliability single-flip，并新增：

```text
--bounded-two-flip
--two-flip-candidates 4
```

策略：

```text
1. 只在 B flagged 时触发；
2. 从相邻 A boundary 中选 reliability 最低的前 m 个变量；
3. 先试最低 reliability 的 1-flip；
4. 若仍 flagged，则枚举前 m 个变量里的 2-flip；
5. 只接受 residual weight 下降的 repair。
```

本次使用：

```text
m = 4
最多 C(4,2)=6 个 2-flip
```

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
  --window-shorten \
  --flag-triggered-single-flip \
  --bounded-two-flip \
  --two-flip-candidates 4 \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_bounded2flip_m4_a_noisy_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

输出文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_bounded2flip_m4_a_noisy_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

### 15.1 flagged 对比

| p | sliding flagged | plain parallel flagged | single-flip flagged | bounded-2flip flagged | bounded repair attempts/accepted/success | twoflip attempts/accepted/success |
|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 0/100 | 1/100 | 1/100 | 1/100 | 2/2/1 | 1/1/0 |
| 0.004 | 0/100 | 5/100 | 3/100 | 3/100 | 10/9/5 | 6/5/1 |
| 0.005 | 0/100 | 17/100 | 16/100 | 15/100 | 34/28/11 | 29/18/6 |
| 0.007 | 0/100 | 43/100 | 43/100 | 43/100 | 86/60/6 | 84/45/4 |

### 15.2 LER/time 对比

| p | sliding noisy+shorten | single-flip | bounded-2flip |
|---:|---:|---:|---:|
| 0.003 | 0.000/5.954s | 0.010/3.138s | 0.010/3.287s |
| 0.004 | 0.020/6.415s | 0.040/3.771s | 0.040/4.943s |
| 0.005 | 0.120/7.173s | 0.190/5.869s | 0.190/11.677s |
| 0.007 | 0.450/9.202s | 0.590/11.097s | 0.590/29.654s |

### 15.3 结论

从 flagged 看：

```text
p=0.004: plain 5 -> single 3 -> bounded2 3
p=0.005: plain 17 -> single 16 -> bounded2 15
p=0.007: plain 43 -> single 43 -> bounded2 43
```

bounded-2flip 对中等 p 有很小的 flagged 改善，但高 p 下没有降低最终 flagged。

原因：

```text
p=0.007 虽然 twoflip accepted=45, success=4，
但最终 flagged 仍是 43/100。
```

说明很多高 p flagged 不是两个 A boundary bit 就能修掉，或者 repair 降低了当前 B residual 但全局 residual / 相邻窗口仍然不一致。

当前判断：

```text
bounded-2flip 是合理诊断实验；
但作为主路线，它的 flagged 收益太小，延迟增加明显。
下一步更值得尝试 A boundary penalty，从源头减少坏 boundary。
```

### 15.4 加入 Top-K=2 的 flagged 对比

Top-K=2 使用已有结果：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_parallel_topk2_a_noisy_shorten_a_solve5_b5_sliding3_process_workers4_p003_to_007.csv
```

flagged 对比：

| p | sliding flagged | plain parallel | single-flip | bounded-2flip | Top-K=2 |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 0/100 | 1/100 | 1/100 | 1/100 | 0/100 |
| 0.004 | 0/100 | 5/100 | 3/100 | 3/100 | 4/100 |
| 0.005 | 0/100 | 17/100 | 16/100 | 15/100 | 15/100 |
| 0.007 | 0/100 | 43/100 | 43/100 | 43/100 | 41/100 |

Top-K=2 的 LER/time：

| p | Top-K=2 LER/time | Top-K=2 flagged |
|---:|---:|---:|
| 0.003 | 0.000/7.458s | 0/100 |
| 0.004 | 0.050/11.173s | 4/100 |
| 0.005 | 0.190/20.100s | 15/100 |
| 0.007 | 0.580/44.355s | 41/100 |

结论：

```text
Top-K=2 对 flagged 有小幅改善：
p=0.007: 43/100 -> 41/100
p=0.003: 1/100 -> 0/100
```

但它的延迟代价很大：

```text
p=0.007: Top-K=2 44.355s
bounded-2flip 29.654s
single-flip 11.097s
plain noisy+shorten 10.609s
```

因此 Top-K=2 能说明“更多边界候选确实可能减少 flagged”，但性价比不高。bounded-2flip 达到部分 Top-K 的 flagged 改善，但仍不足以明显降低高 p flagged。

## 16. A boundary penalty 实验

### 16.1 策略

目标是从源头减少坏的 A boundary，而不是在 B flagged 后再 repair。

对 A window 中会影响相邻 B residual 的 commit 变量提高 LLR 绝对值：

```text
LLR_j' = scale * LLR_j,  j in A/B boundary commit variables
```

由于 p < 0.5，放大 LLR 等价于降低这些 boundary 变量被置 1 的先验概率。直观上：

```text
除非 A window 内证据足够强，否则不要轻易把错误解释成跨 A/B boundary。
```

本轮控制变量：

```text
N=144, repeat=12, shots=100
p = 0.003,0.004,0.005,0.007
sliding_width = 3
a_solve_size = 5
b_width = 5
A noisy boundary = on
B noisy boundary = on
window shortening = on
top_k_boundary = 1
只改变 a_boundary_weight_scale
```

命令示例：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 --num-repeat 12 --num-shots 100 \
  --p-list 0.003,0.004,0.005,0.007 \
  --decoders parallel \
  --a-size 3 --a-solve-size 5 --b-width 5 --sliding-width 3 \
  --top-k-boundary 1 \
  --a-noisy-boundary --b-noisy-boundary --window-shorten \
  --a-boundary-weight-scale 1.2 \
  --parallel-workers 4 --parallel-backend process
```

### 16.2 结果

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_a_boundary_penalty_scale1p1_ab_noisy_shorten_parallel_only.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_a_boundary_penalty_scale1p2_ab_noisy_shorten_parallel_only.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_a_boundary_penalty_scale1p5_ab_noisy_shorten_parallel_only.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_a_boundary_penalty_scale2p0_ab_noisy_shorten_parallel_only.csv
```

flagged / LER / time 对比：

| p | scale=1.0 baseline | scale=1.1 | scale=1.2 | scale=1.5 | scale=2.0 |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 4.224s | 0/100, 0.000, 3.210s | 0/100, 0.000, 3.158s | 2/100, 0.020, 3.299s | 8/100, 0.080, 4.251s |
| 0.004 | 5/100, 0.060, 5.122s | 8/100, 0.080, 3.620s | 8/100, 0.090, 3.537s | 10/100, 0.100, 5.698s | 20/100, 0.250, 4.246s |
| 0.005 | 17/100, 0.200, 7.324s | 15/100, 0.180, 5.461s | 8/100, 0.130, 5.039s | 25/100, 0.310, 5.434s | 55/100, 0.630, 6.387s |
| 0.007 | 43/100, 0.590, 10.609s | 43/100, 0.630, 9.397s | 48/100, 0.620, 9.976s | 53/100, 0.720, 7.597s | 82/100, 0.920, 6.690s |

### 16.3 结论

A boundary penalty 的效果不是单调的。

有效的地方：

```text
p=0.003: scale 1.1/1.2 把 flagged 从 1/100 降到 0/100
p=0.005: scale 1.2 把 flagged 从 17/100 降到 8/100，LER 从 0.200 降到 0.130
```

失效的地方：

```text
p=0.004: 所有 penalty 都比 baseline 更差
p=0.007: scale 1.2 以后 flagged 明显变差，scale 2.0 直接恶化到 82/100
```

解释：

```text
boundary penalty 只是在 A 层降低 boundary bit 被置 1 的倾向。
如果真实错误确实落在 boundary 上，过强 penalty 会迫使 A 选择错误的局部解释，
然后 B 得到更差的 residual。
```

所以当前判断：

```text
A boundary penalty 可以作为一个可调 regularizer，
但不能单独作为降低 high-p flagged 的主路线。
```

如果继续使用，比较合理的是只保留较小 scale：

```text
scale = 1.1 或 1.2
```

并且需要做 p-dependent 或 reliability-dependent 的自适应版本，而不是固定强惩罚。

## 17. A-B-A boundary feedback 实验

### 17.1 策略

这个实验尝试让 B 反过来影响 A boundary，但仍然保持 layer-wise parallel：

```text
并行 A
-> 并行 B
-> 根据 stitched global residual 生成 B -> A feedback
-> 翻转被接受的 A boundary variables
-> 并行重跑受影响的 B
```

第一版实现是 flagged-only、保守搜索：

```text
1. A 正常解码并 commit。
2. B 正常解码并 commit。
3. 计算 stitched global residual。
4. 如果某个 B rows 上 residual 非零，则在相邻 A commit boundary columns 中找候选 flip。
5. 枚举最多 max_flips 个 boundary flip。
6. 接受能降低该 B rows residual weight 的 feedback。
7. 对受影响的 B window 重跑一次。
```

注意：最开始尝试用 B local decoder residual 作为触发条件，但在 `B noisy boundary + shortening` 下，B local residual 全部为 0，因此不会触发 feedback。后来改为使用 stitched global residual 触发，这才对应最终 flagged 的来源。

新增参数：

```text
--aba-boundary-feedback
--aba-max-flips
--aba-candidate-cols
```

### 17.2 结果

控制变量：

```text
N=144, repeat=12, shots=100
p = 0.003,0.004,0.005,0.007
a_solve_size = 5
b_width = 5
sliding_width = 3
A noisy boundary = on
B noisy boundary = on
window shortening = on
top_k_boundary = 1
```

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_aba_feedback_global_m2_c16_ab_noisy_shorten_parallel_only.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_aba_feedback_global_m3_c24_ab_noisy_shorten_parallel_only.csv
```

普通 parallel vs A-B-A feedback：

| p | baseline flagged/LER/time | ABA m2 c16 flagged/LER/time | ABA m3 c24 flagged/LER/time |
|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 4.224s | 1/100, 0.010, 3.449s | 1/100, 0.010, 3.328s |
| 0.004 | 5/100, 0.060, 5.122s | 5/100, 0.060, 3.971s | 5/100, 0.060, 4.023s |
| 0.005 | 17/100, 0.200, 7.324s | 17/100, 0.200, 5.932s | 17/100, 0.200, 5.653s |
| 0.007 | 43/100, 0.590, 10.609s | 43/100, 0.590, 10.595s | 43/100, 0.590, 9.937s |

feedback diagnostics：

| p | m2 attempts/proposed/accepted/rerun | m3 attempts/proposed/accepted/rerun |
|---:|---:|---:|
| 0.003 | 2/2/2/2 | 2/2/2/2 |
| 0.004 | 10/10/10/10 | 10/10/10/10 |
| 0.005 | 34/34/34/34 | 34/34/34/34 |
| 0.007 | 86/86/86/86 | 86/86/86/86 |

### 17.3 结论

A-B-A feedback 在这个实现下没有降低 final flagged：

```text
p=0.003: 1/100 -> 1/100
p=0.004: 5/100 -> 5/100
p=0.005: 17/100 -> 17/100
p=0.007: 43/100 -> 43/100
```

关键诊断是：

```text
所有 stitched global residual 都触发了 feedback；
所有 feedback 都产生了 proposal；
所有 proposal 都被接受并触发 B rerun；
但 final flagged 没变。
```

这说明当前 flagged 不是简单的：

```text
某个 A boundary bit 错了
-> 翻回来
-> B 重跑
-> residual 消失
```

而更像是：

```text
A/B 各自局部都能满足自己的窗口方程，
但拼接后的 correction 落在全局不一致或错误等价类中。
```

因此，单个 B 对相邻 A boundary 的局部反馈仍然不够。若不增大 A/B 矩阵大小，下一步更可能有效的是：

```text
把所有 flagged rows 周围的 boundary variables 收集起来，
做一次小型 global boundary system repair，
而不是每个 B 独立反馈、独立重跑。
```

也就是从：

```text
B_i -> A_left/A_right
```

升级为：

```text
all flagged B rows -> shared boundary repair system
```

这仍然不扩大 A/B window，但会把多个窗口的边界修正作为一个整体求解。

## 18. Shared boundary repair system 实验

### 18.1 策略

这一版把上一节的 local B feedback 升级为 shared repair：

```text
1. A layer 正常并行解码。
2. B layer 正常并行解码。
3. 计算 stitched global residual。
4. 找出所有 residual 非零的 B rows。
5. 收集这些 flagged B rows 相邻的 A commit boundary columns。
6. 建立一个小型 GF(2) repair system：

   H_boundary x = r_flagged

7. 用 BP+OSD 解 boundary flip x。
8. 临时翻转这些 A boundary variables。
9. 重跑受影响的 B windows。
10. 只有 global residual weight 下降时才接受。
```

和上一节区别：

```text
上一节：每个 B 独立反馈给左右 A。
这一节：所有 flagged B rows 共享一个 boundary repair system。
```

这仍然不扩大 A/B window，只是在窗口解码后加入一个 flagged-only boundary 协调层。

新增参数：

```text
--shared-boundary-repair
--shared-boundary-max-cols
```

### 18.2 结果

控制变量：

```text
N=144, repeat=12, shots=100
p = 0.003,0.004,0.005,0.007
a_solve_size = 5
b_width = 5
sliding_width = 3
A noisy boundary = on
B noisy boundary = on
window shortening = on
top_k_boundary = 1
```

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_shared_boundary_repair_c512_ab_noisy_shorten_parallel_only.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_shared_boundary_repair_c2048_ab_noisy_shorten_parallel_only.csv
```

普通 parallel vs shared boundary repair：

| p | baseline flagged/LER/time | shared c512 flagged/LER/time | shared c2048 flagged/LER/time |
|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 4.224s | 1/100, 0.010, 3.125s | 1/100, 0.010, 2.810s |
| 0.004 | 5/100, 0.060, 5.122s | 5/100, 0.060, 3.621s | 5/100, 0.060, 3.357s |
| 0.005 | 17/100, 0.200, 7.324s | 17/100, 0.200, 4.813s | 17/100, 0.200, 4.802s |
| 0.007 | 43/100, 0.590, 10.609s | 43/100, 0.590, 7.164s | 43/100, 0.590, 7.271s |

diagnostics：

| p | c512 init/attempt/accepted/success | c2048 init/attempt/accepted/success |
|---:|---:|---:|
| 0.003 | 1/1/0/0 | 1/1/0/0 |
| 0.004 | 5/5/0/0 | 5/5/0/0 |
| 0.005 | 17/17/0/0 | 17/17/0/0 |
| 0.007 | 43/43/0/0 | 43/43/0/0 |

平均参与 repair 的 boundary columns：

```text
c512: 512 columns per attempted shot
c2048: about 2016 columns per attempted shot
```

### 18.3 结论

shared boundary repair 仍然没有降低 final flagged：

```text
p=0.003: 1/100 -> 1/100
p=0.004: 5/100 -> 5/100
p=0.005: 17/100 -> 17/100
p=0.007: 43/100 -> 43/100
```

重要诊断：

```text
所有 flagged shot 都触发了 shared repair attempt；
但没有任何一次 trial repair 能让 global residual weight 下降；
把候选列从 512 增加到约 2016 后仍然没有改善。
```

这说明当前 final flagged 不是简单的：

```text
只改 A boundary variables 就能修复。
```

更可能是：

```text
1. B window 内部 committed variables 也需要和 A boundary 一起联合变化；
2. 或者 residual 对应的是更长程的 qLDPC coupling，不在当前 boundary variable span 内；
3. 或者 BP+OSD 给出的局部解已经满足 local window，但 global stitching 落入了错误的整体一致性结构。
```

因此，在不扩大 A/B 矩阵大小的前提下，只修改 A boundary 的策略目前看起来不够。

如果继续沿着“不扩大 window”做，下一步应该从：

```text
repair only A boundary variables
```

升级为：

```text
repair A boundary variables + adjacent B edge variables
```

也就是 shared repair system 的变量集合不仅包含 A commit boundary，还要包含每个 flagged B window 两端的一小部分 B committed variables。这样仍然不扩大 A/B 解码矩阵，但 repair 层拥有足够自由度去真正改变 stitching residual。

## 19. A top-2 neighbor residual rerank 实验

### 19.1 策略

这个实验对应新的边界决策方向：

```text
不要等 B flagged 后 repair；
而是在 A commit 前，让 A 的候选选择参考相邻 B residual risk。
```

普通 parallel 的 A 选择是：

```text
A candidate = argmin W_A
```

本实验改成：

```text
A candidate = argmin [ W_A + beta * R_neighbor ]
```

其中：

```text
R_neighbor = 相邻 B rows 上 raw syndrome 加上该 A candidate boundary contribution 后的 popcount
```

实现方式：

```text
1. A window 生成 top-2 candidates。
2. 不做 B 的 K^2 stitching。
3. 对每个 A candidate 计算 neighbor residual risk。
4. A 只 commit rerank 后的一个 candidate。
5. B window 正常只解码一次。
```

新增参数：

```text
--a-neighbor-rerank
--a-rerank-top-k 2
--a-neighbor-beta
```

这和 full Top-K stitching 的区别：

```text
Top-K stitching:
A top-K -> B 枚举左右 K^2 -> B 多次 decode

A neighbor rerank:
A top-2 -> A 内部 rerank -> B decode 一次
```

### 19.2 结果

控制变量：

```text
N=144, repeat=12, shots=100
p = 0.003,0.004,0.005,0.007
a_solve_size = 5
b_width = 5
sliding_width = 3
A noisy boundary = on
B noisy boundary = on
window shortening = on
top_k_boundary = 1
```

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_a_neighbor_rerank_top2_beta0p25_ab_noisy_shorten_parallel_only.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_a_neighbor_rerank_top2_beta0p5_ab_noisy_shorten_parallel_only.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_a_neighbor_rerank_top2_beta1p0_ab_noisy_shorten_parallel_only.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_a_neighbor_rerank_top2_beta2p0_ab_noisy_shorten_parallel_only.csv
```

flagged / LER / time 对比：

| p | baseline | beta=0.25 | beta=0.5 | beta=1.0 | beta=2.0 |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 4.224s | 1/100, 0.010, 4.635s | 1/100, 0.010, 4.658s | 1/100, 0.010, 4.637s | 0/100, 0.000, 4.523s |
| 0.004 | 5/100, 0.060, 5.122s | 4/100, 0.050, 5.312s | 5/100, 0.060, 5.366s | 5/100, 0.060, 5.332s | 5/100, 0.060, 5.452s |
| 0.005 | 17/100, 0.200, 7.324s | 17/100, 0.200, 7.038s | 17/100, 0.200, 7.057s | 17/100, 0.200, 6.839s | 17/100, 0.200, 6.846s |
| 0.007 | 43/100, 0.590, 10.609s | 43/100, 0.600, 10.406s | 43/100, 0.600, 10.080s | 43/100, 0.600, 10.763s | 44/100, 0.610, 10.712s |

A rerank 选择非 top-1 candidate 的次数：

| p | beta=0.25 | beta=0.5 | beta=1.0 | beta=2.0 |
|---:|---:|---:|---:|---:|
| 0.003 | 2 | 3 | 5 | 7 |
| 0.004 | 5 | 9 | 10 | 9 |
| 0.005 | 6 | 8 | 12 | 12 |
| 0.007 | 13 | 14 | 15 | 20 |

### 19.3 结论

A top-2 neighbor residual rerank 有轻微信号，但整体没有稳定降低 flagged。

有效的地方：

```text
p=0.003: beta=2.0 把 flagged 从 1/100 降到 0/100
p=0.004: beta=0.25 把 flagged 从 5/100 降到 4/100
```

无效或变差的地方：

```text
p=0.005: 所有 beta 都保持 17/100，没有改善
p=0.007: beta=0.25/0.5/1.0 flagged 不变，但 LER 轻微变差；
         beta=2.0 flagged 从 43/100 变成 44/100
```

诊断：

```text
rerank 确实会选择非 top-1 的 A candidate；
但这些改动没有稳定降低 final flagged。
```

说明当前使用的 neighbor risk：

```text
R_neighbor = popcount(s_B + H_boundary b)
```

作为 B 代价代理太粗。它能改变 A 边界选择，但不一定让最终 stitched correction 更一致。

因此，这个实验支持如下判断：

```text
边界决策方向是对的；
但简单 residual popcount 不是足够好的 M_B(b) 近似。
```

如果继续优化 A boundary decision，下一步需要更强的 neighbor score，例如：

```text
1. 只看 B 的真正左右 boundary rows，而不是整个 B rows；
2. 用 B pre-decoder 的 syndrome weight / BP convergence 信息作为 score；
3. 在 A top-2 中加入 A-boundary penalty + neighbor rerank 的组合 score；
4. 用 top-2 margin 判断是否启用 rerank，避免高 p 下过度改动。
```

## 20. A two-color / odd-even residual propagation 实验

### 20.1 策略

这个实验不再继续修 A/B boundary，而是尝试给 parallel 恢复一部分 sliding 的 residual 因果传播。

普通 parallel：

```text
A_all -> B_all
```

two-color parallel：

```text
A_odd -> residual update -> A_even -> residual update -> B_all
```

或：

```text
A_even -> residual update -> A_odd -> residual update -> B_all
```

这里的 odd/even 按文档中的 A 编号理解：

```text
odd-first:  A1,A3,... -> A2,A4,... -> B
even-first: A2,A4,... -> A1,A3,... -> B
```

实现注意：

```text
residual update 只使用 A window 最终 commit 的变量；
不会使用 A solve window 内部未 commit 的临时变量；
B 解码时直接使用被所有 A commit 更新后的 residual syndrome。
```

新增参数：

```text
--a-two-color-order odd-first
--a-two-color-order even-first
```

### 20.2 结果

控制变量：

```text
N=144, repeat=12, shots=100
p = 0.003,0.004,0.005,0.007
a_solve_size = 5
b_width = 5
sliding_width = 3
A noisy boundary = on
B noisy boundary = on
window shortening = on
top_k_boundary = 1
```

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_a_two_color_odd_first_ab_noisy_shorten_parallel_only.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_a_two_color_even_first_ab_noisy_shorten_parallel_only.csv
```

flagged / LER / time 对比：

| p | baseline | odd-first | even-first |
|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 4.224s | 1/100, 0.010, 3.720s | 1/100, 0.010, 3.625s |
| 0.004 | 5/100, 0.060, 5.122s | 5/100, 0.060, 4.229s | 5/100, 0.060, 4.243s |
| 0.005 | 17/100, 0.200, 7.324s | 17/100, 0.200, 5.433s | 17/100, 0.200, 5.428s |
| 0.007 | 43/100, 0.590, 10.609s | 43/100, 0.590, 7.440s | 43/100, 0.590, 7.459s |

A/B local flagged diagnostics：

| p | odd first/second A flagged | even first/second A flagged | B local flagged odd/even |
|---:|---:|---:|---:|
| 0.003 | 0/0 | 0/0 | 0/0 |
| 0.004 | 0/0 | 0/0 | 0/0 |
| 0.005 | 0/0 | 0/0 | 0/0 |
| 0.007 | 0/0 | 0/0 | 0/0 |

### 20.3 结论

A two-color residual propagation 没有改善 final flagged：

```text
p=0.003: 1/100 -> 1/100
p=0.004: 5/100 -> 5/100
p=0.005: 17/100 -> 17/100
p=0.007: 43/100 -> 43/100
```

odd-first 和 even-first 完全一致，说明：

```text
A layer 缺少 odd/even 级别的 residual 因果传播，
不是当前 final flagged 的主要来源。
```

更重要的诊断是：

```text
A local flagged = 0
B local flagged = 0
final flagged > 0
```

这意味着每个局部 window 在自己的方程中都能闭合，但拼接成全局 correction 后仍然有 residual / logical failure。换句话说，问题不是 A 或 B 单个窗口“解不出来”，也不是 A-A 的两色传播缺失，而是：

```text
局部窗口解都成立，但 global stitching 不成立。
```

因此 two-color 是一个有价值的 negative result。它排除了一个假设：

```text
parallel 主要差在 A_all 同时解码、缺少一层 A-to-A causal residual。
```

下一步如果继续找结构性方向，更应该考虑：

```text
B-first / B-centered ownership
```

或者重新定义最终 commit 的变量归属，而不是继续在现有 A-boundary ownership 下做局部 residual 修正。

## 21. Strict seam diagnostics

### 21.1 目的

这个实验不改变 decoder，只增加 diagnostics，用来解释：

```text
A local flagged = 0
B local flagged = 0
但 final flagged 很高
```

新增参数：

```text
--seam-diagnostics
```

诊断内容：

```text
1. A commit 后 residual:
   rA = s + H e_A

2. final residual:
   rFinal = s + H(e_A + e_B)

3. 按 seam row ownership 统计 residual:
   A-owned rows
   B-owned rows
   boundary rows
   unowned rows

4. 比较 B local real residual 和 final global residual 在 B rows 上是否 mismatch。
```

本次 row ownership 识别为：

```text
A-owned rows: s1, s7, s13
B-owned rows: s2-s6, s8-s12
boundary rows: s1, s13
unowned rows: none
```

### 21.2 运行命令

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_plain_parallel_seam_diagnostics_ab_noisy_shorten.csv
```

### 21.3 结果

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_plain_parallel_seam_diagnostics_ab_noisy_shorten.csv
```

residual 分类统计：

| p | flagged | LER | rA weight A/B/boundary/unowned | final weight A/B/boundary/unowned | final flagged A/B/boundary/unowned | B local-global mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 1/100 | 0.010 | 0/6247/0/0 | 0/6/0/0 | 0/1/0/0 | 0 |
| 0.004 | 5/100 | 0.060 | 0/8272/0/0 | 0/28/0/0 | 0/5/0/0 | 0 |
| 0.005 | 17/100 | 0.200 | 0/9937/0/0 | 0/106/0/0 | 0/17/0/0 | 0 |
| 0.007 | 43/100 | 0.590 | 0/13297/0/0 | 0/274/0/0 | 0/43/0/0 | 0 |

final residual block histogram：

| p | nonzero final residual blocks |
|---:|---|
| 0.003 | s2:3, s8:3 |
| 0.004 | s2:14, s8:14 |
| 0.005 | s2:55, s8:51 |
| 0.007 | s2:137, s8:137 |

### 21.4 结论

这个 diagnostics 回答了当前问题：

```text
A local flagged = 0
B local flagged = 0
final flagged > 0
```

原因不是 A-owned seam rows 没清掉：

```text
final residual on A-owned rows = 0
```

也不是 B local real residual 和 final global residual 不一致：

```text
B local-global mismatch = 0
```

真正的问题是：

```text
final residual 全部落在 B-owned rows，
而且集中在每个 B window 的第一行：
s2, s8
```

这说明 `b_noisy_boundary` 的 augmented/noisy boundary columns 让 B 的 augmented local problem 可以闭合，所以 `B local flagged = 0`；但这些 noisy boundary columns 不会被写回全局 correction，因此真实 global residual 仍然留在 B rows 上。

换句话说：

```text
B local flagged=0 是 augmented local equation 的成功；
final flagged>0 是真实 global equation 中 noisy boundary contribution 没有对应 physical correction。
```

因此当前 final flagged 高的直接来源不是 A seam row，而是：

```text
B noisy boundary 吸收了 B 左边界 residual，
但该吸收项没有进入最终 global correction。
```

下一步如果要降低 flagged，应该先验证：

```text
1. 关闭 b_noisy_boundary 后，B local flagged 与 final flagged 是否一致；
2. 或者把 noisy boundary columns 的使用改成只用于 BP/shortening 辅助，
   不允许它们吸收最终必须由真实变量解释的 residual；
3. 或者把 noisy boundary 的被选中情况记录并转化为明确的边界 failure，而不是 local success。
```

## 22. b_noisy_boundary ablation and z_B usage

### 22.1 目的

上一节发现：

```text
B local flagged = 0
final flagged > 0
```

主要来自 `b_noisy_boundary` 的 augmented/noisy boundary columns。为了确认这一点，这里做两个实验：

```text
实验 A：关闭 --b-noisy-boundary，只保留 --a-noisy-boundary 和 --window-shorten
实验 B：保留 --b-noisy-boundary，并统计 B 解码中 noisy tail z_B 是否非零
```

两个实验都使用同样参数：

```text
N=144
num_repeat=12
shots=100
p-list=0.003,0.004,0.005,0.007
a_solve_size=5
b_width=5
sliding_width=3
top_k_boundary=1
parallel_workers=4
parallel_backend=process
seam_diagnostics=True
```

### 22.2 关闭 b_noisy_boundary

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_parallel_seam_diagnostics_a_noisy_shorten_no_b_noisy.csv
```

| p | final flagged | LER | A local flagged | B local flagged | final B-row flagged | final residual blocks |
|---:|---:|---:|---:|---:|---:|---|
| 0.003 | 1/100 | 0.010 | 0 | 2 | 1 | s6:2, s12:2 |
| 0.004 | 5/100 | 0.060 | 0 | 10 | 5 | s6:15, s12:15 |
| 0.005 | 17/100 | 0.200 | 0 | 34 | 17 | s6:50, s12:50 |
| 0.007 | 43/100 | 0.590 | 0 | 86 | 43 | s6:128, s12:128 |

这里 `B local flagged` 是按 B window-shot 计数；当前每个 failed shot 通常两个 B window 都失败，所以它正好是 `2 * final flagged`。

结论：

```text
关掉 b_noisy_boundary 后，B local flagged 不再是 0，
而是和 final flagged 完全对齐。
```

因此之前的 `B local flagged=0` 不是物理 B 方程真的成功，而是 noisy boundary slack 把物理 residual 吃掉了。

### 22.3 保留 b_noisy_boundary 并统计 z_B

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_parallel_seam_diagnostics_ab_noisy_shorten_zstats.csv
```

| p | final flagged | B local flagged | z_B used shots | z_B final-overlap | final without z_B | z_B without final | z_B total weight | z_B left/right weight |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 1/100 | 0 | 1 | 1 | 0 | 0 | 6 | 6/0 |
| 0.004 | 5/100 | 0 | 5 | 5 | 0 | 0 | 28 | 28/0 |
| 0.005 | 17/100 | 0 | 17 | 17 | 0 | 0 | 106 | 106/0 |
| 0.007 | 43/100 | 0 | 43 | 43 | 0 | 0 | 274 | 274/0 |

window-level 统计显示，`z_B` 全部来自两个 B window 的左侧 noisy boundary：

```text
p=0.007:
B1=s2-s6  used_shots=43, left_weight=137, right_weight=0
B2=s8-s12 used_shots=43, left_weight=137, right_weight=0
```

结论非常明确：

```text
z_B 非零 shot 与 final flagged shot 100% 重合。
```

也就是说：

```text
final flagged 的准确触发信号就是 B noisy boundary tail z_B != 0。
```

### 22.4 对优化方向的影响

这组实验说明，当前 flagged 高不是一个普通的 A/B 拼接 bug，也不是 A seam row 未清零。更准确的解释是：

```text
B 的物理变量无法解释某些 seam residual；
如果打开 b_noisy_boundary，B 会用 z_B 把这些 residual 局部吞掉；
但 z_B 不属于物理 correction，所以 final global check 仍然失败。
```

因此后续优化不应该继续把 `b_noisy_boundary` 当作成功条件，而应该把它当作 boundary-conflict detector。

优先方向：

```text
1. 修改统计口径：
   B augmented local pass 但 z_B != 0 时，不再记作真正 B physical pass。

2. 用 z_B 触发 targeted repair：
   只在 z_B != 0 的 B window 上修边界，不再盲目 single-flip。

3. 优先修左侧边界：
   当前 z_B weight 全在 left boundary，说明冲突集中在 B window 左边界。

4. 如果实验目标包含 noisy syndrome：
   可以把 z_B 显式纳入全局 syndrome-noise correction；
   但这会改变最终校验目标，从 H e = s 变成 H e + m = s。
```

短期最合理的下一步是：

```text
parallel + b_noisy_boundary
→ B 解码
→ 如果 z_B == 0，正常接受
→ 如果 z_B != 0，把该 B window 记为 boundary conflict
→ 只针对 z_B 支持的边界行做 targeted A-boundary / B-boundary repair
```

## 23. z_B-triggered physical seam repair 实验

### 23.1 策略

本实验实现上一节的最小 physical repair：

```text
--z-boundary-repair
--z-repair-edge-width 2/3
```

核心思想：

```text
b_noisy_boundary 仍然用于定位 seam conflict；
但 z_B 不再被当作真实 correction。
当 z_B != 0 时，收集真实物理变量做局部 seam repair。
```

对每个触发的 B window-shot：

```text
left z_B:
    left A boundary region
    B window left edge_width regions

right z_B:
    B window right edge_width regions
    right A boundary region
```

然后进一步只保留会影响 `z_B` 支持 detector rows 的物理列，构造：

```text
H[R_ext, C_z] Delta e = r[R_ext]
```

其中：

```text
C_z = A boundary + B edge physical variables
R_ext = C_z 中所有变量会影响到的 detector rows
```

接受条件：

```text
只在 global final residual weight 下降时接受 repair。
```

### 23.2 运行命令

edge width 2：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --z-boundary-repair \
  --z-repair-edge-width 2 \
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_z_boundary_repair_edge2_ab_noisy_shorten_parallel_only.csv
```

edge width 3：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_z_boundary_repair_edge3_ab_noisy_shorten_parallel_only.csv
```

### 23.3 结果

baseline 使用：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_parallel_seam_diagnostics_ab_noisy_shorten_zstats.csv
```

flagged / LER / runtime：

| p | baseline | z-repair edge=2 | z-repair edge=3 |
|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 3.493s | 1/100, 0.010, 3.609s | 1/100, 0.010, 3.603s |
| 0.004 | 5/100, 0.060, 3.973s | 5/100, 0.060, 3.973s | 5/100, 0.060, 3.992s |
| 0.005 | 17/100, 0.200, 4.785s | 17/100, 0.200, 5.020s | 17/100, 0.200, 5.055s |
| 0.007 | 43/100, 0.590, 8.015s | 43/100, 0.590, 6.943s | 43/100, 0.590, 7.565s |

repair diagnostics：

| p | edge | attempts | accepted | success-to-zero | final B residual weight | final residual blocks |
|---:|---:|---:|---:|---:|---:|---|
| 0.003 | 2 | 2 | 0 | 0 | 6 | s2:3, s8:3 |
| 0.004 | 2 | 10 | 4 | 0 | 24 | s2:7, s3:5, s8:7, s9:5 |
| 0.005 | 2 | 34 | 11 | 0 | 81 | s2:29, s3:13, s8:30, s9:9 |
| 0.007 | 2 | 86 | 36 | 0 | 202 | s2:64, s3:37, s8:64, s9:37 |
| 0.003 | 3 | 2 | 0 | 0 | 6 | s2:3, s8:3 |
| 0.004 | 3 | 10 | 4 | 0 | 24 | s2:7, s3:5, s8:7, s9:5 |
| 0.005 | 3 | 34 | 11 | 0 | 81 | s2:29, s3:13, s8:30, s9:9 |
| 0.007 | 3 | 86 | 36 | 0 | 202 | s2:64, s3:37, s8:64, s9:37 |

### 23.4 结论

`z_B`-triggered physical seam repair 没有降低 final flagged：

```text
p=0.003: 1/100 -> 1/100
p=0.004: 5/100 -> 5/100
p=0.005: 17/100 -> 17/100
p=0.007: 43/100 -> 43/100
```

但它确实改变了 residual 结构：

```text
baseline p=0.007 final B residual weight = 274
z-repair p=0.007 final B residual weight = 202
```

也就是说，repair 能降低一部分 residual weight，但不能把任何 failed shot 修到 zero residual：

```text
success-to-zero = 0
```

同时 residual 从：

```text
s2, s8
```

被推到了：

```text
s2, s3, s8, s9
```

这说明当前最小候选集合：

```text
A boundary + B edge variables
```

可以局部移动 seam residual，但自由度仍然不够，不能形成完整的物理闭合 correction。

edge width 2 和 3 结果完全一致，是因为本实现先过滤到“会影响 z_B 支持 rows 的物理列”；额外 edge region 没有直接作用到这些 z rows，因此没有进入实际 repair candidate set。

### 23.5 下一步方向

这个实验排除了一个窄假设：

```text
final flagged 不能只靠 z_B 支持行附近的少量 A-boundary + B-edge variables 修掉。
```

如果继续利用 `z_B`，下一步不应该只扩大 edge_width，而应该改变 repair system 的目标：

```text
从 local z-row repair
升级为 flagged B window 的 physical joint retry。
```

也就是对 `z_B != 0` 的 B window，联合重解：

```text
left A boundary
+ full B committed variables
+ right A boundary
```

并且检查整个 B rows 或更大的 affected rows，而不是只围绕 `z_B` 支持行。这个方向更接近：

```text
B window 不再被 A boundary hard condition 完全锁死；
但只在 z_B 触发时开放额外自由度。
```

这仍然不改变常规 A/B window size，也不做全量 Top-K；它只是把 `z_B` flagged shot 作为少量 fallback case 处理。

## 24. z_B-triggered B-window physical joint retry 实验

### 24.1 策略

上一节的窄版 seam repair 只使用：

```text
A boundary + B edge variables
```

它可以降低 residual weight，但无法把 failed shot 清零。因此本实验升级为：

```text
z_B != 0
→ 对该 B window 做一次 physical joint retry
```

候选变量集合为：

```text
left A boundary variables
+ full B committed variables
+ right A boundary variables
```

注意这里的 A boundary variables 不是整个 A commit，而是相邻 A commit 中真正作用到该 B rows 的物理列。

实现参数：

```text
--z-joint-retry
```

流程：

```text
1. A layer 正常 decode + commit。
2. B layer 使用 b_noisy_boundary decode，并记录 z_B。
3. 若 z_B == 0，则接受 B physical part。
4. 若 z_B != 0，则构造 joint variable set:

       C_joint = E_B ∪ E_A_boundary_left ∪ E_A_boundary_right

5. 固定其它变量，重新解：

       H[B_rows, C_joint] x = s_B + H_other e_other

6. 用解出的 x 替换当前 C_joint 上的 physical correction。
7. 只有 global residual weight 下降时才接受。
```

这个策略和 full Top-K 不同：

```text
不是所有 B 都多次 decode；
只在 z_B != 0 的 B window-shot 上触发一次 joint retry。
```

### 24.2 运行命令

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --z-joint-retry \
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_z_joint_retry_ab_noisy_shorten_parallel_only.csv
```

### 24.3 结果

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_z_joint_retry_ab_noisy_shorten_parallel_only.csv
```

flagged / LER / runtime 对比：

| p | baseline | z-row repair edge=2 | z-joint retry |
|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 3.493s | 1/100, 0.010, 3.609s | 1/100, 0.010, 3.879s |
| 0.004 | 5/100, 0.060, 3.973s | 5/100, 0.060, 3.973s | 4/100, 0.050, 4.321s |
| 0.005 | 17/100, 0.200, 4.785s | 17/100, 0.200, 5.020s | 17/100, 0.200, 5.980s |
| 0.007 | 43/100, 0.590, 8.015s | 43/100, 0.590, 6.943s | 43/100, 0.590, 9.560s |

joint retry diagnostics：

| p | attempts | accepted | success-to-zero | local flagged | final B residual weight | final residual blocks |
|---:|---:|---:|---:|---:|---:|---|
| 0.003 | 2 | 0 | 0 | 0 | 6 | s2:3, s8:3 |
| 0.004 | 10 | 4 | 1 | 0 | 13 | s2:3, s7:4, s8:10 |
| 0.005 | 34 | 2 | 0 | 0 | 97 | s2:46, s7:2, s8:51 |
| 0.007 | 86 | 2 | 0 | 0 | 266 | s2:129, s7:3, s8:137 |

### 24.4 结论

`z-joint retry` 有一个小幅正信号：

```text
p=0.004:
flagged 5/100 -> 4/100
LER     0.060 -> 0.050
success-to-zero = 1
```

这说明：

```text
把 full B committed variables 和 A boundary 一起开放，
确实可以修掉一部分低 p 的 seam conflict。
```

但高 p 下没有明显改善：

```text
p=0.005: 17/100 -> 17/100
p=0.007: 43/100 -> 43/100
```

诊断上，joint retry 的 local B equation 都能闭合：

```text
z_joint_local_flagged = 0
```

但接受后的 residual 会出现在 A seam row：

```text
s7
```

这说明只按单个 B window 重解时，B rows 可以被修好，但会把不一致转移到相邻 A seam row。也就是说，问题已经不是：

```text
B 内部变量是否足够
```

而是：

```text
单个 B window retry 没有同时约束相邻 A seam equation。
```

### 24.5 下一步方向

如果继续沿着 `z_B` 触发优化，下一步不应该只对单个 B window 做 retry，而应该做：

```text
z_B-triggered A-B-A joint patch
```

对一个 flagged B_i，联合变量应该包含：

```text
left A boundary/seam variables
+ full B_i variables
+ right A boundary/seam variables
```

联合 rows 不只包含 B rows，还应包含：

```text
left A seam row
+ B rows
+ right A seam row
```

这样可以避免当前现象：

```text
B rows 修好了，但 residual 被推到 A seam row s7。
```

换句话说，本实验说明：

```text
z_B 是有效触发器；
full B joint retry 比窄版 z-row repair 更有信息；
但单个 B window 的约束仍然不够，需要把相邻 A seam rows 一起纳入 joint patch。
```

## 25. A-center commit + B-expanded ownership 实验

> 说明：这个尝试结果较差，相关代码入口已经移除；本节只保留 negative result 记录。

### 25.1 策略

这个实验测试新的 ownership 方向：

```text
A 不再 hard commit 左右边界变量；
A 只 commit 最中心 seam/gap 变量；
B 负责更多物理变量，包括原来由 A 提供给 B 的左右边界。
```

原始 ownership：

```text
A1 commit: e0,e1
B1 commit: e2-e10
A2 commit: e11,e12,e13
B2 commit: e14-e22
A3 commit: e23,e24
```

新 ownership：

```text
A1 commit: e0
B1 commit: e1-e11
A2 commit: e12
B2 commit: e13-e23
A3 commit: e24
```

也就是 interior A 从：

```text
A2 commit: e11,e12,e13
```

变成：

```text
A2 commit: e12
```

同时 B 从：

```text
B1 commit: e2-e10
```

变成：

```text
B1 commit: e1-e11
```

这个不是纯 B-first，因为 A 仍然先解并给出 center commit。为了检查 “A seam 太窄” 的担心，同时测试两个版本：

```text
1. --a-center-b-expanded
   A center commit + B expanded ownership，不做后处理。

2. --a-center-b-expanded --a-center-refine
   B expanded 后，再根据 B 给出的左右边界重新解 A center seam。
```

### 25.2 运行命令

无 refine：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --a-center-b-expanded \
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_a_center_b_expanded_ab_noisy_shorten_parallel_only.csv
```

带 A-center refine：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --a-center-b-expanded \
  --a-center-refine \
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_a_center_b_expanded_refine_ab_noisy_shorten_parallel_only.csv
```

### 25.3 结果

flagged / LER / runtime 对比：

| p | baseline | A-center+B-expanded | A-center+B-expanded+refine |
|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 3.493s | 92/100, 0.920, 3.632s | 1/100, 0.020, 4.544s |
| 0.004 | 5/100, 0.060, 3.973s | 98/100, 0.980, 3.971s | 10/100, 0.170, 5.203s |
| 0.005 | 17/100, 0.200, 4.785s | 99/100, 0.990, 4.273s | 21/100, 0.320, 5.858s |
| 0.007 | 43/100, 0.590, 8.015s | 100/100, 1.000, 4.954s | 74/100, 0.860, 6.641s |

diagnostics：

| p | mode | A commit | B commit | z_B used | final residual owner | main residual blocks |
|---:|---|---|---|---:|---|---|
| 0.003 | no-refine | e0,e12,e24 | e1-e11,e13-e23 | 0 | A-owned rows | s1:171, s7:352, s13:220 |
| 0.004 | no-refine | e0,e12,e24 | e1-e11,e13-e23 | 0 | A-owned rows | s1:189, s7:491, s13:378 |
| 0.005 | no-refine | e0,e12,e24 | e1-e11,e13-e23 | 0 | A-owned rows | s1:290, s7:717, s13:529 |
| 0.007 | no-refine | e0,e12,e24 | e1-e11,e13-e23 | 0 | A-owned rows | s1:587, s7:1067, s13:669 |
| 0.003 | refine | e0,e12,e24 | e1-e11,e13-e23 | 0 | A-owned rows | s1:2, s7:2 |
| 0.004 | refine | e0,e12,e24 | e1-e11,e13-e23 | 0 | A-owned rows | s1:9, s7:24, s13:15 |
| 0.005 | refine | e0,e12,e24 | e1-e11,e13-e23 | 0 | A-owned rows | s1:13, s7:65, s13:52 |
| 0.007 | refine | e0,e12,e24 | e1-e11,e13-e23 | 0 | A-owned rows | s1:144, s7:231, s13:145 |

A-center refine 统计：

| p | refine attempts | refine flagged | center changed |
|---:|---:|---:|---:|
| 0.003 | 300 | 2 | 159 |
| 0.004 | 300 | 20 | 194 |
| 0.005 | 300 | 42 | 236 |
| 0.007 | 300 | 165 | 272 |

### 25.4 结论

这个实验非常明确：

```text
把 A commit 缩小、让 B 负责更多，确实解决了 B rows 的 residual；
但 residual 直接转移到了 A-owned seam rows。
```

无 refine 版本几乎全失败：

```text
p=0.003: 92/100 flagged
p=0.007: 100/100 flagged
```

原因是 B 现在能完整闭合自己的 B rows：

```text
B final residual = 0
z_B used = 0
```

但 A 只 commit 一个 center 变量块，无法保证：

```text
s_A = H_left e_left + H0 e_center + H_right e_right
```

在 B 选完左右边界后仍然可解。

加上 `--a-center-refine` 后，A seam rows 明显减少，但仍然比 baseline 差：

```text
p=0.004: baseline 5/100, refine 10/100
p=0.007: baseline 43/100, refine 74/100
```

这验证了之前的担心：

```text
A seam 太窄。
```

也就是说，单独把 A 从 3 个 commit 变量块缩成 1 个，会让 B 的局部一致性变好，但 A seam equation 失去足够自由度。

### 25.5 下一步方向

这个方向不应该继续用：

```text
A commit only center 1 block
```

作为主方案。

更合理的折中是：

```text
A 不 hard commit 会直接作为 B residual boundary 的最外侧变量；
但 A seam 至少保留 2 个或更多 center-side 变量块。
```

例如把 interior A 从：

```text
旧：A2 commit e11,e12,e13
过窄：A2 commit e12
```

改成 asymmetric ownership：

```text
方案 A: A2 commit e12,e13，B1 owns e11
方案 B: A2 commit e11,e12，B2 owns e13
方案 C: A2 commit e12 plus a small A-seam auxiliary retry with adjacent A/B rows
```

从 diagnostics 看，`A-center+B-expanded` 已经把问题从 B rows 转移到 A rows，因此下一步如果继续改 ownership，需要给 A seam 多一点自由度，而不是把 A 压到只剩一个 center block。

## 26. a_solve=9 non-overlap schedule check

### 26.1 目的

测试：

```text
a_solve_size = 9
```

时，A solve windows 是否可以保持不重叠。

结论先写在前面：

```text
可以，但需要 b_width 也同步设为 9。
```

如果：

```text
a_solve_size=9
b_width=9
```

代码会走 buffer-aligned schedule：

```text
step = a_solve_size + 1 = 10
```

A windows 不重叠。

如果：

```text
a_solve_size=9
b_width=5
```

代码会走 staggered schedule，A windows 会重叠。

### 26.2 schedule check

#### a_solve=9, b_width=9

结果文件：

```text
ParallelWindowDecoder/results/schedule_check_a9_b9.csv
```

schedule：

```text
schedule = buffer_aligned
step = 10

A rows:
    A1 solve s1-s5
    A2 solve s7-s13

B rows:
    B1 solve s2-s10
    B2 solve s12-s13

A commit:
    A1 commit e0-e1
    A2 commit e19-e21

B commit:
    B1 commit e2-e18
    B2 commit e22-e24
```

所以 A rows：

```text
s1-s5
s7-s13
```

没有重叠，中间隔着 `s6`。

#### a_solve=9, b_width=5

结果文件：

```text
ParallelWindowDecoder/results/schedule_check_a9_b5.csv
```

schedule：

```text
schedule = staggered
step = 6

A rows:
    A1 solve s1-s5
    A2 solve s3-s11
    A3 solve s9-s13

B rows:
    B1 solve s2-s6
    B2 solve s8-s12
```

这里 A windows 明显重叠：

```text
A1: s1-s5
A2: s3-s11
overlap: s3-s5

A2: s3-s11
A3: s9-s13
overlap: s9-s11
```

因此若实验要求 A 之间不重叠，不能使用 `a_solve=9,b_width=5`。

### 26.3 100-shot test: a_solve=9, b_width=9

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
  --p-list 0.003,0.004,0.005,0.007 \
  --decoders parallel \
  --a-size 3 \
  --a-solve-size 9 \
  --b-width 9 \
  --sliding-width 3 \
  --top-k-boundary 1 \
  --a-noisy-boundary \
  --b-noisy-boundary \
  --window-shorten \
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_a_solve9_b9_ab_noisy_shorten_parallel_only.csv
```

结果：

| p | flagged | LER | runtime | z_B used |
|---:|---:|---:|---:|---:|
| 0.003 | 0/100 | 0.000 | 4.115s | 0 |
| 0.004 | 0/100 | 0.000 | 5.113s | 0 |
| 0.005 | 0/100 | 0.040 | 7.710s | 0 |
| 0.007 | 0/100 | 0.430 | 15.516s | 0 |

和 `a_solve=5,b_width=5` baseline 对比：

| p | a5/b5 flagged/LER/time | a9/b9 flagged/LER/time |
|---:|---:|---:|
| 0.003 | 1/100, 0.010, 3.493s | 0/100, 0.000, 4.115s |
| 0.004 | 5/100, 0.060, 3.973s | 0/100, 0.000, 5.113s |
| 0.005 | 17/100, 0.200, 4.785s | 0/100, 0.040, 7.710s |
| 0.007 | 43/100, 0.590, 8.015s | 0/100, 0.430, 15.516s |

diagnostics：

```text
A local flagged = 0
B local flagged = 0
z_B used = 0
final residual = 0
```

### 26.4 结论

`a_solve=9,b_width=9` 下，A windows 可以不重叠，而且 final flagged 降到 0：

```text
p=0.003: 1/100 -> 0/100
p=0.004: 5/100 -> 0/100
p=0.005: 17/100 -> 0/100
p=0.007: 43/100 -> 0/100
```

这说明大 A context 确实解决了 seam residual / z_B 问题。

但代价也明显：

```text
p=0.007 runtime: 8.015s -> 15.516s
```

并且 high-p 下 LER 仍然不低：

```text
p=0.007 LER = 0.430
```

所以结论是：

```text
如果允许 b_width 也设为 9，A 可以保持不重叠；
但这已经不是小 window 对比，矩阵规模和 runtime 都明显增大。
```

## 27. TSAE: Tri-Shift A Ensemble 实验

### 27.1 目的

上一节说明：

```text
a_solve=9,b_width=9
→ A windows 可以不重叠
→ final flagged = 0
→ 但单个局部矩阵变大，runtime 增加明显
```

本实验尝试把大的 A solve 拆成多个小 A solve。后续把这个策略命名为：

```text
TSAE = Tri-Shift A Ensemble
中文：三偏移 A 集成
```

含义是：对同一个 interior A commit，同时使用 left / center / right 三个同规模 shifted A 小窗口给出候选，然后选择一个 candidate commit。

```text
不用一个 9-row A 矩阵；
而是用多个 5-row shifted A 矩阵提供不同上下文。
```

新增参数：

```text
--a-shifted-ensemble
--a-shift-offsets -2,0,2
--a-shifted-beta
```

### 27.2 策略

baseline A2 是：

```text
A2 solve:  s5-s9
A2 commit: e11-e13
```

TSAE 使用同样大小的 5-row 小窗口：

```text
A2-left   solve: s3-s7
A2-center solve: s5-s9
A2-right  solve: s7-s11
```

每个窗口都只输出同一个 commit：

```text
e11,e12,e13
```

然后在每个 shot 上选择一个 shifted candidate：

```text
score = local_cost + beta * neighbor_B_residual_risk
```

其中：

```text
neighbor_B_residual_risk
= candidate 对相邻 B rows 造成的 residual popcount
```

边界 A1/A3 因为 commit 变量必须落在 solve cols 内，只有原始 offset：

```text
A1 valid offsets: 0
A2 valid offsets: -2,0,2
A3 valid offsets: 0
```

所以总 candidate 数为：

```text
每个 shot: 1 + 3 + 1 = 5 个 A 小窗口候选
```

单个 A 子矩阵行数仍然不超过：

```text
a_solve_size = 5
```

### 27.3 运行命令

TSAE beta=1：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_TSAE_tri_shift_A_beta1_ab_noisy_shorten_parallel_only.csv
```

beta=0 ablation：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_a_shifted_ensemble_m2_0_p2_beta0_ab_noisy_shorten_parallel_only.csv
```

### 27.4 结果

flagged / LER / runtime：

| p | baseline a5/b5 | shifted beta=0 | shifted beta=1 | a9/b9 |
|---:|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 3.493s | 0/100, 0.000, 3.786s | 0/100, 0.000, 3.610s | 0/100, 0.000, 4.115s |
| 0.004 | 5/100, 0.060, 3.973s | 3/100, 0.040, 4.062s | 3/100, 0.040, 3.964s | 0/100, 0.000, 5.113s |
| 0.005 | 17/100, 0.200, 4.785s | 4/100, 0.150, 4.798s | 4/100, 0.140, 4.675s | 0/100, 0.040, 7.710s |
| 0.007 | 43/100, 0.590, 8.015s | 24/100, 0.570, 6.661s | 25/100, 0.560, 6.402s | 0/100, 0.430, 15.516s |

z_B / residual diagnostics：

| p | baseline z_B used/weight | shifted beta=0 z_B used/weight | shifted beta=1 z_B used/weight |
|---:|---:|---:|---:|
| 0.003 | 1 / 6 | 0 / 0 | 0 / 0 |
| 0.004 | 5 / 28 | 3 / 14 | 3 / 14 |
| 0.005 | 17 / 106 | 4 / 14 | 4 / 14 |
| 0.007 | 43 / 274 | 24 / 152 | 25 / 166 |

shift 选择统计：

| p | beta=0 selected nonzero shift | beta=1 selected nonzero shift |
|---:|---:|---:|
| 0.003 | 67 | 63 |
| 0.004 | 76 | 74 |
| 0.005 | 79 | 74 |
| 0.007 | 74 | 68 |

这里 `selected nonzero shift` 说明 ensemble 确实经常选择 left/right shifted A，而不是只退化成原始 center A。

### 27.5 结论

TSAE 明显优于 baseline：

```text
p=0.003: flagged 1/100  -> 0/100
p=0.004: flagged 5/100  -> 3/100
p=0.005: flagged 17/100 -> 4/100
p=0.007: flagged 43/100 -> 24/100 或 25/100
```

它没有达到 `a_solve=9,b_width=9` 的 zero-flagged 效果，但资源代价更小：

```text
单个 A 子矩阵仍然是 5-row；
runtime 接近 baseline，远低于 a9/b9 的 high-p runtime。
```

尤其在 `p=0.005`：

```text
baseline flagged = 17/100
shifted flagged  = 4/100
a9/b9 flagged    = 0/100
```

说明这个策略确实在用多个小 A 近似大 A context。

beta=0 和 beta=1 差别很小，说明主要收益来自：

```text
shifted 多视角 A evidence
```

而不是当前简单的：

```text
neighbor_B_residual_risk
```

下一步如果继续优化 shifted ensemble，可以尝试：

```text
1. 改进 selection score，而不是只用 local cost + B residual popcount；
2. 只对边界变量 e11/e13 使用 shifted estimate，中间 e12 仍用 center；
3. 增加 offset -1,+1 或比较 [-1,0,1] 与 [-2,0,2]；
4. 和 z_B-triggered C layer 组合，只在 shifted 后仍 z_B != 0 时进入 C repair。
```

### 27.6 与 sliding / a_solve=9,b_width=9 对比

为了把 TSAE 放到更清楚的位置，补跑同样 `repeat=12, shots=100` 的：

```text
sliding width = 3 + shortening
parallel a_solve=9,b_width=9 + noisy boundary + shortening
```

命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
  --p-list 0.003,0.004,0.005,0.007 \
  --decoders sliding,parallel \
  --a-size 3 \
  --a-solve-size 9 \
  --b-width 9 \
  --sliding-width 3 \
  --top-k-boundary 1 \
  --a-noisy-boundary \
  --b-noisy-boundary \
  --window-shorten \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_compare_sliding_vs_a9b9_ab_noisy_shorten.csv
```

综合对比：

| p | mode | flagged | LER | logical/flagged | runtime |
|---:|---|---:|---:|---:|---:|
| 0.003 | baseline a5/b5 | 1/100 | 0.010 | 1/100 | 3.493s |
| 0.003 | TSAE a5 tri-shift | 0/100 | 0.000 | 0/100 | 3.610s |
| 0.003 | sliding w3 | 0/100 | 0.000 | 0/100 | 5.637s |
| 0.003 | parallel a9/b9 | 0/100 | 0.000 | 0/100 | 3.511s |
| 0.004 | baseline a5/b5 | 5/100 | 0.060 | 6/100 | 3.973s |
| 0.004 | TSAE a5 tri-shift | 3/100 | 0.040 | 4/100 | 3.964s |
| 0.004 | sliding w3 | 0/100 | 0.020 | 2/100 | 6.283s |
| 0.004 | parallel a9/b9 | 0/100 | 0.000 | 0/100 | 4.468s |
| 0.005 | baseline a5/b5 | 17/100 | 0.200 | 20/100 | 4.785s |
| 0.005 | TSAE a5 tri-shift | 4/100 | 0.140 | 14/100 | 4.675s |
| 0.005 | sliding w3 | 0/100 | 0.120 | 12/100 | 7.051s |
| 0.005 | parallel a9/b9 | 0/100 | 0.040 | 4/100 | 7.213s |
| 0.007 | baseline a5/b5 | 43/100 | 0.590 | 59/100 | 8.015s |
| 0.007 | TSAE a5 tri-shift | 25/100 | 0.560 | 56/100 | 6.402s |
| 0.007 | sliding w3 | 0/100 | 0.450 | 45/100 | 9.089s |
| 0.007 | parallel a9/b9 | 0/100 | 0.430 | 43/100 | 14.791s |

结论：

```text
1. sliding 和 a9/b9 都能把 final flagged 降到 0/100。
2. a9/b9 的 LER 在 p=0.005 和 p=0.007 都优于 sliding。
3. TSAE 明显降低 flagged，但仍没有达到 sliding / a9/b9 的 physical closure。
4. TSAE 的价值在于单个 A 子矩阵仍是 5-row，资源形态更接近 FPGA 小核并行；
   a9/b9 的价值在于准确率最好，但单个局部矩阵明显变大。
```

### 27.7 Boundary-aware TSAE 尝试

代码状态：该分支已从当前代码中移除，仅保留实验记录；当前主线保留最初的 TSAE。

上面的 TSAE 只对 interior A2 做三偏移：

```text
A1: center
A2: left / center / right
A3: center
```

进一步尝试边界感知版本：

```text
Boundary-aware TSAE

A1: center / right
A2: left / center / right
A3: left / center
```

对应矩阵：

```text
A1-center solve: s1-s3   -> commit e0-e1
A1-right  solve: s1-s5   -> commit e0-e1

A2-left   solve: s3-s7   -> commit e11-e13
A2-center solve: s5-s9   -> commit e11-e13
A2-right  solve: s7-s11  -> commit e11-e13

A3-left   solve: s9-s13  -> commit e23-e24
A3-center solve: s11-s13 -> commit e23-e24
```

新增参数：

```text
--boundary-aware-tsae
```

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --boundary-aware-tsae \
  --a-shift-offsets=-2,0,2 \
  --a-shifted-beta 1.0 \
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_boundary_aware_TSAE_beta1_ab_noisy_shorten_parallel_only.csv
```

结果：

| p | baseline a5/b5 | TSAE | Boundary-aware TSAE | sliding w3 | a9/b9 |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 3.493s | 0/100, 0.000, 3.610s | 0/100, 0.000, 4.562s | 0/100, 0.000, 5.637s | 0/100, 0.000, 3.511s |
| 0.004 | 5/100, 0.060, 3.973s | 3/100, 0.040, 3.964s | 4/100, 0.060, 4.907s | 0/100, 0.020, 6.283s | 0/100, 0.000, 4.468s |
| 0.005 | 17/100, 0.200, 4.785s | 4/100, 0.140, 4.675s | 16/100, 0.230, 5.597s | 0/100, 0.120, 7.051s | 0/100, 0.040, 7.213s |
| 0.007 | 43/100, 0.590, 8.015s | 25/100, 0.560, 6.402s | 49/100, 0.660, 8.075s | 0/100, 0.450, 9.089s | 0/100, 0.430, 14.791s |

diagnostics：

```text
valid offsets:
    A1: center,right
    A2: -2,0,2
    A3: left,center

candidate count:
    7 per shot

selected_nonzero:
    200 / 100 shots
```

解释：

```text
Boundary-aware TSAE 的 naive selection 反而变差。
主要原因是 A1/A3 的 center 是 3-row，而 anchored right/left 是 5-row；
当前 score = local_cost + beta * neighbor_B_residual_risk
没有对不同 row-count / 不同局部矩阵规模做归一化，
导致边界 A 的 anchored 候选被系统性选中。
```

因此这个版本暂时不作为主结果。若继续尝试，需要先修正 selection rule，例如：

```text
1. 对 A1/A3 的 center/right/left 候选使用归一化 cost；
2. 对 boundary A 的 anchored 候选加 penalty；
3. 只允许 anchored 候选在 center local flagged 或 z_B 触发时参与选择；
4. 分别统计 A1/A2/A3 的 selected label，再做 ablation。
```

### 27.8 Boundary anchor veto-only

代码状态：该分支已从当前代码中移除，仅保留实验记录。

上一节的问题是 A1-right / A3-left 直接替换 center commit 后会污染 B residual。
因此改成更保守的版本：

```text
1. A1/A3 最终 commit 永远来自 center；
2. A1-right / A3-left 仍然并行解码；
3. anchored 候选只作为 reliability / veto 诊断，不直接写入 correction。
```

新增参数：

```text
--boundary-anchor-veto-only
```

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --boundary-aware-tsae \
  --boundary-anchor-veto-only \
  --a-shift-offsets=-2,0,2 \
  --a-shifted-beta 1.0 \
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_boundary_anchor_veto_only_TSAE_beta1_ab_noisy_shorten_parallel_only.csv
```

结果：

| p | TSAE | Boundary-aware commit | Boundary anchor veto-only |
|---:|---:|---:|---:|
| 0.003 | 0/100, 0.000, 3.610s | 0/100, 0.000, 4.562s | 0/100, 0.000, 4.433s |
| 0.004 | 3/100, 0.040, 3.964s | 4/100, 0.060, 4.907s | 3/100, 0.040, 4.873s |
| 0.005 | 4/100, 0.140, 4.675s | 16/100, 0.230, 5.597s | 4/100, 0.140, 5.671s |
| 0.007 | 25/100, 0.560, 6.402s | 49/100, 0.660, 8.075s | 25/100, 0.560, 7.660s |

anchor disagreement：

| p | A1/A3 anchor disagree |
|---:|---:|
| 0.003 | 7/200 |
| 0.004 | 10/200 |
| 0.005 | 23/200 |
| 0.007 | 54/200 |

结论：

```text
Boundary anchor veto-only 的 correction 与原 TSAE 完全一致；
flagged 和 LER 没有改善，只增加了 A1-right/A3-left 的额外解码成本。
```

但 disagreement 统计有用：

```text
p 越高，A1/A3 的 anchored view 越经常和 center view 不一致。
```

这说明 A1/A3 的辅助窗口确实能提供“边界不确定性”信息，但不能直接替换 commit。
后续如果继续利用它，应把 disagreement 作为触发信号，例如：

```text
if A1-center != A1-right or A3-center != A3-left:
    标记 boundary unreliable
    只在对应 B window z_B != 0 时触发 C repair / rerank
```

### 27.9 Boundary anchor partial commit

代码状态：该分支已从当前代码中移除，仅保留实验记录。

继续尝试一个介于 whole-commit 和 veto-only 之间的版本：

```text
A1-right 只允许覆盖 e1，不覆盖 e0；
A3-left   只允许覆盖 e23，不覆盖 e24。
```

也就是说：

```text
A1:
    e0 <- A1-center
    e1 <- A1-right

A3:
    e23 <- A3-left
    e24 <- A3-center
```

A2 仍然使用原 TSAE：

```text
A2-left / A2-center / A2-right 选择一个 candidate commit e11-e13
```

新增参数：

```text
--boundary-anchor-partial-commit
```

运行命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --boundary-aware-tsae \
  --boundary-anchor-partial-commit \
  --a-shift-offsets=-2,0,2 \
  --a-shifted-beta 1.0 \
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_boundary_anchor_partial_commit_TSAE_beta1_ab_noisy_shorten_parallel_only.csv
```

结果：

| p | TSAE | Boundary-aware whole commit | Boundary anchor veto-only | Boundary anchor partial commit |
|---:|---:|---:|---:|---:|
| 0.003 | 0/100, 0.000, 3.610s | 0/100, 0.000, 4.562s | 0/100, 0.000, 4.433s | 4/100, 0.040, 4.577s |
| 0.004 | 3/100, 0.040, 3.964s | 4/100, 0.060, 4.907s | 3/100, 0.040, 4.873s | 8/100, 0.080, 4.919s |
| 0.005 | 4/100, 0.140, 4.675s | 16/100, 0.230, 5.597s | 4/100, 0.140, 5.671s | 10/100, 0.160, 5.658s |
| 0.007 | 25/100, 0.560, 6.402s | 49/100, 0.660, 8.075s | 25/100, 0.560, 7.660s | 39/100, 0.600, 7.608s |

diagnostics：

```text
partial_used = 200 / 100 shots
```

每个 shot 都会对 A1 的 e1 和 A3 的 e23 使用 anchored partial commit。

结论：

```text
partial commit 比 whole commit 温和，但仍明显差于原 TSAE。
```

说明问题不是只来自 `e0/e24` 这些最外侧变量；即使只改：

```text
A1-right -> e1
A3-left  -> e23
```

仍然会污染相邻 B residual。当前结果支持一个判断：

```text
A1/A3 anchored view 可以作为不确定性诊断，
但不应该无条件写入 physical correction。
```

因此边界 anchored 信息后续更适合作为 gated repair 的触发条件，而不是直接 commit。

### 27.10 Boundary anchor partial commit: same-size version

代码状态：该分支已从当前代码中移除，仅保留实验记录。

上一节 partial commit 使用的是 anchored 大小：

```text
A1-right solve: s1-s5 -> e1 only
A3-left  solve: s9-s13 -> e23 only
```

进一步按同规模窗口改成：

```text
A1-center solve: s1-s3 -> e0,e1
A1-right  solve: s2-s4 -> e1 only

A3-left   solve: s10-s12 -> e23 only
A3-center solve: s11-s13 -> e23,e24
```

这样 A1-right / A3-left 和 A1/A3 center 一样都是 3-row 矩阵。

整体变量来源：

```text
e0        <- A1-center
e1        <- A1-right partial
e2-e10    <- B1
e11-e13   <- selected A2 TSAE
e14-e22   <- B2
e23       <- A3-left partial
e24       <- A3-center
```

结果：

| p | TSAE | partial anchored 5-row | partial same-size |
|---:|---:|---:|---:|
| 0.003 | 0/100, 0.000, 3.610s | 4/100, 0.040, 4.577s | 71/100, 0.720, 4.481s |
| 0.004 | 3/100, 0.040, 3.964s | 8/100, 0.080, 4.919s | 89/100, 0.890, 5.057s |
| 0.005 | 4/100, 0.140, 4.675s | 10/100, 0.160, 5.658s | 99/100, 0.990, 6.134s |
| 0.007 | 25/100, 0.560, 6.402s | 39/100, 0.600, 7.608s | 99/100, 0.990, 7.967s |

disagreement：

| p | partial anchored 5-row | partial same-size |
|---:|---:|---:|
| 0.003 | 7/200 | 97/200 |
| 0.004 | 10/200 | 128/200 |
| 0.005 | 23/200 | 156/200 |
| 0.007 | 54/200 | 177/200 |

结论：

```text
same-size partial commit 明显不可行。
```

原因是：

```text
A1-right: s2-s4 -> e1 only
A3-left:  s10-s12 -> e23 only
```

这些窗口不包含 A boundary 的完整 center equation：

```text
A1 center row s1 couples e0,e1
A3 center row s13 couples e23,e24
```

因此 partial 窗口给出的 `e1/e23` 和 center 给出的 `e0/e24` 拼接后，往往不再满足 A seam 自身的一致性。
这直接把 residual 留在 A/B 边界附近，导致 final flagged 极高。

这个实验说明：

```text
边界变量不能从缺少配对 seam equation 的 shifted partial 窗口中无条件抽取。
```

所以 A1/A3 的 right/left 小窗口仍然更适合作为 reliability diagnostic，而不是 commit source。

### 27.11 TSAE offset sweep

回到最初 TSAE，只改变 interior A2 的 shifted offsets：

```text
方案 A: -1,0,1
方案 B: -2,0,2
方案 C: -2,-1,0,1,2
```

其它参数保持一致：

```text
N=144
repeat=12
shots=100
a_solve=5
b_width=5
sliding_width=3
a_noisy_boundary=True
b_noisy_boundary=True
window_shorten=True
beta=1.0
```

结果：

| p | baseline a5/b5 | TSAE -1,0,1 | TSAE -2,0,2 | TSAE -2,-1,0,1,2 |
|---:|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 3.493s | 1/100, 0.010, 4.778s | 0/100, 0.000, 3.610s | 0/100, 0.000, 5.606s |
| 0.004 | 5/100, 0.060, 3.973s | 4/100, 0.050, 4.932s | 3/100, 0.040, 3.964s | 3/100, 0.040, 5.546s |
| 0.005 | 17/100, 0.200, 4.785s | 9/100, 0.120, 5.760s | 4/100, 0.140, 4.675s | 4/100, 0.140, 6.903s |
| 0.007 | 43/100, 0.590, 8.015s | 30/100, 0.540, 8.092s | 25/100, 0.560, 6.402s | 24/100, 0.550, 8.650s |

candidate 统计：

| offsets | valid offsets | candidates / shot | p=0.007 selected nonzero |
|---|---|---:|---:|
| -1,0,1 | A1: 0, A2: -1/0/1, A3: 0 | 5 | 74 |
| -2,0,2 | A1: 0, A2: -2/0/2, A3: 0 | 5 | 68 |
| -2,-1,0,1,2 | A1: 0, A2: -2/-1/0/1/2, A3: 0 | 7 | 86 |

结论：

```text
1. -1,0,1 明显弱于 -2,0,2。
2. -2,-1,0,1,2 与 -2,0,2 的 flagged 很接近；
   只在 p=0.007 从 25/100 降到 24/100。
3. 五偏移 candidate 更多，runtime 明显增加。
4. 当前最好的性价比仍然是 -2,0,2。
```

解释：

```text
-2,0,2 提供了更分散的上下文视角：
    s3-s7, s5-s9, s7-s11

-1,0,1 的三个视角更接近：
    s4-s8, s5-s9, s6-s10

所以 -1,0,1 的 ensemble 多样性不足，不能像 -2,0,2 那样明显降低 boundary hard decision 错误。
```

### 27.12 TSAE-stitch: component / pairwise stitching

继续尝试不增加 BP-OSD 次数，只在 TSAE 已有候选上做更细的 stitching。

原 TSAE 是：

```text
A2-left, A2-center, A2-right
    -> 选一个完整 candidate commit e11,e12,e13
```

新增两个模式：

```text
--tsae-stitch-mode component
--tsae-stitch-mode pairwise
```

#### component stitching

分量级选择：

```text
e11 从 left/center 中按左 B boundary residual 选择；
e12 固定从 center 取；
e13 从 center/right 中按右 B boundary residual 选择。
```

然后检查 A seam row：

```text
s7 = H2 e11 + H0 e12 + H1 e13
```

如果 A seam residual 为 0，则接受拼接；否则 fallback 到原 TSAE 的完整 candidate。

#### pairwise stitching

枚举小组合：

```text
e11 source in {left, center}
e12 source = center
e13 source in {center, right}
```

打分：

```text
score = wt(A_center_residual) + beta * wt(left_B_boundary + right_B_boundary)
```

选择 score 最小的组合。

#### 结果

| p | baseline | TSAE | TSAE-component | TSAE-pairwise |
|---:|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 3.493s | 0/100, 0.000, 3.610s | 0/100, 0.000, 4.815s | 0/100, 0.000, 4.872s |
| 0.004 | 5/100, 0.060, 3.973s | 3/100, 0.040, 3.964s | 4/100, 0.050, 5.631s | 5/100, 0.060, 5.534s |
| 0.005 | 17/100, 0.200, 4.785s | 4/100, 0.140, 4.675s | 4/100, 0.140, 6.423s | 15/100, 0.180, 6.447s |
| 0.007 | 43/100, 0.590, 8.015s | 25/100, 0.560, 6.402s | 26/100, 0.550, 8.338s | 44/100, 0.600, 8.725s |

stitch diagnostics：

| p | component accepted/fallback/non-payload | pairwise accepted/fallback/non-payload |
|---:|---:|---:|
| 0.003 | 95 / 5 / 63 | 100 / 0 / 66 |
| 0.004 | 88 / 12 / 66 | 100 / 0 / 71 |
| 0.005 | 68 / 32 / 49 | 100 / 0 / 64 |
| 0.007 | 37 / 63 / 19 | 100 / 0 / 39 |

结论：

```text
1. component stitching 没有明显改善；
   p=0.005 与 TSAE 相同，p=0.007 略差。

2. pairwise stitching 明显变差；
   p=0.007 甚至 44/100，接近 baseline 43/100。

3. 当前 residual-only stitching score 不够可靠。
```

解释：

```text
完整 TSAE candidate 来自同一个 BP-OSD 局部解，内部一致性较强；
component/pairwise 把不同局部解的 e11/e12/e13 拼起来，
即使 A seam residual 或 B boundary residual 局部较小，
也可能破坏更大范围的 correction consistency。
```

因此当前主线仍然保持：

```text
TSAE -2,0,2，完整 candidate 三选一。
```

### 27.13 TSAE + Top-K=2

继续尝试：

```text
TSAE 提供 shifted context diversity；
Top-K 提供每个 context 内的 solution diversity。
```

实现方式：

```text
--tsae-top-k 2
--tsae-boundary-top-k 1
```

也就是：

```text
A1: top1
A2-left:   top2
A2-center: top2
A2-right:  top2
A3: top1
```

每个 shot 的 A candidate 数从：

```text
TSAE top1: 1 + 3 + 1 = 5
TSAE top2: 1 + 3*2 + 1 = 8
```

保持完整 candidate 粒度，不做 bit-level stitching。

命令：

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 144 \
  --num-repeat 12 \
  --num-shots 100 \
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
  --tsae-top-k 2 \
  --tsae-boundary-top-k 1 \
  --seam-diagnostics \
  --parallel-workers 4 \
  --parallel-backend process \
  --out ParallelWindowDecoder/results/N144_repeat12_shots100_TSAE_topk2_offsets_m2_0_p2_beta1_ab_noisy_shorten_parallel_only.csv
```

结果：

| p | baseline | TSAE top1 | TSAE top2 | sliding | a9/b9 |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 1/100, 0.010, 3.493s | 0/100, 0.000, 3.610s | 0/100, 0.000, 3.724s | 0/100, 0.000, 5.637s | 0/100, 0.000, 3.511s |
| 0.004 | 5/100, 0.060, 3.973s | 3/100, 0.040, 3.964s | 3/100, 0.040, 4.067s | 0/100, 0.020, 6.283s | 0/100, 0.000, 4.468s |
| 0.005 | 17/100, 0.200, 4.785s | 4/100, 0.140, 4.675s | 4/100, 0.140, 4.621s | 0/100, 0.120, 7.051s | 0/100, 0.040, 7.213s |
| 0.007 | 43/100, 0.590, 8.015s | 25/100, 0.560, 6.402s | 25/100, 0.550, 6.672s | 0/100, 0.450, 9.089s | 0/100, 0.430, 14.791s |

candidate 统计：

| mode | candidates / shot | p=0.007 selected nonzero | p=0.007 z_B used |
|---|---:|---:|---:|
| TSAE top1 | 5 | 68 | 25 |
| TSAE top2 | 8 | 68 | 25 |

结论：

```text
TSAE + Top-K=2 没有降低 flagged。
```

虽然候选数从 5 增加到 8，但：

```text
p=0.004: 3/100 -> 3/100
p=0.005: 4/100 -> 4/100
p=0.007: 25/100 -> 25/100
```

说明 per-shift top2 没有提供当前 selector 能利用的有效新 commit diversity。
可能原因：

```text
1. OSD-list 的 top2 在 commit 区域 e11-e13 上经常和 top1 相同或等价；
2. 新 candidate 虽然不同，但 local_cost + neighbor_B_risk 没有选中；
3. 剩余失败可能不是单个 A2 候选池扩大能解决，而需要 A/B 联合一致性或更大上下文。
```

## TSAE interface branching, GDG-style

目的：

```text
不扩大单个 A 小窗口；
只在 interior A 的左右 interface region 中各选 1 个最相关物理列；
对这两个 interface bit 做 0/1 hard-fix branching；
每个 branch 下仍然用 shifted 小 A window 解码；
最后用原 TSAE score 选择 candidate。
```

参数：

```bash
--a-shifted-ensemble
--a-shift-offsets=-2,0,2
--a-shifted-beta 1.0
--tsae-interface-branch
--tsae-interface-cols-per-side 1
```

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_TSAE_baseline_after_interface_branch_refactor.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_TSAE_interface_branch_c1_ab_noisy_shorten_parallel_only.csv
```

结果：

| p | TSAE baseline | TSAE interface branch | runtime baseline | runtime branch |
|---:|---:|---:|---:|---:|
| 0.003 | 0/100, LER 0.000 | 0/100, LER 0.000 | 4.975s | 8.500s |
| 0.004 | 3/100, LER 0.040 | 3/100, LER 0.050 | 4.926s | 9.248s |
| 0.005 | 4/100, LER 0.140 | 4/100, LER 0.130 | 5.776s | 10.449s |
| 0.007 | 25/100, LER 0.560 | 21/100, LER 0.530 | 8.692s | 10.873s |

candidate 统计：

```text
TSAE baseline candidate count: 500
TSAE interface branch candidate count: 1400
interface branch payloads: 12
```

结论：

```text
interface branching 在 p=0.007 有小幅改善：
flagged 25/100 -> 21/100
LER 0.560 -> 0.530

但 p=0.003/0.004/0.005 基本没有改善，runtime 增加明显。
说明“在 interface bit 上主动制造 diversity”有一点有效信号，
但当前每侧只选 1 个物理列的 hard-fix branch 仍然不足以模拟 a_solve=9 的大上下文。
```

### 与普通 parallel / sliding / oracle 对比

普通 parallel、普通 TSAE、TSAE interface、sliding、oracle 使用同一组参数和 seeds 对比。

普通 parallel、sliding、oracle 使用同一组参数重新运行：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_compare_sliding_plain_parallel_oracle_after_interface_branch.csv
```

普通 TSAE 和 TSAE interface branch 使用：

```text
ParallelWindowDecoder/results/N144_repeat12_shots100_TSAE_baseline_after_interface_branch_refactor.csv
ParallelWindowDecoder/results/N144_repeat12_shots100_TSAE_interface_branch_c1_ab_noisy_shorten_parallel_only.csv
```

对比：

| p | mode | flagged | LER | runtime |
|---:|---|---:|---:|---:|
| 0.003 | sliding | 0/100 | 0.000 | 7.168s |
| 0.003 | plain parallel | 1/100 | 0.010 | 3.676s |
| 0.003 | TSAE | 0/100 | 0.000 | 4.975s |
| 0.003 | TSAE interface | 0/100 | 0.000 | 8.500s |
| 0.003 | oracle parallel | 0/100 | 0.000 | 2.300s |
| 0.004 | sliding | 0/100 | 0.020 | 6.459s |
| 0.004 | plain parallel | 5/100 | 0.060 | 4.072s |
| 0.004 | TSAE | 3/100 | 0.040 | 4.926s |
| 0.004 | TSAE interface | 3/100 | 0.050 | 9.248s |
| 0.004 | oracle parallel | 0/100 | 0.000 | 3.082s |
| 0.005 | sliding | 0/100 | 0.120 | 7.923s |
| 0.005 | plain parallel | 17/100 | 0.200 | 4.578s |
| 0.005 | TSAE | 4/100 | 0.140 | 5.776s |
| 0.005 | TSAE interface | 4/100 | 0.130 | 10.449s |
| 0.005 | oracle parallel | 0/100 | 0.020 | 2.760s |
| 0.007 | sliding | 0/100 | 0.450 | 9.023s |
| 0.007 | plain parallel | 43/100 | 0.590 | 6.377s |
| 0.007 | TSAE | 25/100 | 0.560 | 8.692s |
| 0.007 | TSAE interface | 21/100 | 0.530 | 10.873s |
| 0.007 | oracle parallel | 0/100 | 0.130 | 3.832s |

结论：

```text
普通 TSAE 已经相比 plain parallel 明显降低 flagged：
p=0.005: 17/100 -> 4/100
p=0.007: 43/100 -> 25/100

TSAE interface branch 相比普通 TSAE 只在高 p 有小幅额外改善：
p=0.003: 0/100 -> 0/100
p=0.004: 3/100 -> 3/100
p=0.005: 4/100 -> 4/100
p=0.007: 25/100 -> 21/100

但它仍然没有接近 sliding/oracle 的 0 flagged，
并且 runtime 已经高于 sliding。
```

这说明主要收益仍来自 TSAE 的 shifted context；
interface branching 确实缓解了一部分高 p 的 A boundary hard decision 错误，
但当前实现的额外收益有限，且代价偏高。

## repeat=18 对比

目的：

```text
把 num_repeat 从 12 改成 18，
保持 N=144、shots=100、p-list=0.003,0.004,0.005,0.007、
a_solve=5、b_width=5、sliding_width=3、noisy boundary + shorten 不变。
```

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat18_shots100_compare_sliding_plain_parallel_oracle.csv
ParallelWindowDecoder/results/N144_repeat18_shots100_TSAE_baseline.csv
ParallelWindowDecoder/results/N144_repeat18_shots100_TSAE_interface_branch_c1.csv
```

结果：

| p | mode | flagged | LER | runtime |
|---:|---|---:|---:|---:|
| 0.003 | sliding | 0/100 | 0.000 | 34.652s |
| 0.003 | plain parallel | 3/100 | 0.030 | 8.122s |
| 0.003 | TSAE | 0/100 | 0.000 | 11.403s |
| 0.003 | TSAE interface | 0/100 | 0.000 | 17.896s |
| 0.003 | oracle parallel | 0/100 | 0.000 | 10.867s |
| 0.004 | sliding | 0/100 | 0.020 | 30.504s |
| 0.004 | plain parallel | 6/100 | 0.060 | 7.387s |
| 0.004 | TSAE | 4/100 | 0.050 | 10.579s |
| 0.004 | TSAE interface | 5/100 | 0.060 | 20.010s |
| 0.004 | oracle parallel | 0/100 | 0.000 | 5.236s |
| 0.005 | sliding | 0/100 | 0.150 | 25.032s |
| 0.005 | plain parallel | 32/100 | 0.350 | 7.696s |
| 0.005 | TSAE | 14/100 | 0.270 | 12.738s |
| 0.005 | TSAE interface | 13/100 | 0.270 | 25.785s |
| 0.005 | oracle parallel | 0/100 | 0.020 | 5.532s |
| 0.007 | sliding | 0/100 | 0.620 | 28.301s |
| 0.007 | plain parallel | 68/100 | 0.830 | 9.571s |
| 0.007 | TSAE | 43/100 | 0.800 | 19.873s |
| 0.007 | TSAE interface | 38/100 | 0.780 | 23.445s |
| 0.007 | oracle parallel | 0/100 | 0.220 | 6.557s |

结论：

```text
repeat=18 后，plain parallel 的 flagged 明显上升：
p=0.005: 32/100
p=0.007: 68/100

TSAE 仍然能明显降低 flagged：
p=0.005: 32/100 -> 14/100
p=0.007: 68/100 -> 43/100

TSAE interface branch 只有小幅额外改善：
p=0.005: 14/100 -> 13/100
p=0.007: 43/100 -> 38/100

sliding 和 oracle 仍保持 0 flagged，
但 sliding runtime 随 repeat 增长明显，约 25-35s。
```

## repeat=12, shots=1000: sliding / plain parallel / TSAE

目的：

```text
用 1000 shots 重新测试 repeat=12，
比较 sliding、普通 parallel、普通 TSAE。
```

参数：

```text
N=144
num_repeat=12
shots=1000
p-list=0.003,0.004,0.005,0.007
a_solve=5
b_width=5
sliding_width=3
a_noisy_boundary=True
b_noisy_boundary=True
window_shorten=True
parallel_workers=4
parallel_backend=process
```

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots1000_compare_sliding_plain_parallel.csv
ParallelWindowDecoder/results/N144_repeat12_shots1000_TSAE_baseline.csv
```

结果：

| p | mode | flagged | logical_or_flagged | LER | runtime |
|---:|---|---:|---:|---:|---:|
| 0.003 | sliding | 0/1000 | 1/1000 | 0.001 | 71.426s |
| 0.003 | plain parallel | 10/1000 | 11/1000 | 0.011 | 29.389s |
| 0.003 | TSAE | 4/1000 | 7/1000 | 0.007 | 29.244s |
| 0.004 | sliding | 0/1000 | 21/1000 | 0.021 | 75.332s |
| 0.004 | plain parallel | 32/1000 | 38/1000 | 0.038 | 31.263s |
| 0.004 | TSAE | 11/1000 | 22/1000 | 0.022 | 40.071s |
| 0.005 | sliding | 0/1000 | 93/1000 | 0.093 | 70.182s |
| 0.005 | plain parallel | 119/1000 | 148/1000 | 0.148 | 33.977s |
| 0.005 | TSAE | 38/1000 | 111/1000 | 0.111 | 42.902s |
| 0.007 | sliding | 0/1000 | 513/1000 | 0.513 | 92.773s |
| 0.007 | plain parallel | 423/1000 | 585/1000 | 0.585 | 56.715s |
| 0.007 | TSAE | 275/1000 | 556/1000 | 0.556 | 67.522s |

结论：

```text
1000 shots 下，TSAE 相比 plain parallel 持续降低 flagged：
p=0.003: 10/1000 -> 4/1000
p=0.004: 32/1000 -> 11/1000
p=0.005: 119/1000 -> 38/1000
p=0.007: 423/1000 -> 275/1000

TSAE 的 LER 在 p=0.004 和 p=0.005 接近 sliding：
p=0.004: sliding 0.021, TSAE 0.022
p=0.005: sliding 0.093, TSAE 0.111

但高 p=0.007 下，TSAE 仍明显有 final flagged，
LER 也没有接近 sliding。
```

## TSAE + z_B targeted seam correction, shots=1000

目的：

```text
围绕当前 TSAE 的最强诊断信号 z_B 做 targeted physical seam correction。

触发条件：
    B noisy boundary z_B != 0

候选变量：
    z_B 对应 seam 附近的 A boundary vars + B edge vars

接受条件：
    global final residual weight 下降；
    记录 success_to_zero 表示是否真正把 final residual 清零。
```

实现补充：

```text
在原 z_boundary_repair 基础上增加 residual-gain 的 1/2-flip 小枚举；
如果枚举不能清零，再回退到小 BP-OSD repair。
允许 z_boundary_repair 与 TSAE 同时使用。
```

参数：

```bash
--a-shifted-ensemble
--a-shift-offsets=-2,0,2
--a-shifted-beta 1.0
--z-boundary-repair
--z-repair-edge-width 2
```

结果文件：

```text
ParallelWindowDecoder/results/N144_repeat12_shots1000_TSAE_z_boundary_repair_edge2.csv
```

对比：

| p | mode | flagged | logical_or_flagged | LER | runtime |
|---:|---|---:|---:|---:|---:|
| 0.003 | sliding | 0/1000 | 1/1000 | 0.001 | 71.426s |
| 0.003 | plain parallel | 10/1000 | 11/1000 | 0.011 | 29.389s |
| 0.003 | TSAE | 4/1000 | 7/1000 | 0.007 | 29.244s |
| 0.003 | TSAE + zRepair | 4/1000 | 7/1000 | 0.007 | 29.531s |
| 0.004 | sliding | 0/1000 | 21/1000 | 0.021 | 75.332s |
| 0.004 | plain parallel | 32/1000 | 38/1000 | 0.038 | 31.263s |
| 0.004 | TSAE | 11/1000 | 22/1000 | 0.022 | 40.071s |
| 0.004 | TSAE + zRepair | 11/1000 | 22/1000 | 0.022 | 32.834s |
| 0.005 | sliding | 0/1000 | 93/1000 | 0.093 | 70.182s |
| 0.005 | plain parallel | 119/1000 | 148/1000 | 0.148 | 33.977s |
| 0.005 | TSAE | 38/1000 | 111/1000 | 0.111 | 42.902s |
| 0.005 | TSAE + zRepair | 38/1000 | 111/1000 | 0.111 | 40.830s |
| 0.007 | sliding | 0/1000 | 513/1000 | 0.513 | 92.773s |
| 0.007 | plain parallel | 423/1000 | 585/1000 | 0.585 | 56.715s |
| 0.007 | TSAE | 275/1000 | 556/1000 | 0.556 | 67.522s |
| 0.007 | TSAE + zRepair | 275/1000 | 556/1000 | 0.556 | 61.377s |

z repair diagnostics：

| p | attempts | accepted | success_to_zero | decode_flagged | weight_before | weight_after | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 8 | 2 | 0 | 8 | 23 | 25 | -2 |
| 0.004 | 22 | 15 | 0 | 22 | 84 | 71 | 13 |
| 0.005 | 76 | 60 | 0 | 76 | 396 | 312 | 84 |
| 0.007 | 550 | 428 | 0 | 550 | 2920 | 2376 | 544 |

结论：

```text
z_B targeted repair 能降低部分 residual weight，
但 success_to_zero 始终为 0，
所以 final flagged 和 LER 完全没有改善。

这说明当前候选集合和小 BP-OSD repair 只能减轻 seam residual，
不能把 z_B witness 转化成完整的物理闭合 correction。
```

下一步判断：

```text
不能继续只做 residual-weight 降低型 repair；
如果目标是降低 flagged，acceptance 应该要求 residual 清零，
或者改成重新求解相关 B window + A boundary 的 joint physical problem。
```
