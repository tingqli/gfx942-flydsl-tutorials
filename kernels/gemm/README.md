# CDNA3 (gfx942) BF16 GEMM：用 FlyDSL 从 v0 写到 v2

参考：

- [FlyDSL kernel tuning guide](https://github.com/ROCm/FlyDSL/blob/main/docs/kernel_tuning_guide.md)
- [gfx950-gluon-tutorials / gfx942 / intra_wave](https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/gfx942/intra_wave)：v2 的设计来源

问题：`C[M,N] (fp32) = A[M,K] (bf16) · B[N,K]^T (bf16)`，A、B 都是 K 连续，`M=8192, N=9728, K=8192`。
`M/256 × N/256 = 32 × 38 = 1216 = 4 × 304` 个 CTA，正好是 304 个 CU 的整数倍，所以不讨论尾块。

| 文件 | 内容 |
|---|---|
| [v0.py](v0.py) | 1 wave/CTA，直接从显存读到寄存器做 MFMA |
| [v1.py](v1.py) | 4 wave/CTA，A/B 经 LDS 共享，XOR swizzle |
| [v2.py](v2.py) | v1 + 单 wave 软件流水（4 region、寄存器预取、`sched_group_barrier`） |
| [bench.py](bench.py) | 正确性 + 计时（含 torch.matmul / hipBLASLt） |
| [slides/](slides/) | 培训 PPT：`make_figs.py` 画图，`make_ppt.py` 生成 `CDNA3_GEMM_FlyDSL.pptx`，`measure.sh` 采数据 |

```bash
python3 bench.py                    # 所有版本：正确性 + TFLOPS
python3 bench.py v1 v2              # 指定版本
```

---

## 0. 先记住的硬件事实（CDNA3 / MI300X、MI325X）

| 项目 | 数值 | 对 kernel 的含义 |
|---|---|---|
| CU | 8 XCD × 38 = 304，每 CU 4 个 SIMD，wave64 | 一个 CTA 最多用满 1 个 CU；4 wave/CTA = 每 SIMD 1 个 wave |
| 寄存器 | 每 SIMD 512 KB；1 wave/SIMD 时每 lane 256 VGPR + 256 AGPR | 累加器可放 AGPR，不挤占 VGPR；occupancy 越高每 wave 寄存器越少 |
| MFMA | `v_mfma_f32_16x16x16_bf16`：8192 FLOP，每 SIMD 16 cycle | 峰值 304 × 4 × 512 FLOP/clk × 2.1 GHz = **1307 TFLOPS** |
| LDS | 64 KB/CU，32 bank × 4 B，128 B/clk，4 个 SIMD 共享 | 256×256×64 的 A+B 正好一级，**无法双缓冲** |
| L1 | 32 KB/CU | |
| L2 | 4 MB/XCD，XCD 之间不共享 | CTA 落在哪个 XCD 影响复用 |
| MALL | 256 MB Infinity Cache | |
| HBM | MI325X HBM3E ~6 TB/s | |
| global → LDS 直通 | 只有 4 B/lane（`buffer_load_dword ... lds`），16 B 版本 CDNA4 才有 | 实际路径是 global → VGPR → `ds_write` → LDS |
| 同步 | `s_barrier`（CTA 内）；`s_waitcnt vmcnt/lgkmcnt`（按发射顺序的计数器） | 编译器按“最老的第 N 条”插入等待；没有 mbarrier / 事务计数 |

### 与 NVIDIA 的对照（只说与本例相关的）

| | CDNA3 | NVIDIA Ampere / Hopper | 本例中的体现 |
|---|---|---|---|
| 执行粒度 | wave64，SIMD | warp32，SM 子分区 | MFMA 操作数排布以 64 lane 为单位（§1.2） |
| 矩阵指令 | MFMA，操作数在 VGPR/AGPR，同步发射 | Ampere `mma.sync` 同样在寄存器；Hopper `wgmma` 可直接读 smem、异步 | CDNA3 每个 K 步都要先 `ds_read` 到寄存器 |
| 片上共享存储 | LDS 64 KB | A100 164 KB、H100 228 KB | NVIDIA 常见 3~5 级流水；CDNA3 只能 1 级（§3） |
| 异步拷贝 | 无宽的 global→LDS 异步拷贝 | `cp.async`（Ampere）、TMA（Hopper），不占寄存器 | v2 用 64 个 VGPR 暂存下一块，并在指令流里发 `ds_write`（§3.6） |
| 累加器 | 独立的 AGPR 文件 | 统一寄存器堆，每线程 ≤ 255 | 128×128/wave 的 fp32 累加器正好占满 256 AGPR |
| 等待机制 | `s_waitcnt` 计数器，编译器显式插入 | 寄存器记分板（硬件自动）+ `cp.async.wait_group` / mbarrier | 读 ISA 时看 `vmcnt(N)` 判断流水深度 |
| bank conflict | 32 bank × 4 B | 32 bank × 4 B | XOR swizzle 原理相同（CUTLASS `Swizzle<3,4,3>`） |
| L2 | 每 XCD 独立 | 全芯片统一 | CTA → XCD 映射会影响 L2 命中（下一步优化） |

---

## 1. v0：最直观的实现

任务均分到 304 个 CU：每个 CTA 只有 1 个 wave，算 C 的 `[BM, BN]` 块，直接从显存读入寄存器，然后做 MFMA。SIMD 之间不协作。

### 1.1 FlyDSL 基本结构

```python
A = fx.make_view(fx.get_iter(A), fx.make_ordered_layout((8, K//8, 16, BM//16, grid_m), 0))
A = A[None, None, None, None, bid_m]                 # None = 保留该维，其余维取定值
row, kg = tid % 16, tid // 16
for k, acc in range(fx.Index(0), fx.Index(K//32), fx.Index(1), init=acc_init):   # 运行时 scf.for
    kk = fx.Int32(k) * 4 + kg                        # 循环变量是 index 类型，tid 是 Int32
    a_tile = [A[None, kk, row, i].load().bitcast(fx.Int16) for i in range_constexpr(NI)]   # 16 B load
    ...
    results = yield acc                               # 累加器作为 loop-carried 值
```

- `make_ordered_layout(shape, 0)`：第 0 维最连续。把 `A[M, K]` 看成 `(8, K/8, 16, BM/16, grid_m)` 的 5 维张量，切片和下标就直接对应“哪个 lane 读哪 16 字节”，不需要手写地址计算。
- `range_constexpr` 是编译期展开；`range(fx.Index(...), init=...)` 是运行时循环。边界必须是 DSL 类型，写成 Python int 会被展开。

### 1.2 MFMA 操作数排布

`v_mfma_f32_16x16x16_bf16`：lane `l` 提供 A 第 `l%16` 行、K 组 `l//16` 的 4 个元素，B 同理；结果 lane `l` 持有 C 的 `rows 4*(l//16)+0..3, col l%16`。

技巧：每个 lane 一次 load **8 个连续 K（16 B，`global_load_dwordx4` / `ds_read_b128`）**，前 4 个和后 4 个分别喂两条 MFMA。A、B 用同样的 K 排列，`Σk A·B` 不变。所以一个 wave 每步覆盖 32 个 K，所有 load 都是 16 B 宽。

### 1.3 优化原理：寄存器复用

一个 wave 算 `BM×BN`，每个 K 读 `BM+BN` 个元素、做 `2·BM·BN` 次 FLOP：寄存器层的算术强度 `I = BM·BN/(BM+BN)` flop/B（bf16）。

| wave tile | I (flop/B) | 累加器 / lane | 结果 |
|---|---|---|---|
| 64×64 | 32 | 64 | |
| 128×128 | 64 | 256（AGPR 用满） | A/B 各 32 VGPR |
| 128×256 | 85 | 512 | spill |

在不 spill 的前提下把 BM、BN 加到最大（128×128）。副作用是 occupancy 降到 1 wave/SIMD，没法靠 wave 切换（TLP）来掩盖延迟。

### 1.4 观察硬件行为

```bash
FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python3 v0.py   # IR + 最终 ISA
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i trace.yaml -- python3 v0.py       # ATT thread trace
rocprofv3 --stats --kernel-trace -f csv -- python3 v0.py                         # 基本信息
```

- ISA：每步 `global_load_dwordx4` 之后紧跟 `s_waitcnt vmcnt(0)` 才开始 MFMA，访存和计算串行。
- ATT `occupancy.json`：每个 CU 的 4 个 wave（每 SIMD 一个）在开始时同时上线、到结束才下线，中间没有新 wave 可以切换。
- 问题：同一 CU 的 4 个 wave 各读各的 A、B 行，没有共享。

---

## 2. v1：引入 LDS

CDNA3 的 4 个 SIMD 共享一个 LDS，所以 4 wave/CTA 可以通过 LDS + barrier 协作加载 A、B，再各自从 LDS 读取复用。目的和 v0 加大 wave tile 一样，是提高数据复用；LDS 的作用是把复用从 wave 级扩大到 CTA 级，进一步降低外存访问。

### 2.1 切分

- 64 KB LDS 尽量占满：CTA 算 256×256，4 个 wave 2×2 均分，每个 wave 仍算 128×128（保持寄存器复用）。
- 每个 K 步 `BK=64`：A、B 各 256×64 bf16 = 32 KB。
- L2 层强度：每 K 步读 64 KB、做 `2·256·256·64` FLOP → 128 flop/B，是 v0 的 2 倍。
- LDS 层：每 K 步读 128 KB（每 wave 32 KB）+ 写 64 KB，约 48 B/clk/CU，是 LDS 带宽 128 B/clk 的 37%。

### 2.2 每个 K 步的三段

```python
a_st = [A[None, st_c, kt, st_r, p].load() for p in range_constexpr(BM // 32)]   # ① global -> 暂存寄存器
gpu.barrier()                                                                   # 所有 wave 读完上一块
for p in range_constexpr(BM // 32):
    lds_a[None, st_c_sw, p * 32 + st_r].store(a_st[p])                         # ② 暂存寄存器 -> LDS
gpu.barrier()                                                                   # 本块对所有 wave 可见
a_tile = [lds_a[None, ck, wm * WM + i * 16 + row].load() ...]                    # ③ LDS -> 寄存器 -> MFMA
```

- **合并访存**：`st_c = tid % 8, st_r = tid // 8`，8 个相邻线程读一行的 128 B，一个 wave 一条指令读 8 行 = 1024 B 连续。
- **申请 LDS**：`@fx.struct` 声明 `fx.Array[fx.BFloat16, n, 16]`，`fx.SharedAllocator().allocate(...)`，再 `.view(layout)` 得到可切片的 LDS 张量。

### 2.3 bank conflict 与 XOR swizzle

LDS 按 `[行][8 个 16 B chunk]` 存放。一行 64 个 bf16 = 128 B = 正好 32 个 bank，所以不同行的同一 chunk 落在同一组 bank 上。

- `ds_read_b128` 每 8 个 lane 一拍（8×16 B = 128 B）。读操作数时，这 8 个 lane 读的是 8 行的同一 chunk，不做处理就是 8 路冲突。
- swizzle：物理 chunk = `chunk ^ (row % 8)`。8 行同一 chunk → 8 个不同 chunk → 8 组不同 bank，无冲突。
- 写的时候 8 个 lane 写同一行的 8 个 chunk，XOR 只是行内置换，仍然无冲突。
- XOR 不占额外空间；padding 要占空间，LDS 满 64 KB 时用不了。
- 实测 `SQ_LDS_BANK_CONFLICT = 0`。

### 2.4 v1 的问题

此时没有任何流水，相对 v0 的改动尽量少。每个 K 步严格串行：发 load → 等 vmcnt(0) → barrier → 写 LDS → barrier → 读 LDS + MFMA。1 wave/SIMD 下没有别的 wave 来填这些空档。v1 比 v0 快（外存读取减少到 1/4.5（5.59 → 1.24 GB）），但 MFMA 大部分时间仍在等访存。

---

## 3. v2：重叠搬运和计算 —— 单 wave 软件流水

前两步为了最大数据复用、降低外存访问，把 occupancy 做到了 1，没有 TLP 可用了。这时只能用 **ILP**：在同一个指令流里并行发出彼此无关的 vmem / MFMA / ds / valu 指令。要做到这一点，需要软件流水。

设计完全仿照 [gfx942/intra_wave](https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/gfx942/intra_wave)，代码从 v1 修改而来。

### 3.1 要藏哪些延迟

| 延迟来源 | 量级 | 藏法 |
|---|---|---|
| global load（L2/MALL/HBM） | 数百~上千 cycle | 提前 2 个 K 步发出（GR k+2） |
| 寄存器 → LDS | 等 global 数据到达 | 提前 1 个 K 步写入（LW k+1） |
| LDS → 寄存器 | ~100 cycle | 提前 1 个 region（~1024 cycle）读出（LR） |

### 3.2 为什么不是 LDS 双缓冲

- 256×256×64 一级就是 64 KB，LDS 正好放一级。
- 把 BK 降到 32 做双缓冲（8 KB × 8 = 64 KB）：每行只有 64 B，一条 `global_load_dwordx4` 要碰 16 条 cache line，TCP 处理时间加倍。参考实现实测 400 vs 476 TFLOPS，更慢。
- 结论：**保持 BK=64，用寄存器当第二级 buffer**，LDS 原地刷新。NVIDIA 上对应的做法是 TMA/cp.async + 多级 smem 环形缓冲，在 CDNA3 上做不到。

### 3.3 半 tile 切分与 4 个 region

A 沿 M 切成 `A_top / A_bot`，B 沿 N 切成 `B_left / B_right`，每个 16 KB，占一个 LDS slot（4 个 slot 正好 64 KB）。`(A 半, B 半)` 组成 C 的一个 128×128 象限；每个 wave 在每个象限各算 64×64。

一个 K 步 = 4 个 region，每个 region = 一个象限的 64 条 MFMA（约 1024 cycle），再穿插三种访存：

| region | DOT | LR（LDS→寄存器） | LW（寄存器→LDS） | GR（global→寄存器） |
|---|---|---|---|---|
| 0 | `C_tl = A_top·B_left` | `A_bot(k)` | `B_left(k+1)` | `B_left(k+2)` |
| 1 | `C_bl = A_bot·B_left` | `B_right(k)` | `A_top(k+1)` | `A_top(k+2)` |
| — | **barrier X** | | | |
| 2 | `C_tr = A_top·B_right` | `B_left(k+1)` | `A_bot(k+1)` | `A_bot(k+2)` |
| 3 | `C_br = A_bot·B_right` | `A_top(k+1)` | `B_right(k+1)` | `B_right(k+2)` |
| — | **barrier Y** | | | |

- LR 读的总是**下一个** region 要用的操作数，所以 LDS 延迟有一个 region 的时间来藏。
- 相邻 region 共享一个操作数：`b_left` 用于 R0、R1，`a_top` 用于 R0、R2，依此类推。
- 某个 slot 读完以后，过一个 region 就重写；重写以后，再过一个 region 才被读。所以**单级 LDS 足够**。
- 流水深度：GR(k+2) → LW(k+1) 隔一整个 K 步（约 4096 cycle），用来藏 HBM/L2 延迟；LR(k+1) → DOT(k+1) 隔一个 region（约 1024 cycle），用来藏 LDS 延迟。这和 gfx950 双缓冲的深度相同。

### 3.4 正确性：为什么每 K 步只要 2 个 barrier

LDS 是 4 个 wave 共享的：一个 wave 写的数据要被其它 wave 读，所以同一 slot 的读和写之间必须隔一个 barrier。逐个 slot 检查：

| slot | 上次读（WAR） | 写 | 下次读（RAW） | 隔开它们的 barrier |
|---|---|---|---|---|
| B_left | 上一 K 步 R2 | R0 | R2 | Y（上一步末尾） / X |
| A_top | 上一 K 步 R3 | R1 | R3 | Y / X |
| A_bot | R0 | R2 | 下一 K 步 R0 | X / Y |
| B_right | R1 | R3 | 下一 K 步 R1 | X / Y |

每一对读写都恰好跨过 X 或 Y。prologue 末尾还需要一个 barrier：prologue 里读了 `b_left(0)`、`a_top(0)`，要等它们读完，R0 才能重写 B_left。

`gpu.barrier()` 会先等本 wave 的 LDS 操作完成（`s_waitcnt lgkmcnt(0)`），再发 `s_barrier`。

### 3.5 代码结构

```python
def kstep(acc, a_top, b_left, g, kt2, last):          # last: 0 稳态 / 1 不再 GR / 2 不再 GR、LW
    g_bl, g_at, g_ab, g_br = g
    acc[0] = dot(acc[0], a_top, b_left)                # region 0
    a_bot = lr(lds_a, 1, wm)                           #   LR A_bot(k)
    lw(lds_b, 0, g_bl)                                 #   LW B_left(k+1)
    g_bl = gr(B, 0, kt2)                               #   GR B_left(k+2)
    interleave(8, 4, 4)
    acc[1] = dot(acc[1], a_bot, b_left)                # region 1 ...
    ...
    gpu.barrier()                                      # X
    ...                                                # region 2, 3
    gpu.barrier()                                      # Y
    return acc, a_top, b_left, [g_bl, g_at, g_ab, g_br]

# prologue: tile0 -> LDS; tile1 -> 寄存器（顺序与循环一致）; LR b_left(0), a_top(0)
for k, st in range(fx.Index(0), fx.Index(NK - 2), fx.Index(1), init=state0):
    acc, a_top, b_left, g = unpack(st)                 # loop-carried：64 累加器 + 下一步操作数 + 16 暂存
    acc, a_top, b_left, g = kstep(acc, a_top, b_left, g, fx.Int32(k) + 2, 0)
    results = yield ...
kstep(..., last=1)                                     # peel：LW 最后一块，不再 GR
kstep(..., last=2)                                     # drain：只做 DOT
```

### 3.6 寄存器预算（每 lane，1 wave/SIMD）

| 文件 | 存放 | 个数 |
|---|---|---|
| AGPR 256 | C 累加器：4 象限 × 64×64 f32 / 64 lane | 256（满） |
| VGPR 256 | DOT 操作数：4 个半 tile × 每 wave 64×64 bf16 | 4 × 32 = 128 |
| | GR 暂存：4 个半 tile × 128×64 bf16 / 256 lane | 4 × 16 = 64 |
| | 地址、循环变量 | 其余 |

编译结果：`.vgpr_count 512`（含 AGPR 256），`.vgpr_spill_count 28`，**spill 只出现在循环外的 peel 段**；循环内没有 scratch。LLVM 会把一部分操作数放进 AGPR、一部分累加器放进 VGPR（gfx90a 以后 MFMA 可以直接读写 AGPR）。

这 64 个暂存 VGPR 是“经寄存器中转”的代价。NVIDIA Hopper 的 TMA 直接写 smem、wgmma 直接读 smem，所以没有这部分寄存器压力。

### 3.7 指令调度：`sched_group_barrier`

数据依赖只保证正确，不保证穿插：LLVM 可能把一个 region 的访存挤在一起。用 `rocdl.sched_group_barrier` 告诉调度器每个 region 的发射配比：

```python
def interleave(n_dsrd, n_dswr, n_vmem):
    for _ in range_constexpr(n_dsrd):
        rocdl.sched_mfma(2); rocdl.sched_dsrd(1)       # 每条 ds_read 配 2 条 MFMA
    for _ in range_constexpr(n_dswr):
        rocdl.sched_mfma(2); rocdl.sched_dswr(1)       # 每条 ds_write 配 2 条 MFMA
    for _ in range_constexpr(n_vmem):
        rocdl.sched_mfma(1); rocdl.sched_vmem(1)
    rocdl.sched_mfma(64 - 2 * n_dsrd - 2 * n_dswr - n_vmem)
```

- 为什么每次 LDS 访问配 2 条 MFMA：LDS 带宽 128 B/clk 由 4 个 SIMD 共享，一条 b128 访问（1 KB）平均每个 SIMD 占 32 cycle，正好是 2 条 16 cycle 的 MFMA。
- 编译后的循环体：256 MFMA、32 `ds_read_b128`、16 `ds_write_b128`、16 `global_load_dwordx4`、2 `s_barrier`。每次 ds_write 前是 `s_waitcnt vmcnt(15/14/13/12)`，也就是 16 条 global load 在途，每次只等最老的那几条。这正是 2 个 K 步的预取深度。
- prologue 的 GR 顺序必须和循环一致（B_left, A_top, A_bot, B_right）。否则 `SIInsertWaitcnts` 在循环入口合并两条路径的计数时会变保守，`vmcnt(15)` 会塌成 `vmcnt(3)` 甚至更小，流水被排空。

### 3.8 剩下的瓶颈

参考实现的消融实验：去掉循环里的 16 条 `ds_write_b128` 后，MFMA 效率从约 91% 升到 98%。一条 `ds_write_b128` 发射约需 20 cycle（4 地址 + 16 数据），而一条 MFMA 只有 16 cycle，所以一条 MFMA 盖不住一条 ds_write。这是 CDNA3“经寄存器中转”的固有成本，单个指令流里排不掉。

---

## 4. 结果

`bench.py`，单卡 MI325X（空闲时），HIP event 计时取 20 次中位数；计数器为 `rocprofv3 --pmc` 每次 dispatch 的值（`slides/measure.sh` 自动采集，原始数据在 `slides/data/`）：

| 版本 | 时间 (us) | TFLOPS | 相对上一版 | L2 命中 | L2 之外读取 | LDS bank 冲突 |
|---|---|---|---|---|---|---|
| v0 64×64 | 8051 | 162 | | | | |
| v0 128×128 | 3863 | 338 | 2.08x | 72.0% | 5.59 GB | 0 |
| v1 | 2965 | 440 | 1.30x | 74.7% | 1.24 GB | 0 |
| v2（不加 `sched_group_barrier`） | 2259 | 578 | 1.31x | | | |
| v2 | 2013 | **649** | 1.12x | 72.6% | 1.31 GB | 0 |
| torch.matmul（hipBLASLt，bf16 输出） | 2061 | 633 | | | | |

- **v0 64→128**：寄存器层强度 32 → 64 flop/B，性能约翻倍，说明 v0 受访存限制。
- **v1**：L2 之外（MALL/HBM）的读取从 5.59 GB 降到 1.24 GB；A+B 本身只有 0.29 GB，也就是每字节平均被读 4.2 次（v0 是 19 次）。这就是 CTA 级共享的效果。剩下的瓶颈是串行等待。
- **v2**：流水把访存藏到 MFMA 后面，比 v1 快 47%。其中 `sched_group_barrier` 让访存均匀穿插，单这一项贡献 12%。最终略高于 hipBLASLt（+2.4%），约为 2.1 GHz 峰值的 50%。满载时频率会低于 2.1 GHz，所以按实际频率算的 MFMA 效率更高。
- 外存流量和 L2 命中率在 v1、v2 之间基本不变：v2 的收益完全来自延迟隐藏，而不是更少的数据。

## 5. 下一步

- **inter_wave**：8 wave/CTA，每 SIMD 2 个 wave 分两组乒乓，一组在 MFMA 上时另一组做访存，把 ds_write 从关键路径上拿掉（参考实现约 99.7% MFMA 效率）。
- **XCD 感知的 CTA 重映射 + GROUP_SIZE_M**：CTA 按 id 轮流分到 8 个 XCD，每个 XCD 有独立的 L2。重排 CTA id，让同一 XCD 上的 CTA 读相同的 A 行和 B 行。
- **epilogue 加宽**：现在每个 lane 4 个标量 store。改用转置的 MFMA 布局（`transposed=True`），每个 lane 就持有同一行的 4 个连续 N，可以用 dwordx4。
- **buffer_load**：利用硬件越界检查，支持任意 M/N/K。
- **频率**：750 W 功耗墙下 MFMA 密集时会降频，比较 TFLOPS 时要考虑实际频率。
